"""Analyze legacy predictions_*.npz history files with pass-at-n accuracy."""

import argparse
import bisect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ValueSortedDict:
    """Dictionary with keys ordered by ascending value."""

    def __init__(self):
        self.sorted_list = []
        self.key_to_value = {}

    def insert(self, key, value):
        if key in self.key_to_value:
            self.remove(key)
        bisect.insort(self.sorted_list, (value, key))
        self.key_to_value[key] = value

    def get(self, key, default=0):
        return self.key_to_value.get(key, default)

    def remove(self, key):
        if key not in self.key_to_value:
            return
        value = self.key_to_value.pop(key)
        index = bisect.bisect_left(self.sorted_list, (value, key))
        if index < len(self.sorted_list) and self.sorted_list[index] == (value, key):
            self.sorted_list.pop(index)

    def items(self):
        return [(key, value) for value, key in self.sorted_list]

    def get_by_index(self, index):
        if -len(self.sorted_list) <= index < len(self.sorted_list):
            value, key = self.sorted_list[index]
            return key, value
        raise IndexError("Index out of range")

    def find_key(self, key):
        if key not in self.key_to_value:
            return -1
        return [sorted_key for _, sorted_key in self.sorted_list].index(key)


def get_accuracy(true_solution_hashes, fname="predictions.npz"):
    """Return pass-at-n values at each step from a legacy prediction history."""
    with np.load(fname, allow_pickle=True) as stored_data:
        solution_logs = stored_data["solution_contribution_logs"]

    n_tasks = len(solution_logs)
    if n_tasks == 0:
        raise ValueError("The predictions file contains no task histories.")
    if len(true_solution_hashes) != n_tasks:
        raise ValueError(
            f"Found histories for {n_tasks} tasks but {len(true_solution_hashes)} "
            "ground-truth solutions."
        )

    n_iterations = len(solution_logs[0])
    n_attempts = max(2 * n_iterations, 1)
    pass_at_n = np.zeros((n_iterations, n_attempts), dtype=np.float64)

    for task_num, history in enumerate(solution_logs):
        true_hash = int(true_solution_hashes[task_num]) >> 16
        solution_scores = ValueSortedDict()
        for iteration_num, iteration in enumerate(history):
            for hashed, score in iteration:
                hashed = int(hashed) >> 16
                updated_score = np.logaddexp(
                    float(score),
                    solution_scores.get(hashed, -10000.0),
                )
                solution_scores.insert(hashed, float(updated_score))
            solution_index = solution_scores.find_key(true_hash)
            if solution_index != -1:
                guess_rank = len(solution_scores.sorted_list) - 1 - solution_index
                if guess_rank < n_attempts:
                    pass_at_n[iteration_num, guess_rank] += 1

    return np.cumsum(pass_at_n, axis=1) / n_tasks


def plot_accuracy(pass_at_n, output_path="accuracy_curve_at_n.png"):
    """Save a plot of selected pass-at-n curves available in the data."""
    n_iterations, n_attempts = pass_at_n.shape
    figure, axis = plt.subplots()
    for attempts, color in (
        (1, "r"),
        (2, "k"),
        (5, "g"),
        (10, "b"),
        (100, "c"),
        (1000, "m"),
        (2000, "y"),
    ):
        if attempts <= n_attempts:
            axis.plot(
                np.arange(n_iterations),
                pass_at_n[:, attempts - 1],
                color + "-",
                label=f"pass@{attempts}",
            )
    axis.legend()
    axis.set_xlabel("step")
    axis.set_ylabel("accuracy")
    axis.grid(which="both", color="0.65", linewidth=0.8, linestyle="-")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return Path(output_path)


def print_accuracy(pass_at_n):
    """Print selected pass-at-n values at steps present in the history."""
    n_iterations, n_attempts = pass_at_n.shape
    iteration_points = [100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 2000]
    attempt_points = [1, 2, 5, 10, 100, 1000]
    for iteration in iteration_points:
        if iteration > n_iterations:
            continue
        for attempts in attempt_points:
            if attempts <= n_attempts:
                value = float(pass_at_n[iteration - 1, attempts - 1])
                print(
                    f"iteration {iteration}, {attempts} attempts: "
                    f"accuracy = {value}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solutions_file", type=Path, help="Legacy predictions_*.npz file")
    parser.add_argument("split", choices=("training", "evaluation"))
    parser.add_argument("--output", type=Path, default=Path("accuracy_curve_at_n.png"))
    args = parser.parse_args()

    from preprocessing import preprocess_tasks

    with np.load(args.solutions_file, allow_pickle=True) as stored_data:
        number_of_tasks = len(stored_data["solution_contribution_logs"])
    tasks = preprocess_tasks(args.split, range(number_of_tasks))
    accuracy = get_accuracy(
        [task.solution_hash for task in tasks],
        fname=args.solutions_file,
    )
    print(f"Plotting accuracy for {len(tasks)} tasks.")
    print_accuracy(accuracy)
    print(f"Saved {plot_accuracy(accuracy, args.output)}")


if __name__ == "__main__":
    main()