"""List tasks solved within a requested guess count from legacy NPZ history."""

import argparse
from pathlib import Path

import numpy as np

from .plot_accuracy import ValueSortedDict


def probe_solutions(solutions_file, split, iteration_num, task_nums=None):
    """Return tuples of (task index, task id, guess rank) solved by that step."""
    if iteration_num < 0:
        raise ValueError("iteration must be non-negative")

    with np.load(solutions_file, allow_pickle=True) as stored_data:
        solution_logs = stored_data["solution_contribution_logs"]

    task_nums = list(range(len(solution_logs))) if task_nums is None else list(task_nums)
    if len(task_nums) != len(solution_logs):
        raise ValueError(
            f"The NPZ has {len(solution_logs)} task histories but "
            f"{len(task_nums)} task indices were selected."
        )

    from preprocessing import preprocess_tasks

    tasks = preprocess_tasks(split, task_nums)
    if len(tasks) != len(solution_logs):
        raise ValueError("Could not match all NPZ histories to tasks in the selected split.")

    solved = []
    for task_index, (task, history) in enumerate(zip(tasks, solution_logs)):
        true_hash = int(task.solution_hash) >> 16
        solution_scores = ValueSortedDict()
        for iteration in history[: iteration_num + 1]:
            for hashed, score in iteration:
                hashed = int(hashed) >> 16
                solution_scores.insert(
                    hashed,
                    float(np.logaddexp(float(score), solution_scores.get(hashed, -10000.0))),
                )
        rank = solution_scores.find_key(true_hash)
        if rank != -1:
            guess_number = len(solution_scores.sorted_list) - rank
            solved.append((task_index, task.task_name, guess_number))
    return solved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solutions_file", type=Path, help="Legacy predictions_*.npz file")
    parser.add_argument("split", choices=("training", "evaluation"))
    parser.add_argument("iteration", type=int, help="Zero-based training step")
    parser.add_argument("--max-guesses", type=int, default=2)
    args = parser.parse_args()

    solved = probe_solutions(args.solutions_file, args.split, args.iteration)
    within_limit = sum(guesses <= args.max_guesses for _, _, guesses in solved)

    print(f"Tasks solved at iteration {args.iteration}:")
    print(f"Total solved: {len(solved)}")
    print(f"Solved with {args.max_guesses} or fewer guesses: {within_limit}")
    if solved:
        print()
        print("Task #  | Task ID                                  | Required guesses")
        print("-" * 75)
        for task_index, task_id, guesses in solved:
            print(f"{task_index:7d} | {task_id:40s} | {guesses:16d}")
    else:
        print("No tasks were solved at this iteration.")


if __name__ == "__main__":
    main()