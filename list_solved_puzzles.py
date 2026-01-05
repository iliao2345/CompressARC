import json
import argparse

import torch
import numpy as np

import preprocessing
from plot_accuracy import ValueSortedDict


def probe_solutions(solutions_file, split, iteration_num, task_nums=None):
    """
    Probe which tasks were solved at a given iteration and at which guess number.

    Args:
        solutions_file (str): Path to the NPZ file containing solution logs
        split (str): The split name ('training' or 'evaluation')
        iteration_num (int): The iteration number to check (0-indexed)
        task_nums (list, optional): List of task numbers to check. If None, uses all tasks.

    Returns:
        list: List of tuples (task_id, task_name, guess_number) for solved tasks
    """
    # Load the solution data
    stored_data = np.load(solutions_file, allow_pickle=True)
    solution_contribution_logs = stored_data['solution_contribution_logs']

    n_tasks = len(solution_contribution_logs)

    # Load tasks to get ground truth solution hashes and task names
    if task_nums is None:
        task_nums = list(range(n_tasks))

    tasks = preprocessing.preprocess_tasks(split, task_nums)
    true_solution_hashes = [task.solution_hash for task in tasks]
    task_names = [task.task_name for task in tasks]

    # Check which tasks were solved at the given iteration
    solved_tasks = []

    for task_num in range(n_tasks):
        true_hash = true_solution_hashes[task_num] >> 16
        solution_scores = ValueSortedDict()

        # Build up the solution scores up to the specified iteration
        for iter_idx in range(min(iteration_num + 1, len(solution_contribution_logs[task_num]))):
            for i in range(2):
                hashed, score = solution_contribution_logs[task_num][iter_idx][i]
                hashed = int(hashed) >> 16
                original_score = torch.tensor(solution_scores.get(hashed, default=-10000))
                score = torch.tensor(score)
                new_score = float(torch.logaddexp(score, original_score))
                solution_scores.insert(hashed, new_score)

        # Find the rank of the true solution
        solution_index = solution_scores.find_key(true_hash)
        if solution_index != -1:
            # Convert to guess number (1-indexed, ranked from best to worst)
            guess_number = len(solution_scores.sorted_list) - solution_index
            solved_tasks.append((task_num, task_names[task_num], guess_number))

    return solved_tasks


def main():
    parser = argparse.ArgumentParser(
        description='Probe which tasks were solved and at which guess number'
    )
    parser.add_argument(
        'solutions_file',
        type=str,
        help='Path to the solutions NPZ file (e.g., predictions_training.npz)'
    )
    parser.add_argument(
        'split',
        type=str,
        choices=['training', 'evaluation'],
        help='Dataset split to analyze'
    )
    parser.add_argument(
        'iteration',
        type=int,
        help='Iteration number to check (0-indexed)'
    )

    args = parser.parse_args()

    # Probe the solutions
    solved_tasks = probe_solutions(args.solutions_file, args.split, args.iteration)

    # Calculate pass@2 (solved with 2 or fewer guesses)
    solved_with_2_guesses = sum(1 for _, _, guess_number in solved_tasks if guess_number <= 2)

    # Print results
    print(f"\nTasks solved at iteration {args.iteration}:")
    print(f"Total solved: {len(solved_tasks)}")
    print(f"Solved with 2 or fewer guesses: {solved_with_2_guesses}\n")

    if solved_tasks:
        print("Task #  | Task ID                                  | Number of Required Guesses")
        print("-" * 75)
        for task_id, task_name, guess_number in solved_tasks:
            print(f"{task_id:7d} | {task_name:40s} | {guess_number:12d}")
    else:
        print("No tasks were solved at this iteration.")


if __name__ == "__main__":
    main()
