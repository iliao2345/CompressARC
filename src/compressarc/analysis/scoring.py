"""Score a submission against an ARC solutions JSON file."""

import argparse
import json
from pathlib import Path

DEFAULT_SOLUTIONS_FILE = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "arc-agi_training_solutions.json"
)


def score_submission(submission_file_name, solutions_file_name, include_task_scores=False) -> dict:
    """Score each submitted test pair against its ground-truth solution."""
    with open(submission_file_name, "r", encoding="utf-8") as file:
        submission = json.load(file)

    with open(solutions_file_name, "r", encoding="utf-8") as file:
        solutions = json.load(file)

    total_score = 0
    total_tasks = 0
    task_scores = {}

    for task_id, task_submission in submission.items():
        total_tasks += 1
        task_score = 0
        num_pairs = len(task_submission)

        for pair_index, pair_attempts in enumerate(task_submission):
            pair_correct = any(
                attempt == solutions[task_id][pair_index]
                for attempt in pair_attempts.values()
            )
            if pair_correct:
                task_score += 1

        task_score /= num_pairs
        total_score += task_score
        task_scores[task_id] = task_score

    result = {
        "total_score": total_score,
        "total_tasks_scored": total_tasks,
    }
    if include_task_scores:
        result["task_scores"] = task_scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to the submission JSON file")
    parser.add_argument(
        "--solutions",
        type=Path,
        default=DEFAULT_SOLUTIONS_FILE,
        help=f"Ground-truth solutions JSON (default: {DEFAULT_SOLUTIONS_FILE})",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-task scores from the output",
    )
    args = parser.parse_args()

    score = score_submission(
        args.submission,
        args.solutions,
        include_task_scores=not args.summary_only,
    )
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()