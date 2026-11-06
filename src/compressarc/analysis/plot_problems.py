"""Render ARC puzzle grids from a CompressARC dataset split."""

import argparse
from pathlib import Path

from .visualization import plot_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("training", "evaluation", "test"),
        required=True,
    )
    parser.add_argument("--task", action="append", help="Task ID; may be repeated")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/problems"))
    args = parser.parse_args()

    from preprocessing import preprocess_tasks

    task_selection = args.task if args.task else range(10000)
    tasks = preprocess_tasks(args.split, task_selection)
    if not tasks:
        raise ValueError(f"No tasks found for split={args.split!r}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for task in tasks:
        figure = plot_problem(task, fname=False)
        output_path = args.output_dir / f"{task.task_name}_problem.png"
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(figure)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()