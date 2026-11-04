"""Run the remaining frozen jobs of a prepared experiment folder, skipping done ones.

Works on any box with CUDA: Slurm allocation, Colab, Kaggle. Idempotent - re-running
after a disconnect picks up where it stopped.

    python experiments/run_remaining.py --folder experiments/seed2_finish_optimised \
        --device cuda:0 --cuda-graph

    # inspect first
    python experiments/run_remaining.py --folder experiments/seed2_finish_optimised --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_repo(folder):
    for candidate in [Path.cwd(), *folder.parents]:
        if (candidate / "src/train.py").is_file():
            return candidate
    raise SystemExit("Run from the ARC-fafo repository root (src/train.py not found)")


def finished(checkpoint, steps):
    if not checkpoint.is_file():
        return False
    try:
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return int(payload.get("completed_steps", -1)) == steps
    except Exception:  # noqa: BLE001 - a corrupt file should be retrained, not skipped
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", action="append", help="only these arms (repeatable)")
    parser.add_argument("--limit", type=int, help="run at most N jobs")
    parser.add_argument("--cuda-graph", action="store_true", help="add training.cuda_graph=true")
    parser.add_argument("--no-wandb", action="store_true",
                        help="disable W&B logging (needed on Colab/Kaggle without login)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failure")
    args = parser.parse_args()

    folder = args.folder.resolve()
    repo = find_repo(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    steps = manifest.get("iterations", 1500)
    jobs = manifest["jobs"]
    if args.arm:
        jobs = [job for job in jobs if job["arm"] in set(args.arm)]
    if args.limit:
        jobs = jobs[: args.limit]

    logs = folder / "logs"
    logs.mkdir(exist_ok=True)
    ran = skipped = failed = 0
    failures = []
    started = time.monotonic()

    for position, job in enumerate(jobs, 1):
        out = repo / "outputs" / folder.name / "checkpoints" / "manual" / job["task"] / job["arm"] / f"seed_{job.get('seed', 0)}"
        if finished(out / f"{job['task']}.pt", steps):
            skipped += 1
            continue
        command = [sys.executable, str(repo / "src/train.py"),
                   "--config", str(folder / job["config"]),
                   "--task", job["task"],
                   "--set", f"training.device={args.device}",
                   "--save-checkpoints", "--output-dir", str(out)]
        if args.cuda_graph:
            command += ["--set", "training.cuda_graph=true", "--set", "training.step_timing=wall"]
        if args.no_wandb:
            command += ["--set", "logging.wandb=false"]
        print(f"[{position}/{len(jobs)}] {job['task']} {job['arm']}", flush=True)
        if args.dry_run:
            print("   ", " ".join(command), flush=True)
            continue
        began = time.monotonic()
        with (logs / f"{job['task']}-{job['arm']}.log").open("w") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        elapsed = (time.monotonic() - began) / 60
        if result.returncode == 0:
            ran += 1
            print(f"    ok in {elapsed:.1f} min", flush=True)
        else:
            failed += 1
            failures.append((job["task"], job["arm"], result.returncode))
            print(f"    FAILED rc={result.returncode} after {elapsed:.1f} min; see {logs.name}/", flush=True)
            if not args.keep_going:
                break

    total = (time.monotonic() - started) / 60
    print(f"\nran {ran}, skipped {skipped} (already complete), failed {failed}, "
          f"{total:.1f} min elapsed")
    for task, arm, code in failures:
        print(f"  failed: {task} {arm} rc={code}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
