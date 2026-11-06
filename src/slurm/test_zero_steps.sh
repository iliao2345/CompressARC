#!/bin/bash -l
#SBATCH --job-name=arc-zero-steps
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-zero-steps-%j.out
#SBATCH --error=slurm-zero-steps-%j.err

set -euo pipefail
# Interactive allocations can retain a stale SLURM_SUBMIT_DIR.
START_DIR="$PWD"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT=""
for candidate in "$START_DIR" "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR/../.." "$SCRIPT_DIR"; do
    if [[ -n "$candidate" && -f "$candidate/src/run_training_timing.py" ]]; then
        REPO_ROOT="$(cd -- "$candidate" && pwd -P)"
        break
    fi
done
if [[ -z "$REPO_ROOT" ]]; then
    printf 'Cannot locate src/run_training_timing.py.\nCurrent directory: %s\nSLURM_SUBMIT_DIR: %s\nScript directory: %s\n' \
        "$START_DIR" "${SLURM_SUBMIT_DIR:-<unset>}" "$SCRIPT_DIR" >&2
    printf 'Check that the updated ARC-fafo src/ directory is present on this machine.\n' >&2
    exit 1
fi
cd -- "$REPO_ROOT"
if [[ "${1:-}" == "--check-root" ]]; then
    printf 'Repository root: %s\n' "$REPO_ROOT"
    exit 0
fi
if [[ "${ARC_SKIP_ENV:-0}" != "1" ]]; then
    module load miniforge3
    mamba activate SymARC
fi
PYTHON_BIN="${ARC_PYTHON:-$(command -v python)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - "$@" <<'PY'
import argparse
import gc
from pathlib import Path
import random
import sys

import numpy as np
import torch
from config import load_config
from preprocessing import preprocess_tasks
from compressarc.model.model import ARCCompressor

parser = argparse.ArgumentParser(description="Zero optimizer steps; one no-grad forward per variant/seed")
parser.add_argument("--task", default="74dd1130", help="Small task from the selected 20")
parser.add_argument("--seed", type=int, action="append", help="Repeat to test selected seeds; default 0,1,2")
parser.add_argument("--device", default="cuda:0", help="Default cuda:0; cpu is available for local checks")
parser.add_argument("--init-only", action="store_true", help="Skip even the no-grad forward")
args = parser.parse_args()
seeds = args.seed if args.seed is not None else [0,1,2]
if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
    parser.error("Use distinct nonnegative seeds")
device = torch.device(args.device)
if device.type == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: no CUDA GPU visible; run inside an allocated GPU job")
    torch.cuda.set_device(device)
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
print("Python:", sys.executable, "Torch:", torch.__version__, flush=True)
print("ZERO optimizer updates; W&B/checkpoint/model uploads disabled", flush=True)
configs = [Path("src/config/models") / (name + ".yaml") for name in (
    "final_original_20", "final_projected_conv_20", "final_projected_d4_20", "final_projected_lse_20")]
# Preprocessing explicitly uses torch.get_default_device(); a context alone
# does not reliably update that global default.
torch.set_default_device(device)
expected_device = torch.get_default_device()
completed = 0
with torch.device(device):
    for path in configs:
        config = load_config(path)
        assert len(set(config.experiment.task_ids)) == 20
        assert list(config.experiment.seeds) == [0,1,2]
        if args.task not in config.experiment.task_ids:
            raise SystemExit(f"Task {args.task} is not in the final selected 20")
        config.logging.wandb = False
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            tasks = preprocess_tasks(config.training.split, [args.task],
                                     multitensor_constraints=config.model.multitensor_constraints)
            if len(tasks) != 1:
                raise SystemExit(f"FAIL: task not found: {args.task}")
            task = tasks.pop()
            model = ARCCompressor(task, config.model)
            placed = {"task.problem": task.problem, "task.masks": task.masks}
            if task.solution is not None:
                placed["task.solution"] = task.solution
            placed.update({f"weight[{i}]": parameter for i, parameter in enumerate(model.weights_list)})
            for kind in ("shift", "cummax", "direction_share"):
                layer = getattr(model, kind)
                if isinstance(layer, torch.nn.Module):
                    placed.update({f"{kind}.{name}": value for name, value in layer.named_buffers()})
            wrong = {name: str(value.device) for name, value in placed.items() if value.device != expected_device}
            if wrong:
                raise SystemExit(f"FAIL: expected all model/task tensors on {expected_device}; mismatches: {wrong}")
            del placed, wrong
            outputs = None
            if not args.init_only:
                with torch.no_grad():
                    outputs = model.forward()
                tensors = (*outputs[:3], *outputs[3])
                if not all(bool(torch.isfinite(value).all()) for value in tensors):
                    raise SystemExit(f"FAIL: nonfinite forward result: {path.stem}, seed {seed}")
                del tensors
            assert all(parameter.grad is None for parameter in model.weights_list)
            count = sum(parameter.numel() for parameter in model.weights_list)
            details = "initialization" if args.init_only else "initialization + finite no-grad forward"
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                details += f"; peak allocated {torch.cuda.max_memory_allocated(device)/2**20:.1f} MiB"
            print(f"PASS {path.stem} | task={args.task} | seed={seed} | params={count} | {details}", flush=True)
            completed += 1
            del task, model, outputs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
assert "wandb" not in sys.modules, "Unexpected W&B import during smoke test"
print(f"PASS: {completed} variant/seed checks; optimizer steps=0; W&B runs=0; uploads=0", flush=True)
print("Full plan remains 20 tasks x 4 variants x 3 seeds = 240 runs; none launched.", flush=True)
PY
