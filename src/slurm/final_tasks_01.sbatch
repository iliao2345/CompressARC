#!/bin/bash -l
#SBATCH --job-name=arc-tasks_01
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-59%1
#SBATCH --output=slurm-tasks_01-%A_%a.out
#SBATCH --error=slurm-tasks_01-%A_%a.err

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
module load miniforge3
mamba activate SymARC
PYTHON_BIN="$(command -v python)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
TASKS=("3eda0437" "0dfd9992" "95990924" "7f4411dc" "6f8cd79b")
TASK_ARGS=()
for task in "${TASKS[@]}"; do
    TASK_ARGS+=(--task "$task")
done
OUTPUT_ROOT="${ARC_OUTPUT_ROOT:-outputs/final_ablation/tasks_01/${SLURM_ARRAY_JOB_ID}}"
"$PYTHON_BIN" -c 'import sys, torch; print(sys.executable, torch.__version__); assert torch.cuda.is_available(), "No CUDA GPU visible"; print(torch.cuda.get_device_name(0))'
srun "$PYTHON_BIN" src/run_training_timing.py --mode train \
    --config src/config/models/final_original_20.yaml \
    --config src/config/models/final_projected_conv_20.yaml \
    --config src/config/models/final_projected_d4_20.yaml \
    --config src/config/models/final_projected_lse_20.yaml \
    "${TASK_ARGS[@]}" \
    --shard-count 60 --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --set training.device=cuda:0 \
    --save-checkpoints --output-dir "$OUTPUT_ROOT"
