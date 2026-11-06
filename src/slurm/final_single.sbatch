#!/bin/bash -l
#SBATCH --job-name=arc-single-test
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-single-%j.out
#SBATCH --error=slurm-single-%j.err

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
TASK="${1:-74dd1130}"
VARIANT="${2:-original}"
SEED="${3:-0}"
case "$VARIANT" in
    original) CONFIG=final_original_20 ;;
    projected-conv) CONFIG=final_projected_conv_20 ;;
    projected-d4) CONFIG=final_projected_d4_20 ;;
    projected-lse) CONFIG=final_projected_lse_20 ;;
    *) printf 'Variant must be original, projected-conv, projected-d4 or projected-lse.\n' >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'Seed must be a nonnegative integer.\n' >&2
    exit 2
fi
module load miniforge3
mamba activate SymARC
PYTHON_BIN="$(command -v python)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
OUTPUT_ROOT="${ARC_OUTPUT_ROOT:-outputs/final_single/${SLURM_JOB_ID}/${CONFIG}/seed_${SEED}}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU visible"; print(torch.cuda.get_device_name(0))'
srun "$PYTHON_BIN" src/train.py \
    --config "src/config/models/${CONFIG}.yaml" --task "$TASK" \
    --set "training.seed=$SEED" --set training.device=cuda:0 \
    --save-checkpoints --output-dir "$OUTPUT_ROOT"
