#!/bin/bash -l
#SBATCH --job-name=arc-pilot-variants
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%2
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the ARC-fafo repository root}"
if [[ ! -f src/run_training_timing.py ]]; then
    printf 'Submit from the ARC-fafo repository root.\n' >&2
    exit 1
fi
module load miniforge3
mamba activate SymARC
PYTHON_BIN="$(command -v python)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# Prevent different array submissions from overwriting one another's weights.
OUTPUT_ROOT="${ARC_OUTPUT_ROOT:-outputs/m3_variants/${SLURM_ARRAY_JOB_ID}}"
"$PYTHON_BIN" -c 'import sys, torch; print(sys.executable, torch.__version__); assert torch.cuda.is_available(), "No CUDA GPU visible in this allocation"; print(torch.cuda.get_device_name(0))'
srun "$PYTHON_BIN" src/run_training_timing.py --mode train \
    --config src/config/models/conv_d4_cummax_pilot_8_corrected.yaml \
    --config src/config/models/all_projected_triton_pilot_8_corrected.yaml \
    --shard-count 16 --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --save-checkpoints --output-dir "$OUTPUT_ROOT" "$@"
