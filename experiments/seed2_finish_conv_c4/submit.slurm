#!/bin/bash -l
#SBATCH --job-name=arc-seed2-conv-c4
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-21%4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --output=slurm-seed2-finish-%A_%a.out
#SBATCH --error=slurm-seed2-finish-%A_%a.err
set -euo pipefail
if [[ ! -f src/train.py ]]; then
    printf 'Submit from the ARC-fafo repository root.\n' >&2
    exit 1
fi
module load miniforge3
mamba activate SymARC
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export ARC_CUDA_GRAPH=1   # CUDA Graph replay: speed only, parity gated in CUDA_GRAPH.md
srun python experiments/seed2_finish_conv_c4/run_job.py --index "$SLURM_ARRAY_TASK_ID"
