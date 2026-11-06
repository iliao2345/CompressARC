#!/bin/bash -l
#SBATCH --job-name=arc-seed2-40
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-159%1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --output=slurm-seed2-40-%A_%a.out
#SBATCH --error=slurm-seed2-40-%A_%a.err
set -euo pipefail
# Submit from the ARC-fafo root; ignore any inherited stale submit directory.
if [[ ! -f src/train.py ]]; then
    printf 'Submit from the ARC-fafo repository root.\n' >&2
    exit 1
fi
module load miniforge3
mamba activate SymARC
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
srun python experiments/seed2_40/run_job.py --index "$SLURM_ARRAY_TASK_ID"
