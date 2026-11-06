#!/bin/bash -l
#SBATCH --job-name=arc-d4conv-s2
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-39%2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --output=slurm-arc-d4conv-s2-%A_%a.out
#SBATCH --error=slurm-arc-d4conv-s2-%A_%a.err
set -euo pipefail
[[ -f src/train.py ]] || { printf 'Submit from ARC-fafo root.\n' >&2; exit 1; }
module load miniforge3
mamba activate SymARC
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
srun python experiments/seed2_d4conv_priority/run_job.py --index "$SLURM_ARRAY_TASK_ID"
