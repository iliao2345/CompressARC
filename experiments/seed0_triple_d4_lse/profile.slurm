#!/bin/bash -l
#SBATCH --job-name=arc-triple-latency
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --output=slurm-arc-triple-latency-%A_%a.out
#SBATCH --error=slurm-arc-triple-latency-%A_%a.err
set -euo pipefail
[[ -f src/train.py ]] || { printf 'Submit from ARC-fafo root.\n' >&2; exit 1; }
module load miniforge3
mamba activate SymARC
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
srun python -m pytest src/tests/correctness/training/test_triple_d4_lse_model.py -q
srun python experiments/seed0_triple_d4_lse/profile.py --index "$SLURM_ARRAY_TASK_ID"
