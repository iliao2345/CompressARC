#!/bin/bash
set -euo pipefail
[[ -f src/train.py ]] || { printf 'Run from ARC-fafo root.\n' >&2; exit 1; }
python experiments/prepare_seed2_final_remaining.py
LSE_COUNT=$(python -c 'import json; print(json.load(open("experiments/seed2_final_remaining_counts.json"))["lse"])')
C4_COUNT=$(python -c 'import json; print(json.load(open("experiments/seed2_final_remaining_counts.json"))["c4"])')
LSE_JOB=""
if (( LSE_COUNT > 0 )); then
  LSE_JOB=$(sbatch --parsable experiments/seed2_final_lse_remaining/submit.slurm)
  LSE_JOB=${LSE_JOB%%;*}
  printf 'LSE first: %s (%s tasks)\n' "$LSE_JOB" "$LSE_COUNT"
fi
if (( C4_COUNT > 0 )); then
  if [[ -n "$LSE_JOB" ]]; then
    sbatch --dependency="afterany:$LSE_JOB" experiments/seed2_final_c4_remaining/submit.slurm
  else
    sbatch experiments/seed2_final_c4_remaining/submit.slurm
  fi
fi
