#!/bin/bash
set -euo pipefail
[[ -f src/train.py ]] || { printf 'Run from ARC-fafo root.\n' >&2; exit 1; }
d4_job=$(sbatch --parsable experiments/seed2_d4conv_priority/submit.slurm)
d4_job=${d4_job%%;*}
printf 'D4-conv array submitted: %s\n' "$d4_job"
follow_job=$(sbatch --parsable --dependency="afterany:${d4_job}" experiments/seed2_original_d4_remaining/submit.slurm)
follow_job=${follow_job%%;*}
printf 'Original + D4-sharing array submitted: %s (after D4-conv array %s finishes)\n' "$follow_job" "$d4_job"
printf 'LSE was not submitted. Check failed/timeout D4 elements separately; afterany does not require success.\n'
