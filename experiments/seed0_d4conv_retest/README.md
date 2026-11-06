# Optimised seed-0 retest: projected_conv_d4

40 fresh runs, 1500 steps each, seed0. The same frozen pilot20 + additional20
cohorts are retained. Already completed legacy seed0 tasks are deliberately
rerun to test the optimised implementation. No task is selected by accuracy.

Submission: `sbatch experiments/seed0_d4conv_retest/submit.slurm` from ARC-fafo root.
Array 0-39%2; one GPU per element, at most two running concurrently.
Submitting both packages permits four concurrent GPU runs in total.

W&B name suffix: `projected_conv_d4_seed0_optimised_retest40-seed0`.
W&B tag: `seed0-optimised-retest`. Original completed runs are not changed or
removed. Keep legacy and retest results separate when assessing backend effects;
do not select the best run by outcome. No model/checkpoint weights are saved.
Effective config and code commit are logged; execution JSON is saved locally.

Dry-run: `python experiments/seed0_d4conv_retest/run_job.py --index 0 --dry-run`.
Omit `--dry-run` only inside an allocated GPU session to run that one task.
