# Seed-2 priority submission

Use `bash experiments/submit_seed2_priority.sh` from ARC-fafo root after pulling.

1. `seed2_d4conv_priority`: 40 D4-conv runs, array 0-39%2.
2. `seed2_original_d4_remaining`: 22 Original + 23 projected D4-sharing runs,
   array 0-44%2. The helper gives this array an `afterany` dependency on the
   D4-conv array, so it starts only after every D4-conv element has terminated.
3. Projected LSE is deferred and is not submitted.

Every array element requests one GPU. At most two elements run concurrently;
this is an array submission, not one process looping through every task. Slurm
still decides availability and start times. `afterany` allows the followup even
if a D4-conv element failed or timed out; those failed tasks still need retrying.
The helper prints the submitted job IDs. No jobs have been submitted locally.

The same frozen 40 tasks, cohorts, training seed2 and 1500 iterations are kept.
Completion was audited from W&B: Original 18/40, D4-sharing 17/40, D4-conv 0/40.
Already finished tasks are skipped regardless of accuracy. Duplicate completed
runs count once. New jobs use compatible optimised settings, ordinary W&B logs,
effective-config/commit recording and no saved model/checkpoint weights.
Completed legacy runs retain their original execution backends in provenance.

Keep older seed2 arrays held to avoid overlap. Do not submit the earlier mixed
`seed2_40_optimised_remaining/submit.slurm`; it contains LSE and overlapping work.
For D4-conv only: `sbatch experiments/seed2_d4conv_priority/submit.slurm`.
Do not run both this manual submission and the helper, which would duplicate it.
