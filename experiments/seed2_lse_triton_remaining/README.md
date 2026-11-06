# Projected LSE: learned Triton, seed2

23 unfinished tasks from the same frozen 40-task set. The 17 legacy completed
seed2 tasks are retained separately, not rerun or changed. This package uses
model.cummax_implementation=triton, the learned LSE in cummax/triton_op.py;
it does NOT use triton_hard or triton_hard_grouped. Tau, learned kernel shapes,
projection dimensions and initialisation remain unchanged. Common accelerations
match the benchmark: vectorised direction sharing, fused softmax/sharing norm,
fused Adam, direct numerical checks and explicit runtime device dispatch.

Start with ONE full task: 6d75e8bb, 1500 steps, seed2:

    sbatch experiments/seed2_lse_triton_remaining/submit_first.slurm

Inside an existing allocated GPU session, the equivalent command is:

    python experiments/seed2_lse_triton_remaining/run_job.py --index 0

Check completion, numerical health and timing before the rest. Do not gate the
remaining task selection on whether the first task was solved. Then submit:

    sbatch experiments/seed2_lse_triton_remaining/submit_remaining.slurm

The remaining array is indices1-22%2, one GPU each, at most two simultaneous.
Do NOT also submit submit.slurm after the first task: that schedules all23,
including index0, and would duplicate the first run. submit.slurm is available
only if choosing to schedule the whole pending set instead of a separate first run.

No model weights/checkpoints are saved or uploaded. All normal metrics and
candidate tables are retained. Effective settings and code commit are logged.
New tag: seed2-lse-triton-remaining; backend tag: cummax-triton-lse.
Existing seed0 D4 retest packages remain separate and have not been submitted.
No jobs were submitted by this preparation. Wait for runs to complete before
combining results; report legacy-vs-Triton provenance when retaining both.
