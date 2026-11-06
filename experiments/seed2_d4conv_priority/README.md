# Remaining optimised seed-2 runs

40 pending runs over the same fixed 40 tasks. Counts:

{
  "projected_conv_d4": {
    "completed": 0,
    "active": 0,
    "remaining": 40
  }
}

Complete means W&B finished and final logged train_step >=1499, matched by task,
training split, seed2 and variant tag. Duplicate run IDs are retained as evidence
but each task is counted once. Accuracy is not used. Only the variants listed in the counts above are scheduled. Active runs are recorded separately and excluded.

New runs use compatible optimised settings, 1500 steps, seed2, ordinary metrics
and candidate logging. LSE retains its learned optimised operator; projected D4
sharing retains its d4 operator; hard-cummax variants use grouped hard Triton.
No weights/checkpoints are saved or uploaded. This restarts unfinished runs from
initialisation, not deleted checkpoints. Legacy completed runs keep their older
backends; report that provenance when combining results for the paper.

Submit from repository root: `sbatch experiments/seed2_d4conv_priority/submit.slurm`.
One GPU per array element; at most two elements run concurrently. Any GPU in the
gpu partition; normal/vf38, 4 CPUs, 16 GB RAM, 12 hours per run, existing email.
Keep the old seed2_40 array held to avoid launching overlapping pending jobs.
No Slurm submission or cancellation was performed by the preparation script.

Dry-run one command: `python experiments/seed2_d4conv_priority/run_job.py --index 0 --dry-run`.
Inside an existing GPU allocation, omit `--dry-run` to run that one pair.
Use a fresh package via prepare_seed2_40_optimised.py if the completion state changes;
the manifest is a snapshot, not a live scheduler. Execution JSON stores the code
commit and config digest locally. Source configs and both cohorts are preserved.
