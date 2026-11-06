# Finish LSE seed 2 first

This frozen snapshot contains the five unfinished LSE tasks. The generator refreshes the
snapshot against W&B and excludes full completed and active runs, regardless of accuracy.

The model exactly matches the existing projected_lse_optimised arm: learned LSE
cummax_implementation=optimised, original shift and direction sharing, tau 0.1.
This is the existing PyTorch learned LSE backend, not hard cummax. CUDA Graphs are
disabled because previous attempts stopped around the capture boundary at step 100;
that timing alone does not establish the exact failure cause. No other model or
optimiser recipe changes. Seed 2, training split, 1500 steps. No saved weights.

Submit from repo root after cancelling obsolete pending submissions:

```bash
scancel --state=PENDING 61904226
bash experiments/submit_seed2_final_lse_first.sh
```

The helper refreshes W&B before submission and queues C4 with afterany on the LSE
array. Thus C4 starts after LSE exits, even if an LSE task fails. Each array allows
four concurrent jobs, one GPU per job, generic gpu partition/normal QoS/vf38,
4 CPUs, 16 GiB memory, 12 hours, existing email alerts. This prepares/submits Slurm
jobs on M3; no remote Slurm access is available from the local Codex workspace.

C4 retains the existing shift=triton C4 arm and original other operators. The new
C4 script calls its own worker. The old C4 submission script's mistaken call to the
LSE worker has also been corrected. Check squeue for any other old arrays before
submission to avoid duplicates that have not connected to W&B yet.
