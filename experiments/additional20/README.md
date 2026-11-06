# Additional 20 training tasks

Sampling seed 43 by default; uniform sampling without replacement from all training tasks excluding the previous 20. No performance, difficulty, shape, or runtime filters. All four variants use the same tasks, training seed 0, and 1500 steps. This sample should be frozen before viewing its model results.

This directory is tracked at `experiments/additional20` in ARC-fafo. From that repository root:

```bash
sbatch experiments/additional20/submit.slurm
```

This is an 80-element array with one concurrent run and a 12-hour limit per run. Existing A40/gpu/normal/vf38 settings and email alerts are retained. W&B uses the existing project, with an `additional20` tag and distinct run names. No jobs are submitted by the preparation script.

For one selected job in an allocated GPU session:

```bash
python experiments/additional20/run_job.py --index 0
```

Print a command without training:

```bash
python experiments/additional20/run_job.py --index 0 --dry-run
```

Do not substitute tasks after inspecting outcomes. This samples training tasks, not an evaluation split. GPU and numeric runtime differences can still affect results.
