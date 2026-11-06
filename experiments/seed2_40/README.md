# Same 40 tasks, training seed 2

40 fixed tasks = the original 20 plus the independently sampled additional 20. No new sampling. Four variants, 1500 steps, seed 2 only: 160 runs. Same model/training controls as the additional20 package, apart from the training seed. Existing W&B project; `sample40` and `seed2-rerun` tags and distinct names.

Copy this directory to `experiments/seed2_40` in ARC-fafo. From its repository root:

```bash
sbatch experiments/seed2_40/submit.slurm
```

Array 0..159, one concurrent run, 12-hour limit per run, A40/gpu/normal/vf38, existing email alerts. Do not submit the older additional20 array as well unless you also intend to run that seed-0 batch.

Allocated GPU session, one run:

```bash
module load miniforge3
mamba activate SymARC
python experiments/seed2_40/run_job.py --index 0
```

Add `--dry-run` to print the command without training. The manifest maps all indices to task, variant and seed. Configurations and the task cohorts are frozen in this folder. Checkpoint directories use seed_2. No job is submitted by the preparation script.
