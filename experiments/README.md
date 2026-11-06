# Fixed-task ablation batches

These folders contain frozen task manifests, resolved model/training configurations, and M3 launchers. Preparation never submits jobs or reads model performance.

| Batch | Tasks | Training seed | Runs |
| --- | --- | --- | --- |
| additional20 | Uniform sample of 20 tasks excluding the original 20 | 0 | 80 |
| seed2_40 | Same original 20 plus the additional 20 | 2 | 160 |

Sampling uses seed 43 over sorted eligible training IDs. The original 20 were performance-selected; the additional 20 were sampled without performance filters. The 40-task union is therefore not a fully independent random sample. Report these cohorts separately as well as together.

From the ARC-fafo repository root:

```bash
sbatch experiments/additional20/submit.slurm
sbatch experiments/seed2_40/submit.slurm
```

Each array runs at most one job concurrently and allows 12 hours per job. Settings: A40, gpu partition, vf38 account, normal QoS; email notifications use the existing recipient. Four variants and 1500 steps, with only the designated layer varying. W&B retains the existing project with distinct names and batch tags. Checkpoints go under outputs/<batch>/checkpoints, not into these tracked experiment folders.

For a command preview (no training):

```bash
python experiments/seed2_40/run_job.py --index 159 --dry-run
```

All configs and manifests are already generated. Do not rerun preparation to substitute tasks after looking at results. The generators refuse to overwrite existing samples; source dataset hashes and the sample seed are recorded in the manifests. Generated configs are JSON, accepted by the normal OmegaConf loader.
