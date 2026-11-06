# Final controlled ablation

Four variants use the same 20 task IDs, 1,500 optimizer updates, training seeds 0, 1 and 2,
strict tensor constraints, four layers, shared/grid widths 16/8, all other
projection widths, Adam settings, loss weights and annealing, and logging.
The 20 training tasks were sampled without replacement using seed 42 from the
132 tasks solved by the author's original CompressARC at step 1,500. This is a
comparison on that selected set, not a full-dataset solve-rate estimate.

| Variant | Shift | Direction sharing | Cummax |
|---|---|---|---|
| Control | Original primitives | Original primitives | Original primitives |
| Projected convolution only | Projected tied convolution | Original primitives | Original primitives |
| Projected D4 only | Original primitives | Projected D4 mixer | Original primitives |
| Projected LSE only | Original primitives | Original primitives | Projected PyTorch LSE (`optimised`, tau 0.1) |

Each replacement is applied at every model depth. Its own parameterization may
change parameter count; unrelated settings and operations stay identical.
Previous combined variants remain available for historical comparisons but are
excluded from this final experiment. No training jobs have been launched.

Preview the 240 task/variant/seed runs from the repository root:

```bash
python src/run_training_timing.py --mode train --dry-run
```

Run when ready:

```bash
python src/run_training_timing.py --mode train --set training.device=cuda:0
```

W&B is enabled in these four configs, targeting `arc_agi/compressarc-final-ablation`.
Run names identify the variant and training seed. Metrics, candidate tables, kernel/numerical
checks and final inference model artifacts use the current shared trainer.
Checkpoints are separated under `outputs/training/<config-name>/seed_<seed>/`.
The launcher accepts `--shard-count`, `--shard-index`, repeated `--task`, and
`--set` overrides; apply the same overrides to all variants for a controlled run.

## Tasks

- `3eda0437`
- `0dfd9992`
- `95990924`
- `7f4411dc`
- `6f8cd79b`
- `445eab21`
- `3bdb4ada`
- `36fdfd69`
- `d43fd935`
- `1c786137`
- `1bfc4729`
- `3aa6fb7a`
- `6cdd2623`
- `74dd1130`
- `f8c80d96`
- `623ea044`
- `d406998b`
- `6d75e8bb`
- `d9fac9be`
- `97999447`

The seed sweep is expanded by the batch launcher. Direct calls to `train.py`
run one seed. Use repeated `--seed` flags to override the sweep, for example
`--seed 0 --seed 1 --seed 2`; `--set training.seed=1` selects one seed unless
explicit `--seed` flags are supplied. Sharding distributes all 240 jobs exactly
once. Sampling seed 42 chooses the task subset; training seeds 0/1/2 control
model initialization and stochastic training independently on the same tasks.

## Slurm on A40

Four task-group arrays are ready under `src/slurm/final_tasks_*.sbatch`.
Each owns five tasks and runs all four variants and three seeds for those tasks.
See [submission instructions](slurm/FINAL_SLURM.md). Each has 60 elements, one
GPU per active element and one active element per array by default.

## W&B tags

Shared tags: `arc-agi`, `training`, `final-ablation`, `sample20`, `steps1500`,
`strict`. Each run also has `seed0`, `seed1` or `seed2`, and one variant tag:
`original`, `projected-conv`, `projected-d4` or `projected-lse`.

All final jobs now use A40 in the `gpu` partition, account `vf38`, QoS `normal`.
