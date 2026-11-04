# Finish the seed-2 cells for the two incomplete arms

Only the missing (task, seed 2) cells are queued: 20 for the LSE arm and
22 for the C4-conv arm, 42 runs total. Everything else at seed 2 already exists in W&B.

## Arms

| arm | config | model | note |
|---|---|---|---|
| `projected_lse_optimised` | `configs/projected_lse_optimised.json` | `cummax: optimised` (the PyTorch LSE already used at seed 0) | completes the LSE column |
| `projected_conv_c4_triton` | `configs/projected_conv_c4_triton.json` | `shift: triton` (Triton C4 conv) | optimised C4 conv |
| `projected_conv_c4_reference` | `configs/projected_conv_c4_reference.json` | `shift: tied_conv` (PyTorch C4 conv) | **not queued**; swap in if you want the implementation to match seed 0 exactly |

All arms: `iterations=1500`, `split=training`, `seed=2`, same model/training controls as the
frozen `final_*_20.yaml` bases.

## Submit

```bash
sbatch experiments/seed2_finish_optimised/submit.slurm
```

Array `0-41`, one concurrent run, 12 h limit, A40/gpu/normal/vf38, existing alerts.

One job in an allocated session:

```bash
python experiments/seed2_finish_optimised/run_job.py --index 0
python experiments/seed2_finish_optimised/run_job.py --index 0 --dry-run
```

## CUDA Graphs

The worker appends `--set training.cuda_graph=true --set training.step_timing=off` when
`ARC_CUDA_GRAPH=1`. That is a speed change only; `cuda_graph_start` (100) already equals
`annealing_steps` in these configs. Leave it unset to keep these runs identical in shape to the
seed-0 batch.

## Caveat to state in the paper

`projected_conv_c4_triton` changes both the training seed (0 -> 2) and the conv implementation
(`tied_conv` -> `triton`) relative to the seed-0 column. If the two implementations are
numerically equivalent that is harmless, but the table must say so, or use the
`projected_conv_c4_reference` config instead, which changes only the seed.
