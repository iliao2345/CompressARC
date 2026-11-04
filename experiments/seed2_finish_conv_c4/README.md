# Finish the seed-2 C4-conv cells

Queued: the 22 (task, seed 2) cells the projected C4 convolution arm is missing.

| arm | config | model |
|---|---|---|
| `projected_conv_c4_triton` | `configs/projected_conv_c4_triton.json` | `shift: triton` (Triton C4 conv) |
| `projected_conv_c4_reference` | `configs/projected_conv_c4_reference.json` | `shift: tied_conv`, the seed-0 recipe — **not queued** |

Both use `share=primitives`, `cummax=primitives`, seed 2, 1500 steps, `split=training`.

```bash
sbatch experiments/seed2_finish_conv_c4/submit.slurm
```

Array `0-21%4` with `ARC_CUDA_GRAPH=1`, 12 h limit, A40/gpu/normal/vf38, existing alerts.
Worker: `python experiments/seed2_finish_conv_c4/run_job.py --index 0 [--dry-run]`.

## Caveat to state in the paper

`projected_conv_c4_triton` changes the training seed (0 -> 2) *and* the conv implementation
(`tied_conv` -> `triton`) relative to the seed-0 column. Use `projected_conv_c4_reference`
instead if you want a seed-only change: swap `configs/projected_conv_c4_triton.json` for
`configs/projected_conv_c4_reference.json` in every job of the manifest.
