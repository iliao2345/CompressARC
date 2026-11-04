# Slurm experiment map

Derived from `src/slurm/*.sbatch` and `experiments/*/configs/*.json` (each config carries its
`logging.wandb_tags` and training settings). Use this instead of guessing from run names.

## Launchers

| script | array | shards | configs | tasks | purpose |
|---|---|---|---|---|---|
| `final_tasks_01..04.sbatch` | `0-59%1` | 60 | `final_original_20`, `final_projected_conv_20`, `final_projected_d4_20`, `final_projected_lse_20` | 5 per script, 20 total | the controlled 20-task ablation, 4 variants x 5 tasks x 3 seeds |
| `final_single.sbatch` | none | none | `$CONFIG` | one | re-run a single config |
| `pilot_variants.sbatch` | `0-15%2` | 16 | `conv_d4_cummax_pilot_8_corrected`, `all_projected_triton_pilot_8_corrected` | 8 | early pilot, two variants |

60 shards = 4 variants x 5 tasks x 3 seeds, so each array runs every (variant, task, seed)
exactly once. All arrays use A40, `--gres=gpu:A40:1`, 4 h wall, `outputs/final_ablation/tasks_0N/<jobid>`.

## Tag glossary

| tag | meaning |
|---|---|
| `final-ablation` | belongs to the controlled experiment family |
| `sample20` | pilot 20 tasks (chosen from tasks the original solved) |
| `additional20` | the second 20 tasks; `sample20` + `additional20` = the 40-task set |
| `sample40` | the 40-task set run as a single batch |
| `steps1500`, `strict` | protocol markers (1500 updates, strict tensor constraints) |
| `seed2-rerun` | the seed-2 replication pass |
| `optimised-backends` | vectorised / Triton implementations instead of primitives |
| `latency-only`, `triple-latency-reference` | timing references, **not** accuracy results |
| `uniform-sample` | how the additional 20 tasks were drawn |
| batch markers | `seed0-combined-d4`, `seed0-optimised-retest`, `seed2-optimised-remaining`, `d4-conv-pilot20`, `d4-conv-additional20`, `seed2-lse-triton-remaining`, `combined-d4-packed-remaining`, `seed0-triple-d4-lse` |

## Batches

| experiment dir | variants (config files) | tasks | seed | batch tags | backends |
|---|---|---|---|---|---|
| `additional20` | original, projected_conv, projected_d4, projected_lse | 20 | 0 | `additional20`, `uniform-sample` | primitives |
| `d4_conv_pilot20` | projected_conv_d4 | 20 | 0 | `d4-conv-pilot20` | primitives |
| `d4_conv_additional20` | projected_conv_d4 | 20 | 0 | `d4-conv-additional20` | primitives |
| `seed0_combined_d4` (+`_packed_remaining`) | combined_d4 | 40 | 0 | `sample40`, `seed0-combined-d4` | Triton + fused |
| `seed0_d4conv_retest` | projected_conv_d4 | 40 | 0 | `sample40`, `seed0-optimised-retest` | vectorized + Triton |
| `seed0_d4sharing_retest` | projected_d4 | 40 | 0 | `sample40`, `seed0-optimised-retest` | Triton |
| `seed0_triple_d4_lse` | triple_d4_lse; `hard_reference` is `latency-only` | 40 | 0 | `sample40`, `seed0-triple-d4-lse` | Triton LSE |
| `seed2_40` | original, projected_conv, projected_d4, projected_lse | 40 | 2 | `sample40`, `seed2-rerun` | primitives |
| `seed2_original_d4_remaining` | original, projected_d4 | 40 | 2 | `sample40`, `seed2-rerun`, `seed2-optimised-remaining` | vectorized + Triton |
| `seed2_d4conv_priority` | projected_conv_d4 | 40 | 2 | `sample40`, `seed2-rerun`, `seed2-optimised-remaining` | vectorized + Triton |
| `seed2_40_optimised_remaining` | projected_conv_d4, projected_d4, projected_lse | 40 | 2 | `sample40`, `seed2-rerun`, `seed2-optimised-remaining` | vectorized + Triton |
| `seed2_lse_triton_remaining` | projected_lse | 40 | 2 | `sample40`, `seed2-rerun`, `seed2-optimised-remaining`, `lse-triton` | Triton LSE |

## The trap: variant tags do not identify an implementation

Every variant exists in **two backend generations**, and both reuse the same variant tag:

| variant | primitives-era config | optimised-era config |
|---|---|---|
| original | `shift=primitives, share=primitives, cummax=primitives` | `share=vectorized, cummax=triton_hard_grouped` |
| projected_conv | `tied_conv + primitives` | — |
| projected_d4 | `share=d4, cummax=primitives` | `share=d4, cummax=triton_hard_grouped` |
| projected_conv_d4 | `d4_tied_conv + primitives` | `share=vectorized, cummax=triton_hard_grouped` |
| projected_lse | `cummax=optimised` | `cummax=triton` (LSE Triton kernel) |

Consequences for any table:

1. **Seed 0, 40 tasks (`sample20` + `additional20`) is clean** — all four arms come from the
   primitives-era `final_*_20.yaml` and `additional20` configs.
2. **Seed 2, 40 tasks is not clean** — `seed2_40` ran the primitives-era configs while
   `seed2_*_remaining` and `seed2_d4conv_priority` ran the optimised-era ones, under the same
   `seed2-rerun` tag. Any per-variant seed-2 rate silently pools both.
3. `hard_reference` in `seed0_triple_d4_lse` is tagged `latency-only` and must be excluded.
4. Before quoting a number, group by `(variant, optimised-backends present?, seed)`.
