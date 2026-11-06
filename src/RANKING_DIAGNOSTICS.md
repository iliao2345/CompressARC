# Inspect raw/EMA candidates and selected guesses

These diagnostics observe the existing CompressARC ranking rule without changing decisions. Enable with `logging.solution_diagnostics=true`.

The prepared configs train only `50cb2852`, using seed 0, 1500 updates and the corrected model code:

```bash
python src/run_training_timing.py --mode train \
  --config src/config/models/original_guess_diagnostics.yaml \
  --config src/config/models/conv_d4_guess_diagnostics.yaml \
  --save-checkpoints --output-dir outputs/guess_diagnostics
```

The projected configuration is conv2d shift + projected D4 + hard cummax. Each config creates its own W&B run/checkpoint folder. Add `--task TASK_ID` for another task. Commit/push and pull the changes before starting.

## Durable candidate registry

A candidate ID is the full SHA-256 of canonical JSON representing the original-color test output grids, including their shapes and test-example ordering. IDs are stable across runs and models. The legacy Python hash remains an internal ranking lookup only; it is never the persisted candidate ID.

Each unique candidate's full grids appear once in `grids.jsonl`. Events and selected-rank records contain only IDs referencing that registry. W&B's `ranking/grids` tables log only newly encountered registry entries. Existing temporal image panels may visualize a known grid again; canonical grid data is deduplicated.

## Every contribution, every iteration

`events.jsonl` has one record per iteration with zero-based `train_step` and one-based `iteration`. Each record contains two events, in raw then EMA order, with:

- `contribution_index`, source and stable candidate ID.
- Uncertainty and `confidence_score = -10 * uncertainty`.
- Separate signed penalties: `early_step_penalty` is -10 before train_step 150, then 0; `ema_penalty` is -4 for EMA and 0 for raw.
- The actual score contribution and accumulated score before/after adding it.
- Both selected IDs and scores immediately after that contribution, full source-summary snapshots in `top_two_after`, and rank-change flags.

The final `selected` records describe the top two after both events. Source summaries contain raw/EMA counts and separate log score masses, last observed source/uncertainty/contribution, and the source responsible for the most recent promotion to top-1.

A pattern may receive contributions from both raw and EMA. `last_source` and `last_top1_promotion_source` are different concepts. A selected guess need not have been generated in the current iteration.

W&B history logs source uncertainties, separate penalties, contributions, accumulated scores, intermediate selected IDs/scores, final selected IDs/scores and score gap every iteration. `ranking/candidates` and `ranking/selected` tables batch all events since the previous image log, preserving every step.

## Every 50 iterations and the final iteration

- `ranking/raw_vs_ema`: current raw grids on the left and EMA grids on the right.
- `ranking/top_two`: the selected top two test guesses.
- `ranking/highest_scoring`: the 20 highest-scoring candidates, including appearance counts and first/last steps.
Training examples keep their separate cadence of every 10 iterations and the final iteration:

- `train/predictions`: provided training examples' expected outputs and current predicted outputs, exact-match flags and original-color grids.
- `training_predictions.jsonl`: the same training y_true/y_pred, example numbers and selected crop coordinates, with step/iteration numbers.

Known training output shapes are used for training prediction crops. These are output-mode forward samples; they do not modify candidate ranking or use test answers to choose guesses. Correct reference outputs are also logged once at run start.

## Reproducibility metadata and artifacts

Files are under `<output-dir>/<task>-<wandb-run-id>.ranking/`:

```text
metadata.json
Grids: grids.jsonl
Events: events.jsonl
Training predictions: training_predictions.jsonl
```

`metadata.json` saves the fully resolved configuration after overrides, seed, task, run ID, config path, Git commit, source-dirty flag, source-code fingerprint, Python/executable, PyTorch/NumPy/CUDA versions and processing order. If Git information cannot be read, its fields are null and `git_error` records the failure instead of inventing a commit. A dirty checkout is not reproducible from the commit alone; the fingerprint identifies its actual source state.

`final_candidates.jsonl` stores all unique candidate IDs, final ranks, accumulated scores, raw/EMA appearance counts, total appearance counts and first/last zero-based train steps. Grids remain saved once in `grids.jsonl`, joined by candidate ID. Appearances count raw and EMA contributions separately.

On successful completion, all five files upload as artifact `ranking-<run-id>`, type `ranking-diagnostics`. Files are flushed locally during training so partially completed runs retain their trace even before final upload.

The old observed transition was at train_step 1063 (iteration 1064). A rerun need not switch at exactly that step. The prior run did not save logits or score events, so these diagnostics cannot reconstruct that old score gap retroactively. Diagnostics add logging overhead and are not clean timing measurements.
