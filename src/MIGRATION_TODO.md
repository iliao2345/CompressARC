# Migration follow-ups

## Review solution tracking in metrics.py

- Review `compressarc/train/metrics.py` (`SolutionTracker`) against the former legacy solution-selection behavior: decoding/cropping, EMA predictions, accumulated scores, and top-two guesses.
- Decide whether to add prediction-history recording and `.npz` export for `compressarc.analysis.list_solved_puzzles` and `compressarc.analysis.plot_accuracy`.
- `compress stuff/solution_selection.py` was removed intentionally. The legacy `compress stuff/train.py` and `src/compressarc/analysis/solve_task.py` still import it and cannot run until migrated to `SolutionTracker` or retired.
- The current `src/train.py` already uses `SolutionTracker` and does not import the removed module.
- The former implementation is recoverable from Git history for comparison.

## Deferred logging and training review

- User will decide later which additional metrics and outputs are needed. Do not change logging behavior until that review.
- Review `compressarc/train/metrics.py`, `compressarc/train/logging.py`, and `config/default.yaml` together: scalar metrics, prediction images, final correctness, prediction histories, and optional performance metrics.
- W&B is disabled by default. Prediction tracking runs every 50 steps by default; the legacy trainer tracked every step. Review whether `prediction_every: 1` is wanted for legacy-like candidate accumulation and EMA behavior.
- The principal training and model-size YAML defaults match the legacy constants; initialization/training order differs and can change the random-number sequence.
- Both current and legacy training loops train tasks sequentially. The legacy trainer constructs all models/optimizers upfront; the current trainer constructs each as its task begins. Neither implements parallel task training.
- Parallel training, if later wanted, requires an explicit worker/scheduling design and GPU-memory limits; retaining all models alone does not provide it.
