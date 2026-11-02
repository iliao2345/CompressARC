# Projected convolution on ARC-AGI-2 in Kaggle

1. Upload `projected_conv_arcagi2.ipynb` as a Kaggle notebook.
2. Create a private Kaggle Dataset containing `arc_fafo_projected_conv_source.zip`; attach it to the notebook.
3. Attach the official ARC-AGI-2 competition data. Accept competition access requirements in Kaggle if needed.
4. Enable a GPU accelerator. No W&B secret or internet connection is required for execution.
5. The notebook locates `arc-agi_test_challenges.json`. Set `CHALLENGES_PATH` explicitly if several versions are attached. Never attach ARC-AGI-1 files as a substitute.
6. Run the two-step smoke test. Then run one full-length task with `MAX_TASKS=1` to measure time before attempting all tasks in a fresh session. Set `MAX_TASKS=None` to attempt all tasks.
7. The final cell writes `/kaggle/working/submission.json` and an explicit task-coverage audit. Save a notebook version and submit through Kaggle's interface only after checking outputs.

The notebook uses existing Kaggle PyTorch. The code/configuration is your strict projected-conv-only variant, 4 layers, seed 0, 1500 steps by default. PyTorch/CUDA/GPU differences can change numeric results relative to M3. It trains per puzzle; it does not load a universal pretrained model. W&B records offline with tag `kaggle run 1`. Download kaggle_run_1_logs.zip and sync on an Internet-connected machine. Smoke tests do not log W&B runs. Runtime versions and source hashes are recorded.

Full task coverage at 1500 steps is not guaranteed. Default global training budget is 6 hours minus setup/reserve, with a 2-hour task timeout. Set budgets to the actual competition limits. Timed-out or unprocessed tasks receive copy-input fallback guesses; the audit makes these distinguishable from model outputs. The trainer saves only final checkpoints, so timed-out tasks lose their intermediate guesses. The script starts each task from scratch. Outputs are saved after each task; rerunning requires a fresh output directory. All guesses must be grids of color integers 0..9 and two attempts are written per test input.

No data, hidden solutions, API keys, or W&B run logs are bundled. Public labeled challenges can also be tested by setting `CHALLENGES_PATH`, but this workflow deliberately strips test output labels and does not calculate reference correctness.

## Rebuild the source ZIP after code changes

Use a local environment with pip and an internet connection to prepare the pure-Python wheels once:

```bash
python -m pip wheel --no-deps --wheel-dir outputs/kaggle_wheels omegaconf==2.3.0 antlr4-python3-runtime==4.9.3
python kaggle/build_bundle.py --repo-root . --wheels outputs/kaggle_wheels --output outputs/arc_fafo_projected_conv_source.zip
```

Kaggle supplies PyYAML, numpy, matplotlib, tqdm, and GPU PyTorch. The notebook checks these instead of replacing the GPU stack.

Official references:

- https://arcprize.org/competitions/2026/arc-agi-2
- https://github.com/arcprize/ARC-AGI-2
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

Competition rules and compute limits must be checked for the actual event; this package does not submit anything automatically.


## Parallel execution and verification evidence

The full run now defaults to MAX_TASKS=None and dispatches a shared task queue across up to four visible GPUs. Each puzzle uses one GPU and its own process. Four-GPU execution requires four GPUs actually allocated by Kaggle; the notebook prints the detected count and uses fewer when necessary. The smoke test exercises one task per available worker.

After execution, the notebook saves smoke_test_result.json, submission_validation.json, task_status.json, run_manifest.json and submission.json. Save a notebook version with execution outputs to retain evidence. Source code alone is not evidence of successful Kaggle GPU execution. Full model coverage remains dependent on available compute; validation explicitly records fallback counts and full_model_coverage.

W&B is offline with tag kaggle run 1. Smoke tests do not log to W&B. Download kaggle_run_1_logs.zip to sync training logs later from an Internet-connected machine.
