# Training timing metrics: first-pass specification

Status: metric definitions for review. No launcher or timing instrumentation is
implemented by this document. The existing training calculations remain in
`train.py` and `compressarc/train/objective.py`; W&B logging remains in
`compressarc/train/logging.py` and `metrics.py`.

## Run design

Use a thin launcher that calls the existing `train.py` for each real ARC task
and chosen factory variant. Use the same model/training YAML as exploratory
runs. Each task/variant/repeat gets a separate W&B run with the actual resolved
configuration. Weights come from the normal model initializer and internal
activations come from the normal latent decoder; no synthetic benchmark inputs.

Start with one GPU, the previously selected five varied tasks, 10 complete
warmup training iterations, and a measured block of 30 complete iterations
(configurable to 20-50). Warmup updates the actual model and optimizer and is
excluded from the measured averages. Normal annealing step indices continue
through warmup and measurement. The intended default is 40 total iterations
per task/variant, not 10 total.

| Task ID | Train | Test | Total examples | H | W | W/H |
|---|---:|---:|---:|---:|---:|---:|
| 694f12f3 | 2 | 1 | 3 | 10 | 10 | 1.00 |
| 760b3cac | 3 | 1 | 4 | 6 | 9 | 1.50 |
| 94f9d214 | 4 | 1 | 5 | 8 | 4 | 0.50 |
| 3428a4f5 | 4 | 2 | 6 | 13 | 5 | 0.38 |
| dae9d2b5 | 5 | 2 | 7 | 3 | 6 | 2.00 |

The current full-model cummax variants are `primitives`, `d4`, and `optimised`.
They are runtime alternatives; primitive cummax and LSE need not produce
numerically equal losses. The optional `triton` factory variant uses Triton axis and diagonal LSE; its explicit YAML requires Linux CUDA/FP32.

## Existing W&B training metrics

| Key | Definition | Current status |
|---|---|---|
| train_step | Zero-based optimizer-update index | Existing |
| train/loss | Weighted total KL plus reconstruction objective | Existing |
| train/reconstruction_error | Reconstruction term before its loss weight | Existing |
| train/total_KL | Sum of KL components before its loss weight | Existing |
| train/kl_components/<axes> | KL contribution of a multitensor component | Existing |
| top_1_correct, pass_2_correct | Final task-level correctness, when answers exist | Existing run summaries |
| guess_1, guess_2 | Final decoded predictions | Existing run summaries |

Keep the existing metric names and definitions. Prediction images and optional
PCA are existing outputs; their computation/upload is accounted separately from
core training timing.

## Proposed setup metrics (milliseconds; run summaries)

| Key | Exact boundary |
|---|---|
| setup/preprocessing_ms | Dataset loading and normal task preprocessing/device transfers for the selected task |
| setup/model_ms | Normal ARCCompressor construction, including parameter initialization |
| setup/optimizer_ms | Adam construction; excludes its lazily allocated first-update state |
| setup/tracker_ms | Normal SolutionTracker construction |
| setup/core_total_ms | Preprocessing through tracker readiness; excludes W&B startup and image logging |
| setup/wandb_init_ms | W&B run initialization |
| timing/first_train_step_wall_ms | First complete take_step, including lazy Adam state creation |

CUDA setup stages require completed device work at their timing boundaries.
Python process/import startup should be identified separately if measured;
it must not be silently attributed to a particular task's model setup.

## Proposed complete-step timing metrics

| Key | Definition |
|---|---|
| timing/warmup_iterations | Number of completed warmup updates excluded from measured statistics |
| timing/measured_iterations | Number of complete updates in the measured block |
| timing/block_wall_ms | Synchronized elapsed time of the complete training block |
| timing/mean_train_step_wall_ms | block_wall_ms / measured_iterations |
| timing/block_cuda_elapsed_ms | CUDA-event elapsed span of the same block |
| timing/mean_train_step_cuda_ms | block_cuda_elapsed_ms / measured_iterations |

A complete update includes gradient clearing, actual model forward, actual
KL/reconstruction loss, backward, Adam update, and the current take_step metric
collection. Timing must identify and exclude separate prediction decoding,
W&B logging, plotting, PCA, checkpoints, and console reporting. It must also
identify steps requesting prediction-output transfers, which can cost more
than scalar-only steps. There is no separate validation pass in this objective.

## Proposed operation breakdown

For the training-stage option, use `timing/<phase>_host_ms` and
`timing/<phase>_cuda_ms`, aggregated over measured iterations, for:

- gradient_clear: optimizer.zero_grad.
- forward: the complete ARCCompressor.forward.
- loss: background logits plus KL/reconstruction objective; the existing
  KL-component scalar transfers occur in this path and must be included.
- backward: loss.backward.
- optimizer: optimizer.step.
- metrics: final scalar/output conversion in take_step.

On CUDA, host times measure CPU submission and any waits; event spans measure
elapsed stream time and can include CPU dispatch gaps. Neither is pure GPU busy
time. Do not add the two clocks together. Synchronize at block boundaries;
per-operation synchronization would alter the training execution pattern.
Instrumentation overhead must be identified in timing results.

Individual-layer profiling is a separate detail level: share_up/share_down,
softmax, cummax, shift, direction_share, and nonlinear, distinguished by depth.
It is pending the user's choice of stages, individual layers, or both.

## Proposed memory and run metadata

Log `memory/peak_allocated_mib` and `memory/peak_reserved_mib` over the measured
block. Record task ID/split, train/test/total counts, canvas H/W and W/H, colors,
parameter count, full resolved model/training config, implementation names,
seed, repeat, dtype, actual GPU name, and software versions in run configuration.
GPU-utilization telemetry can be inspected through W&B system metrics; it is
sampled telemetry rather than an exact per-iteration measurement.

Gradient norms, parameter-update norms, and prediction-history export remain
optional follow-ups. Their selection was previously deferred; this timing work
should not silently add them.
