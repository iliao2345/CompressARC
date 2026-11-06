# Training

Install dependencies from the repository root:

```bash
python -m pip install -r src/requirements.txt
```

Train one task, or omit `--task` to train the configured split:

```bash
python src/train.py --task 50cb2852 --set training.iterations=1500
```

Defaults live in `src/config/default.yaml`. Use `--config <yaml>` and repeat
`--set KEY=VALUE` to override them. Select strict or relaxed preprocessing with
`model.multitensor_constraints`.

The solution tracker updates every optimizer step. Console output shows progress,
final loss, both guesses, and correctness when reference solutions are available.
W&B logs `train/reconstruction_error` (unweighted reconstruction CE objective)
and `train/total_KL` every step, plus `predictions/top_1_correct` and
`predictions/pass_2_correct` when reference answers exist.

The solution tracker updates every step. Every 50 updates, W&B logs `candidates/cumulative_unique_count`,
`candidates/cumulative_correct_count`, `candidates/best_correct_rank`,
`candidates/best_correct_score`, and `candidates/top_ranked_score`.
Correctness fields require reference answers. Before a correct candidate exists,
its rank and score are null (not zero). These rank and score fields use the current ranking at each logged step, not
historical best values. A correct candidate can fall from rank 1 to rank 2 or
lower. Rank 1 is the highest accumulated
selection score; equal scores follow first appearance. Candidates are exact
full tuples of test-output grids, deduplicated across the run. Correct candidate
count therefore normally stays at 0 or 1 for a task with one accepted answer.

`candidates/first_correct_generation_step` is stored in the summary immediately
on first generation, using 1-based steps. If never generated, it is absent.
At normal completion, `candidates/all_unique` contains one row per unique output
hash with exactly four columns: `hash`, `first_step` (1-based), `occurrences`,
and `is_correct` (exact match, null when reference unavailable). Occurrences
count a candidate at most once per step when raw and EMA agree, and count
reappearances on later steps. Final candidate metrics remain in the summary.

```bash
wandb login
python src/train.py --task 50cb2852 --set logging.wandb=true --set logging.wandb_entity=arc_agi
```

The default project is `compressarc`. Configure the account, project and mode with
`logging.wandb_entity`, `logging.wandb_project` and `logging.wandb_mode`.
Logging is disabled by default.

## Local checkpoints

```bash
python src/train.py --task 50cb2852 --save-checkpoints --output-dir outputs/training
```

This saves final weights, model configuration, task information, and scalar
metrics in a local `.pt` file. It is a final model snapshot, not a resumable
optimizer checkpoint.

## Modal

Install and authenticate Modal, then use the same trainer:

```bash
pip install modal
modal setup
python src/train.py --backend modal --task 50cb2852 --save-checkpoints
```

The default GPU is T4; select another with `--modal-gpu`. Saved checkpoints are
returned to the local output directory. When W&B is enabled, configure a Modal
secret named `wandb-secret` containing `WANDB_API_KEY`.

## Benchmarks

```bash
python src/train.py --mode benchmark --task 694f12f3 --benchmark-pass clean
python src/train.py --mode benchmark --task 694f12f3 --benchmark-pass operations
```

Both passes use the existing model, objective, and optimizer. Clean reports setup,
complete-step timing and memory; operations profiles phases. Results print as JSON
to the console. Defaults are 10 warmup and 30 measured updates. Benchmark mode does
not save model checkpoints.

The batch launcher supports configured variants and task sharding:

```bash
python src/run_training_timing.py --set training.device=cuda:0
python src/run_training_timing.py --mode train --save-checkpoints
```

Use `--help` on the trainer or launcher for configuration and sharding options.

## Per-run summaries

With W&B enabled, each task records `run/wall_clock_seconds`, `run/status`
(`running`, `finished`, `failed`, or `interrupted`), and on CUDA,
`run/peak_gpu_allocated_mib` and `run/peak_gpu_reserved_mib`.
Wall time covers model/optimizer setup, training, tracking and local checkpoint
saving; it excludes dataset loading, W&B initialization and final upload/finish.
GPU peaks reset before constructing each task model. They measure PyTorch-managed
allocations/reservations, not total device memory. CPU runs omit GPU fields.
Caught failures finish W&B with a nonzero exit code. Forced termination cannot
write final summaries and is reflected by W&B's own run state.

## Kernel and numerical diagnostics

With W&B enabled, trainable directional cores are monitored individually by
layer type, depth and parameter name (including the direction-share `theta`).
Primitive-only paths have no trainable core kernels and omit these measurements.
The parameter tensor count is stored as `kernels/parameter_tensor_count`.

Every 50 updates (50, 100, ...), collection and logging include:
- `kernels/<layer>/depth_<n>/<parameter>/distance_from_initialization`: absolute L2 distance to the initialization snapshot.
- `kernels/<layer>/depth_<n>/<parameter>/gradient_norm`: L2 gradient norm after backward, before Adam. Missing gradients are null.
- `kernels/<layer>/depth_<n>/<parameter>/parameter_norm`: L2 parameter norm after Adam.

Every training update checks raw model outputs and KL contributions, loss terms,
all available parameter gradients before Adam, and all parameters after Adam.
Nonfinite events emit `numerics/phase`, `numerics/nan_count` and
`numerics/inf_count` with `train_step`. Finite observations emit no event.
Final summaries always include `numerics/nonfinite_detected`, `nan_detected`,
`inf_detected`, `nonfinite_event_count`, and `nonfinite_step_count` under the
`numerics/` prefix. `first_nonfinite_step` is 1-based and present only if detected.
Event counts count observed phases, so the same nonfinite value propagated
through several phases can generate multiple events in one step. Checks are
observational; they do not skip optimizer updates or sanitize values.

## Final model upload

Each completed training run with W&B enabled automatically saves a final `.pt`
checkpoint under `--output-dir` and attaches it as a W&B artifact of type `model`,
named `model-<run-id>` with aliases `final` and `latest`. No `--save-checkpoints`
flag is needed for W&B runs; that flag still controls local-only runs.
The versioned inference file contains final parameter values and layer buffers,
model configuration, task inputs/training examples, color/shape metadata,
training seed/iteration count, scalar losses, and Python/NumPy/PyTorch runtime
versions. Selection state includes its EMA tensors, candidate scores and records,
selected guesses, first-correct step, EMA decay, uncertainty weight, early-step
penalty/cutoff, EMA penalty, and tie/crop rules. CPU/CUDA Torch, NumPy and Python
RNG states are included because decoder inference is stochastic.

Logging configuration, credentials and optimizer state are excluded. This is an
inference checkpoint, not a resumable training checkpoint. Benchmark mode does
not upload a final model. Bitwise inference reproduction is tested on the same
CPU/runtime; CUDA kernels and different hardware/software can differ.

```python
from compressarc.train.checkpoint import load_inference_checkpoint

saved = load_inference_checkpoint("outputs/training/50cb2852.pt", device="cpu")
print(saved.tracker.solution_most_frequent)  # Original selected final guess.
print(saved.tracker.solution_second_most_frequent)
# saved.model.forward() reproduces the next sampled model forward on the same
# runtime/device, from the RNG state saved after training.
```

The saved model weights are after the final optimizer update; the stored selected
guesses include the last forward pass made before that update. These are distinct
states, both retained. Use the same code/backend implementation as the saved
configuration, with `src` importable, to reconstruct the model.



## Per-step latency and bottleneck profiling

`training.step_timing` accepts `off` (the default), `wall`, or `phases`. The D4-conv pilot now enables `wall` in its config for future launches. Already-running processes are not changed.

- `wall` records total step time, including the model update, output transfers, candidate tracking, diagnostics and normal scalar logging.
- `phases` additionally records host wall durations for forward, loss, backward, optimizer, metric extraction, progress display, post-step diagnostics, candidate tracking and logging. On CUDA it also records stream elapsed times with events. Only step boundaries synchronize; phase boundaries do not.
- CUDA stream elapsed includes CPU-dispatch gaps and is not GPU kernel-active time. Host and stream times overlap; never add them together.
- Numerics checks inside take_step are included in their enclosing phase; the separate diagnostics phase measures diagnostics.after_step only.

Every measured step is written to `<output-dir>/<task>.step_timing.jsonl` and, when enabled, W&B under `timing/*` with train_step as the x-axis. Summary statistics go to `<task>.step_timing.summary.json`. The summary excludes the first ten completed measurements from steady statistics when more than ten are available. First-step and periodic 50-step overhead remain visible in the per-step file. Timing-export overhead is excluded from the step metric and recorded separately, including an all-step mean with export overhead included. Data loading, model construction, final checkpoints/artifacts and final candidate-table uploads are outside the per-step timing.

For a short D4-only performance experiment in an allocated GPU session:

```bash
python src/train.py --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json --task 00576224 --set training.split=evaluation --set training.device=cuda:0 --set training.iterations=100 --set training.step_timing=phases --set experiment.name=projected_conv_d4_latency_profile --output-dir outputs/d4_conv_latency_profile
```

The public ARC-AGI-1 evaluation fixture above shares the same task ID/inputs as a Kaggle public task; use the actual Kaggle dataset/runtime for a deployment-representative benchmark. For a full existing experiment worker, add `--step-timing phases` or `--step-timing wall` after `--index N`. Retest production latency with wall mode after finding bottlenecks. Phase instrumentation adds overhead, and CPU tests cannot establish a 0.15-second GPU target. This does not change the model operators or training objective.


### Individual operator profiling

Use a separate short run to inspect forward/backward operators. This instruments
normal training, including candidate tracking and enabled W&B diagnostics. It
captures zero-based step 10 by default, after ten warmup steps, and records
phase wall/stream timings with `timing/operator_profile_active` marking capture steps.
Use a separate clean pass for baseline latency; instrumented times contain overhead.

```bash
python src/train.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set training.device=cuda:0 --set training.iterations=20 \
  --set training.operator_profile=true \
  --set experiment.name=d4_conv_operator_profile \
  --output-dir outputs/d4_conv_operator_profile
```

Outputs are `<task>.operators.*`: `.cuda.txt` and `.cpu.txt` rank operators by
self time; `.operators.csv` and `.by_phase.json` attribute operations to training
phases, with nested numerical-check scopes; `.kernels.json` lists CUDA kernels;
`.trace.json` is a Chrome/Perfetto timeline; `.manifest.json` records the capture.
Inspect GPU self time for expensive kernels, CPU self time and timeline gaps for
dispatch/synchronization overhead, and operator call counts for repeated small
operations. Inclusive parent and child times overlap: do not add them together.
Use the clean StepTimer run again to verify any eventual optimization.

Optional overrides: `training.operator_profile_start=10`,
`training.operator_profile_steps=1`, `training.operator_profile_shapes=true`,
`training.operator_profile_memory=false`, and `training.operator_profile_stack=false`.
Shape, stack, and memory instrumentation add overhead. GPU tracing requires a
working PyTorch/CUPTI installation. CPU tests do not verify CUDA tracing.


### Layer contribution and forward/backward comparison

From the repository root, run two independent, identically seeded passes:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --device cuda:0 --clean-steps 40 --warmup-steps 10 --capture-steps 1 \
  --output-dir outputs/d4_conv_layer_profile
```

The first pass records full training-step wall latency and host/CUDA-stream
phase spans without PyTorch Profiler. The second warms up the same model anew,
then captures the requested steps with labels for latent decoding, every depth's
share-up, softmax, cummax, shift, direction sharing, nonlinearity, share-down,
normalization, and output heads. Numerical checks run if W&B is enabled in the
configuration, exactly as in normal training. W&B run names are distinct:
`layer_profile_clean` and `layer_profile_profile`. Profiling outputs are local;
upload the directory explicitly if desired.

Read `comparison.txt` first: it contains clean latency/phase means plus a layer
ranking from the instrumented capture. Machine-readable results are in
`comparison.json`, `profile/<task>.operators.layers.json`, and `.layers.csv`.
The layer reports count actual GPU kernel records linked to originating CPU
operations by trace IDs. They also count scalar reads and synchronization APIs,
with host duration. They exclude GPU user annotations from GPU execution totals.
Backward attribution follows autograd sequence numbers across worker threads;
ambiguous or missing links remain `unassigned`, and coverage is reported.

Forward label wall times include Python dispatch and waits. Backward layer CPU
self time is attributed operator time, not isolated per-layer elapsed latency.
GPU kernel durations exclude dispatch gaps; CUDA-stream spans include them.
CPU/GPU times overlap, so do not add them. No per-layer CUDA barriers are added.
Step timing adds boundary barriers: synchronization counts include these and
should not automatically be blamed on the model. Inspect counts and time together.

Options: `--no-shapes` reduces instrumentation, `--profile-memory` investigates
allocation, and `--with-stack` requests supported CPU stacks (some PyTorch builds
lack stack data). `--set` passes normal configuration overrides, e.g.
`--set logging.wandb=false` for a run without production W&B diagnostics; do not
compare that latency directly with a diagnostics-enabled baseline. For meaningful
before/after optimization, keep task, seed, layers, GPU, diagnostics, and settings
identical. The driver prints several optimization experiments to consider; it
does not change operators, optimizer settings, or mathematical behaviour.

```bash
WANDB_ENTITY=arc_agi WANDB_MODE=online python -m wandb artifact put \
  outputs/d4_conv_layer_profile \
  --name compressarc-final-ablation/d4-conv-layer-profile --type profile
```


To upload the complete report directory automatically after both profiling
passes succeed, add `--upload-wandb` to `src/profile_layers.py`. The default
artifact is `arc_agi/compressarc-final-ablation/d4-conv-layer-profile`, type
`profile`. Optional flags: `--wandb-entity`, `--wandb-project`, `--artifact-name`.
The upload uses online mode and prints the W&B link. Upload failure preserves
all local outputs; retry with the standalone artifact command above rather than
rerunning training. Upload is outside the measured training intervals.


### Equivalent vectorized primitive direction sharing

`model.direction_share_implementation=vectorized` selects packed-GEMM evaluation
of the original primitive's pair-weight equations. It does not select the
projected D4 direction-sharing architecture. Original pair matrices, biases,
coefficient pattern, normalization, initialization, and tied parameter objects
are retained. No new trainable parameters are introduced. Temporary block
matrices are rebuilt each forward so optimizer updates and autograd sharing are
preserved. Non-directional components pass through untouched. The default stays
`primitives`, and existing experiment configs are unchanged.

Floating-point summation order changes, so outputs/gradients agree within tested
tolerances, not bitwise. Long training trajectories can diverge numerically;
compare accuracy and speed rather than expecting identical candidate histories.
CUDA tests are included but require a GPU; local CPU timing is not a GPU claim.

Profile it against the existing baseline on the same GPU/task/settings:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --device cuda:0 --output-dir outputs/d4_conv_vectorized_profile \
  --upload-wandb --artifact-name d4-conv-vectorized-direction-profile
```

W&B adds `direction-share-vectorized` to these runs. Run the GPU contract before
profiling if desired:

```bash
python -m pytest src/tests/correctness/layers/direction_share/test_direction_share_vectorization_contract.py -q
```


### Experimental hard-cummax Triton backend

`model.cummax_implementation=triton_hard` accelerates the original **hard max**
primitive. It is separate from `triton`, which remains the learned LSE backend.
There are no learned scan kernel weights to freeze; surrounding projection
weights still train. Masks, original normalization constants, directional
orientation/channel ordering, first-tie max/min reductions, latest-tie cardinal
cummax, and the diagonal tree's half-gradient tie routing are retained.

V1: finite FP32 CUDA inputs, fixed FP32 masks, first-order gradients, cardinal
axis lengths up to 128. Rectangular and singleton grids are supported. Diagonal
backward saves max/min histories (O(log(min(H,W))) full grids), so check peak
memory. Cardinal backward gathers contributions without atomics, trading extra
arithmetic for fewer launches. Speedup is not established. No defaults change.

**Run the CUDA contracts before profiling or full training.** Development on the
local CPU checked syntax/reference regressions; it did not compile/run Triton.

```bash
python -m pytest src/tests/correctness/layers/lse/test_hard_cummax_triton.py -q -x
```

Only after those pass, compare on the same GPU as the vectorized baseline:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --device cuda:0 --output-dir outputs/d4_conv_hard_triton_profile \
  --upload-wandb --artifact-name d4-conv-hard-cummax-profile
```

These runs receive the W&B tag `cummax-triton-hard`. Compare clean full-step
latency, cummax forward/backward GPU duration, launch count, and peak memory.
A successful forward alone is insufficient: the gradient and tie cases must pass.


### Native fused semantic-subset softmax

`model.softmax_implementation=fused` keeps all original nonempty semantic-axis
subsets, excludes examples/channels from normalization, and preserves subset
concatenation order. Single-axis subsets call native `torch.softmax` directly;
multiple-axis subsets move/flatten their reduction axes, call native softmax,
then restore the axis layout. The original projections, normalization options,
biases, residual, initialization, and parameter sharing remain unchanged.
No parameters are added. The default remains `primitives`.

Native forward/backward replace separate max, exp, sum, and divide graphs.
Multi-axis permutations may need copies: speedup is not guaranteed. Floating-point
reduction order changes, so comparisons are numerical, not bitwise.

On M3, first validate the GPU cases (requires src on the Python path):

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/layers/softmax/test_softmax_fused.py -q -x
```

Then compare to the hard-cummax + vectorized-direction-sharing baseline:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --device cuda:0 --output-dir outputs/d4_conv_fused_softmax_profile \
  --upload-wandb --artifact-name d4-conv-fused-softmax-profile
```

W&B adds the `softmax-fused` tag. Use the same GPU and diagnostics settings for
before/after clean wall timings; check memory and the detailed softmax launches.


### Fused Adam and optimizer subphase profiling

`training.optimizer_implementation=default` preserves the previous Adam
construction. `fused` explicitly requests native `torch.optim.Adam(fused=True)`
for CUDA parameters. Learning rate, betas, epsilon, weight decay, and other Adam
defaults are unchanged. Unsupported modes/devices raise instead of silently
falling back. Fused updates can differ in floating-point rounding and require
GPU parity tests; a speedup is not assumed. Existing configs and defaults stay
on the original selection. W&B tags fused runs with `adam-fused`.

Phase timing now separates `optimizer_gradient_checks`, `optimizer_adam`, and
`optimizer_parameter_checks`, inside the existing optimizer phase. The first
includes any configured gradient-norm diagnostics; the third checks updated
parameters. Checks and ordering are retained. Subphases add CUDA events, not
synchronization barriers. They are parts of the optimizer total: do not add both
parent and child times. No-diagnostics runs naturally show near-zero check cost.
Detailed artifacts also report optimizer subcomponent launch/synchronization/
scalar-read counts in `.layers.json` and `.layers.txt`.

On M3, validate the CUDA optimizer contract before comparing speeds:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_optimizer_modes.py -q -x
```

Run the existing profiling command twice, preserving task, GPU, seed and diagnostics:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set training.optimizer_implementation=default \
  --device cuda:0 --output-dir outputs/d4_conv_adam_default_profile \
  --upload-wandb --artifact-name d4-conv-adam-default-profile

python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set training.optimizer_implementation=fused \
  --device cuda:0 --output-dir outputs/d4_conv_adam_fused_profile \
  --upload-wandb --artifact-name d4-conv-adam-fused-profile
```

`comparison.txt` shows clean full-step and optimizer subphase timings. Each run
contains a clean pass and a separate short instrumented capture. Compare latency
and peak memory on multiple task sizes before using fused mode for a large batch.


### Direct-tensor numerical checks and diagnostic detail

`training.diagnostics_implementation=reference` retains the original numerical
checks. `triton` scans contiguous CUDA float16/bfloat16/float32/float64 tensors
through pointer-table metadata instead of creating per-tensor detached views and
concatenating their values. Each scan counts NaN and Inf elements separately;
aliases/duplicate list entries are still counted repeatedly, exactly as before.
Checks run every step, emit events only on detection, and preserve final summaries
and the existing kernel-norm cadence. They never modify parameters or gradients.
CPU tensors, complex values, and noncontiguous layouts use the reference path.

Metadata is reused only while object identity, storage address, and element count
match; replaced grad buffers or changed parameter storage rebuild it. Source
tensors stay alive through the count transfer. Cache entries hold metadata, not
old gradient tensors. There is no full parameter-data concatenation on the fast
path, but count partials and metadata still allocate some memory. Speedup is not
assumed. The local machine cannot compile/test the CUDA kernel.

With phase timing, `comparison.txt` now separates `numerics_<category>_prepare`,
`concatenate` (reference/fallback), `metadata` (fast path), `reduce`, and `transfer`.
Categories are outputs, loss, gradients, and parameters. Host/stream spans overlap
and these detail spans are already included in their parent training phase.
No extra explicit synchronization barriers are added. W&B tags the direct-scan
mode `numerics-triton`; `numerics/implementation` also records the selected mode.

Validate GPU contracts before profiling:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_nonfinite_counter.py -q -x
```

First rerun the previous profiling command with
`--set training.diagnostics_implementation=reference` to measure the detailed
reference costs. Then run the fast path, holding all other settings fixed:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set training.optimizer_implementation=fused \
  --set training.diagnostics_implementation=triton \
  --device cuda:0 --output-dir outputs/d4_conv_direct_checks_profile \
  --upload-wandb --artifact-name d4-conv-direct-checks-profile
```

Keep W&B diagnostics enabled in both runs. Turning logging off disables these
checks and would invalidate the reference/direct comparison. Inspect clean step
latency, parameter/gradient check detail, launch counts, and peak GPU memory.


The direct-check mode also limits default-device dispatch overhead during
metadata inspection. It materializes source iterators first, then temporarily
disables TorchFunction dispatch **only for metadata reads on ordinary Tensor and
Parameter objects**. Tensor subclass hooks, fallback views, allocations, and GPU
operations execute with normal dispatch. Dtype/device/contiguity and storage
signatures are collected in one pass; tile metadata is built from those integer
signatures without reading tensor pointers again. Storage validation still occurs
every observation. The optimization applies only to the opt-in `triton` mode.
A local 4,000-tensor metadata microbenchmark improved from about 27 ms to 2 ms
under default-device mode; this is not a measured full GPU training speedup.


### Measure share-up/down before changing their implementation

Add `--sharing-stages` to the profiling driver to time sharing stages in the
clean pass. During detailed capture the original layers also emit stage labels.
No projections, masks, broadcast/reduction order, normalization, or parameters
are changed. Stages are weight selection, source projection, communication,
normalization, destination projection, and residual addition.

Clean timings `timing/host_sharing_share_up_<stage>_seconds` (and share_down)
aggregate forward work across model depths. CUDA-stream spans are also recorded,
without per-stage barriers. These spans are already part of forward time; do not
add parent and child totals. Enabling this detail introduces measurement overhead,
so use the same flag for before/after optimization comparisons.

`comparison.txt` and `.layers.json` include `sharing_stages`: kernel launches,
GPU kernel duration, CPU operator self time, and forward host ranges. Backward
stage attribution follows autograd sequence numbers; it is attributed operator
work, not an isolated per-stage backward wall-clock benchmark. Missing/ambiguous
links stay unassigned. Coverage is reported by `sharing_stage_attributed_launches`
and `sharing_layer_launches`. Existing layer totals are retained separately.

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set training.optimizer_implementation=fused \
  --set training.diagnostics_implementation=triton \
  --sharing-stages --device cuda:0 \
  --output-dir outputs/d4_conv_sharing_stages_profile \
  --upload-wandb --artifact-name d4-conv-sharing-stages-profile
```

This experiment identifies whether grouped projections or communication/reduction
fusion should be attempted next. It does not replace sharing with a dense block
matrix or otherwise change model equations.


### Reuse and optional batching for multitensor sharing

`model.share_implementation=primitives` remains the default. `reuse` keeps the
same projections and normalizations but plans compatible source/destination
communication once, then reuses identical reduction/view prefixes within each
forward. Mask unions, broadcast mask layouts, and +1e-4 denominators are reused
only when their exact original layouts/reduction axes match. `batched` adds
stacked evaluation of independent reduction nodes with matching operation,
mask specification, shape, dtype, and device. The original destination addition
order and reduction-axis order are retained. No parameters are added or changed.
Share-up reuses insertion/broadcast views; it does not batch new matmuls.

Plans cache only layout metadata, never values or autograd graphs. All values,
masks, and denominators are reconstructed each forward, so optimizer updates,
mask edits, and different tasks are observed. Inference checkpoints preserve the
backend setting. W&B tags distinguish `share-reuse` and `share-batched`.

`model.share_batch_max_elements` defaults to 262144 and bounds the element count
of each stacked batch; oversized/single groups use individual reductions. This
bounds temporary stacks, not total peak memory. Cached intermediate values and
batched copies can increase peak memory. Floating-point accumulation in backward
and batched reductions may differ; output/gradient parity is numerical, not a
promise of bitwise identical training trajectories. CUDA cases require M3.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/layers/share/test_share_reuse.py -q -x
```

Profile reuse first, keeping all other accelerations and the same GPU:

```bash
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set model.share_implementation=reuse \
  --set training.optimizer_implementation=fused \
  --set training.diagnostics_implementation=triton \
  --sharing-stages --device cuda:0 \
  --output-dir outputs/d4_conv_share_reuse_profile \
  --upload-wandb --artifact-name d4-conv-share-reuse-profile
```

Then change `reuse` to `batched`, using a distinct output directory and artifact
name (`d4_conv_share_batched_profile` / `d4-conv-share-batched-profile`). Compare
clean latency, communication forward/backward launches, and peak GPU memory.
Batching may lose to reuse alone when copying costs outweigh launch savings.

### Runtime device dispatch experiment

`training.runtime_device_dispatch=explicit` removes PyTorch's global default-device
mode after task/model/optimizer setup. Runtime factories use tensor-derived or
explicit devices; tensor-subclass dispatch and autograd stay enabled. The prior
default device is restored on normal completion and exceptions. The default
`global` retains the original path. This changes dispatch overhead, not equations.

CPU tests compare three real training steps, including outputs, weights, gradients
and RNG state. CUDA cases also exercise fused Adam and hard Triton cummax; run
those on M3 before measuring performance. A 0.15 s step time is a target, not a
verified result. Compare the same task and GPU against the 0.349 s reuse baseline.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_device_dispatch.py -q -x

python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set model.share_implementation=reuse \
  --set training.optimizer_implementation=fused \
  --set training.diagnostics_implementation=triton \
  --set training.runtime_device_dispatch=explicit \
  --sharing-stages --device cuda:0 \
  --output-dir outputs/d4_conv_explicit_dispatch_profile \
  --upload-wandb --artifact-name d4-conv-explicit-dispatch-profile
```

### Diagnose backward before changing implementation

The short operator capture now labels vectorised direction sharing as
`normalization`, `weight_packing`, `input_layout`, `matmul`, and `residual`.
D4 shift labels distinguish `d4_projection`, `kernel_expansion`, and
`convolution`. These labels are active only during the instrumented capture;
model equations, gradients, and saved-tensor policy are unchanged.

The `.layers.json` report adds `operation_stages` and `backward_nodes`.
Autograd sequence numbers link backward nodes to forward stages, including
across worker threads. Inspect `StackBackward`, `CloneBackward`, `CopyBackward`,
`PermuteBackward`, `MmBackward`, and `ConvolutionBackward` alongside their stage,
CPU operator self time, GPU kernel time and actual kernel-launch count. Nodes
without an unambiguous link remain unassigned; views are not necessarily copies.
Do not interpret a backward node's name alone as proof of materialisation.

Existing `sharing_stages` still contains only share-up/down stages. Generic
operation stages include those rows too; do not add both tables. CPU self times
and kernel sums are diagnostic attribution, not isolated clean stage latency.

Run the usual explicit-dispatch profile with `--profile-memory` and a distinct
output directory/artifact name, for example `d4_conv_backward_stages_profile`
and `d4-conv-backward-stages-profile`. Memory events in the raw Chrome trace help
identify allocations and copies; allocation traffic does not equal peak live
memory or saved activation memory. `--with-stack` is optional and adds overhead.
The separate clean pass remains the source of wall-clock latency. No checkpointing
or custom backward optimisation has been introduced by these profiling labels.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_backward_stage_profiling.py -q -x
```

### Isolated backward completion benchmark

`src/profile_backward.py` measures a full-loss backward reference and isolated
VJPs for share-up, share-down, direction sharing, shift and cummax at every depth.
It prepares the model with ten training steps, then fixes the weights. One normal
loss backward captures actual component inputs and upstream gradients. Each
isolated sample rebuilds its forward graph, then times backward only. Tied
parameters, input devices and strides are preserved. No weights/checkpoints are
saved; the optional W&B artifact contains JSON/text reports only.

Warmup is separate for every component. JSON includes every boundary's shapes,
strides, output-gradient coverage, median/mean/min/max completion time and CUDA
stream spans. CUDA barriers isolate each backward from pending forward work;
stream spans include dispatch gaps and are not GPU kernel-active time. Use the
existing operator trace for kernel-active execution and launch counts.

These are controlled microbenchmarks at one prepared weight snapshot, not a
replacement for the clean whole-step benchmark. Detached input boundaries change
scheduling and accumulation; do not sum component times as full-model time or
interpret a difference as an exact removable cost. Identity outputs can share
gradient paths. Confirm any eventual optimisation with full-step timing and
correctness checks. No benchmark result establishes full-training accuracy.

```bash
module load miniforge3
mamba activate SymARC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_backward_latency.py -q -x && \
python src/profile_backward.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard \
  --set model.softmax_implementation=fused \
  --set model.share_implementation=reuse \
  --set training.optimizer_implementation=fused \
  --set training.runtime_device_dispatch=explicit \
  --device cuda:0 --prepare-steps 10 --warmup-steps 5 --measured-steps 20 \
  --output-dir outputs/d4_conv_backward_latency --upload-wandb
```

`--components direction_share shift` limits the isolated cases when desired.
This script does not enable per-step W&B logging or numerical checks, so the
`training.diagnostics_implementation` setting does not affect the timed VJPs.

### Complete backward coverage: model versus loss

The default isolated set now also includes softmax, nonlinear, standalone
normalisation, latent decoder, output heads, and the reconstruction/KL objective.
A four-depth model should report 35 isolated boundaries (8 per depth, plus the
latent decoder, output heads and loss). The report lists expected and observed
boundary counts; selecting `--components` deliberately reduces that coverage.

Three measurements answer different questions:

- `full_backward`: the actual connected loss-to-parameter backward reference.
- `connected_model_backward`: a connected VJP through the entire model, using
  captured real gradients for logits, masks AND KL amounts, with loss arithmetic
  excluded. This preserves cross-layer connectivity and shared parameters.
- `loss`: objective-only backward into detached logits, masks and KL amounts.
  Objective construction uses the same function as normal training.

Each isolated component still uses its real captured loss gradients. Latent
parameters remain the original tied leaves, not detached copies. Exact CPU tests
verify connected-model parameter gradients and loss-boundary gradients against a
full training backward. The loss extraction also matched the pre-refactor trainer
bitwise for three steps each on fixed and uncertain output shapes.

Do not call `full - connected - loss` pure graph overhead: each timing has a
separate graph, gradient accumulation, barriers and scheduling. These experiments
help locate the missing work and compare connected versus isolated behaviour;
only an eventual clean end-to-end A/B test proves an optimisation's saving.
The raw operator trace remains the source for kernel-active execution times.

Run the prior `profile_backward.py` command with a fresh output directory:
`--output-dir outputs/d4_conv_backward_coverage --upload-wandb`. Uploads now use
artifact `d4-conv-backward-coverage`, preserving the earlier five-component report.
Reports contain JSON/text only; no model weights or checkpoints are uploaded.

### Hard-cummax launch-layout sweep (experimental)

`src/benchmark_cummax_launch.py` sweeps axis lengths, row counts, output positions
per cardinal backward program, diagonal spatial sizes/batches, block sizes and
warp counts. It does not modify the training factory or production hard-cummax
backend. `positions=1,warps=4` is the current cardinal kernel; diagonal
`block=256,warps=4` is the current diagonal pipeline layout.

Grouped cardinal backward programs compute several output gradients while
sharing the row input loads; this remains the same quadratic gradient formula,
not a new prefix-gradient algorithm. Larger groups may reduce repeated loads
but increase register pressure. Performance must be measured, not assumed.
Programs are workgroups inside a launch, not additional kernel launches.
Diagonal block size changes programs per scan stage; the original tree and its
equal-value gradient splitting are retained.

Every layout is checked against primitive PyTorch outputs AND input gradients
on the timed shape with random masked inputs, ties, and +/-1000 padding ties.
Invalid layouts are recorded and excluded from winner selection. Compilation
and validation are outside timing. Preallocated contiguous buffers exclude
allocation, movedim/contiguous adapters and autograd scheduling. These timings
are amortised dispatch-to-completion cost, not model layer latency. CUDA-event
spans include dispatch gaps, not just kernel-active execution. Compare any
winner in a whole-step A/B experiment before adopting it. Reports include the
baseline comparison, launch/program counts, shapes and GPU metadata.

```bash
module load miniforge3
mamba activate SymARC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/layers/lse/test_cummax_launch_sweep.py -q -x && \
python src/benchmark_cummax_launch.py \
  --device cuda:0 --kind both \
  --lengths 6 16 30 64 --rows 32 256 2048 \
  --positions 1 2 4 8 --warps 4 8 \
  --spatial-sizes 6x6 16x16 30x30 --batches 8 64 \
  --blocks 128 256 512 \
  --warmup 10 --samples 10 --repeats 20 \
  --output-dir outputs/hard_cummax_launch_sweep --upload-wandb
```

Use `--dry-run` to inspect configurations without CUDA. The default sweep has
132 layouts. Compilation and correctness validation can take several minutes;
GPU timing begins after those steps. Results upload as report-only artifact
`hard-cummax-launch-sweep` in `arc_agi/compressarc-final-ablation`; no weights.

### Whole-step A/B: grouped cardinal hard-cummax backward

`model.cummax_implementation=triton_hard_grouped` is an opt-in backend using
8 positions per cardinal backward program and 4 warps. Cardinal forward and all
diagonal scans use the original hard backend unchanged. It introduces no weights
or changed initialisation. The ordinary `triton_hard` backend remains unchanged.
Factory/checkpoint configuration distinguishes the backends, and W&B tags the
experiment `cummax-triton-hard-grouped8`.

The frozen-reference scan tests now cover both backends, including masks, ties,
rectangular grids, input gradients and full-layer projection-weight gradients.
Local CPU integration tests pass; CUDA tests must run on M3. Microbenchmark
speedups are not evidence of a full-step improvement, especially on small tasks.

`profile_layers.py` now forces `logging.save_model=false` for both passes.
Normal training retains `logging.save_model=true` by default; explicitly asking
for `--save-checkpoints` still saves a checkpoint. Profiling uploads reports,
traces and candidate tables, with no model checkpoint artifacts or bundle weights.
Use a fresh output directory so old local checkpoints are not re-uploaded.

Run both backends sequentially on the SAME GPU/job to compare clean latency:

```bash
module load miniforge3
mamba activate SymARC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/layers/lse/test_hard_cummax_triton.py \
  -q -x -k triton_hard_grouped && \
for backend in triton_hard triton_hard_grouped; do
  python src/profile_layers.py \
    --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
    --task 00576224 --set training.split=evaluation \
    --set model.direction_share_implementation=vectorized \
    --set model.cummax_implementation="$backend" \
    --set model.softmax_implementation=fused \
    --set model.share_implementation=reuse \
    --set training.optimizer_implementation=fused \
    --set training.diagnostics_implementation=triton \
    --set training.runtime_device_dispatch=explicit \
    --sharing-stages --device cuda:0 \
    --output-dir "outputs/d4_conv_${backend}_ab" \
    --upload-wandb --artifact-name "d4-conv-${backend}-ab" || break
done
```

Compare `clean_mean_seconds`, `clean_median_seconds` and clean backward phase time
in each `comparison.json`. Do not use the instrumented capture's step time as the
speed result. Repeat on a larger task before choosing a default.

### Opt-in sharing normalisation fusion

`model.share_implementation=reuse_fused_norm` keeps the existing reused sharing
DAG, exact masked reduction order/denominators, tied projection weights and
residual structure. Only the normalisation between communication and destination
projection changes: two Triton kernels for forward (statistics/output), and two
for backward (gradient statistics/input gradients), replacing multiple separate
reductions and elementwise autograd operations per multitensor component.
This is normalisation fusion, not a fused implementation of the entire sharing
layer. Matrix multiplications and communication reductions are unchanged.

The formulas retain centered population variance over all non-channel axes and
epsilon 1e-8. Backward explicitly subtracts the mean of the centered-input gradient;
it does not assume floating-point mean(centered input) is exactly zero. Reduction
order and rsqrt-based arithmetic can differ numerically from native PyTorch.
No parameters, initialisation or weight tying are changed.

The fused path supports contiguous FP32 CUDA tensors with rank >=2 and at most
8192 rows (all non-channel dimensions flattened). Other layouts, larger cases,
CPU and FP64 use original normalisation. Statistics use eight warps at N>=4096,
four otherwise; output/gradient application uses blocks of 256 elements. This
initial layout is experimental, not an established fastest configuration.
CUDA fusion supports first-order training gradients only. CPU tests verify
fallback equations and unchanged parameters; M3 tests compare fused outputs,
input/projection/mask gradients, constant and low-variance inputs and fallbacks.

After committing/pulling the code, run on the SAME GPU as the prior profile:

```bash
module load miniforge3
mamba activate SymARC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_fused_share_normalization.py -q -x && \
python src/profile_layers.py \
  --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation \
  --set model.direction_share_implementation=vectorized \
  --set model.cummax_implementation=triton_hard_grouped \
  --set model.softmax_implementation=fused \
  --set model.share_implementation=reuse_fused_norm \
  --set training.optimizer_implementation=fused \
  --set training.diagnostics_implementation=triton \
  --set training.runtime_device_dispatch=explicit \
  --sharing-stages --device cuda:0 \
  --output-dir outputs/d4_conv_fused_share_norm_profile \
  --upload-wandb --artifact-name d4-conv-fused-share-norm-profile
```

Compare clean full-step and backward timing with the grouped-cardinal + plain
reuse baseline (~0.268 s on the small L4 task). Fusion may increase register
pressure or be inefficient for small channel counts: no speedup is assumed.
Profiling saves reports/traces, with no checkpoints or model-weight uploads.

### Compare compatible acceleration across all variants

`src/benchmark_variants.py` compares reference versus accelerated execution for
original CompressARC, projected C4-conv, projected D4-conv, projected D4 direction
sharing, and projected LSE. This is measurement using existing backends, not a
new optimisation or experiment configuration change. It runs sequentially on the
same GPU with the same task, split and seed. By default that is ten short runs:
five variants x two modes, each 40 steps with the first ten omitted from timing.

The accelerated settings are fused softmax, reuse + fused sharing normalisation,
fused Adam, direct numerical checks, and explicit runtime device dispatch.
Primitive direction sharing becomes the equivalent vectorised implementation;
projected D4 sharing remains `d4`. Primitive hard cummax becomes grouped hard
Triton; learned LSE stays `optimised`, with its original tau and learned weights.
All shifts retain their original choice: fixed shifts, C4 conv or D4 conv.
No change is made to the source experiment configs, model dimensions or seeds.

Reference mode uses primitive sharing/softmax, the default Adam and diagnostics,
and global device dispatch. The variant's own variable operator is preserved.
Numerical reduction order can differ; short parity tests do not establish
identical 1500-step trajectories or equal task accuracy. CUDA tests compare the
actual accelerated implementations; CPU tests substitute hard primitives for
CUDA scans and check native-equivalent paths only.

The profiling driver now has `--clean-only` and `--run-label`. This comparison
uses no operator capture, does not save or upload weights, and retains ordinary
per-step diagnostics, candidate tracking and logging overhead. Each W&B run is
named with the variant and mode and tagged `variant-latency`. The summary artifact
`variant-latency-comparison` contains JSON/CSV comparisons, effective model
settings, GPU metadata and timing files, with no large operator traces.

```bash
module load miniforge3
mamba activate SymARC
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/tests/correctness/training/test_variant_latency.py -q -x && \
python src/benchmark_variants.py \
  --tasks 00576224 --split evaluation --seed 0 --device cuda:0 \
  --clean-steps 40 --warmup-steps 10 \
  --output-dir outputs/variant_latency_comparison --upload-wandb
```

Use `--dry-run` to inspect all commands without GPU execution or file creation.
Use `--variants original projected_d4 projected_lse` for a subset, or
`--modes accelerated` to measure only accelerated versions. Use a fresh output
directory for each repeat. `--tasks` accepts several IDs from the selected split;
include larger tasks before extrapolating full-dataset runtime. Failed jobs are
recorded explicitly, do not get speedup values, and make the script exit with an
error after preserving the completed reports.
