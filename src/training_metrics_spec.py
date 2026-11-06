"""Timing and memory metric definitions; no training or logging is performed."""
import json

# Setup: wall-clock milliseconds, measured once per task/variant.
# Synchronize the GPU before starting and before stopping each setup timer.
SETUP = {
    "setup/preprocessing_ms": "Load and preprocess the actual task, including device transfers.",
    "setup/model_ms": "Construct the model and initialize its parameters.",
    "setup/optimizer_ms": "Construct Adam; its lazy state allocation happens in the first step.",
    "setup/total_ms": "Total core setup before training; excludes W&B initialization.",
}

# Training: wall-clock milliseconds from the clean, uninstrumented run.
# Synchronize the GPU at the first-step and measured-block timer boundaries.
# Warmup includes the first step and is excluded from the measured block.
# Each complete update includes gradient clearing; it has no separate metric.
TRAINING = {
    "timing/first_step_ms": "First complete training step, including lazy Adam state allocation.",
    "timing/block_ms": "Complete warmed-up training block, synchronized at its GPU boundaries.",
    "timing/mean_step_ms": "Block time divided by the number of measured iterations.",
}

# Operations: per-step CUDA-event milliseconds in W&B history on GPU.
# Run summaries contain the mean of the measured step values.
# History is buffered and replayed in step order after profiling finishes.
# Collect in a separate instrumented run with the same task, configuration,
# seed, warmup count, and measured count. TRAINING comes from the clean run.
# Event spans include dispatch gaps; they are not pure GPU busy time.
# Operation spans need not sum to block wall time: the clocks differ, and
# gradient clearing, loop overhead, and other work may be outside these spans.
OPERATIONS = {
    "timing/forward_ms": "Full model forward, including latent decoding.",
    "timing/loss_ms": "KL/reconstruction loss computation and KL-component scalar transfers performed in the loss path; excludes final metric/output transfers.",
    "timing/backward_ms": "loss.backward.",
    "timing/optimizer_ms": "optimizer.step.",
    "timing/metrics_ms": "Final total-KL, reconstruction-error, loss-scalar, and optional output transfers after the optimizer update; excludes KL-component transfers counted in loss_ms.",
}

# Memory: MiB from the clean run. Complete warmup and synchronize, then call
# torch.cuda.reset_peak_memory_stats(device) immediately before the block.
# Peaks include retained model/Adam state and allocator reservations; earlier
# setup/first-step high-water marks are excluded, but retained memory still counts.
MEMORY = {
    "memory/peak_allocated_mib": "Peak GPU memory allocated to PyTorch tensors.",
    "memory/peak_reserved_mib": "Peak GPU memory reserved by PyTorch's allocator.",
}


COUNTS = {
    "timing/warmup_iterations": "Warmup updates, including the first step.",
    "timing/measured_iterations": "Updates in the measured block.",
}


def build_spec():
    return {
        "setup": SETUP, "training": TRAINING, "operations": OPERATIONS,
        "memory": MEMORY, "counts": COUNTS,
    }


if __name__ == "__main__":
    print(json.dumps(build_spec(), indent=2))
