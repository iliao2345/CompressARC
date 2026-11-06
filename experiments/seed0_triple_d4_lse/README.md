# Triple D4 convolution, direction sharing, and LSE

One model: `shift_implementation=d4_tied_conv`, `direction_share_implementation=d4`,
`direction_share_packed=true`, `cummax_implementation=triton`, `lse_tau=0.1`.
Same frozen 40 training tasks and original20/additional20 cohorts as the existing combined experiment.
Training seed 0 only, 1500 steps per task. No checkpoints or model artifacts.
Existing candidate, correctness, kernel, numerical, and wall-time metrics remain enabled.

## 1. Profile first on M3

Sync these files to the M3 checkout before submission. From the ARC-fafo repository root:

```bash
sbatch experiments/seed0_triple_d4_lse/profile.slurm
```

This submits **three array elements**, at most one concurrently. Each element compares
the hard-cummax combined model and the new triple model on the **same allocated GPU**.
Tasks are selected only by max-height times max-width among the frozen 40:

| Index | Task | Size proxy | Role |
|---|---|---:|---|
| 0 | 25ff71a9 | 9 | smallest |
| 1 | 6d75e8bb | 154 | upper median |
| 2 | 1190e5a7 | 729 | largest |

The size proxy is a shape-based sampling rule, not a claim about runtime ordering.
For each variant, three fresh processes run 120 real training steps with seed 0.
The first 20 steps are omitted; mean, median and p90 use the remaining 100.
Variant order alternates across repeats. Full-step wall latency includes candidate decoding,
numerical diagnostics and normal W&B logging; timing-export overhead is excluded.
There are no operator instrumentation spans in these passes.
Afterwards, a separate triple-model process captures one operator-profile step after 20 warmup steps.
Use that trace for attribution, not as baseline latency. Forward/backward and other phase scopes,
layer reports and CUDA traces come from the existing OperatorProfiler.

Output:
`outputs/seed0_triple_d4_lse/latency/<array_job_id>/<task>/comparison.json`.
Each task also has execution provenance (Git commit, config hashes, GPU, Torch/CUDA versions),
per-repeat timing JSONL, aggregate wall results and separate operator traces.
Reports show the triple/hard latency ratio and a per-task 1500-step extrapolation.
That extrapolation excludes setup and cannot establish total runtime over all 40 tasks.
GPU models may differ between array elements; compare variants within each element.

W&B profile runs use `triple_latency_*` names and `latency-only`/`triple-d4-lse-profile` tags.
They are shortened profiling runs, not final accuracy results. Traces and aggregate reports
are local M3 outputs; scalar histories are logged to W&B. No model weights are saved.

The Slurm scripts preserve the existing gpu partition, vf38 account, normal QoS, one GPU,
four CPUs and 16 GB host memory. Profiling has a four-hour limit per task; training has twelve.
If a fixed GPU model is required, supply an allocation-supported typed `--gres` override.

Preview commands without a GPU or submission:

```bash
python experiments/seed0_triple_d4_lse/profile.py --index 1 --dry-run
python experiments/seed0_triple_d4_lse/run_job.py --index 39 --dry-run
```

## 2. Review latency, then run the final experiment

The profile job never submits training automatically. After reviewing the reports:

```bash
sbatch experiments/seed0_triple_d4_lse/submit.slurm
```

This is exactly 40 independent task-specific models, seed 0, 1500 steps, at most two GPUs
concurrently. W&B project: `arc_agi/compressarc-final-ablation`.
Experiment name: `triple_d4_conv_sharing_lse_seed0_sample40`.
Distinct variant/tag: `triple-d4-conv-sharing-lse`.
Workers record the effective config, Git commit and execution command, and do not resume checkpoints.
Report original20 and additional20 separately; the original20 were performance-selected.

## Validation

Local verification: **121 passed, 15 skipped** across triple-model integration, existing combined
model, packed direction sharing and LSE suites. CPU integration checks finite loss, nonzero finite
gradients and parameter updates in all three operators together. It uses the mathematical PyTorch
LSE backend and unfused Adam; the deployed CUDA/Triton path is covered by the CUDA integration
case and must pass on M3. The profiling Slurm job runs that integration suite before measurement.

Manual CUDA preflight:

```bash
module load miniforge3
mamba activate SymARC
PYTHONPATH=src python -m pytest src/tests/correctness/training/test_triple_d4_lse_model.py -q
```

No GPU latency was measured locally: local Torch is CPU-only.
