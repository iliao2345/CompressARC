# M3 Slurm pilot variants

Independent task training: each array element owns one task-specific model and optimizer on one GPU. Parameters/gradients are not shared between jobs. This is not DDP training of a single model.

The array covers the original eight pilot tasks and two corrected seed-0 configurations:

- `conv_d4_cummax_pilot_8_corrected.yaml`: projected conv2d shift + projected D4 + original hard cummax.
- `all_projected_triton_pilot_8_corrected.yaml`: projected conv2d shift + projected D4 + projected Triton LSE (axis and diagonal), tau 0.1.

Both use 1500 updates, strict multitensors, per-iteration top-1/pass@2, images every 10 updates, final .pt checkpoints and W&B artifacts. Pull the corrected code before submitting; commit your local changes first.

## Prepare and submit

From the repository root on M3, load Miniforge and activate your existing `SymARC` environment, then install dependencies once and log into W&B:

```bash
module load miniforge3
mamba activate SymARC
python --version
python -m pip install -r src/requirements.txt
sbatch src/slurm/pilot_variants.sbatch
```

The script uses a login Bash shell, runs `module load miniforge3` and `mamba activate SymARC` in each job, and uses that environment's Python. It requests `--partition=gpu --gres=gpu:1`, four CPUs, 16 GB host RAM and four hours per task. Supply your usual account/QoS flags if required, for example `sbatch --account=YOUR_PROJECT ...`. These resources are defaults to adjust for your allocation; no remote submission is performed by the local changes.

The default `--array=0-15%2` runs at most two GPU jobs concurrently. To request up to four concurrent jobs:

```bash
sbatch --array=0-15%4 src/slurm/pilot_variants.sbatch
```

To keep all jobs on one GPU model, add a GPU type permitted by your account, such as `--gres=gpu:A40:1`. Do not assume the L4 used by your previous interactive allocation is present in M3's batch GPU partition. GPU type/software differences can affect runtime and numerical trajectories.

## Mapping and outputs

Task order is `0dfd9992`, `22eb0ac0`, `239be575`, `50cb2852`, `54d82841`, `623ea044`, `c1d99e64`, `ec883f72`.

For task number k (zero-based), array index 2k runs hard cummax and 2k+1 runs Triton LSE. Thus 16 elements train 16 independent models. Each worker receives `--shard-count 16 --shard-index "$SLURM_ARRAY_TASK_ID"`; no worker launches all tasks.

- Logs: `slurm-<array-job-id>_<index>.out` and `.err` at the repository root.
- Weights: `outputs/m3_variants/<array-job-id>/<config-stem>/<task-id>.pt`.
- W&B: one run per task/config, with corrected configuration names and tags.

Each submission has its own output directory, and the launcher separates checkpoint paths by configuration to avoid overwriting between variants. A failed process exits nonzero so Slurm reports it as failed.

Monitor with:

```bash
squeue -u "$USER"
sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode
```

Resubmit only selected failed elements using their original indices, e.g. `sbatch --array=10,11 ...` for the two `623ea044` configurations. Each submission creates new W&B runs and a new checkpoint directory.

## Preview without training or CUDA

```bash
python src/run_training_timing.py --mode train \
  --config src/config/models/conv_d4_cummax_pilot_8_corrected.yaml \
  --config src/config/models/all_projected_triton_pilot_8_corrected.yaml \
  --shard-count 16 --shard-index 1 --save-checkpoints --dry-run
```

This prints only `0dfd9992` with projected Triton LSE. Omit the shard flags to run both variants sequentially through the same trainer.

Sources: [M3 GPU allocation](https://docs.erc.monash.edu/Compute/HPC/M3/RunningJobsOnM3/GPUs/) and [Slurm job arrays](https://slurm.schedmd.com/job_array.html).
