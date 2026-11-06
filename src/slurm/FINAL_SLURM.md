# A40 Slurm submission

The current scripts use the user's GPU-partition template, adapted to one
independent model per Slurm task:

```bash
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
```

A single model uses one GPU; eight Slurm tasks would otherwise launch duplicate
processes with the existing srun commands. The desktopq QoS is no longer used.
Monash documents `--partition=gpu --gres=gpu:A40:1` for A40 batch jobs:
https://docs.erc.monash.edu/Compute/HPC/M3/RunningJobsOnM3/GPUs/
Check your association's normal QoS limits with `mon_qos` if required.

## Single task first

From the updated ARC-fafo repository root:

```bash
sbatch src/slurm/final_single.sbatch
```

This runs task `74dd1130`, original control, seed 0, 1500 steps, with a one-hour
limit and all configured W&B metrics/model uploads. Override its arguments:

```bash
sbatch src/slurm/final_single.sbatch 74dd1130 projected-conv 0
```

Variants: original, projected-conv, projected-d4, projected-lse. Model outputs
use `outputs/final_single/<job-id>/<config-name>/seed_<seed>/`.

## Four task-group arrays

```bash
sbatch src/slurm/final_tasks_01.sbatch
sbatch src/slurm/final_tasks_02.sbatch
sbatch src/slurm/final_tasks_03.sbatch
sbatch src/slurm/final_tasks_04.sbatch
```

Each array owns five distinct tasks and covers all four variants and three seeds
for those tasks: 60 elements per array, 240 independent 1500-update runs overall.
Each element has a four-hour limit. `--array=0-59%1` permits one active GPU per
array (up to four across the submissions), subject to normal QoS limits. All
array elements count towards submitted-job limits even with a concurrency cap.
Index = 12 * local_task_index + 3 * variant_index + seed_index. The variant order
is original, projected convolution, projected D4, projected LSE; seeds are 0,1,2.

| Script | Tasks |
|---|---|
| `final_tasks_01.sbatch` | `3eda0437`, `0dfd9992`, `95990924`, `7f4411dc`, `6f8cd79b` |
| `final_tasks_02.sbatch` | `445eab21`, `3bdb4ada`, `36fdfd69`, `d43fd935`, `1c786137` |
| `final_tasks_03.sbatch` | `1bfc4729`, `3aa6fb7a`, `6cdd2623`, `74dd1130`, `f8c80d96` |
| `final_tasks_04.sbatch` | `623ea044`, `d406998b`, `6d75e8bb`, `d9fac9be`, `97999447` |

Checkpoints:
`outputs/final_ablation/tasks_<group>/<array-job-id>/<config-name>/seed_<seed>/<task>.pt`.
W&B: `arc_agi/compressarc-final-ablation`, with task, variant and seed in run
names/tags. Existing metrics and inference model artifacts are unchanged.
Resubmission is a fresh run; checkpoints do not resume optimizer state.

## Environment and zero-step check

Each script loads `miniforge3` and activates `SymARC`. Install dependencies and
log into W&B once in that environment. These scripts now request A40 uniformly;
do not combine old L4/desktop scripts with the A40 experiment.

```bash
module load miniforge3
mamba activate SymARC
python -m pip install -r src/requirements.txt
wandb login
sbatch src/slurm/test_zero_steps.sh
```

The zero-step test has a 15-minute limit and checks all four variants/seeds on
small task `74dd1130`, with no optimizer updates, W&B runs or uploads. It sets
PyTorch's global device and checks tensor placement. `--init-only` skips forward;
`--seed 0` tests one seed. Inside an existing GPU allocation, run
`bash src/slurm/test_zero_steps.sh`. Set `ARC_SKIP_ENV=1` if your environment is
already active. Use `--check-root` to test repository discovery only.

Monitor with `squeue -u "$USER"` and
`sacct -j JOB_ID --format=JobID,State,ExitCode`. Retry array elements with
`sbatch --array=5,17 src/slurm/final_tasks_01.sbatch`.
No jobs were submitted by creating or validating these scripts.

## Email alerts

All final scripts request BEGIN, END and FAIL alerts. The recipient defaults to teedsingyau@gmail.com. Override it at submission
with --mail-user=YOUR_EMAIL, or export SBATCH_MAIL_USER before submitting.
For arrays these notifications apply to the whole array by default.
