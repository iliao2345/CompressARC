# Packed combined D4 remaining runs

One model: learned D4-tied convolution plus learned projected D4 direction sharing.
`direction_share_implementation=d4`, `direction_share_packed=true`.
The packed matrix is rebuilt differentiably from the existing projection and orbit parameters;
no parameters or checkpoint keys are replaced. Hard cummax remains the original operator math
with the grouped Triton backend. Seed 0, 1500 steps, frozen 40 tasks, no checkpoint/model saving.

The checked-in manifest is a W&B snapshot, not a scheduler lock. Refresh on M3 before submitting.
Cancel only the old combined array's pending tasks first; running tasks can finish normally:

```bash
git pull
module load miniforge3
mamba activate SymARC
scancel --state=PENDING --name=arc-combined-d4-s0
python experiments/prepare_combined_d4_packed_remaining.py
sbatch experiments/seed0_combined_d4_packed_remaining/submit.slurm
```

Do not submit if the generator fails or reports no remaining tasks. The generated array runs
two tasks concurrently, one GPU per task. W&B completed full-length and active runs are excluded;
failed or short runs remain eligible. Selection never uses accuracy. New runs retain the combined
cohort tag and add `combined-d4-packed-remaining`; their config and Git commit are recorded.
W&B may lag live processes; inspect the refreshed active exclusions before submission.

Audit: 59 CPU tests passed, 14 CUDA tests skipped locally. Includes D4 convolution contracts,
packed direction-action symmetry, float32/float64 outputs and gradients versus the old path,
and both operators receiving gradients and updating in one model. This is not a proof of
whole-model spatial equivariance. GPU tests must be run on M3.

One measured task: 269.45 ms existing vs 266.93 ms packed (~0.9% difference).
This is not an established general speedup or a guarantee of identical training trajectories.
