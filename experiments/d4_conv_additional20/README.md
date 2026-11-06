# D4-conv additional20, seed 0

Runs the exact fixed, performance-independent additional20 cohort from
`experiments/additional20/manifest.json`, without resampling. Only the shift
implementation is D4-constrained projected conv; all other model/training
settings match the D4-conv pilot (1500 steps, seed 0, wall timing on).
Operator profiling is disabled. Distinct W&B experiment name:
`projected_conv_d4_additional20`, tag `d4-conv-additional20`.

From the repository root on M3:

```bash
sbatch experiments/d4_conv_additional20/submit.slurm
```

Twenty array elements, at most two concurrently, one GPU per element.
Requests any available GPU type, account vf38, partition gpu, QoS normal,
4 CPUs, 16G memory, 4-hour limit per element. BEGIN/END/FAIL email alerts
are sent to teedsingyau@gmail.com. Hardware-dependent timing may vary.
Checkpoints go to `outputs/d4_conv_additional20/checkpoints/<array-id>/...`.

```bash
python experiments/d4_conv_additional20/run_job.py --index 0 --dry-run
```
