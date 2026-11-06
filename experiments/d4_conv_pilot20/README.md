# D4-constrained convolution pilot

Same original 20 tasks, training seed 0, 1500 steps. The only model-setting difference from the frozen C4 projected-conv config is shift_implementation=d4_tied_conv. The existing C4 version is untouched and remains available as c4_tied_conv or legacy tied_conv. C4 weights are NOT silently converted: each pilot run trains from scratch.

Canonical rightward kernels are projected onto up/down-reflection symmetry: (K + flip_rows(K))/2. Canonical down-right kernels are projected onto diagonal-transpose symmetry: (K + K.T)/2. Each projected kernel then generates its four rotated directions. Constraints are imposed differentiably at every forward pass, so arbitrary updates to raw parameters cannot break them. There are 18 raw scalar parameters per depth but 12 effective kernel degrees of freedom; this is a different hypothesis class from C4, not a behavior-preserving rename.

From ARC-fafo root, after committing, pushing and pulling on M3:

```bash
sbatch experiments/d4_conv_pilot20/submit.slurm
```

20 array elements, at most 2 GPU jobs concurrently, one A40 per run, 12-hour per-run limit, existing gpu/vf38/normal settings and email alerts. These are two NEW GPU jobs alongside existing arrays, subject to scheduler limits. W&B names are <task>-training-projected_conv_d4_pilot20-seed0, with D4-conv and d4-conv-pilot20 tags. Existing C4 names/results are preserved.

Preview one command without training:

```bash
python experiments/d4_conv_pilot20/run_job.py --index 0 --dry-run
```

The test contract covers core and projected-layer D4 actions under joint spatial/directional/mask transforms, optimizer updates, output/input/kernel/projection gradients and state round trips. This does not assert full ARC model prediction equivariance. All local checks use CPU; CUDA remains to be verified in M3. The remaining 60 task/seed combinations are deliberately not submitted as part of this pilot.

Per-step wall timing is now enabled for future launches. Detailed profiling: add `--step-timing phases` to run_job.py. Timing files and summaries are written beside checkpoints, and W&B receives timing/* metrics. Existing running jobs keep their imported code/config.
