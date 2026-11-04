# ARC-AGI-2 compressor: interview/presentation brief

One page. Everything here is checkable in the repo; numbers are measured, not estimated.

## What the system is

A per-task latent compressor for ARC-AGI-2. Each puzzle is trained from scratch for 1500
Adam updates. Four ingredients make it a D4-equivariant compressor:

1. **D4-tied convolution** (`shift_implementation: d4_tied_conv`) - the 8 rotations/reflections
   share one conv kernel instead of 8 free ones.
2. **Projected D4 direction sharing** (`direction_share_implementation: d4`, `packed: true`) -
   one packed GEMM per component, with a fused autograd boundary for the tied leaves.
3. **Cummax scan** - either hard (`triton_hard_grouped`) or a log-sum-exp relaxation
   (`triton`, `lse_tau: 0.1`) used for the LSE variant.
4. **CUDA Graph replay** - forward/loss/backward/grad-zero captured after annealing
   (step 100); 100 eager updates, then 1400 replays. Adam, metrics, geometry decoding and
   logging stay eager.

## The engineering result worth talking about

The graph path is only worth using if replay is *numerically* the same computation as eager.
So there is a preflight gate on a small real model that compares eager against replay:
outputs, loss terms, gradients, Adam state, update counts, RNG advancement, capture cost.

**It failed on the LSE arm, before any graph existed.** Two eager runs from bit-identical
parameters diverged at step 1-2 by 1.4e-4, later 1.4e-2.

Cause, in two parts:

- The LSE backward accumulates the shared scan-kernel gradient with `tl.atomic_add`
  (`axis_lse_bwd_dk_kernel`, `diagonal_lse_bwd_dk_kernel`). The summation order depends on
  kernel scheduling, so the last bits of that gradient are not reproducible run-to-run.
- Adam turns that into a large parameter move. With `lr=0.01`, `betas=(0.5,0.9)`, `eps=1e-8`
  and `|g| ~ 1e-8`, the first update is `|dtheta| = lr*|g|/(|g|+eps)`, which is ~`lr/2` and
  almost independent of `|g|`. A *sign flip* of that gradient is `2e-8` wide - it passes a
  `1e-4` gradient tolerance comfortably - and it moves the parameter by exactly `lr`.
  So the gate's gradient tolerance was far wider than the parameter tolerance it had to survive.

**Conclusion:** strict multi-step parameter equality is not a valid replay check for this
kernel. The fix compares one update at a time from synchronized parameters *and* optimizer
state, keeps the original output/gradient tolerances, verifies Adam against the candidate's
own gradients with a second eager optimizer, and measures the eager/eager noise floor in the
same run so a failure can be classified as roundoff or as a real replay bug. The gate states
explicitly that it does not claim multi-update trajectory identity.

## Second finding (process, not numerics)

The Kaggle notebook does not read the preflight from the attached dataset - it **overlays**
its own embedded copy of 11-12 source files, with a SHA-256 allow-list. A local fix therefore
never reached the GPU unless it was re-embedded. That is why the same AssertionError survived
a "fix". The notebook now carries the gate module and a digest for both the old and new
preflight, so the guard still rejects unknown sources.

## Evidence you can point at

- `src/cuda_graph_preflight.py --variant {conv,combined,combined_lse}`.
- `src/cuda_graph_lse_validation.py` - the same-state gate and `verify_adam_update`.
- `src/tests/correctness/training/test_cuda_graph_lse_validation.py` - includes the test that
  asserts a `1e-8` sign flip (within `atol=1e-4`) changes the parameter by >0.009.
- CPU suite: 18 passed, 1 skipped (CUDA-only).
- `src/CUDA_GRAPH.md` and `kaggle/run_a_lse_4tasks.ipynb`, `kaggle/run_b_hard_4tasks.ipynb`.

## What is *not* claimed

- No local CUDA validation (dev box is CPU-only Torch); the gate runs on the target GPU.
- Graph parity is not an accuracy claim. Coverage, not replay, decides the score.
- Per-task training is stochastic; identical configs give slightly different candidates.

## Five questions you should be able to answer cold

1. **Why CUDA Graphs here?** 1400 of 1500 updates per task are the same shapes; replay removes
   per-kernel launch overhead. Adam stays eager because its state updates are cheap and
   keeping them out makes capture simpler and lets diagnostics read real grads.
2. **Why not just compare losses?** A replay bug can preserve the loss (same kernels, stale
   parameters) and still corrupt gradients. Hence gradients, Adam state and update counts.
3. **Isn't relaxing the gate cheating?** No: the relaxed gate still pins outputs, loss terms,
   gradients, Adam arithmetic, step counts and RNG advancement. What it refuses to assert is
   something the hardware cannot deliver - bit-identical Adam trajectories from an atomic
   reduction. That is a property of the kernel, not of the graph.
4. **How would you prove the graph is right?** Same-state single-step parity against eager,
   plus an independent Adam reference recomputed from the candidate's exact gradients, plus
   the measured eager/eager floor. All three are in the report JSON.
5. **What would you do with more time?** Replace the atomic `dk` accumulation with a
   deterministic two-stage reduction; that removes the nondeterminism at the source and
   restores the strict 8-step trajectory gate. It also makes a submission reproducible.

## Numbers to have ready

| quantity | value |
|---|---|
| updates per task | 1500 (100 eager + 1400 replayed) |
| learning rate / betas / eps | 0.01 / (0.5, 0.9) / 1e-8 |
| observed eager-vs-eager drift at step 1-2 | 1.4e-4 |
| observed later drift | 1.4e-2 |
| parameter move from a sign flip at g=1e-8 | 0.01 (= lr) |
| median step time (L4, eager benchmark) | 0.69 s (p90 1.44 s) |
| full run wall clock (previous) | ~21,500 s (~6 h) |
