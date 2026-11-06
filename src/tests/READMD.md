


Layer tests cover these five main areas, plus optional diagnostics:

| Pillar / Check | Objective | What It Validates | Example Test / Method |
| :--- | :--- | :--- | :--- |
| **1. Special-Case Correctness** | Anchor advanced layers to basic primitives | Guarantees the generalized layer reduces to standard operations (e.g., `torch.cummax` or identity) at structural limits. | Zeroing out the kernel or setting $\tau \to 0$ to match native cumulative max. |
| **2. Equivariance** | Ensure spatial or group symmetry | Verifies predictions transform correctly under symmetry group transformations (rotations/reflections). | Testing outputs against permuted inputs across $D_4$ group actions (`product(range(4), [False, True])`). |
| **3. Group Weight Correctness** | Map parameters to proper geometric orbits | Confirms parameters route correctly to specific directions based on distance and symmetry constraints. | Asserting expected weight values across directional orbits given distinct parameters. |
| **4. Outputs & Gradients** | Ensure numeric validity and backpropagation | Checks that forward passes yield finite values and backward passes produce non-zero, valid gradients. | Asserting `torch.isfinite` and checking gradient shapes for model parameters and inputs. |
| **5. Adapter Logic & Integration** | Handle dimensions, residuals, and pass-throughs | Validates multi-tensor wrappers, slice operations, and clean handling of dimensions with/without target axes. | Passing synthetic multi-tensors and checking if non-target axes bypass or correctly apply residual additions. |
| **6. Extensions & Diagnostics** | Monitor metadata, timing, and debugging flags | Ensures optional profiling/timing flags record accurately without altering core functional outputs. | Monkeypatching timers or checking that timing dictionaries populate correctly when requested. |


The layer suites are grouped by operation:

```text
correctness/layers/
  direction_share/
    test_direction_share.py
  lse/
    test_lse.py
    test_triton_lse.py
  shift/
    test_shift.py
```

Core checks and projected-layer integration share one suite for direction-share
and shift. LSE keeps its optional CUDA suite separate. Shift skips only its
Triton cases when Triton or CUDA is unavailable.

Run from the repository root:

```bash
python -m pytest src/tests/correctness/layers -q
python -m pytest src/tests/correctness/layers/direction_share -q
python -m pytest src/tests/correctness/layers/lse -q
python -m pytest src/tests/correctness/layers/shift -q
```

| Check | LSE | Direction share | Shift |
| :--- | :--- | :--- | :--- |
| Correctness | Coordinate-based LSE reference | Primitive special-case equivalence | Primitive pulse shifts and tied-convolution parity |
| Equivariance | Quarter-turn checks | D4 rotations and reflections | Dedicated check still needed |
| Group weight correctness | Shared kernels, gradient accumulation and optimizer checks | Direction-pair orbit mapping | Pulse direction alignment; dedicated tying/accumulation check still needed |
| Outputs and gradients | Preoptimized/optimized and CUDA reference parity | Core gradients, primitive parity and projected equation | Conv2d/unfold/Triton parity; primitive layer backward reference check still needed |
| Adapter logic and integration | Masks, projections and residuals | Axis layouts, passthrough, projections and optimizer registration | Masks, normalization, biases, projections, residuals and passthrough |
| Extensions and diagnostics | Timing flags and CUDA checks | No dedicated diagnostic checks | Triton input validation |
