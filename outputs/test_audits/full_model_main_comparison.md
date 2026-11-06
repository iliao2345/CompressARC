# Full original-model comparison against main

## Final production verification — 12 October 2026

Softmax and nonlinear now call pre_norm=False to reproduce main's effective wrapper behavior. Direction-share still normalizes its input. Latent normalization now averages normalized_mean**2 + 1e-8, exactly as main does.

The production comparison now passes without diagnostic monkeypatches: **12 passed, 5 diagnostic cases deselected**. It covers exact initialization, full forward outputs/KL, training loss, all captured parameter gradients and Adam-updated weights, using seeds 0 and 42 and both fixed and varying grid shapes on CPU float32.

This validates the original primitive-model path on the tested cases; it does not establish identical training outcomes across hardware, or full-model parity for the combined projected/Triton architecture. The earlier failures below are retained as debugging history.

Reference: frozen CompressARC main snapshot master 8282287. Current: ARC-fafo original primitive model, default four depths and original feature widths. CPU float32. Synthetic strict tasks include fixed and varying grid shapes. Both initializations use the same seed (0 or 42), and sampling RNG is reset before each forward/step.

## Results without changing production behavior

- Initialization: 4 passed. Task tensors, masks, parameter-list lengths and every initial parameter value match exactly.
- Forward: 4 failed. All four cases differ in color logits.
- Training step: 4 failed at the loss comparison. Both backward passes and optimizer steps execute; gradient/updated-weight assertions are not reached after the loss failure.

## First material divergence

For seed 0 on the fixed-shape task:

| Stage | Maximum absolute difference |
| --- | ---: |
| Decoded latents | 4.76837158203125e-07 |
| First share-up | 9.5367431640625e-07 |
| First softmax with configured pre_norm=True | 0.6416597366333008 |
| First softmax with current pre_norm disabled for diagnosis | 1.430511474609375e-06 |

Main's add_residual computes z = normalize(x) when pre_norm=True, then overwrites it with z = affine(x, ...). Pre-normalization therefore has no effect. Current apply_residual correctly calls affine(z, ...). This changes softmax and nonlinear blocks, which the model calls with pre_norm=True.

A separate mathematically equivalent latent-normalization rewrite changes float32 arithmetic ordering: main averages mean**2 + 1e-8, whereas current code adds 1e-8 after averaging mean**2. This creates small rounding differences and prevents strict gradient parity in the diagnostic when only pre-normalization is controlled.

## Controlled diagnostic

A test-only monkeypatch reproduced main's ignored pre-normalization and its original channel_layer arithmetic. On seed 0, fixed-shape task, full forward outputs, KL, loss, every captured parameter gradient and post-Adam weights passed the comparison tolerances. Together with the stage-localization diagnostic: 2 passed.

These temporary patches were reverted by pytest. No production model/layer/training mathematics was changed. Full parity is not established for all tasks/seeds, and no causal claim about pilot solve rates is made.

Test: src/tests/correctness/test_full_model_main_reference.py. Frozen reference sources: src/tests/correctness/reference_main_model/.
