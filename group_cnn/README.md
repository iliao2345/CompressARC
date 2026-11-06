# Full CompressARC shift comparison

`src.CompressARCShift` accepts the same inputs as the original `layers.shift`:
`x` (MultiTensor), `residual_weights` (MultiTensor), `masks`, and the
`use_bias`, `pre_norm`, `post_norm` keyword flags. It returns a MultiTensor
with the same system and component shapes.

From the `group_cnn` directory, with the ARC-fafo environment active:

```powershell
python -m pytest -q
```

Use your existing CompressARC input, weights and masks in both calls:

```python
import torch

from src.native_reference import reference as layers
from src import CompressARCShift

model = CompressARCShift(layers)
flags = dict(use_bias=False, pre_norm=False, post_norm=False)
expected = layers.shift(x, residual_weights, masks, **flags)
actual = model(x, residual_weights, masks, **flags)

for dims in x.multitensor_system:
    torch.testing.assert_close(actual[dims], expected[dims])
```

The adapter uses fixed convolution taps instead of slicing/concatenation for
shifts. It reuses the original direction handling, masks, shape selection,
projections and residual addition. It supports floating tensors and preserves
their dtype/device. Weights remain caller-owned; this module does not register
the supplied residual weights as its parameters.

The active components have shapes `[examples, colors, 8, X, Y, features]` and
`[examples, 8, X, Y, features]`. Other valid components pass through unchanged.
Use an even projected feature width, as required by the original channel split.
The simpler `TiedDirectionalConv` interface `[B, 2, 4, H, W]` is separate.

To compare your own primitive network while preserving the complete layer:

```python
from src import build_shift_layer
candidate = build_shift_layer(layers, cardinal_fn=my_shift,
                             diagonal_fn=my_diagonal_shift)
actual = candidate(x, residual_weights, masks, **flags)
```

The supplied functions must match `shift_(x, dim, masks)` and
`diagonal_shift_(x, dim1, dim2, masks, shift_amount=1, pad_value=0)`.
They must accept arbitrary tensor axes and retain shape. If they are trainable
modules, register them in your own network/optimizer; the builder is a callable
wrapper, not a parameter container.

Compatibility means reproducing the retained src/compressarc implementation, including its normalization
and cardinal feature selection behavior.

Tests compare all components and input/projection gradients against the actual
reference, using multiple examples/colors, rectangular grids, random values,
partial masks and every combination of bias/pre/post-normalization flags.
