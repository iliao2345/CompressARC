"""Original directional sharing with a fused autograd boundary for tied weights.

This changes packing/gradient accumulation, not the directional map or its leaves.
The reference vectorized backend remains available for numerical comparisons.
"""
import torch

from .vectorized import DirectionShareVectorized


class _PackTiedWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, routing, coefficients, *weights):
        channels = weights[0].shape[0]
        ctx.channels = channels
        ctx.save_for_backward(routing)
        # Operations inside this Function do not construct 64 stack-gradient paths.
        blocks = torch.stack(weights).index_select(0, indices)
        blocks = blocks * coefficients[:, None, None]
        return blocks.reshape(8, 8, channels, channels).permute(1, 2, 0, 3).reshape(
            8 * channels, 8 * channels)

    @staticmethod
    def backward(ctx, grad_packed):
        (routing,) = ctx.saved_tensors
        channels = ctx.channels
        # Restore [out, in, Cin, Cout], then accumulate each shared leaf once.
        blocks = grad_packed.reshape(8, channels, 8, channels).permute(2, 0, 1, 3)
        blocks = blocks.reshape(64, channels * channels)
        # Explicit multiply/reduce avoids TF32 changes and atomic scatter sums.
        gradients = (routing[:, :, None] * blocks[None, :, :]).sum(dim=1)
        gradients = gradients.reshape(-1, channels, channels)
        return (None, None, None, *gradients.unbind(0))


class DirectionShareVectorizedPacked(DirectionShareVectorized):
    """Cache only tying/layout metadata; rebuild packed values every forward.

    Original leaf objects, optimizer state, initialization, coefficient values,
    checkpoint parameter ordering, normalization, and biases are preserved.
    Floating-point gradient summation order can differ from the reference path.
    """
    def __init__(self, multify):
        self._packing_plans = {}
        super().__init__(multify)

    def _pack_weights(self, weights, coefficients, channels):
        matrices = tuple(pair[0] for row in weights for pair in row)
        anchor = matrices[0]
        key = (tuple(id(weight) for weight in matrices), anchor.device, anchor.dtype)
        if key not in self._packing_plans:
            unique = []
            lookup = {}
            indices = []
            for weight in matrices:
                if id(weight) not in lookup:
                    lookup[id(weight)] = len(unique)
                    unique.append(weight)
                indices.append(lookup[id(weight)])
            c = (1., .2, .4, .2, 1., .2, .4, .2)
            factors = [c[(i-o) % 8] for o in range(8) for i in range(8)]
            routing = [[factor if indices[block] == leaf else 0.
                        for block, factor in enumerate(factors)]
                       for leaf in range(len(unique))]
            # Build metadata on CPU once. No scalar GPU reads in the hot path.
            to_tensor = lambda data, dtype: torch.tensor(data, dtype=dtype, device='cpu').to(anchor.device)
            self._packing_plans[key] = (
                tuple(unique), to_tensor(indices, torch.long),
                to_tensor(routing, anchor.dtype), to_tensor(factors, anchor.dtype))
        unique, indices, routing, factors = self._packing_plans[key]
        return _PackTiedWeights.apply(indices, routing, factors, *unique)
