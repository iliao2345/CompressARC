"""Equivalent primitive direction sharing using one packed GEMM per component."""
import torch

from ..helper import normalize, only_do_for_certain_shapes
from ...train.layer_profiling import profile_operation_stage
from .primitives import DirectionSharePrimitives, directional_dims


class DirectionShareVectorized(DirectionSharePrimitives):
    """Pack original pair matrices without introducing or copying parameters.

    Each block (input direction, output direction) contains the original
    coefficient times its channel matrix. Autograd through stack accumulates
    gradients into the original, possibly shared, leaf tensors. The temporary
    packed matrix is rebuilt on every forward; it is never a trainable leaf.
    The reduction order differs from the loop, so parity is numerical, not bitwise.
    """

    def __init__(self, multify):
        self._coefficients = {}

        def apply_one(dims, x, weights, pre_norm=True, use_bias=False):
            with profile_operation_stage("direction_share", "normalization"):
                z = normalize(x) if pre_norm else x
            direction_dim = -2 - dims[3] - dims[4]
            key = (x.device, x.dtype)
            if key not in self._coefficients:
                c = (1., .2, .4, .2, 1., .2, .4, .2)
                self._coefficients[key] = x.new_tensor(
                    [[c[(i-o)%8] for i in range(8)] for o in range(8)])
            coefficients = self._coefficients[key]
            channels = x.shape[-1]
            # [out, in, Cin, Cout] -> [in*Cin, out*Cout]
            with profile_operation_stage("direction_share", "weight_packing"):
                packed = self._pack_weights(weights, coefficients, channels)
            with profile_operation_stage("direction_share", "input_layout"):
                moved = z.movedim(direction_dim, -2)
                flattened = moved.reshape(-1, 8*channels)
            with profile_operation_stage("direction_share", "matmul"):
                delta = (flattened @ packed).reshape(moved.shape)
            if use_bias:
                biases = torch.stack([pair[1] for row in weights for pair in row]).reshape(
                    8, 8, channels)
                delta = delta + (biases * coefficients[..., None]).sum(dim=1)
            with profile_operation_stage("direction_share", "residual"):
                return x + delta.movedim(-2, direction_dim)

        self._apply = multify(only_do_for_certain_shapes(*directional_dims)(apply_one))

    def _pack_weights(self, weights, coefficients, channels):
        blocks = torch.stack([pair[0] for row in weights for pair in row]).reshape(
            8, 8, channels, channels)
        return (blocks * coefficients[..., None, None]).permute(1, 2, 0, 3).reshape(
            8*channels, 8*channels)
