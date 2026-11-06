"""Primitive directional cumulative-maximum operation and layer class."""

import math

import torch

from ..helper import apply_residual, make_directional_layer, only_do_for_certain_shapes
from ..shift.primitives import diagonal_shift_


def cummax_(x, dim, masks):
    """Compute a masked, normalized cumulative maximum along one axis."""
    masks = 1e3 * (1 - masks)
    max_ = torch.max(x - masks, dim=dim, keepdim=True)[0] + masks + 1e-3
    min_ = torch.min(x + masks, dim=dim, keepdim=True)[0] - masks - 1e-3
    x = torch.cummax(x - masks, dim=dim)[0] + masks
    return (x - min_) / (max_ - min_) * 2 - 1


def diagonal_cummax_(x, dim1, dim2, masks):
    """Compute diagonal cumulative maxima with a parallel associative scan."""
    masks_ = 1e3 * (1 - masks)
    min_dim = min(x.shape[dim1], x.shape[dim2])
    n_iters = int(math.ceil(math.log2(min_dim)))

    max_x = x - masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign * 2**i
            shifted_x = diagonal_shift_(
                max_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=-1e3
            )
            max_x = torch.max(max_x, shifted_x)
        if sign == 1:
            cummax_x = max_x + masks_
    max_x = max_x + masks_

    min_x = x + masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign * 2**i
            shifted_x = diagonal_shift_(
                min_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=1e3
            )
            min_x = torch.min(min_x, shifted_x)
    min_x = min_x - masks_
    return ((cummax_x - min_x) / (max_x - min_x + 1e-5) * 2 - 1) * masks


class CummaxPrimitives:
    """Directional cummax layer assembled from primitive scan operations."""

    def __init__(self, multify):
        directional_cummax = make_directional_layer(cummax_, diagonal_cummax_)

        def apply_one(dims, x, weights, masks, **kwargs):
            def operation(projected):
                return directional_cummax(dims, projected, masks)

            return apply_residual(x, weights, operation, **kwargs)

        active_shapes = ((1, 1, 1, 1, 1), (1, 0, 1, 1, 1))
        self._apply = multify(
            only_do_for_certain_shapes(*active_shapes)(apply_one)
        )

    def forward(self, x, weights, masks, **kwargs):
        """Apply primitive cummax to active multitensor shapes."""
        return self._apply(x, weights, masks, **kwargs)

    __call__ = forward