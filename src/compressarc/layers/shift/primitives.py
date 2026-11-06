"""Primitive cardinal and diagonal shift operations and their layer class."""

import torch

from ..helper import (
    apply_residual,
    make_directional_layer,
    only_do_for_certain_shapes,
)


def shift_(x, dim, masks):
    """Shift a tensor by one cell along ``dim``, padding with zeros."""
    padding = torch.zeros_like(torch.narrow(x, dim, 0, 1))
    narrowed = torch.narrow(x, dim, 0, x.shape[dim] - 1)
    return torch.cat([padding, narrowed], dim=dim)


def diagonal_shift_(x, dim1, dim2, masks, shift_amount=1, pad_value=0):
    """Shift along both spatial axes, padding exposed cells with ``pad_value``."""
    for dim in (dim1, dim2):
        padding = pad_value + torch.zeros_like(
            torch.narrow(x, dim, 0, abs(shift_amount))
        )
        if shift_amount >= 0:
            narrowed = torch.narrow(x, dim, 0, x.shape[dim] - shift_amount)
            x = torch.cat([padding, narrowed], dim=dim)
        else:
            narrowed = torch.narrow(x, dim, -shift_amount, x.shape[dim] + shift_amount)
            x = torch.cat([narrowed, padding], dim=dim)
    return x


class ShiftPrimitives:
    """Shift layer implementation assembled from primitive operations."""

    def __init__(self, multify):
        directional_shift = make_directional_layer(shift_, diagonal_shift_)

        def apply_one(dims, x, weights, masks, **kwargs):
            def operation(projected):
                return directional_shift(dims, projected, masks)

            return apply_residual(x, weights, operation, **kwargs)

        active_shapes = ((1, 1, 1, 1, 1), (1, 0, 1, 1, 1))
        self._apply = multify(
            only_do_for_certain_shapes(*active_shapes)(apply_one)
        )

    def forward(self, x, weights, masks, **kwargs):
        """Apply the primitive shift to active multitensor shapes."""
        return self._apply(x, weights, masks, **kwargs)

    __call__ = forward