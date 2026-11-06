"""Primitive directional communication and its layer implementation."""

import torch

from ..helper import affine, normalize, only_do_for_certain_shapes


# Share only tensors that contain the direction axis.
directional_dims = [
    (example, color, 1, height, width)
    for example in range(2)
    for color in range(2)
    for height in range(2)
    for width in range(2)
]


class DirectionSharePrimitives:
    """Mix information between directional channels using learned projections."""

    def __init__(self, multify):
        def apply_one(dims, x, weights, pre_norm=True, use_bias=False):
            z = normalize(x) if pre_norm else x

            n_directions = dims[3] + dims[4]
            direction_dim = -2 - n_directions
            x_list = list(torch.unbind(x, dim=direction_dim))
            z_list = list(torch.unbind(z, dim=direction_dim))
            coefficients = (1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2)

            for d1 in range(8):
                for d2 in range(8):
                    coefficient = coefficients[(d2 - d1) % 8]
                    x_list[d1] = x_list[d1] + coefficient * affine(
                        z_list[d2], weights[d1][d2], use_bias=use_bias
                    )

            return torch.stack(x_list, dim=direction_dim)

        self._apply = multify(
            only_do_for_certain_shapes(*directional_dims)(apply_one)
        )

    def forward(self, x, weights, *, pre_norm=True, use_bias=False):
        """Apply directional sharing to eligible multitensor components."""
        return self._apply(x, weights, pre_norm=pre_norm, use_bias=use_bias)

    __call__ = forward