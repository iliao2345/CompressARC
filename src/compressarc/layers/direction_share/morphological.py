import torch
import torch.nn as nn

from ..helper import apply_residual, normalize


class D4DirectionShare(nn.Module):
    """
    Most general D4-equivariant linear map on 8 directional channels.

    Input:
        x: [B, 8, H, W]

    Output:
        y: [B, 8, H, W]
    """

    def __init__(self):
        super().__init__()

        # 10 D4 orbits of (d_out, d_in)
        self.theta = nn.Parameter(torch.randn(10) * 0.01)

        keys = [
            (0, 0, 0),  # cardinal -> cardinal, same
            (0, 0, 2),  # cardinal -> cardinal, 90 deg
            (0, 0, 4),  # cardinal -> cardinal, opposite

            (1, 1, 0),  # diagonal -> diagonal, same
            (1, 1, 2),
            (1, 1, 4),

            (0, 1, 1),  # diagonal -> cardinal, 45 deg
            (0, 1, 3),  # diagonal -> cardinal, 135 deg

            (1, 0, 1),  # cardinal -> diagonal, 45 deg
            (1, 0, 3),  # cardinal -> diagonal, 135 deg
        ]

        key_to_id = {key: i for i, key in enumerate(keys)}

        orbit_id = torch.empty(8, 8, dtype=torch.long)

        for d_out in range(8):
            for d_in in range(8):

                delta = (d_in - d_out) % 8

                # reflection identifies +theta and -theta
                distance = min(delta, 8 - delta)

                key = (
                    d_out % 2,
                    d_in % 2,
                    distance,
                )

                orbit_id[d_out, d_in] = key_to_id[key]

        self.register_buffer("orbit_id", orbit_id)

    def weight(self):
        # [8, 8]
        return self.theta[self.orbit_id]

    def forward(self, x):
        """Mix direction axis 1 for tensors shaped ``[batch, 8, ...]``."""
        W = self.weight()

        # W[out_direction, in_direction]
        return torch.einsum(
            "oi,bi...->bo...",
            W,
            x,
        )


def compressarc_direction_share(dims, x, model):
    """Apply D4 direction mixing with CompressARC's axis layout and residual."""
    if not dims[2]:
        return x
    direction_axis = sum(dims[:2])
    if direction_axis == 0:
        mixed = model(x.unsqueeze(0)).squeeze(0)
    else:
        moved = x.movedim(direction_axis, 1)
        mixed = model(moved).movedim(1, direction_axis)
    return x + mixed


class D4DirectionShareLayer(nn.Module):
    """x + W2 D4(W1 normalize(x)); one D4 mixer per model depth."""

    def __init__(self, multify, n_layers=1, packed=False):
        super().__init__()
        self.models = nn.ModuleList([D4DirectionShare() for _ in range(n_layers)])

        def apply_one(dims, x, weights, *, pre_norm=True, use_bias=False, layer_index=0):
            if not dims[2]:
                return x
            axis = sum(dims[:2])
            model = self.models[layer_index]

            if packed:
                # Direction and channel maps commute. Retain the original leaf
                # parameters; rebuild the differentiable packed map each call.
                z = normalize(x) if pre_norm else x
                channels = x.shape[-1]
                channel_map = weights[0][0] @ weights[1][0]
                direction_map = model.weight()
                packed_weight = (
                    direction_map.T[:, None, :, None]
                    * channel_map[None, :, None, :]
                ).reshape(8 * channels, 8 * channels)
                moved = z.movedim(axis, -2)
                delta = (moved.reshape(-1, 8 * channels) @ packed_weight).reshape(moved.shape)
                return x + delta.movedim(-2, axis)

            def mix(projected):
                if axis == 0:
                    return model(projected.unsqueeze(0)).squeeze(0)
                return model(projected.movedim(axis, 1)).movedim(1, axis)

            return apply_residual(x, weights, mix, pre_norm=pre_norm, use_bias=False)

        self._apply_multitensor = multify(apply_one)

    def forward(self, x, weights, *, pre_norm=True, use_bias=False, layer_index=0):
        if use_bias:
            raise ValueError("D4 direction sharing does not support bias")
        return self._apply_multitensor(
            x, weights, pre_norm=pre_norm, use_bias=False, layer_index=layer_index,
        )
