import torch
import torch.nn as nn


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
