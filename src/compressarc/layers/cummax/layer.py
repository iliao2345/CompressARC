"""Multitensor projected residual wrappers for independent LSE implementations."""
import torch
from torch import nn

from ..helper import apply_residual, only_do_for_certain_shapes


def directional_lse(dims, x, masks, model):
    """Apply the matching LSE direction to each CompressARC direction/feature half."""
    if x.shape[-1] % 2:
        raise ValueError('LSE directional features must have even width')
    direction_axis = sum(dims[:2])
    height_axis = direction_axis + 1
    width_axis = direction_axis + 2
    valid = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
    for _ in range(sum(dims[1:3])):
        valid = valid[:, None]
    valid = valid[..., None]
    masked = x * valid
    mappings = ((6, 7, 0, 1, 2, 3, 4, 5), (2, 3, 4, 5, 6, 7, 0, 1))
    halves = []
    for half, mapping in enumerate(mappings):
        outputs = []
        for direction, output_direction in enumerate(mapping):
            grid = torch.select(masked, direction_axis, direction)[..., half::2]
            grid = grid.movedim(-1, -3)
            shape = grid.shape
            selected = model.direction(
                grid.reshape(-1, shape[-2], shape[-1]), output_direction
            ).reshape(shape).movedim(-3, -1)
            outputs.append(selected)
        halves.append(torch.stack(outputs, dim=direction_axis))
    result = torch.cat(halves, dim=-1)
    return result * valid



class LSELayer(nn.Module):
    """Adapt either LSE core to the model's layer interface."""

    def __init__(self, multify, height, width, *, model_type, n_layers=1, tau=0.1, timing=False):
        super().__init__()
        self.models = nn.ModuleList([
            model_type(height, width, tau=tau, timing=timing) for _ in range(n_layers)
        ])

        def apply_one(dims, x, weights, masks, *, layer_index=0, **kwargs):
            return apply_residual(
                x, weights,
                lambda z: directional_lse(dims, z, masks, self.models[layer_index]),
                **kwargs,
            )

        self._apply_multitensor = multify(
            only_do_for_certain_shapes((1, 1, 1, 1, 1), (1, 0, 1, 1, 1))(apply_one)
        )

    def forward(self, x, weights, masks, *, layer_index=0, **kwargs):
        return self._apply_multitensor(x, weights, masks, layer_index=layer_index, **kwargs)
