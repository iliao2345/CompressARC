"""Fixed up/down communication operations over the multitensor system."""

import torch
from ...model import multitensor_systems

from ..helper import affine, normalize
from ...train.layer_profiling import profile_share_stage


class MultitensorShare:
    """Primitive multitensor sharing operations used by the model."""
    def share_up(self, residual, share_up_weights):
        """Gather information from lower-dimensional multitensor components."""
        return self.share_direction(residual, share_up_weights, direction=1)
    
    
    def share_down(self, residual, share_down_weights):
        """Gather information from higher-dimensional multitensor components."""
        return self.share_direction(residual, share_down_weights, direction=-1)

    
    def share_direction(self, residual, share_weights, direction):
        """Communicate between ordered multitensor components.

        ``direction=1`` gathers from lower-dimensional components; ``direction=-1``
        gathers from higher-dimensional components. Projection weights remain owned
        by the model and are supplied at call time.
        """
        if direction not in (1, -1):
            raise ValueError("direction must be 1 (up) or -1 (down)")

        component = 'share_up' if direction == 1 else 'share_down'
        with profile_share_stage(component, 'weight_selection'):
            down_project_weights = multitensor_systems.multify(
                lambda dims, weights: weights[0]
            )(share_weights)
            up_project_weights = multitensor_systems.multify(
                lambda dims, weights: weights[1]
            )(share_weights)

        multitensor_system = residual.multitensor_system
        with profile_share_stage(component, 'source_projection'):
            projected = affine(residual, down_project_weights, use_bias=False)

        if direction == 1:
            def share(dims, _):
                lower_xs = []
                for lower_dims in multitensor_system:
                    if all(lower_axis <= axis for lower_axis, axis in zip(lower_dims, dims)):
                        lower_x = projected[lower_dims]
                        for dim, (lower_axis, axis) in enumerate(zip(lower_dims, dims)):
                            if lower_axis < axis:
                                lower_x = torch.unsqueeze(lower_x, sum(dims[:dim], 0))
                        lower_xs.append(lower_x)
                return sum(lower_xs)
        else:
            def share(dims, _):
                higher_xs = []
                task = projected.multitensor_system.task
                known_shared_size = task.in_out_same_size or task.all_out_same_size
                for higher_dims in multitensor_system:
                    if all(higher_axis >= axis for higher_axis, axis in zip(higher_dims, dims)):
                        higher_x = projected[higher_dims]
                        for dim, (higher_axis, axis) in reversed(
                            list(enumerate(zip(higher_dims, dims)))
                        ):
                            if higher_axis > axis:
                                reduction_axis = sum(higher_dims[:dim], 0)
                                if higher_dims[0] == 1 and known_shared_size and dim == 3:
                                    masks = task.masks
                                    masks = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
                                    for _ in range(sum(higher_dims[1:3])):
                                        masks = masks[:, None, ...]
                                    if dims[4] == 0:
                                        masks = masks[..., 0]
                                    masks = masks[..., None]
                                    higher_x = torch.sum(
                                        higher_x * masks, dim=reduction_axis
                                    ) / (torch.sum(masks, dim=reduction_axis) + 1e-4)
                                elif higher_dims[0] == 1 and known_shared_size and dim == 4:
                                    masks = task.masks
                                    masks = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
                                    for _ in range(sum(higher_dims[1:3])):
                                        masks = masks[:, None, ...]
                                    if higher_dims[3] == 0:
                                        masks = masks[..., 0, :]
                                    masks = masks[..., None]
                                    higher_x = torch.sum(
                                        higher_x * masks, dim=reduction_axis
                                    ) / (torch.sum(masks, dim=reduction_axis) + 1e-4)
                                else:
                                    higher_x = torch.mean(higher_x, dim=reduction_axis)
                        higher_xs.append(higher_x)
                return sum(higher_xs)

        with profile_share_stage(component, 'communication'):
            communicated = multitensor_systems.multify(share)(projected)
        with profile_share_stage(component, 'normalization'):
            communicated = normalize(communicated)
        with profile_share_stage(component, 'destination_projection'):
            communicated = affine(communicated, up_project_weights, use_bias=False)
        with profile_share_stage(component, 'residual'):
            return multitensor_systems.multify(lambda dims, x, y: x + y)(residual, communicated)

