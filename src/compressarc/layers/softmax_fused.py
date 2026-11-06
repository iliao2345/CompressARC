"""Native fused softmax over the original semantic-axis subsets."""
import itertools
import math
import torch

from .helper import apply_residual
from ..model.multitensor_systems import multify


def softmax_operation_fused(dims, x):
    axes = list(range(sum(dims)))
    if dims[0]:
        axes.pop(0)  # Keep examples independent, exactly as the original.
    outputs = []
    for size in range(1, len(axes)+1):
        for subset in itertools.combinations(axes, size):
            if len(subset)==1:
                result = torch.softmax(x, dim=subset[0])
            else:
                # The channel axis stays outside the reduction. Each row is
                # one joint semantic subset; restore axes before concatenating.
                remaining = [axis for axis in range(x.ndim) if axis not in subset]
                order = remaining + list(subset)
                moved = x.permute(order)
                reduction = math.prod(x.shape[axis] for axis in subset)
                rows = moved.reshape(-1, reduction)
                result = torch.softmax(rows, dim=-1).reshape(moved.shape)
                inverse = [order.index(axis) for axis in range(x.ndim)]
                result = result.permute(inverse)
            outputs.append(result)
    return torch.cat(outputs, dim=-1) if outputs else x[..., :0]


@multify
def softmax_fused(dims,x,weights,pre_norm=True,post_norm=False,use_bias=False):
    return apply_residual(x,weights,lambda z:softmax_operation_fused(dims,z),
                          pre_norm=pre_norm,post_norm=post_norm,use_bias=use_bias)
