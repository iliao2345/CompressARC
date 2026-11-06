# layers/helper.py
from contextlib import contextmanager
from contextvars import ContextVar
import time
import itertools
import torch
from ..model import multitensor_systems


def only_do_for_certain_shapes(*shapes):
    def decorate(operation):
        def wrapped(dims, x, *args, **kwargs):
            if tuple(dims) in shapes:
                return operation(dims, x, *args, **kwargs)
            return x
        return wrapped
    return decorate

@multitensor_systems.multify
def normalize(dims, x, debias=True):
    axes = tuple(range(x.ndim - 1))
    if debias:
        x = x - x.mean(dim=axes)
    return x / torch.sqrt(1e-8 + (x**2).mean(dim=axes))


@multitensor_systems.multify
def affine(dims, x, weights, use_bias=False):
    x = x @ weights[0]
    if use_bias:
        x = x + weights[1]
    return x


def apply_residual(x, weights, operation, *, pre_norm=False,
                   post_norm=False, use_bias=False):
    """Apply projections and an operation, then add the residual."""
    z = normalize(x) if pre_norm else x
    z = affine(z, weights[0], use_bias=use_bias)
    z = operation(z)
    if post_norm:
        z = normalize(z)
    return x + affine(z, weights[1], use_bias=use_bias)



def softmax_operation(dims, x):
    """Concatenate softmax outputs over all non-empty semantic-axis subsets."""
    axes = list(range(sum(dims)))
    if dims[0] == 1:
        axes.pop(0)  # Keep examples independent.

    outputs = []
    for subset_size in range(1, len(axes) + 1):
        for subset in itertools.combinations(axes, subset_size):
            shifted = x - torch.amax(x, dim=subset, keepdim=True)
            probabilities = torch.exp(shifted)
            probabilities = probabilities / torch.sum(
                probabilities, dim=subset, keepdim=True
            )
            outputs.append(probabilities)

    if not outputs:
        # Example-only tensors have no semantic axis to normalize.
        return x[..., :0]
    return torch.cat(outputs, dim=-1)


@multitensor_systems.multify
def softmax(dims, x, weights, pre_norm=True, post_norm=False, use_bias=False):
    """Apply the fixed softmax operation with the shared residual projections."""
    return apply_residual(
        x,
        weights,
        lambda projected: softmax_operation(dims, projected),
        pre_norm=pre_norm,
        post_norm=post_norm,
        use_bias=use_bias,
    )



@multitensor_systems.multify
def nonlinear(dims, x, weights, pre_norm=True, post_norm=False, use_bias=False):
    """Apply the fixed SiLU activation inside the residual projection block."""
    return apply_residual(
        x,
        weights,
        torch.nn.functional.silu,
        pre_norm=pre_norm,
        post_norm=post_norm,
        use_bias=use_bias,
    )

def make_directional_layer(fn, diagonal_fn):
    """
    Take a directional function (one version made for cardinal directions and another for diagonal)
    and use it to create a directional layer that works on tensors that have a direction
    dimension.
    Args:
        fn (Callable): A directional function that takes a tensor and a dim argument.
        diagonal_fn (Callable): A directional function that takes a tensor and two dim arguments.
    Returns:
        Callable: A function that takes a tensor with a direction dimension and applies fn and
                diagonal_fn in a different direction for each slice of the tensor along the
                direction dimension.
    """
    def directional_layer(dims, x, masks):
        """
        Args:
            dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
            x (MultiTensor[Tensor]): The input to the directional layer.
            masks (Tensor): A (example, x, y, in/out) tensor of zeros and ones telling you which pixels are in-bounds.
        Returns:
            MultiTensor[Tensor]: The output of the directional layer.
        """

        # rearrange mask to fit same shape as x
        masks = 1-(1-masks[...,0])*(1-masks[...,1])
        if dims[4]==0:
            masks = masks[:,:,0]
        if dims[3]==0:
            masks = masks[:,0,...]
        for i in range(sum(dims[1:3])):
            masks = masks[:,None,...]
        masks = masks[...,None]
        # mask out x
        x = x*masks

        # figure out which dimension the direction dimension is
        n_directions = dims[3]+dims[4]
        direction_dim = sum(dims[:2])

        # make a default output tensor in case we try to do cumulative ops on a dimension that
        # is not present in the tensor x
        zero_tensor = torch.zeros_like(torch.select(x, direction_dim, 0))

        # split the channel dimension into two.
        # split the direction dimension into two.
        # for each half of the direction dimension, each index of the direction dimension corresponds
        # to either x or y, and we accumulate in those respective dimensions.
        # do the other half of the channel dimension in the reverse direction.
        # do the other half of the direction dimension in the reverse direction.
        result_tensors = []
        for channel_split in range(2):  # forward, backward
            result_list = []
            for direction_split in range(2):  # forward, backward
                for direction_ind in range(4):  # x, x+y, y, y-x
                    if direction_ind % 2 == 0:  # cardinal direction
                        cardinal_direction_ind = int(direction_ind//2)
                        if dims[3+cardinal_direction_ind]>0:
                            x_slice = torch.select(x, direction_dim, 4*direction_split+direction_ind)
                            x_slice = x_slice[...,channel_split::2]
                            masks_flipped = torch.select(masks, direction_dim, 0)
                            if direction_split + channel_split == 1:
                                # below: decrement index to account for slicing, increment index to go from direction to x
                                x_slice = torch.flip(x_slice, [direction_dim+cardinal_direction_ind])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim+cardinal_direction_ind])
                            result = fn(x_slice, direction_dim+cardinal_direction_ind, masks_flipped)
                            if direction_split + channel_split == 1:
                                result = torch.flip(result, [direction_dim+cardinal_direction_ind])
                        else:
                            result = zero_tensor
                    else:  # diagonal direction
                        if dims[3] == 1 and dims[4] == 1:
                            diagonal_direction_ind = int(direction_ind//2)  # 0 for x+y, 1 for y-x
                            x_slice = torch.select(x, direction_dim, 4*direction_split+direction_ind)
                            x_slice = x_slice[...,channel_split::2]
                            masks_flipped = torch.select(masks, direction_dim, 0)
                            if (direction_split + channel_split + diagonal_direction_ind) % 2 == 1:
                                # below: decrement index to account for slicing, increment index to go from direction to x
                                x_slice = torch.flip(x_slice, [direction_dim])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim])
                            if direction_split + channel_split == 1:
                                x_slice = torch.flip(x_slice, [direction_dim+1])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim+1])
                            result = diagonal_fn(x_slice, direction_dim, direction_dim+1, masks_flipped)
                            if (direction_split + channel_split + diagonal_direction_ind) % 2 == 1:
                                result = torch.flip(result, [direction_dim])
                            if direction_split + channel_split == 1:
                                result = torch.flip(result, [direction_dim+1])
                        else:
                            result = zero_tensor
                    result_list.append(result)
            result_list = torch.stack(result_list, dim=direction_dim)  # stack direction dim together
            result_tensors.append(result_list)
        return torch.cat(result_tensors, dim=-1)  # cat channel dim together
    return directional_layer


_timings = ContextVar('morphology_timings', default=None)


@contextmanager
def measure_morphology(enabled=False):
    """Collect synchronized forward wall time, excluding rotations and backward."""
    totals = {}
    token = _timings.set(totals if enabled else None)
    try:
        yield totals
    finally:
        _timings.reset(token)


def _measure_call(fn, x, name, totals=None):
    if totals is None:
        totals = _timings.get()
    if totals is None:
        return fn(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    result = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    seconds = time.perf_counter() - started
    key = f'timing/{name}_forward_seconds'
    totals[key] = totals.get(key, 0.0) + seconds
    key = f'timing/{name}_calls'
    totals[key] = totals.get(key, 0) + 1
    return result



