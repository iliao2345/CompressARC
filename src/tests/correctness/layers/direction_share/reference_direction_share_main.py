"""Frozen direction-share equations from main snapshot master 8282287."""
import torch
from compressarc.model.multitensor_systems import multify

def normalize(dims, x, debias=True):
    """
    Normalize the tensor to have variance one, for every index along the channel dimension.
    Args:
        dims (list[int]): Tells you which tensor in the multitensor system we're normalizing
        x (Tensor): Tensor to normalize.
    Returns:
        Tensor: Normalized tensor.
    """
    all_but_last = list(range(len(x.shape)-1))
    if debias:
        x = x - torch.mean(x, dim=all_but_last)
    x = x / torch.sqrt(1e-8+torch.mean(x**2, dim=all_but_last))
    return x

normalize = multify(normalize)

def affine(dims, x, weight, use_bias=False):
    """
    Apply a linear layer to a tensor, along the channel dimension.
    Args:
        dims (list[int]): Tells you which tensor in the multitensor system we're normalizing
        x (Tensor): Input to the linear layer.
        weight (list[Tensor]): A weight matrix and a bias vector.
    Returns:
        Tensor: Output of the linear layer.
    """
    x = torch.matmul(x, weight[0])
    if use_bias:
        x = x + weight[1]
    return x

affine = multify(affine)

def direction_share(dims, x, weights, pre_norm=True, use_bias=False):
    """
    Apply the directional communication layer.
    Args:
        dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
        x (MultiTensor[Tensor]): The input to the directional communication layer.
        weights (MultiTensor[list[list[list[Tensor]]]]): A multitensor full of linear layer weights
                for every pair of directions.
    Returns:
        MultiTensor[Tensor]: The output of the directional communication layer.
    """
    # Optionally normalize the input
    z = normalize(x) if pre_norm else x

    n_directions = dims[3] + dims[4]
    direction_dim = -2 - n_directions

    # Unbind x and z along the direction dimension to avoid repeated slicing.
    x_list = list(torch.unbind(x, dim=direction_dim))
    z_list = list(torch.unbind(z, dim=direction_dim))

    # Precomputed coefficients for the directional shift.
    coefficients = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]

    # Loop over all pairs of directions.
    for d1 in range(8):
        for d2 in range(8):
            # Determine the appropriate coefficient.
            c = coefficients[(d2 - d1) % 8]
            # Apply the affine transformation for this pair and accumulate.
            x_list[d1] = x_list[d1] + c * affine(z_list[d2], weights[d1][d2], use_bias=use_bias)

    # Reassemble the tensor along the original direction dimension.
    return torch.stack(x_list, dim=direction_dim)
