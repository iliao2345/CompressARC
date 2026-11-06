import itertools

import numpy as np
import torch
import multitensor_systems

"""
This file contains all of the layers of our network. The architecture which puts the layers
together is found in arc_compressor.py.
"""

np.random.seed(0)
torch.manual_seed(0)


@multitensor_systems.multify
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

@multitensor_systems.multify
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

def add_residual(layer):
    """
    Surround a layer/operation with a residual connection, up and down projections,
    and pre/post-norms.
    Args:
        layer (Callable): The layer/operation to modify.
    Returns:
        Callable: Another layer/operation that applies the original layer with the
                above modifications.
    """
    def layer_with_residual(dims, x, residual_weights, *args,
                            use_bias=False, pre_norm=False, post_norm=False, **kwargs):
        if pre_norm:
            z = normalize(x)
        z = affine(x, residual_weights[0], use_bias=use_bias)
        z = layer(dims, z, *args, **kwargs)
        if post_norm:
            z = normalize(z)
        z = affine(z, residual_weights[1], use_bias=use_bias)
        return x + z
    return layer_with_residual

def channel_layer(target_capacity, posterior):
    """
    Assume that z comes from some prior distribution, measure the KL divergence to the
    posterior, and give a sample z from the posterior.
    Args:
        target_capacity (Tensor): Rough attempted KL capacity (reparameterized).
        posterior (tuple[Tensor]): Consists of mean and local_capacity_adjustment. mean
                parameterizes the mean of the posterior, and local_capacity_adjustment
                gives the attempted KL capacity to use for each element in the tensor, in
                log space.
    """
    mean, local_capacity_adjustment = posterior

    all_but_last_dim = tuple(range(len(mean.shape)-1))
    dimensionality = 1  # figure out how many elements there are in the tensor
    for axis_length in mean.shape:
        dimensionality *= axis_length
    min_capacity = 0.5
    init_capacity = 10000
    min_capacity = torch.tensor(min_capacity)
    init_capacity = torch.tensor(init_capacity)

    target_capacity = 10*target_capacity  # this reparameterization is for faster learning

    # Compute some rudimentary post-scaling of z. This output scaling leaks a bit of information that isn't
    # measured by the KL, but luckily the scaling parameter is one-dimensional and probably doesn't have
    # that much information in it.
    # The output is scaled by the sigmoid of a signal-to-noise ratio, where the signal-to-noise ratio is the one
    # that an AWGN channel would use to achieve a channel capacity equal to the desired_global_capacity below.
    # A numerically stable formula for the sigmoid of this signal-to-noise ratio is used to compute output_scaling.
    desired_global_capacity = torch.exp(target_capacity)*init_capacity + min_capacity
    output_scaling = 1-torch.exp(-desired_global_capacity / dimensionality * 2)

    # We make local adjustments to the desired_global_capacity in order to allow different elements to have
    # different variances.
    local_capacity_adjustment = (target_capacity + 
                                 local_capacity_adjustment - 
                                 torch.mean(local_capacity_adjustment, dim=all_but_last_dim))
    desired_local_capacity = torch.exp(local_capacity_adjustment)*init_capacity + min_capacity

    # Figure out what signal-to-noise ratio is required to achieve desired_local_capacity, and compute how much
    # signal and how much noise for them to sum to one. Numerically stable formulae for these are used below.
    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    noise_var = noise_std**2
    stable_sqrt1memx = lambda x: torch.where(x>20, 1, torch.sqrt(1-torch.exp(-x)))
    signal_std = stable_sqrt1memx(desired_local_capacity / dimensionality * 2)
    signal_var = 1-noise_var

    # Don't actually send a signal of variance equal to signal. Instead, normalize the means tensor and send that instead.
    normalized_mean = mean - torch.mean(mean, dim=all_but_last_dim)
    normalized_mean = normalized_mean / torch.sqrt(torch.mean(normalized_mean**2+1e-8, dim=all_but_last_dim))

    # Now we can have a sample of z.
    z = signal_std*normalized_mean + noise_std*torch.randn(normalized_mean.shape)
    z = output_scaling*z  # leaks a tiny bit of unmeasured information, see comment above

    # Calculate the KL directly instead of using the AWGN channel capacity formula, because we didn't
    # actually send a signal of variance equal to signal, so the AWGN channel capacity formula would be wrong
    # here.
    KL = 0.5*(noise_var + signal_var*normalized_mean**2 - 1) + desired_local_capacity/dimensionality
    return z, KL

def decode_latents(target_capacities, decode_weights, multiposteriors):
    """
    Decode the latents z, and give the KL loss for the VAE-like setup. Break the KL down into
    its components for possible analysis later. Apply a linear layer afterwards.
    Args:
        target_capacities (MultiTensor[Tensor]): Rough attempted KL capacities (reparameterized).
        decode_weights (MultiTensor[list[Tensor]]): A set of linear layer weights to apply to the decoded
                outputs for every tensor in the multitensor output of the decoding layer.
        multiposteriors (MultiTensor[tuple[Tensor]]): Consists of mean and local_capacity_adjustment. mean
                parameterizes the mean of the posterior, and local_capacity_adjustment
                gives the attempted KL capacity to use for each element in the tensor, in
                log space. One (mean, local_capacity_adjustment) tuple for every tensor in the multitensor
                system.
    Returns:
        MultiTensor[Tensor]: The output of the decoding layer.
        list[Tensor]: Individual KL components that contribute to the total KL.
        list[str]: Names for individual KL components contributing to the total KL.
    """

    KL_amounts = []
    KL_names = []

    @multitensor_systems.multify
    def decode_latents_(dims, target_capacity, decode_weight, posterior):
        z, KL = channel_layer(target_capacity, posterior)
        x = affine(z, decode_weight, use_bias=True)
        KL_amounts.append(KL)
        KL_names.append(str(dims))
        return x
    x = decode_latents_(target_capacities, decode_weights, multiposteriors)
    return x, KL_amounts, KL_names


def share_direction(residual, share_weights, direction):
    """
    Apply the multitensor communication layer.
    Args:
        residual (MultiTensor[Tensor]): The residual stream.
        share_weights (Multitensor[list[list[Tensor]]]): Multiresidual projection weights.
        direction (int): 1 for up, -1 for down.
    Returns:
        MultiTensor[Tensor]: The output of the multitensor communication layer.
    """
    
    # Split the multiresidual into two multilinears
    down_project_weights = multitensor_systems.multify(lambda dims, weights: weights[0])(share_weights)
    up_project_weights = multitensor_systems.multify(lambda dims, weights: weights[1])(share_weights)

    multitensor_system = residual.multitensor_system

    x = affine(residual, down_project_weights, use_bias=False)  # down-project

    # Define a different communication method depending on which way we're communicating.
    if direction == 1:  # share up
        def share(dims, _):
            lower_xs = []
            for lower_dims in multitensor_system:  # get information from all lower tensors
                # check that lower_dims lower than dims in all indices
                if all([lower_naxes <= naxes for lower_naxes, naxes in zip(lower_dims, dims)]):
                    lower_x = x[lower_dims]
                    # unsqueeze all the dimensions of lower_x until it's the same rank as x
                    for dim, (lower_naxes, naxes) in enumerate(zip(lower_dims, dims)):
                        if lower_naxes < naxes:
                            axis = sum(dims[:dim], 0)
                            lower_x = torch.unsqueeze(lower_x, axis)
                    lower_xs.append(lower_x)
            return sum(lower_xs)
    else:  # share down
        def share(dims, _):
            higher_xs = []
            for higher_dims in multitensor_system:  # get information from all higher tensors
                # check that higher_dims higher than dims in all indices
                if all([higher_naxes >= naxes for higher_naxes, naxes in zip(higher_dims, dims)]):
                    higher_x = x[higher_dims]
                    # aggregate all the dimensions of higher_x until it's the same rank as x
                    for dim, (higher_naxes, naxes) in reversed(list(enumerate(zip(higher_dims, dims)))):
                        if higher_naxes > naxes:
                            axis = sum(higher_dims[:dim], 0)
                            if higher_dims[0] == 1 and (x.multitensor_system.task.in_out_same_size or x.multitensor_system.task.all_out_same_size) and dim==3:  # be careful aggregating the x axis
                                # expand/contract masks to make the dims the same as higher_x
                                masks = x.multitensor_system.task.masks
                                masks = 1-(1-masks[...,0])*(1-masks[...,1])
                                for i in range(sum(higher_dims[1:3])):  # insert color and direction dims
                                    masks = masks[:,None,...]
                                if dims[4] == 0:  # remove y dim
                                    masks = masks[...,0]
                                masks = masks[...,None]  # add channel dim
                                higher_x = torch.sum(higher_x*masks, dim=axis) / (torch.sum(masks, dim=axis)+1e-4)
                            elif higher_dims[0] == 1 and (x.multitensor_system.task.in_out_same_size or x.multitensor_system.task.all_out_same_size) and dim==4:  # be careful aggregating the y axis
                                # expand/contract masks to make the dims the same as higher_x
                                masks = x.multitensor_system.task.masks
                                masks = 1-(1-masks[...,0])*(1-masks[...,1])
                                for i in range(sum(higher_dims[1:3])):  # insert color and direction dims
                                    masks = masks[:,None,...]
                                if higher_dims[3] == 0:  # remove x dim
                                    masks = masks[...,0,:]
                                masks = masks[...,None]  # add channel dim
                                higher_x = torch.sum(higher_x*masks, dim=axis) / (torch.sum(masks, dim=axis)+1e-4)
                            else:
                                higher_x = torch.mean(higher_x, dim=axis)
                    higher_xs.append(higher_x)
            return sum(higher_xs)
    x = multitensor_systems.multify(share)(x)  # perform the cross-tensor communication
    x = normalize(x)  # post-norm
    x = affine(x, up_project_weights, use_bias=False)  # up-project
    residual = multitensor_systems.multify(lambda dims, x, y: x+y)(residual, x)  # add residual
    return residual

def share_up(residual, share_up_weights):
    """
    Apply the multitensor communication layer, upwards.
    Args:
        residual (MultiTensor[Tensor]): The residual stream.
        share_up_weights (Multitensor[list[list[Tensor]]]): Multiresidual projection weights.
    Returns:
        MultiTensor[Tensor]: The output of the multitensor communication layer.
    """
    return share_direction(residual, share_up_weights, 1)

def share_down(residual, share_down_weights):
    """
    Apply the multitensor communication layer, downwards.
    Args:
        residual (MultiTensor[Tensor]): The residual stream.
        share_down_weights (Multitensor[list[list[Tensor]]]): Multiresidual projection weights.
    Returns:
        MultiTensor[Tensor]: The output of the multitensor communication layer.
    """
    return share_direction(residual, share_down_weights, -1)


def only_do_for_certain_shapes(*shapes):
    """
    Decorator which takes a function that is applied to every tensor in a multitensor,
    and replaces that function with the identity for select tensors in the multitensor.
    Args:
        *shapes (list[list[int]]): A list of MultiTensor dims, for which the function
                should be applied. Don't do the function if the dims for the tensor isn't
                in the list.
    """
    def decorator(fn):
        def filtered_fn(dims, x, *args, **kwargs):
            if tuple(dims) in shapes:
                return fn(dims, x, *args, **kwargs)
            else:
                return x
        return filtered_fn
    return decorator


@multitensor_systems.multify
@add_residual
def softmax(dims, x):
    """
    Apply the softmax layer. Take softmax over all combinations of dims, but never include
    the channel dim nor the example dim.
    Args:
        dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
        x (MultiTensor[Tensor]): The input to the softmax layer.
    Returns:
        MultiTensor[Tensor]: The output of the softmax layer.
    """
    axes = list(range(sum(dims)))
    if dims[0]==1:
        axes.pop(0)  # don't softmax over examples
    subsets_of_axes = []
    for subset_size in range(1, len(axes)+1):
        subsets_of_axes = subsets_of_axes + list(itertools.combinations(axes, subset_size))
    softmaxxes = []
    for subset in subsets_of_axes:
        offsets = torch.amax(x, dim=subset, keepdim=True)
        softmax = torch.exp(x-offsets)
        softmax = softmax / torch.sum(softmax, dim=subset, keepdim=True)
        softmaxxes.append(softmax)
    if not softmaxxes:
        # The relaxed ablation permits an example-only tensor. It has no
        # non-example semantic axis to normalize, and its projected softmax
        # channel width is zero, so this branch contributes a zero residual.
        return x[..., :0]
    return torch.cat(softmaxxes, dim=-1)


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

"""
Function cummax

Apply the directional cummax layer.
Args:
    x (MultiTensor[Tensor]): The input to the cummax layer.
    weights (MultiTensor[list[list[Tensor]]]): Multiresidual projection weights surrounding the cummax operations.
            Implicitly introduced by the add_residual decorator.
    masks (Tensor): A (example, x, y, in/out) tensor of zeros and ones telling you which pixels are in-bounds.
    Other boolean kwargs such as pre_norm, post_norm, use_bias, introduced by the add_residual decorator.
Returns:
    MultiTensor[Tensor]: The output of the cummax layer.
"""
def cummax_(x, dim, masks):
    masks = 1e3*(1-masks)
    max_ = torch.max(x-masks, dim=dim, keepdim=True)[0] + masks + 1e-3
    min_ = torch.min(x+masks, dim=dim, keepdim=True)[0] - masks - 1e-3
    x = torch.cummax(x-masks, dim=dim)[0] + masks
    return (x - min_) / (max_-min_) * 2 - 1
def diagonal_cummax_(x, dim1, dim2, masks):
    masks_ = 1e3*(1-masks)
    min_dim = min(x.shape[dim1], x.shape[dim2])
    n_iters = int(np.ceil(np.log2(min_dim)))
    # compute the cummax and max via forward+backward associative scan
    max_x = x - masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(max_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=-1e3)
            max_x = torch.max(max_x, shifted_x)
        if sign == 1:  # save the cummax after the forward associative scan
            cummax_x = max_x + masks_
    max_x = max_x + masks_
    # compute the min via forward+backward associative scan
    min_x = x + masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(min_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=1e3)
            min_x = torch.min(min_x, shifted_x)
    min_x = min_x - masks_
    return ((cummax_x - min_x) / (max_x-min_x+1e-5) * 2 - 1)*masks  # rescale the cummax to fit the max and min
cummax = multitensor_systems.multify(  # apply decorators
         only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))(
         add_residual(
         make_directional_layer(
         cummax_, diagonal_cummax_
         ))))

@multitensor_systems.multify
@only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))
def lse_cummax(dims, x, masks, model, weights=None, projected=False):
    """Apply pure LSE morphology or a projected residual LSE block."""
    from lse import directional_lse
    if not projected:
        return directional_lse(dims, x, masks, model)
    z = affine(x, weights[0], use_bias=False)
    z = directional_lse(dims, z, masks, model)
    z = normalize(z)
    return x + affine(z, weights[1], use_bias=False)

"""
Function shift

Apply the directional shift layer.
Args:
    x (MultiTensor[Tensor]): The input to the shift layer.
    weights (MultiTensor[list[list[Tensor]]]): Multiresidual projection weights surrounding the shift operations.
            Implicitly introduced by the add_residual decorator.
    masks (Tensor): A (example, x, y, in/out) tensor of zeros and ones telling you which pixels are in-bounds.
    Other boolean kwargs such as pre_norm, post_norm, use_bias, introduced by the add_residual decorator.
Returns:
    MultiTensor[Tensor]: The output of the shift layer.
"""
def shift_(x, dim, masks):
    padding = torch.zeros_like(torch.narrow(x, dim, 0, 1))
    narrowed = torch.narrow(x, dim, 0, x.shape[dim]-1)
    return torch.cat([padding, narrowed], dim=dim)
def diagonal_shift_(x, dim1, dim2, masks, shift_amount=1, pad_value=0):
    for dim in (dim1, dim2):
        padding = pad_value+torch.zeros_like(torch.narrow(x, dim, 0, abs(shift_amount)))
        if shift_amount >= 0:
            narrowed = torch.narrow(x, dim, 0, x.shape[dim]-shift_amount)
            x = torch.cat([padding, narrowed], dim=dim)
        else:
            narrowed = torch.narrow(x, dim, -shift_amount, x.shape[dim]+shift_amount)
            x = torch.cat([narrowed, padding], dim=dim)
    return x
shift = multitensor_systems.multify(  # apply decorators
        only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))(
        add_residual(
        make_directional_layer(
        shift_, diagonal_shift_
        ))))

@multitensor_systems.multify
@only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))
def tied_conv_shift(dims, x, masks, conv, weights, projected):
    """Replace only the original shift block; other components pass through."""
    from tied_convolution import tied_directional_shift
    if not projected:
        return tied_directional_shift(x, masks, conv)
    z = affine(x, weights[0], use_bias=False)
    z = tied_directional_shift(z, masks, conv)
    # Retain the original shift block's post-normalization for this comparison.
    z = normalize(z)
    return x + affine(z, weights[1], use_bias=False)


directional_dims = [(i,j,1,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
@multitensor_systems.multify
@only_do_for_certain_shapes(*directional_dims)
def d4_direction_share(dims, x, model, projection_weights=None, projected=False):
    """Apply pure D4 mixing or a separately projected residual D4 block."""
    from direction_share import apply_d4
    if not projected:
        return apply_d4(dims, x, model)
    z = normalize(x)
    z = affine(z, projection_weights[0], use_bias=False)
    z = apply_d4(dims, z, model)
    return x + affine(z, projection_weights[1], use_bias=False)


@multitensor_systems.multify
@only_do_for_certain_shapes(*directional_dims)
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

@multitensor_systems.multify
@add_residual
def nonlinear(dims, x):
    """
    Apply the nonlinear layer.
    Args:
        dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
        x (MultiTensor[Tensor]): The input to the nonlinear layer.
        weights (MultiTensor[list[list[Tensor]]]): Multiresidual projection weights surrounding the nonlinear operations.
                Implicitly introduced by the add_residual decorator.
        Other boolean kwargs such as pre_norm, post_norm, use_bias, introduced by the add_residual decorator.
    Returns:
        MultiTensor[Tensor]: The output of the nonlinear layer.
    """
    return torch.nn.functional.silu(x)

def postprocess_mask(task, x_mask, y_mask):
    """
    Apply postprocessing to the masks outputted by the network. If masks are already determined
    by the task because the output shapes follow a known hardcoded structure, then enforce the
    known structure.
    Args:
        task (Task): The task that is being solved by the network.
        x_mask (Tensor): The x mask that is outputted by the network that we must modify to fit
                the task's structure.
        y_mask (Tensor): The y mask that is outputted by the network that we must modify to fit
                the task's structure.
    Returns:
        Tensor: Modified x mask that fits the task's structure.
        Tensor: Modified y mask that fits the task's structure.
    """

    # Make an additive modifier mask that has large negative values for out of bounds pixels.
    x_mask_modifier = np.zeros([task.n_examples, task.n_x, 2])
    y_mask_modifier = np.zeros([task.n_examples, task.n_y, 2])
    for example_num in range(task.n_examples):
        max_length = max(task.shapes[example_num][0][0], task.shapes[example_num][1][0])
        for in_out_mode in range(2):
            x_mask_modifier[example_num,max_length:,in_out_mode] = -1000
        max_length = max(task.shapes[example_num][0][1], task.shapes[example_num][1][1])
        for in_out_mode in range(2):
            y_mask_modifier[example_num,max_length:,in_out_mode] = -1000
    x_mask = x_mask+torch.from_numpy(x_mask_modifier).to(x_mask.device).to(x_mask.dtype)
    y_mask = y_mask+torch.from_numpy(y_mask_modifier).to(y_mask.device).to(y_mask.dtype)
    return x_mask, y_mask
