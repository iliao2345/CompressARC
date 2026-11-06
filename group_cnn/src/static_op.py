import numpy as np
import torch



#######################################################################################################
# original operation
def shift_(x, dim, masks=None):
    padding = torch.zeros_like(torch.narrow(x, dim, 0, 1))
    narrowed = torch.narrow(x, dim, 0, x.shape[dim] - 1)
    return torch.cat([padding, narrowed], dim=dim)

def diagonal_shift_(x, dim1, dim2, masks=None, shift_amount=1, pad_value=0):
    for dim in (dim1, dim2):
        padding = pad_value + torch.zeros_like(
            torch.narrow(x, dim, 0, abs(shift_amount))
        )

        if shift_amount >= 0:
            narrowed = torch.narrow(
                x, dim, 0, x.shape[dim] - shift_amount
            )
            x = torch.cat([padding, narrowed], dim=dim)

        else:
            narrowed = torch.narrow(
                x, dim, -shift_amount, x.shape[dim] + shift_amount
            )
            x = torch.cat([narrowed, padding], dim=dim)

    return x
#######################################################################################################

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
                            x_slice = x_slice[...,::2]
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



shift = make_directional_layer(
        shift_, diagonal_shift_
        )


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

cummax = make_directional_layer(
         cummax_, diagonal_cummax_
)
