"""Initialization of final color and mask head weights."""


def initialize_output_weights(initializer, channel_dim_fn):
    head_weights = initializer.initialize_head()
    mask_dims = [1, 0, 0, 1, 0]
    mask_weights = initializer.initialize_linear(
        mask_dims, [channel_dim_fn(mask_dims), 2]
    )
    return head_weights, mask_weights