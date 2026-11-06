"""Symmetry constraints for ARCCompressor weights."""


def apply_weight_symmetry(initializer, layer_weights, decode_weights):
    initializer.symmetrize_xy(decode_weights)

    xy_symmetric_names = (
        "share_up_weights",
        "share_down_weights",
        "softmax_weights",
        "cummax_weights",
        "shift_weights",
        "nonlinear_weights",
    )
    for name in xy_symmetric_names:
        for weights in layer_weights[name]:
            initializer.symmetrize_xy(weights)

    for weights in layer_weights["direction_share_weights"]:
        if weights is not None:
            initializer.symmetrize_direction_sharing(weights)