"""Initialization of repeated ARCCompressor layer weights."""


def initialize_layer_weights(initializer, config):
    weights = {
        "share_up_weights": [],
        "share_down_weights": [],
        "softmax_weights": [],
        "cummax_weights": [],
        "shift_weights": [],
        "direction_share_weights": [],
        "nonlinear_weights": [],
    }

    for _ in range(config.n_layers):
        # Preserve the original model's registration order in weights_list.
        weights["share_up_weights"].append(
            initializer.initialize_multiresidual(config.share_up_dim, config.share_up_dim)
        )
        weights["share_down_weights"].append(
            initializer.initialize_multiresidual(config.share_down_dim, config.share_down_dim)
        )
        output_scaling_fn = lambda dims: config.softmax_dim * (
            2 ** (dims[1] + dims[2] + dims[3] + dims[4]) - 1
        )
        weights["softmax_weights"].append(
            initializer.initialize_multiresidual(config.softmax_dim, output_scaling_fn)
        )
        weights["cummax_weights"].append(
            initializer.initialize_multiresidual(config.cummax_dim, config.cummax_dim)
        )
        weights["shift_weights"].append(
            initializer.initialize_multiresidual(config.shift_dim, config.shift_dim)
        )
        weights["direction_share_weights"].append(
            initializer.initialize_multidirection_share()
        )
        weights["nonlinear_weights"].append(
            initializer.initialize_multiresidual(config.nonlinear_dim, config.nonlinear_dim)
        )

    return weights