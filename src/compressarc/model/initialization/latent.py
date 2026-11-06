"""Initialization of latent posterior and decoder weights."""


def initialize_latent_weights(initializer, config):
    multiposteriors = initializer.initialize_multiposterior(config.decoding_dim)
    decode_weights = initializer.initialize_multilinear(
        [config.decoding_dim, initializer.channel_dim_fn]
    )
    target_capacities = initializer.initialize_multizeros([config.decoding_dim])
    return multiposteriors, decode_weights, target_capacities