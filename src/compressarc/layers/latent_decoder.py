"""Sampling and decoding for the model's initial multitensor latents."""

import torch

from .helper import affine


def channel_layer(target_capacity, posterior):
    """Sample one latent tensor and compute its KL contribution."""
    mean, local_capacity_adjustment = posterior
    reduction_axes = tuple(range(mean.ndim - 1))
    dimensionality = mean.numel()

    # Python scalars avoid synchronous CPU-to-GPU scalar copies during graph capture.
    min_capacity = 0.5
    init_capacity = 10000.0
    target_capacity = 10 * target_capacity

    desired_global_capacity = torch.exp(target_capacity) * init_capacity + min_capacity
    output_scaling = 1 - torch.exp(
        -desired_global_capacity / dimensionality * 2
    )

    local_capacity_adjustment = (
        target_capacity
        + local_capacity_adjustment
        - local_capacity_adjustment.mean(dim=reduction_axes)
    )
    desired_local_capacity = torch.exp(local_capacity_adjustment) * init_capacity + min_capacity

    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    noise_var = noise_std**2

    def stable_sqrt_one_minus_exp_neg(value):
        return torch.where(
            value > 20,
            torch.ones_like(value),
            torch.sqrt(1 - torch.exp(-value)),
        )

    signal_std = stable_sqrt_one_minus_exp_neg(
        desired_local_capacity / dimensionality * 2
    )
    signal_var = 1 - noise_var

    normalized_mean = mean - mean.mean(dim=reduction_axes)
    normalized_mean = normalized_mean / torch.sqrt(
        (normalized_mean**2 + 1e-8).mean(dim=reduction_axes)
    )

    z = signal_std * normalized_mean + noise_std * torch.randn_like(normalized_mean)
    z = output_scaling * z

    kl = 0.5 * (noise_var + signal_var * normalized_mean**2 - 1)
    kl = kl + desired_local_capacity / dimensionality
    return z, kl


class LatentDecoder:
    """Decode all latent posteriors in a multitensor and collect their KL terms."""

    def __init__(self, multify):
        self._multify = multify

    def forward(self, target_capacities, decode_weights, multiposteriors):
        """Return decoded tensors, per-component KL values, and component names."""
        kl_amounts = []
        kl_names = []

        def decode_one(dims, target_capacity, decode_weight, posterior):
            z, kl = channel_layer(target_capacity, posterior)
            decoded = affine(z, decode_weight, use_bias=True)
            kl_amounts.append(kl)
            kl_names.append(str(dims))
            return decoded

        decode_all = self._multify(decode_one)
        decoded = decode_all(target_capacities, decode_weights, multiposteriors)
        return decoded, kl_amounts, kl_names

    __call__ = forward