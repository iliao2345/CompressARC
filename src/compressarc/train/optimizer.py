"""Explicit Adam implementation selection without changing update settings."""
import torch


def create_optimizer(parameters, config):
    implementation = config.get('optimizer_implementation', 'default')
    if implementation not in ('default', 'fused'):
        raise ValueError('training.optimizer_implementation must be default or fused')
    parameters = list(parameters)
    options = dict(lr=config.learning_rate, betas=tuple(config.optimizer_betas))
    if implementation == 'fused':
        if not parameters or any(p.device.type != 'cuda' for p in parameters):
            raise ValueError('Fused Adam mode requires CUDA parameters; use default on CPU')
        # Keep eps, weight decay, bias correction and other Adam defaults intact.
        # Unsupported combinations raise rather than silently selecting another mode.
        options['fused'] = True
    return torch.optim.Adam(parameters, **options)
