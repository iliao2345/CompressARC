"""Opt-in removal of global default-device dispatch during the training pass."""
from contextlib import contextmanager
import torch


@contextmanager
def runtime_device_dispatch(mode='global'):
    """Setup still uses the configured device; runtime factories must be explicit.

    Tensor operations keep their input device. This removes only the global
    default-device TorchFunction mode, not autograd or tensor-subclass dispatch.
    The previous default device is restored even when a pass raises.
    """
    if mode not in ('global','explicit'):
        raise ValueError('training.runtime_device_dispatch must be global or explicit')
    if mode=='global':
        yield
        return
    previous=torch.get_default_device()
    torch.set_default_device(None)
    try:
        yield
    finally:
        torch.set_default_device(previous)
