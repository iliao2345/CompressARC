"""Explicit C4 naming must not change the already deployed convolution."""
import pytest
import torch

from compressarc.layers.shift.morphological import C4DirectionalConv, TiedDirectionalConv
from compressarc.model.layer_factory import LayerFactory


def test_legacy_class_name_is_an_exact_alias():
    assert C4DirectionalConv is TiedDirectionalConv
    assert issubclass(C4DirectionalConv, torch.nn.Module)
    assert C4DirectionalConv.__name__ == 'C4DirectionalConv'
    assert set(C4DirectionalConv().state_dict()) == {'cardinal_kernel', 'diagonal_kernel'}


@pytest.mark.parametrize('shape', [(2, 2, 4, 3, 5), (1, 2, 4, 1, 5)])
def test_factory_new_and_legacy_names_have_identical_outputs_and_gradients(shape):
    factory = LayerFactory()
    bind = lambda operation: operation
    legacy = factory.create_shift('tied_conv', multify=bind, n_layers=2).double()
    explicit = factory.create_shift('c4_tied_conv', multify=bind, n_layers=2).double()
    generator = torch.Generator().manual_seed(883)
    with torch.no_grad():
        for parameter in legacy.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert len(explicit.models) == 2
    for a, b in zip(legacy.models, explicit.models):
        x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
        z = x.detach().clone().requires_grad_()
        y, w = a(x), b(z)
        torch.testing.assert_close(y, w, rtol=0, atol=0)
        ga = torch.autograd.grad(y.square().sum(), [x, *a.parameters()])
        gb = torch.autograd.grad(w.square().sum(), [z, *b.parameters()])
        for observed, expected in zip(ga, gb):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.parametrize('implementation,expected', [
    ('primitives', False), ('tied_conv', True), ('c4_tied_conv', True),
    ('unfold', True), ('triton', True), ('d4_tied_conv', False),
])
def test_logging_labels_only_c4_convolution_implementations(monkeypatch, implementation, expected):
    import sys
    from types import SimpleNamespace
    from config import load_config
    from compressarc.train.logging import initialize_wandb

    captured = {}
    run = SimpleNamespace(define_metric=lambda *args, **kwargs: None)
    def initialize(**kwargs):
        captured.update(kwargs)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=initialize, Settings=lambda **kwargs: kwargs))
    config = load_config(overrides=['logging.wandb=true', 'logging.wandb_mode=offline',
                                   f'model.shift_implementation={implementation}'])
    assert initialize_wandb(config, SimpleNamespace(task_name='tag-test')) is run
    assert ('C4' in captured['tags']) is expected
