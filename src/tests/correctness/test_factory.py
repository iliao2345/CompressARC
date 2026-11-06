"""Exercise factory-selected implementations through an optimizer update."""

from itertools import product

import pytest
import torch

from config import load_config
from preprocessing import Task
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.model import ARCCompressor
from compressarc.model.multitensor_systems import MultiTensorSystem, multify
from compressarc.train import take_step


@pytest.mark.parametrize('cummax,shift,direction_share', list(product(
    ['primitives', 'd4', 'optimised'], ['primitives', 'tied_conv'], ['primitives', 'd4'],
)))
def test_factory_model_training_step(cummax, shift, direction_share):
    torch.manual_seed(23)
    config = load_config(overrides=[
        'model.n_layers=2', 'model.channel_dim_shared=2', 'model.channel_dim_grid=2',
        'model.share_up_dim=2', 'model.share_down_dim=2', 'model.decoding_dim=2',
        'model.softmax_dim=2', 'model.cummax_dim=2', 'model.shift_dim=2',
        'model.nonlinear_dim=2', f'model.cummax_implementation={cummax}',
        f'model.shift_implementation={shift}',
        f'model.direction_share_implementation={direction_share}',
    ])
    problem = {'train': [{'input': [[0, 1, 0], [1, 0, 1]],
                          'output': [[0, 1, 0], [1, 0, 1]]}],
               'test': [{'input': [[1, 0, 1], [0, 1, 0]]}]}
    task = Task('factory_smoke', problem, [[[1, 0, 1], [0, 1, 0]]])
    model = ARCCompressor(task, config.model)
    parameters = []
    for name in ['cummax', 'shift', 'direction_share']:
        layer = getattr(model, name)
        if isinstance(layer, torch.nn.Module):
            assert len(layer.models) == 2
            assert layer.models[0] is not layer.models[1]
            for parameter in layer.parameters():
                assert sum(parameter is p for p in model.weights_list) == 1
                parameters.append(parameter)
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(model.weights_list, lr=0.001)
    take_step(task, model, optimizer, 0, config.training)
    assert all(torch.isfinite(p).all() for p in model.weights_list)
    for parameter, old in zip(parameters, before):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert not torch.equal(parameter, old)


@pytest.mark.parametrize('has_colors', [False, True])
@pytest.mark.parametrize('post_norm', [False, True])
def test_tied_shift_wrapper_matches_primitive_glue(has_colors, post_norm):
    system = MultiTensorSystem(2, 2, 3, 4, task=None)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    generator = torch.Generator().manual_seed(41)
    active = {(1, int(has_colors), 1, 1, 1)}
    for dims in system:
        if tuple(dims) in active:
            inputs[dims] = torch.randn(system.shape(dims, extra_dim=4),
                                      generator=generator, dtype=torch.float64)
            weights[dims] = [[torch.eye(4, dtype=torch.float64), None],
                            [torch.eye(4, dtype=torch.float64), None]]
        else:
            inputs[dims] = None
            weights[dims] = None
    # Populate the other active shape too so both wrappers traverse a real MultiTensor.
    for dims in [(1, 0, 1, 1, 1), (1, 1, 1, 1, 1)]:
        if inputs[dims] is None:
            inputs[dims] = torch.randn(system.shape(dims, extra_dim=4),
                                      generator=generator, dtype=torch.float64)
            weights[dims] = [[torch.eye(4, dtype=torch.float64), None],
                            [torch.eye(4, dtype=torch.float64), None]]
    masks = torch.ones(2, 3, 4, 2, dtype=torch.float64)
    masks[0, 1, 1] = 0
    factory = LayerFactory()
    primitive = factory.create_shift('primitives', multify=multify)
    tied = factory.create_shift('tied_conv', multify=multify).double()
    expected = primitive(inputs, weights, masks, post_norm=post_norm)
    actual = tied(inputs, weights, masks, post_norm=post_norm)
    for dims in system:
        if inputs[dims] is not None:
            torch.testing.assert_close(actual[dims], expected[dims])
        else:
            assert actual[dims] is None
