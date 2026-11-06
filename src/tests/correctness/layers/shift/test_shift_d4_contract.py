"""Test-first contract for a NEW d4_tied_conv implementation.

The existing tied_conv C4 implementation must remain unchanged. Tests selecting
d4_tied_conv intentionally fail until that separate factory entry is implemented.
Direction permutations below are derived from movement vectors, independently of
the implementation's kernel rotations or adapter index tables.
"""
import io
from itertools import product

import pytest
import torch

from compressarc.layers.shift.morphological import TiedDirectionalConv
from compressarc.layers.shift.primitives import ShiftPrimitives
from compressarc.model.layer_factory import LayerFactory

GROUP = list(product(range(4), [False, True]))
RING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
CARDINAL = [(-1, 0), (0, -1), (1, 0), (0, 1)]
DIAGONAL = [(1, 1), (-1, 1), (-1, -1), (1, -1)]


def spatial(x, group, axes):
    rotation, reflection = group
    if reflection:
        x = torch.flip(x, [axes[1]])
    return torch.rot90(x, rotation, axes)


def permutation(directions, group):
    rotation, reflection = group
    result = [None] * len(directions)
    for source, (dr, dc) in enumerate(directions):
        if reflection:
            dc = -dc
        for _ in range(rotation):
            dr, dc = -dc, dr
        result[directions.index((dr, dc))] = source
    return result


def core_action(x, group):
    x = spatial(x, group, (-2, -1))
    return torch.stack([x[:, 0, permutation(CARDINAL, group)],
                        x[:, 1, permutation(DIAGONAL, group)]], dim=1)


def feature_action(x, group):
    x = spatial(x, group, (-3, -2))
    return x.index_select(x.ndim - 4, torch.tensor(permutation(RING, group), device=x.device))


def mask_action(mask, group):
    mask = spatial(mask, group, (1, 2))
    return mask[..., [1, 0]] if group[0] % 2 else mask


def bind(dims):
    return lambda operation: lambda *args, **kwargs: operation(dims, *args, **kwargs)


def new_layer(dims=(1, 0, 1, 1, 1), depth=2):
    return LayerFactory().create_shift('d4_tied_conv', multify=bind(dims), n_layers=depth).double()


def randomize(module, seed=701):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * .2)


def sample_core(shape):
    generator = torch.Generator().manual_seed(706)
    return torch.randn(2, 2, 4, *shape, generator=generator, dtype=torch.float64)


def test_group_actions_are_invertible():
    x = sample_core((3, 5))
    y = torch.arange(2 * 8 * 3 * 5 * 4).reshape(2, 8, 3, 5, 4)
    for rotation, reflection in GROUP:
        group = rotation, reflection
        inverse = (rotation if reflection else -rotation % 4), reflection
        torch.testing.assert_close(core_action(core_action(x, group), inverse), x, rtol=0, atol=0)
        torch.testing.assert_close(feature_action(feature_action(y, group), inverse), y, rtol=0, atol=0)


def test_existing_c4_is_rotation_equivariant_but_not_reflection_equivariant():
    """Negative control: the old rotation-only core must not pass our D4 oracle."""
    core = TiedDirectionalConv().double()
    randomize(core)
    x = sample_core((3, 5))
    for rotation in range(4):
        group = rotation, False
        torch.testing.assert_close(core(core_action(x, group)), core_action(core(x), group), rtol=1e-10, atol=1e-10)
    error = (core(core_action(x, (0, True))) - core_action(core(x), (0, True))).abs().max()
    assert error > 1e-3, 'A test using only symmetric initial kernels would miss this defect.'


@pytest.mark.parametrize('colors', [False, True])
def test_fixed_primitive_adapter_passes_all_d4_actions(colors):
    dims = (1, int(colors), 1, 1, 1)
    layer = ShiftPrimitives(bind(dims))
    generator = torch.Generator().manual_seed(707)
    shape = [2] + ([3] if colors else []) + [8, 3, 5, 4]
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    masks = torch.randint(0, 2, (2, 3, 5, 2), generator=generator).double()
    identity = torch.eye(4, dtype=torch.float64)
    weights = [[identity, None], [identity, None]]
    expected = layer(x, weights, masks, pre_norm=False)
    for group in GROUP:
        actual = layer(feature_action(x, group), weights, mask_action(masks, group), pre_norm=False)
        torch.testing.assert_close(actual, feature_action(expected, group), rtol=1e-10, atol=1e-10)


def test_d4_factory_is_separate_and_initialization_preserves_fixed_shift():
    layer = new_layer()
    old = TiedDirectionalConv().double()
    assert type(layer.models[0]) is not type(old)
    assert len(layer.models) == 2
    assert layer.models[0] is not layer.models[1]
    x = sample_core((3, 5))
    for core in layer.models:
        torch.testing.assert_close(core(x), old(x), rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize('after_update', [False, True])
def test_d4_core_with_arbitrary_weights_and_after_optimizer_step(shape, after_update):
    core = new_layer().models[0]
    randomize(core)
    x = sample_core(shape)
    if after_update:
        optimizer = torch.optim.Adam(core.parameters(), lr=.01)
        before = [p.detach().clone() for p in core.parameters()]
        # Deliberately asymmetric data/target: symmetry must be structural.
        target = torch.linspace(-.7, .9, x.numel(), dtype=x.dtype).reshape_as(x)
        (core(x) - target).square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters())
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, core.parameters()))
    base = core(x)
    for group in GROUP:
        torch.testing.assert_close(core(core_action(x, group)), core_action(base, group), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('colors', [False, True])
@pytest.mark.parametrize('pre_norm,post_norm,use_bias', list(product([False, True], repeat=3)))
def test_full_projected_layer_d4_outputs_and_gradients(colors, pre_norm, post_norm, use_bias):
    dims = (1, int(colors), 1, 1, 1)
    layer = new_layer(dims)
    randomize(layer)
    generator = torch.Generator().manual_seed(711)
    shape = [2] + ([3] if colors else []) + [8, 3, 5, 4]
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    parameters = [torch.randn(s, generator=generator, dtype=torch.float64, requires_grad=True)
                  for s in [(4, 6), (6,), (6, 4), (4,)]]
    weights = [parameters[:2], parameters[2:]]
    masks = torch.randint(0, 2, (2, 3, 5, 2), generator=generator).double()
    upstream = torch.randn(shape, generator=generator, dtype=x.dtype)
    options = dict(pre_norm=pre_norm, post_norm=post_norm, use_bias=use_bias, layer_index=1)
    shared = [*parameters, *layer.models[1].parameters()]
    base = layer(x, weights, masks, **options)
    expected_grad = torch.autograd.grad((base * upstream).sum(), [x, *shared], allow_unused=True)
    for group in GROUP:
        transformed_x = feature_action(x.detach(), group).requires_grad_()
        actual = layer(transformed_x, weights, mask_action(masks, group), **options)
        torch.testing.assert_close(actual, feature_action(base.detach(), group), rtol=1e-9, atol=1e-9)
        observed_grad = torch.autograd.grad((actual * feature_action(upstream, group)).sum(),
                                            [transformed_x, *shared], allow_unused=True)
        torch.testing.assert_close(observed_grad[0], feature_action(expected_grad[0], group), rtol=1e-8, atol=1e-8)
        for observed, expected in zip(observed_grad[1:], expected_grad[1:]):
            if expected is None:
                assert observed is None
            else:
                assert observed is not None
                torch.testing.assert_close(observed, expected, rtol=1e-8, atol=1e-8)


def test_d4_state_dict_round_trip_preserves_effective_operator():
    layer = new_layer()
    randomize(layer)
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    restored = new_layer()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    x = sample_core((3, 5))
    for a, b in zip(layer.models, restored.models):
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
