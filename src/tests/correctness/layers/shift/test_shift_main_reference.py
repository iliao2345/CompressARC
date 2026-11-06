"""Compare current ShiftPrimitives to the frozen main-branch shift reference."""

import importlib.util
from pathlib import Path

import pytest
import torch

from compressarc.layers.shift.primitives import ShiftPrimitives, diagonal_shift_, shift_

_reference_spec = importlib.util.spec_from_file_location(
    "shift_main_reference", Path(__file__).with_name("reference_shift_main.py"),
)
reference = importlib.util.module_from_spec(_reference_spec)
_reference_spec.loader.exec_module(reference)


# ============================================================================
# Raw primitive outputs and gradients
# ============================================================================

@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("axis", [1, 2])
def test_cardinal_primitive_matches_main_outputs_and_gradients(shape, axis):
    values = torch.arange(2 * shape[0] * shape[1], dtype=torch.float64).reshape(2, *shape) / 7
    actual_x = values.clone().requires_grad_()
    reference_x = values.clone().requires_grad_()
    actual = shift_(actual_x, axis, None)
    expected = reference.shift_(reference_x, axis, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape) / 11
    got, = torch.autograd.grad((actual * upstream).sum(), actual_x)
    wanted, = torch.autograd.grad((expected * upstream).sum(), reference_x)
    torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("amount", [-1, 1])
@pytest.mark.parametrize("pad", [0, -1000])
def test_diagonal_primitive_matches_main_outputs_and_gradients(shape, amount, pad):
    values = torch.arange(2 * shape[0] * shape[1], dtype=torch.float64).reshape(2, *shape) / 7
    actual_x = values.clone().requires_grad_()
    reference_x = values.clone().requires_grad_()
    actual = diagonal_shift_(actual_x, 1, 2, None, shift_amount=amount, pad_value=pad)
    expected = reference.diagonal_shift_(reference_x, 1, 2, None, shift_amount=amount, pad_value=pad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape) / 11
    got, = torch.autograd.grad((actual * upstream).sum(), actual_x)
    wanted, = torch.autograd.grad((expected * upstream).sum(), reference_x)
    torch.testing.assert_close(got, wanted, rtol=0, atol=0)


# ============================================================================
# Full layer: original adapter, masks, projections and backpropagation
# ============================================================================

def bind_layout(dims):
    def multify(operation):
        def apply(*args, **kwargs):
            return operation(dims, *args, **kwargs)
        return apply
    return multify


def main_normalize(x):
    axes = tuple(range(x.ndim - 1))
    centered = x - x.mean(dim=axes)
    return centered / torch.sqrt(1e-8 + centered.square().mean(dim=axes))


def main_layer(dims, x, weights, masks, post_norm, use_bias):
    # Reproduce the deployed main shift wrapper: pre_norm=False.
    up, down = weights
    projected = x @ up[0]
    if use_bias:
        projected = projected + up[1]
    mixed = reference.make_directional_layer(reference.shift_, reference.diagonal_shift_)(
        dims, projected, masks,
    )
    if post_norm:
        mixed = main_normalize(mixed)
    update = mixed @ down[0]
    if use_bias:
        update = update + down[1]
    return x + update


@pytest.mark.parametrize("has_colors", [False, True])
@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("feature_width", [4, 8])
@pytest.mark.parametrize("check", ["outputs", "gradients"])
def test_shift_primitives_layer_matches_main_outputs_and_gradients(
    has_colors, post_norm, use_bias, feature_width, check,
):
    dims = (1, int(has_colors), 1, 1, 1)
    shape = [2] + ([3] if has_colors else []) + [8, 3, 5, feature_width]
    generator = torch.Generator().manual_seed(311)
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    parameters = [
        torch.randn(size, generator=generator, dtype=torch.float64, requires_grad=True)
        for size in ((feature_width, 4), (4,), (4, feature_width), (feature_width,))
    ]
    weights = [parameters[:2], parameters[2:]]
    masks = torch.ones(2, 3, 5, 2, dtype=torch.float64)
    masks[0, 1, 2] = 0
    masks[1, 0, 1, 0] = 0
    masks[1, 2, 3, 1] = 0
    layer = ShiftPrimitives(bind_layout(dims))
    actual = layer(x, weights, masks, pre_norm=False, post_norm=post_norm, use_bias=use_bias)
    expected = main_layer(dims, x, weights, masks, post_norm, use_bias)
    upstream = torch.randn(shape, generator=generator, dtype=torch.float64)
    got = torch.autograd.grad((actual * upstream).sum(), [x, *parameters], allow_unused=True)
    wanted = torch.autograd.grad((expected * upstream).sum(), [x, *parameters], allow_unused=True)
    # Compute both gradient paths before asserting; this checks the whole layer.
    if check == "outputs":
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        return
    for observed, target in zip(got, wanted):
        if target is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, target, rtol=1e-9, atol=1e-9)


# ============================================================================
# Focused regression: cardinal reverse half must read odd input features
# ============================================================================

def test_cardinal_reverse_half_uses_odd_feature_like_main():
    dims = (1, 0, 1, 1, 1)
    x = torch.zeros(1, 8, 3, 5, 4, dtype=torch.float64)
    x[0, 0, 1, 2, 1] = 1  # An odd-feature pulse in the first cardinal stream.
    identity = torch.eye(4, dtype=torch.float64)
    weights = [[identity, None], [identity, None]]
    masks = torch.ones(1, 3, 5, 2, dtype=torch.float64)
    expected = main_layer(dims, x, weights, masks, post_norm=False, use_bias=False)
    actual = ShiftPrimitives(bind_layout(dims))(x, weights, masks, pre_norm=False)
    # The second output half reverses movement and places feature 1 in slot 2.
    assert expected[0, 0, 0, 2, 2].item() == 1
    assert actual[0, 0, 0, 2, 2].item() == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

# ============================================================================
# Isolated directional adapter: no projections, normalization or residual
# ============================================================================

@pytest.mark.parametrize("has_colors", [False, True])
@pytest.mark.parametrize("direction", range(8))
@pytest.mark.parametrize("half", [0, 1], ids=["first-half", "reverse-half"])
@pytest.mark.parametrize("check", ["outputs", "gradients"])
def test_directional_adapter_matches_main(has_colors, direction, half, check):
    from compressarc.layers.helper import make_directional_layer

    dims = (1, int(has_colors), 1, 1, 1)
    shape = [2] + ([3] if has_colors else []) + [8, 3, 5, 4]
    direction_axis = sum(dims[:2])
    generator = torch.Generator().manual_seed(401)
    values = torch.randn(shape, generator=generator, dtype=torch.float64)
    current_x = values.clone().requires_grad_()
    main_x = values.clone().requires_grad_()
    masks = torch.ones(2, 3, 5, 2, dtype=torch.float64)
    masks[0, 1, 2] = 0
    masks[1, 0, 1, 0] = 0
    masks[1, 2, 3, 1] = 0
    current_adapter = make_directional_layer(shift_, diagonal_shift_)
    main_adapter = reference.make_directional_layer(reference.shift_, reference.diagonal_shift_)
    current_all = current_adapter(dims, current_x, masks)
    main_all = main_adapter(dims, main_x, masks)
    # Check one direction/output half independently so other streams cannot
    # hide the fact that diagonal and first-half cardinal routes agree.
    actual = torch.select(current_all, direction_axis, direction)[..., half * 2:(half + 1) * 2]
    expected = torch.select(main_all, direction_axis, direction)[..., half * 2:(half + 1) * 2]
    if check == "outputs":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
        got, = torch.autograd.grad((actual * upstream).sum(), current_x)
        wanted, = torch.autograd.grad((expected * upstream).sum(), main_x)
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
