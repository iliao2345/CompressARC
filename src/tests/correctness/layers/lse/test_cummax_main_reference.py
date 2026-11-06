"""Hard cummax output/gradient parity against the original main implementation."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest
import torch

from compressarc.layers.cummax.primitives import CummaxPrimitives, cummax_, diagonal_cummax_

_spec = importlib.util.spec_from_file_location("cummax_main_reference", Path(__file__).with_name("reference_cummax_main.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def mask_like(values, mode):
    masks = torch.ones_like(values)
    if mode == "holes":
        masks[..., values.shape[-2] // 2, values.shape[-1] // 2] = 0
    elif mode == "empty":
        masks.zero_()
    return masks


def compare_outputs_and_input_gradients(actual, expected, current_x, main_x):
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.linspace(-0.7, 1.3, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    got, = torch.autograd.grad((actual * upstream).sum(), current_x)
    wanted, = torch.autograd.grad((expected * upstream).sum(), main_x)
    assert torch.isfinite(got).all() and torch.isfinite(wanted).all()
    torch.testing.assert_close(got, wanted, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("mask_mode", ["full", "holes", "empty"])
@pytest.mark.parametrize("tied_values", [False, True], ids=["random", "ties"])
@pytest.mark.parametrize("axis", [1, 2])
def test_cardinal_cummax_matches_main_outputs_and_gradients(shape, mask_mode, tied_values, axis):
    generator = torch.Generator().manual_seed(701)
    values = torch.zeros(2, *shape, dtype=torch.float64) if tied_values else torch.randn(2, *shape, generator=generator, dtype=torch.float64)
    current_x, main_x = values.clone().requires_grad_(), values.clone().requires_grad_()
    masks = mask_like(values, mask_mode)
    actual, expected = cummax_(current_x, axis, masks), reference.cummax_(main_x, axis, masks)
    compare_outputs_and_input_gradients(actual, expected, current_x, main_x)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("mask_mode", ["full", "holes", "empty"])
@pytest.mark.parametrize("tied_values", [False, True], ids=["random", "ties"])
def test_diagonal_cummax_matches_main_outputs_and_gradients(shape, mask_mode, tied_values):
    generator = torch.Generator().manual_seed(709)
    values = torch.zeros(2, *shape, dtype=torch.float64) if tied_values else torch.randn(2, *shape, generator=generator, dtype=torch.float64)
    current_x, main_x = values.clone().requires_grad_(), values.clone().requires_grad_()
    masks = mask_like(values, mask_mode)
    actual = diagonal_cummax_(current_x, 1, 2, masks)
    expected = reference.diagonal_cummax_(main_x, 1, 2, masks)
    compare_outputs_and_input_gradients(actual, expected, current_x, main_x)


def bind_layout(dims):
    def multify(operation):
        def apply(*args, **kwargs):
            return operation(dims, *args, **kwargs)
        return apply
    return multify


@pytest.mark.parametrize("has_colors", [False, True])
@pytest.mark.parametrize("mask_mode", ["full", "holes", "empty"])
@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_cummax_layer_matches_main_outputs_and_gradients(has_colors, mask_mode, post_norm, use_bias):
    dims = (1, int(has_colors), 1, 1, 1)
    generator = torch.Generator().manual_seed(719)
    shape = [2] + ([3] if has_colors else []) + [8, 3, 5, 8]
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    parameters = [torch.randn(size, generator=generator, dtype=torch.float64, requires_grad=True) * 0.2
                  for size in ((8, 4), (4,), (4, 8), (8,))]
    weights = [parameters[:2], parameters[2:]]
    masks = mask_like(torch.ones(2, 3, 5, dtype=torch.float64), mask_mode)[..., None].expand(-1, -1, -1, 2).clone()
    if mask_mode == "holes":
        masks[1, 0, 1, 0] = 0  # Output-only pixels remain valid.
        masks[1, 2, 3, 1] = 0  # Input-only pixels remain valid.
    projected = x @ parameters[0]
    if use_bias:
        projected = projected + parameters[1]
    mixed = reference.make_directional_layer(reference.cummax_, reference.diagonal_cummax_)(dims, projected, masks)
    if post_norm:
        axes = tuple(range(mixed.ndim - 1))
        mixed = mixed - mixed.mean(dim=axes)
        mixed = mixed / torch.sqrt(1e-8 + mixed.square().mean(dim=axes))
    update = mixed @ parameters[2]
    if use_bias:
        update = update + parameters[3]
    expected = x + update
    actual = CummaxPrimitives(bind_layout(dims))(x, weights, masks, pre_norm=False,
                                               post_norm=post_norm, use_bias=use_bias)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn(shape, generator=generator, dtype=torch.float64)
    # Projection inputs are shared by the two graphs; retain the first traversal.
    got = torch.autograd.grad((actual * upstream).sum(), [x, *parameters], allow_unused=True, retain_graph=True)
    wanted = torch.autograd.grad((expected * upstream).sum(), [x, *parameters], allow_unused=True)
    for observed, target in zip(got, wanted):
        if target is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, target, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dims", [dims for dims in product([0, 1], repeat=5)
                                  if dims not in ((1, 0, 1, 1, 1), (1, 1, 1, 1, 1))])
def test_cummax_inactive_layouts_pass_through(dims):
    shape = [size for flag, size in zip(dims, [2, 3, 8, 3, 5]) if flag] + [8]
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    actual = CummaxPrimitives(bind_layout(dims))(x, None, None, pre_norm=False, post_norm=True)
    assert actual is x
    gradient, = torch.autograd.grad(actual.sum(), x)
    torch.testing.assert_close(gradient, torch.ones_like(x), rtol=0, atol=0)
