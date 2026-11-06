"""Compare current projected LSE implementations directly with master 8282287."""

import importlib.util
from pathlib import Path

import pytest
import torch

from compressarc.layers.cummax.morphological import MorphologicalMax as Preoptimized
from compressarc.layers.cummax.optimisation import MorphologicalMax as Optimised
from compressarc.layers.cummax.layer import LSELayer

_spec = importlib.util.spec_from_file_location("lse_main_reference", Path(__file__).with_name("reference_lse_main.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def backend(implementation):
    if implementation == "triton":
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("Triton LSE requires CUDA")
        from compressarc.layers.cummax.triton_op import TritonMorphologicalMax
        return TritonMorphologicalMax, "cuda", torch.float32
    return {"preoptimized": Preoptimized, "optimised": Optimised}[implementation], "cpu", torch.float64


def assert_parity(actual, expected, dtype, gradients=False):
    if dtype == torch.float64:
        tolerance = 1e-9 if gradients else 1e-10
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-4 if gradients else 2e-5,
                                   atol=8e-5 if gradients else 2e-6)


@pytest.mark.parametrize("implementation", ["preoptimized", "optimised", "triton"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("tau", [0.1, 0.7])
def test_lse_core_matches_main_outputs_and_gradients(implementation, shape, tau):
    model_type, device, dtype = backend(implementation)
    height, width = shape
    generator = torch.Generator().manual_seed(503)
    original = reference.MorphologicalMax(height, width, tau=tau).to(device=device, dtype=dtype)
    current = model_type(height, width, tau=tau).to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype).to(device) * 0.2)
    current.load_state_dict(original.state_dict(), strict=True)
    values = torch.randn(2, height, width, generator=generator, dtype=dtype).to(device) * 0.2
    original_x = values.clone().requires_grad_()
    current_x = values.clone().requires_grad_()
    expected, actual = original(original_x), current(current_x)
    assert_parity(actual, expected, dtype)
    upstream = torch.randn(actual.shape, generator=generator, dtype=dtype).to(device)
    wanted = torch.autograd.grad((expected * upstream).sum(), [original_x, *original.parameters()])
    got = torch.autograd.grad((actual * upstream).sum(), [current_x, *current.parameters()])
    for observed, target in zip(got, wanted):
        assert torch.isfinite(observed).all()
        assert_parity(observed, target, dtype, gradients=True)


@pytest.mark.parametrize("implementation", ["preoptimized", "optimised", "triton"])
@pytest.mark.parametrize("has_colors", [False, True])
def test_projected_lse_layer_matches_main_outputs_and_gradients(implementation, has_colors):
    model_type, device, dtype = backend(implementation)
    dims = (1, int(has_colors), 1, 1, 1)
    generator = torch.Generator().manual_seed(509)

    def multify(operation):
        def apply(*args, **kwargs):
            return operation(dims, *args, **kwargs)
        return apply

    layer = LSELayer(multify, 3, 5, model_type=model_type, n_layers=2, tau=0.1).to(device=device, dtype=dtype)
    original = reference.MorphologicalMax(3, 5, tau=0.1).to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype).to(device) * 0.2)
    layer.models[1].load_state_dict(original.state_dict(), strict=True)
    shape = [2] + ([3] if has_colors else []) + [8, 3, 5, 8]
    x = torch.randn(shape, generator=generator, dtype=dtype).to(device).requires_grad_()
    up = torch.randn(8, 4, generator=generator, dtype=dtype).to(device).requires_grad_()
    down = torch.randn(4, 8, generator=generator, dtype=dtype).to(device).requires_grad_()
    masks = torch.ones(2, 3, 5, 2, device=device, dtype=dtype)
    masks[0, 1, 2] = 0
    masks[1, 0, 1, 0] = 0
    masks[1, 2, 3, 1] = 0
    # Reproduce main's projected wrapper, including post-normalization.
    mixed = reference.directional_lse(dims, x @ up, masks, original)
    axes = tuple(range(mixed.ndim - 1))
    centered = mixed - mixed.mean(dim=axes)
    normalized = centered / torch.sqrt(1e-8 + centered.square().mean(dim=axes))
    expected = x + normalized @ down
    actual = layer(x, [[up, None], [down, None]], masks, layer_index=1,
                   pre_norm=False, post_norm=True, use_bias=False)
    assert_parity(actual, expected, dtype)
    upstream = torch.randn(shape, generator=generator, dtype=dtype).to(device)
    wanted = torch.autograd.grad((expected * upstream).sum(), [x, up, down, *original.parameters()])
    got = torch.autograd.grad((actual * upstream).sum(), [x, up, down, *layer.models[1].parameters(),
                                                       *layer.models[0].parameters()], allow_unused=True)
    assert got[-2:] == (None, None)
    for observed, target in zip(got[:-2], wanted):
        assert observed is not None and torch.isfinite(observed).all()
        assert_parity(observed, target, dtype, gradients=True)
