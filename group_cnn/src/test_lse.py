import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

# Load the sibling module explicitly: CompressARC also has an unrelated lse.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('group_cnn_lse_under_test', Path(__file__).with_name('lse.py'))
lse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lse)
MorphologicalMax = lse.MorphologicalMax


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
SHAPES = [(1, 1, 1), (2, 3, 1), (1, 1, 7), (2, 3, 5), (3, 5, 2), (1, 4, 4)]


def loop_shift(x):
    """Original shift construction retained as an independent reference."""
    width = x.shape[-1]
    flat = x.reshape(-1, width)
    shifted = x.new_full((flat.shape[0], width, width), float('-inf'))
    for i in range(width):
        for u in range(i + 1):
            shifted[:, i, u] = flat[:, i - u]
    return shifted


def loop_phi(x, kernel, support, tau):
    width = x.shape[-1]
    scores = loop_shift(x) + kernel[:width].view(1, 1, width)
    scores = scores.masked_fill(~support[:width].view(1, 1, width), float('-inf'))
    smooth = tau * torch.logsumexp(scores / tau, dim=-1)
    exact = scores.max(dim=-1).values
    return smooth.reshape(x.shape), exact.reshape(x.shape)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_pad_unfold_matches_previous_shift(shape, noncontiguous, device):
    torch.manual_seed(17)
    x = torch.randn(*shape, 2, device=device, dtype=torch.float64)[..., 0]
    if not noncontiguous:
        x = x.contiguous()
    width = x.shape[-1]
    padded = F.pad(x.reshape(-1, width), (width - 1, 0), value=float('-inf'))
    actual = padded.unfold(-1, width, 1).flip(-1)
    torch.testing.assert_close(actual, loop_shift(x), rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('masked_support', [False, True])
def test_phi_outputs_and_gradients_match_previous_loop(shape, dtype, masked_support, device):
    torch.manual_seed(23)
    # Strided input also verifies that reshape handles non-contiguous tensors.
    x = torch.randn(*shape, 2, device=device, dtype=dtype)[..., 0].requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    model = MorphologicalMax(shape[-2], shape[-1], tau=0.3).to(device=device, dtype=dtype)
    with torch.no_grad():
        # Nonconstant offsets catch a missing flip even when zero kernels would not.
        model.kernel.copy_(torch.randn_like(model.kernel))
        if masked_support:
            model.support[1::2] = False
    reference_kernel = model.kernel.detach().clone().requires_grad_()
    actual = model.phi(x)
    expected = loop_phi(reference_x, reference_kernel, model.support, model.tau)
    weights = torch.randn_like(x)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference)
        assert torch.isfinite(output).all()
        actual_grads = torch.autograd.grad((output * weights).sum(), (x, model.kernel), retain_graph=True)
        reference_grads = torch.autograd.grad((reference * weights).sum(),
                                              (reference_x, reference_kernel), retain_graph=True)
        for grad, reference_grad in zip(actual_grads, reference_grads):
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, reference_grad)


def test_offset_order_example():
    x = torch.tensor([[[1., 2., 3.]]])
    expected = torch.tensor([[[1., -torch.inf, -torch.inf],
                              [2., 1., -torch.inf], [3., 2., 1.]]])
    padded = F.pad(x.reshape(-1, 3), (2, 0), value=float('-inf'))
    torch.testing.assert_close(padded.unfold(-1, 3, 1).flip(-1), expected)


def loop_diagonal_phi(x, kernel, tau):
    """Original coordinate/shift loops, reducing only real diagonal positions."""
    batch, height, width = x.shape
    out, exact = torch.empty_like(x), torch.empty_like(x)
    for offset in range(-(height - 1), width):
        coords = [(h, h + offset) for h in range(height) if 0 <= h + offset < width]
        values = torch.stack([x[:, h, w] for h, w in coords], dim=-1)
        length = len(coords)
        shifted = x.new_full((batch, length, length), float('-inf'))
        for i in range(length):
            for u in range(i + 1):
                shifted[:, i, u] = values[:, i - u]
        scores = shifted + kernel[:length]
        smooth = tau * torch.logsumexp(scores / tau, -1)
        maxima = scores.max(-1).values
        for i, (h, w) in enumerate(coords):
            out[:, h, w] = smooth[:, i]
            exact[:, h, w] = maxima[:, i]
    return out, exact


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_diagonal_outputs_and_gradients_match_loops(shape, rotation, dtype, device):
    torch.manual_seed(31)
    base = torch.randn(*shape, 2, device=device, dtype=dtype)[..., 0]
    x = torch.rot90(base, rotation, (-2, -1)).detach().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    model = MorphologicalMax(x.shape[-2], x.shape[-1], tau=0.3).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.diagonal_kernel.normal_()
    kernel = model.diagonal_kernel.detach().clone().requires_grad_()
    actual = model.diagonal_phi(x)
    expected = loop_diagonal_phi(reference_x, kernel, model.tau)
    weights = torch.randn_like(x)
    for output, reference in zip(actual, expected):
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference)
        grads = torch.autograd.grad((output * weights).sum(), (x, model.diagonal_kernel), retain_graph=True)
        reference_grads = torch.autograd.grad((reference * weights).sum(), (reference_x, kernel), retain_graph=True)
        for grad, reference_grad in zip(grads, reference_grads):
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, reference_grad)
