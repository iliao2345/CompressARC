"""GPU checks for the Triton diagonal LSE forward, dx and dk."""
import pytest
import torch

pytest.importorskip('triton')
if not torch.cuda.is_available():
    pytest.skip('Triton LSE requires CUDA', allow_module_level=True)

from compressarc.layers.cummax.triton_op import DiagonalLSEFunction, FusedDiagonalLSE


def reference(x, kernel, tau):
    _, height, width = x.shape
    result = torch.empty_like(x)
    for row in range(height):
        for col in range(width):
            u = torch.arange(min(row, col) + 1, device=x.device)
            scores = x[:, row - u, col - u] + kernel[u]
            result[:, row, col] = tau * torch.logsumexp(scores / tau, dim=-1)
    return result


@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (7, 1), (3, 5), (5, 3), (8, 8), (7, 16)])
@pytest.mark.parametrize('tau', [0.03, 0.1, 0.7])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'transposed'])
def test_forward_input_and_kernel_gradients(shape, tau, layout):
    height, width = shape
    torch.manual_seed(71)
    if layout == 'strided':
        base = torch.randn(3, height * 2 + 2, width * 2 + 2, device='cuda') * 0.2
        view = lambda x: x[1:, 1:1 + height * 2:2, 1:1 + width * 2:2]
    elif layout == 'transposed':
        base = torch.randn(2, width, height, device='cuda') * 0.2
        view = lambda x: x.transpose(-1, -2)
    else:
        base = torch.randn(2, height, width, device='cuda') * 0.2
        view = lambda x: x
    reference_base = base.clone().requires_grad_()
    triton_base = base.clone().requires_grad_()
    # Include an unused tail; its gradient must remain exactly zero.
    kernel = torch.randn(2 * (min(height, width) + 3), device='cuda') * 0.15
    reference_kernel = kernel.clone()[::2].detach().requires_grad_()
    triton_kernel = kernel.clone()[::2].detach().requires_grad_()
    expected = reference(view(reference_base), reference_kernel, tau)
    actual = DiagonalLSEFunction.apply(view(triton_base), triton_kernel, tau)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    dy = torch.randn_like(actual).transpose(-1, -2).contiguous().transpose(-1, -2)
    expected.backward(dy)
    actual.backward(dy)
    torch.testing.assert_close(triton_base.grad, reference_base.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(triton_kernel.grad, reference_kernel.grad, rtol=3e-4, atol=8e-5)
    assert torch.count_nonzero(triton_kernel.grad[min(height, width):]) == 0
    assert torch.isfinite(triton_base.grad).all()
    assert torch.isfinite(triton_kernel.grad).all()


@pytest.mark.parametrize('x_grad,k_grad', [(True, False), (False, True), (True, True)])
def test_repeat_backward_zeroes_atomic_buffer(x_grad, k_grad):
    x = torch.randn(2, 3, 5, device='cuda', requires_grad=x_grad)
    kernel = torch.randn(5, device='cuda', requires_grad=k_grad)
    actual = DiagonalLSEFunction.apply(x, kernel, 0.3)
    targets = [p for p in [x, kernel] if p.requires_grad]
    first = torch.autograd.grad(actual.sum(), targets, retain_graph=True)
    second = torch.autograd.grad((actual * 2).sum(), targets)
    for got, previous in zip(second, first):
        torch.testing.assert_close(got, 2 * previous, rtol=3e-4, atol=8e-5)


def test_module_optimizer_and_unused_tail():
    layer = FusedDiagonalLSE(8, tau=0.1).cuda()
    x = torch.randn(2, 3, 5, device='cuda', requires_grad=True)
    before = layer.kernel.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    layer(x).mean().backward()
    assert x.grad.shape == x.shape
    assert layer.kernel.grad.shape == (8,)
    assert torch.count_nonzero(layer.kernel.grad[3:]) == 0
    optimizer.step()
    assert not torch.equal(before[:3], layer.kernel[:3])
    torch.testing.assert_close(before[3:], layer.kernel[3:])


@pytest.mark.parametrize('tau', [0.0, -0.1, float('inf'), float('nan')])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError, match='tau'):
        DiagonalLSEFunction.apply(torch.zeros(1, 2, 3, device='cuda'),
                                  torch.zeros(2, device='cuda'), tau)


def test_validation():
    layer = FusedDiagonalLSE(2).cuda()
    with pytest.raises(ValueError, match='max_length'):
        layer(torch.zeros(1, 3, 4, device='cuda'))
    with pytest.raises(TypeError, match='float32'):
        DiagonalLSEFunction.apply(torch.zeros(1, 2, 3, device='cuda', dtype=torch.float64),
                                  torch.zeros(2, device='cuda'), 0.1)

@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (7, 1), (3, 5), (5, 3), (8, 8)])
@pytest.mark.parametrize('tau', [0.03, 0.1, 0.7])
@pytest.mark.parametrize('transposed', [False, True], ids=['contiguous', 'transposed'])
def test_triton_diagonal_matches_optimisation_output_and_gradients(shape, tau, transposed):
    from compressarc.layers.cummax.optimisation import MorphologicalMax

    height, width = shape
    torch.manual_seed(29)
    if transposed:
        values = (torch.randn(2, width, height, device='cuda') * 0.2).transpose(-1, -2)
    else:
        values = torch.randn(2, height, width, device='cuda') * 0.2
    x_reference = values.detach().clone().requires_grad_()
    x_triton = values.detach().clone().requires_grad_()

    reference_model = MorphologicalMax(height, width, tau=tau, timing=False).cuda()
    triton_model = FusedDiagonalLSE(min(height, width), tau=tau).cuda()
    with torch.no_grad():
        reference_model.diagonal_kernel.normal_(std=0.15)
        triton_model.kernel.copy_(reference_model.diagonal_kernel)

    expected = reference_model._diagonal_lse(x_reference)
    actual = triton_model(x_triton)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    # Use identical, nonuniform  plus gradients to check the full backward map.
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    assert x_reference.grad is not None and x_triton.grad is not None
    assert reference_model.diagonal_kernel.grad is not None and triton_model.kernel.grad is not None
    torch.testing.assert_close(x_triton.grad, x_reference.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(triton_model.kernel.grad, reference_model.diagonal_kernel.grad,
                               rtol=3e-4, atol=8e-5)


@pytest.mark.parametrize("shape", [(1, 5), (3, 5), (5, 3)])
def test_triton_directional_model_matches_optimisation_outputs_and_gradients(shape):
    from compressarc.layers.cummax.optimisation import MorphologicalMax
    from compressarc.layers.cummax.triton_op import TritonMorphologicalMax

    height, width = shape
    torch.manual_seed(137)
    expected_model = MorphologicalMax(height, width, tau=0.3).cuda()
    actual_model = TritonMorphologicalMax(height, width, tau=0.3).cuda()
    with torch.no_grad():
        expected_model.kernel.normal_(std=0.15)
        expected_model.diagonal_kernel.normal_(std=0.15)
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    values = torch.randn(2, height, width, device="cuda") * 0.2
    expected_x = values.clone().requires_grad_()
    actual_x = values.clone().requires_grad_()
    expected, actual = expected_model(expected_x), actual_model(actual_x)
    for direction in range(8):
        torch.testing.assert_close(actual[:, direction], expected[:, direction], rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual)
    expected_grad = torch.autograd.grad((expected * upstream).sum(),
                                       [expected_x, *expected_model.parameters()])
    actual_grad = torch.autograd.grad((actual * upstream).sum(),
                                     [actual_x, *actual_model.parameters()])
    for got, want in zip(actual_grad, expected_grad):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=3e-4, atol=8e-5)


@pytest.mark.parametrize("post_norm", [False, True])
def test_triton_factory_layer_matches_optimisation_masks_projections_and_gradients(post_norm):
    from compressarc.model.layer_factory import LayerFactory
    from compressarc.model.multitensor_systems import MultiTensorSystem, multify

    torch.manual_seed(149)
    system = MultiTensorSystem(2, 2, 3, 5, task=None)
    factory = LayerFactory()
    expected_layer = factory.create_cummax("optimised", multify=multify, height=3, width=5, tau=0.3).cuda()
    actual_layer = factory.create_cummax("triton", multify=multify, height=3, width=5, tau=0.3).cuda()
    with torch.no_grad():
        expected_layer.models[0].kernel.normal_(std=0.15)
        expected_layer.models[0].diagonal_kernel.normal_(std=0.15)
    actual_layer.load_state_dict(expected_layer.state_dict(), strict=True)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    targets, active = [], []
    for dims in system:
        inputs[dims] = (torch.randn(system.shape(dims, extra_dim=4), device="cuda") * 0.2).requires_grad_()
        if tuple(dims) not in ((1, 0, 1, 1, 1), (1, 1, 1, 1, 1)):
            weights[dims] = None
            continue
        active.append(dims)
        w1 = (torch.randn(4, 6, device="cuda") * 0.2).requires_grad_()
        w2 = (torch.randn(6, 4, device="cuda") * 0.2).requires_grad_()
        weights[dims] = [[w1, None], [w2, None]]
        targets.extend([inputs[dims], w1, w2])
    masks = torch.randint(0, 2, (2, 3, 5, 2), device="cuda").float()
    expected = expected_layer(inputs, weights, masks, pre_norm=False, post_norm=post_norm)
    actual = actual_layer(inputs, weights, masks, pre_norm=False, post_norm=post_norm)
    expected_terms, actual_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[dims], rtol=2e-5, atol=2e-6)
        if dims not in active:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn_like(actual[dims])
            expected_terms.append((expected[dims] * upstream).sum())
            actual_terms.append((actual[dims] * upstream).sum())
    expected_grad = torch.autograd.grad(sum(expected_terms), [*targets, *expected_layer.parameters()])
    actual_grad = torch.autograd.grad(sum(actual_terms), [*targets, *actual_layer.parameters()])
    for got, want in zip(actual_grad, expected_grad):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=3e-4, atol=8e-5)


# Axis autograd uses the same validation/first-order contract as diagonal LSE.
def reference_axis(x, kernel, tau):
    result = torch.empty_like(x)
    for index in range(x.shape[-1]):
        scores = x[..., :index + 1].flip(-1) + kernel[:index + 1]
        result[..., index] = tau * torch.logsumexp(scores / tau, dim=-1)
    return result


@pytest.mark.parametrize("shape", [(2, 1), (2, 3), (2, 3, 5), (2, 4, 7), (2, 2, 3, 9)])
@pytest.mark.parametrize("tau", [0.03, 0.1, 0.7])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
def test_axis_forward_input_and_kernel_gradients(shape, tau, layout):
    from compressarc.layers.cummax.triton_op import AxisLSEFunction

    torch.manual_seed(167)
    if layout == "strided":
        base = torch.randn(*shape[:-1], shape[-1] * 2 + 2, device="cuda") * 0.2
        view = lambda value: value[..., 1:1 + 2 * shape[-1]:2]
    elif layout == "transposed":
        base = torch.randn(*shape, device="cuda") * 0.2
        view = lambda value: value.transpose(-1, -2)
    else:
        base = torch.randn(*shape, device="cuda") * 0.2
        view = lambda value: value
    expected_base = base.clone().requires_grad_()
    actual_base = base.clone().requires_grad_()
    width = view(actual_base).shape[-1]
    values = torch.randn(2 * (width + 3), device="cuda") * 0.15
    expected_kernel = values[::2].clone().requires_grad_()
    actual_kernel = values[::2].detach().requires_grad_()
    expected = reference_axis(view(expected_base), expected_kernel, tau)
    actual = AxisLSEFunction.apply(view(actual_base), actual_kernel, tau)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual).transpose(-1, -2).contiguous().transpose(-1, -2)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual_base.grad, expected_base.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual_kernel.grad, expected_kernel.grad, rtol=3e-4, atol=8e-5)
    assert torch.count_nonzero(actual_kernel.grad[width:]) == 0
    assert torch.isfinite(actual_base.grad).all() and torch.isfinite(actual_kernel.grad).all()


@pytest.mark.parametrize("x_grad,k_grad", [(True, False), (False, True), (True, True)])
def test_axis_repeat_backward_zeroes_atomic_buffer(x_grad, k_grad):
    from compressarc.layers.cummax.triton_op import AxisLSEFunction

    x = torch.randn(2, 3, 5, device="cuda", requires_grad=x_grad)
    kernel = torch.randn(8, device="cuda", requires_grad=k_grad)
    output = AxisLSEFunction.apply(x, kernel, 0.3)
    targets = [value for value in (x, kernel) if value.requires_grad]
    first = torch.autograd.grad(output.sum(), targets, retain_graph=True)
    second = torch.autograd.grad((2 * output).sum(), targets)
    for got, previous in zip(second, first):
        torch.testing.assert_close(got, 2 * previous, rtol=3e-4, atol=8e-5)
    if k_grad:
        assert torch.count_nonzero(second[-1][5:]) == 0


@pytest.mark.parametrize("tau", [0., -0.1, float("inf"), float("nan")])
def test_axis_invalid_temperature(tau):
    from compressarc.layers.cummax.triton_op import AxisLSEFunction

    with pytest.raises(ValueError, match="tau"):
        AxisLSEFunction.apply(torch.zeros(2, 3, device="cuda"), torch.zeros(3, device="cuda"), tau)


def test_axis_validation_and_module_optimizer():
    from compressarc.layers.cummax.triton_op import AxisLSEFunction, FusedAxisLSE

    kernel = torch.zeros(8, device="cuda")
    with pytest.raises(ValueError, match="nonempty"):
        AxisLSEFunction.apply(torch.zeros((), device="cuda"), kernel, 0.1)
    with pytest.raises(ValueError, match="nonempty"):
        AxisLSEFunction.apply(torch.zeros(2, 0, device="cuda"), kernel, 0.1)
    with pytest.raises(ValueError, match="at least W"):
        AxisLSEFunction.apply(torch.zeros(2, 9, device="cuda"), kernel, 0.1)
    with pytest.raises(ValueError, match="1D"):
        AxisLSEFunction.apply(torch.zeros(2, 3, device="cuda"), kernel.reshape(2, 4), 0.1)
    with pytest.raises(ValueError, match="same CUDA"):
        AxisLSEFunction.apply(torch.zeros(2, 3, device="cpu"), kernel, 0.1)
    with pytest.raises(TypeError, match="float32"):
        AxisLSEFunction.apply(torch.zeros(2, 3, device="cuda", dtype=torch.float64), kernel, 0.1)
    with pytest.raises(TypeError, match="constant"):
        AxisLSEFunction.apply(torch.zeros(2, 3, device="cuda"), kernel, torch.tensor(0.1))
    layer = FusedAxisLSE(8, tau=0.1).cuda()
    x = torch.randn(2, 3, 5, device="cuda", requires_grad=True)
    before = layer.kernel.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    layer(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(layer.kernel.grad[5:]) == 0
    optimizer.step()
    assert not torch.equal(before[:5], layer.kernel[:5])
    torch.testing.assert_close(before[5:], layer.kernel[5:])
    with pytest.raises(ValueError, match="max_length"):
        layer(torch.zeros(1, 9, device="cuda"))
