import pytest
import torch

from src import CompressARCShift, conv_shift_, conv_diagonal_shift_

from src.native_reference import reference


@pytest.mark.parametrize("shape,dim", [((3, 5), 0), ((3, 5), 1), ((2, 3, 4, 5), -2), ((2, 1), 1)])
def test_cardinal_primitive(shape, dim):
    x = torch.randn(shape, dtype=torch.float64)
    torch.testing.assert_close(conv_shift_(x, dim), reference.shift_(x, dim, None), rtol=0, atol=0)


@pytest.mark.parametrize("amount", [-2, -1, 0, 1, 2])
def test_diagonal_primitive(amount):
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    expected = reference.diagonal_shift_(x, 1, 2, None, amount, 7)
    actual = conv_diagonal_shift_(x, 1, 2, shift_amount=amount, pad_value=7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("pre_norm,post_norm", [(False, False), (True, False), (False, True), (True, True)])
def test_full_shift_outputs_and_gradients(use_bias, pre_norm, post_norm):
    generator = torch.Generator().manual_seed(42)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64, requires_grad=True)

    system = reference.multitensor_systems.MultiTensorSystem(2, 3, 3, 5, task=None)
    x = system.make_multitensor()
    weights = system.make_multitensor()
    active = [(1, 1, 1, 1, 1), (1, 0, 1, 1, 1)]
    differentiable = []
    for dims in system:
        x[dims] = rand(*system.shape(dims, extra_dim=4))
        if tuple(dims) in active:
            weights[dims] = [[rand(4, 6), rand(6)], [rand(6, 4), rand(4)]]
            differentiable.append(x[dims])
            differentiable.extend(w for pair in weights[dims] for w in pair)

    # Different masks per example, including holes and boundaries.
    masks = torch.randint(0, 2, (2, 3, 5, 2), generator=generator).double()
    flags = dict(use_bias=use_bias, pre_norm=pre_norm, post_norm=post_norm)
    expected = reference.shift(x, weights, masks, **flags)
    actual = CompressARCShift(reference)(x, weights, masks, **flags)
    assert actual.multitensor_system is system
    for dims in system:
        assert actual[dims].shape == x[dims].shape
        torch.testing.assert_close(actual[dims], expected[dims], rtol=1e-10, atol=1e-10)
        if tuple(dims) not in active:
            assert actual[dims] is x[dims]

    expected_loss = sum(expected[d].square().sum() for d in active)
    actual_loss = sum(actual[d].square().sum() for d in active)
    expected_grads = torch.autograd.grad(expected_loss, differentiable, allow_unused=True)
    actual_grads = torch.autograd.grad(actual_loss, differentiable, allow_unused=True)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("has_colors", [False, True])
@pytest.mark.parametrize("mask_value", [0, 1, None])
def test_tied_conv_direction_mapping(has_colors, mask_value):
    from src.group_cnn import TiedDirectionalConv
    from src.shift_adapter import tied_directional_shift
    generator = torch.Generator().manual_seed(17)
    shape = (2, 3, 8, 3, 5, 6) if has_colors else (2, 8, 3, 5, 6)
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    masks = torch.randint(0, 2, (2, 3, 5, 2), generator=generator).double()
    if mask_value is not None:
        masks.fill_(mask_value)
    dims = (1, int(has_colors), 1, 1, 1)
    model = TiedDirectionalConv().double()
    expected = reference.make_directional_layer(reference.shift_, reference.diagonal_shift_)(dims, x, masks)
    actual = tied_directional_shift(x, masks, model)
    torch.testing.assert_close(actual, expected)
    expected_grad, = torch.autograd.grad(expected.square().sum(), x)
    actual_grad, = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)
    torch.testing.assert_close(actual_grad, expected_grad)
    kernel_grads = torch.autograd.grad(actual.square().sum(), tuple(model.parameters()))
    assert all(torch.isfinite(grad).all() for grad in kernel_grads)
