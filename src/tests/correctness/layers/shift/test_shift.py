"""Shift primitive, tied-convolution, unfold, and Triton layer checks."""

from itertools import product

import pytest
import torch

from compressarc.layers.shift.primitives import ShiftPrimitives, diagonal_shift_, shift_
from compressarc.layers.shift.morphological import TiedDirectionalConv, UnfoldTiedDirectionalConv
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.multitensor_systems import MultiTensorSystem, multify


# ============================================================================
# Reference helpers
# ============================================================================

@pytest.fixture(
        params=[(2, 3), (0, 0), (4, 5)], ids=["interior", "top_left", "bottom_right"])
def pulse(request):
    position = request.param
    x = torch.zeros(1, 5, 6)
    x[0, position[0], position[1]] = 1
    return x, position


def expected_pulse(x, position, delta):
    expected = torch.zeros_like(x)
    row, col = (position[i] + delta[i] for i in range(2))
    if 0 <= row < x.shape[1] and 0 <= col < x.shape[2]:
        expected[0, row, col] = 1
    return expected


ACTIVE_SHAPES = {(1, 1, 1, 1, 1), (1, 0, 1, 1, 1)}


def reference_normalize(x):
    axes = tuple(range(x.ndim - 1))
    centered = x - x.mean(dim=axes)
    return centered / torch.sqrt(1e-8 + centered.square().mean(dim=axes))


def reference_directions(x, masks, has_colors):
    """Place values at explicit destinations without using the layer helpers."""
    valid = masks.any(dim=-1)
    if has_colors:
        valid = valid[:, None]
    masked = x * valid.unsqueeze(-3).unsqueeze(-1)
    result = torch.zeros_like(x)
    # Direction slots represent down, down-right, right, up-right, then reverses.
    deltas = [(1, 0), (1, 1), (0, 1), (-1, 1),
              (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for direction, (dr, dc) in enumerate(deltas):
        for half in range(2):
            row_delta, col_delta = (dr, dc) if half == 0 else (-dr, -dc)
            # Both orbit types select even/odd input features by output half.
            source_half = half
            source = masked[..., direction, :, :, source_half::2]
            for row, col in product(range(x.shape[-3]), range(x.shape[-2])):
                target_row, target_col = row + row_delta, col + col_delta
                if 0 <= target_row < x.shape[-3] and 0 <= target_col < x.shape[-2]:
                    result[..., direction, target_row, target_col, half] = source[..., row, col, 0]
    return result


@pytest.fixture
def triton_shift_runtime():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("Triton shift requires CUDA")


# ============================================================================
# Correctness
# ============================================================================

@pytest.mark.parametrize(
    "dim,reverse,delta",
    [(2, False, (0, 1)), (2, True, (0, -1)),
     (1, False, (1, 0)), (1, True, (-1, 0))],
    ids=["right", "left", "down", "up"],
)
def test_cardinal_shift(pulse, dim, reverse, delta):
    x, position = pulse
    # shift_ moves toward increasing indices; flipping gives the opposite direction.
    oriented = torch.flip(x, [dim]) if reverse else x
    actual = shift_(oriented, dim=dim, masks=None)
    if reverse:
        actual = torch.flip(actual, [dim])

    torch.testing.assert_close(actual, expected_pulse(x, position, delta), rtol=0, atol=0)



@pytest.mark.parametrize(
    "shift_amount,flip_columns,delta",
    [(1, False, (1, 1)), (1, True, (1, -1)),
     (-1, False, (-1, -1)), (-1, True, (-1, 1))],
    ids=["down_right", "down_left", "up_left", "up_right"],
)
def test_diagonal_shift(pulse, shift_amount, flip_columns, delta):
    x, position = pulse
    # The primitive uses one sign for both axes; flip columns for mixed directions.
    oriented = torch.flip(x, [2]) if flip_columns else x
    actual = diagonal_shift_(oriented, dim1=1, dim2=2, masks=None,
                             shift_amount=shift_amount)
    if flip_columns:
        actual = torch.flip(actual, [2])

    torch.testing.assert_close(actual, expected_pulse(x, position, delta), rtol=0, atol=0)

@pytest.mark.parametrize(
    "orbit,direction,delta",
    [(0, 0, (-1, 0)), (0, 1, (0, -1)),
     (0, 2, (1, 0)), (0, 3, (0, 1)),
     (1, 0, (1, 1)), (1, 1, (-1, 1)),
     (1, 2, (-1, -1)), (1, 3, (1, -1))],
    ids=["up", "left", "down", "right",
         "down_right", "up_right", "up_left", "down_left"],
)
def test_tied_directional_conv_matches_primitives(pulse, orbit, direction, delta):
    grid, _ = pulse
    # Different values in two batches also catch accidental batch mixing.
    grids = torch.cat([grid, 2 * grid], dim=0)
    x = torch.zeros(2, 2, 4, *grid.shape[-2:])
    x[:, orbit, direction] = grids

    # Orient the primitive toward the requested movement.
    flip_dims = [axis + 1 for axis, amount in enumerate(delta) if amount < 0]
    oriented = torch.flip(grids, flip_dims) if flip_dims else grids
    if orbit == 0:
        dim = 1 if delta[0] else 2
        shifted = shift_(oriented, dim=dim, masks=None)
    else:
        shifted = diagonal_shift_(oriented, dim1=1, dim2=2, masks=None)
    if flip_dims:
        shifted = torch.flip(shifted, flip_dims)

    expected = torch.zeros_like(x)
    expected[:, orbit, direction] = shifted
    actual = TiedDirectionalConv()(x)

    # Compare the entire output: untouched direction channels must stay zero.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ============================================================================
# Outputs and gradients
# ============================================================================

@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
@pytest.mark.parametrize("learned", [False, True], ids=["initial", "learned"])
def test_unfold_core_outputs_and_gradients(shape, layout, learned):
    height, width = shape
    generator = torch.Generator().manual_seed(223)
    expected_model = TiedDirectionalConv().double()
    actual_model = UnfoldTiedDirectionalConv().double()
    if learned:
        with torch.no_grad():
            for parameter in expected_model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.2)
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    if layout == "strided":
        values = torch.randn(2, 2, 4, height * 2 + 2, width * 2 + 2, generator=generator, dtype=torch.float64)
        view = lambda value: value[..., 1:1 + 2 * height:2, 1:1 + 2 * width:2]
    elif layout == "transposed":
        values = torch.randn(2, 2, 4, width, height, generator=generator, dtype=torch.float64)
        view = lambda value: value.transpose(-1, -2)
    else:
        values = torch.randn(2, 2, 4, height, width, generator=generator, dtype=torch.float64)
        view = lambda value: value
    expected_base = values.clone().requires_grad_()
    actual_base = values.clone().requires_grad_()
    expected, actual = expected_model(view(expected_base)), actual_model(view(actual_base))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    wanted = torch.autograd.grad((expected * upstream).sum(), [expected_base, *expected_model.parameters()])
    got = torch.autograd.grad((actual * upstream).sum(), [actual_base, *actual_model.parameters()])
    for observed, reference in zip(got, wanted):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3), (8, 8)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.usefixtures("triton_shift_runtime")
def test_tied_core_outputs_and_canonical_kernel_gradients(shape, layout, learned):
    from compressarc.layers.shift.triton_op import TritonTiedDirectionalConv

    torch.manual_seed(239)
    h, w = shape
    reference = TiedDirectionalConv().cuda()
    actual_model = TritonTiedDirectionalConv().cuda()
    if learned:
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=0.2)
    actual_model.load_state_dict(reference.state_dict(), strict=True)
    if layout == "strided":
        values = torch.randn(2, 2, 4, 2 * h + 2, 2 * w + 2, device="cuda") * 0.2
        view = lambda value: value[..., 1:1 + 2 * h:2, 1:1 + 2 * w:2]
    elif layout == "transposed":
        values = torch.randn(2, 2, 4, w, h, device="cuda") * 0.2
        view = lambda value: value.transpose(-1, -2)
    else:
        values = torch.randn(2, 2, 4, h, w, device="cuda") * 0.2
        view = lambda value: value
    expected_x = values.clone().requires_grad_()
    actual_x = values.clone().requires_grad_()
    with torch.backends.cudnn.flags(allow_tf32=False):
        expected = reference(view(expected_x))
    actual = actual_model(view(actual_x))
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    upstream = torch.randn_like(actual).transpose(-1, -2).contiguous().transpose(-1, -2)
    with torch.backends.cudnn.flags(allow_tf32=False):
        wanted = torch.autograd.grad((expected * upstream).sum(), [expected_x, *reference.parameters()])
    got = torch.autograd.grad((actual * upstream).sum(), [actual_x, *actual_model.parameters()])
    for observed, reference_grad in zip(got, wanted):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference_grad, rtol=3e-4, atol=8e-5)


@pytest.mark.parametrize("x_grad,k_grad", [(True, False), (False, True), (True, True)])
@pytest.mark.usefixtures("triton_shift_runtime")
def test_repeat_backward_and_partial_gradients(x_grad, k_grad):
    from compressarc.layers.shift.triton_op import DepthwiseConvFunction

    x = torch.randn(2, 8, 3, 5, device="cuda", requires_grad=x_grad)
    kernel = torch.randn(8, 1, 3, 3, device="cuda", requires_grad=k_grad)
    output = DepthwiseConvFunction.apply(x, kernel)
    targets = [value for value in (x, kernel) if value.requires_grad]
    first = torch.autograd.grad(output.sum(), targets, retain_graph=True)
    second = torch.autograd.grad((2 * output).sum(), targets)
    for actual, previous in zip(second, first):
        torch.testing.assert_close(actual, 2 * previous, rtol=3e-4, atol=8e-5)


# ============================================================================
# Adapter logic and integration
# ============================================================================

@pytest.mark.parametrize("pre_norm,post_norm,use_bias", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("entrypoint", ["forward", "__call__"])
def test_shift_primitives_glue(pre_norm, post_norm, use_bias, entrypoint):
    masks = torch.ones(2, 3, 4, 2, dtype=torch.float64)
    masks[0, 1, 1] = 0  # Neither input nor output includes this pixel.
    masks[1, 1, 2, 0] = 0  # Output-only pixels must still be included.
    masks[1, 2, 1, 1] = 0  # Input-only pixels must still be included.
    system = MultiTensorSystem(2, 2, 3, 4, task=None)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    generator = torch.Generator().manual_seed(17)
    expected = {}
    w1 = torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=torch.float64)
    w2 = torch.tensor([[0.75, 0.5], [-1.0, 1.25]], dtype=torch.float64)
    b1 = torch.tensor([0.2, -0.3], dtype=torch.float64)
    b2 = torch.tensor([-0.4, 0.6], dtype=torch.float64)

    for dims in system:
        key = tuple(dims)
        x = torch.randn(system.shape(dims, extra_dim=2), generator=generator,
                        dtype=torch.float64)
        inputs[dims] = x
        if key not in ACTIVE_SHAPES:
            # Inactive leaves must pass through without trying to read weights.
            weights[dims] = None
            continue
        weights[dims] = [[w1, b1], [w2, b2]]
        projected = (reference_normalize(x) if pre_norm else x) @ w1
        if use_bias:
            projected = projected + b1
        shifted = reference_directions(projected, masks, has_colors=bool(dims[1]))
        if post_norm:
            shifted = reference_normalize(shifted)
        update = shifted @ w2
        if use_bias:
            update = update + b2
        expected[key] = x + update

    layer = ShiftPrimitives(multify)
    actual = getattr(layer, entrypoint)(inputs, weights, masks,
                                       pre_norm=pre_norm, post_norm=post_norm,
                                       use_bias=use_bias)
    assert actual.multitensor_system is system
    for dims in system:
        key = tuple(dims)
        if key in ACTIVE_SHAPES:
            torch.testing.assert_close(actual[dims], expected[key])
        else:
            assert actual[dims] is inputs[dims]


@pytest.mark.parametrize("pre_norm,post_norm,use_bias", list(product([False, True], repeat=3)))
def test_unfold_factory_layer_matches_conv2d_masks_projections_and_gradients(pre_norm, post_norm, use_bias):
    generator = torch.Generator().manual_seed(227)
    system = MultiTensorSystem(2, 2, 3, 5, task=None)
    factory = LayerFactory()
    expected_layer = factory.create_shift("tied_conv", multify=multify).double()
    actual_layer = factory.create_shift("unfold", multify=multify).double()
    with torch.no_grad():
        for parameter in expected_layer.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.2)
    actual_layer.load_state_dict(expected_layer.state_dict(), strict=True)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    targets, active = [], []
    for dims in system:
        inputs[dims] = torch.randn(system.shape(dims, extra_dim=4), generator=generator,
                                   dtype=torch.float64, requires_grad=True)
        if tuple(dims) not in ((1, 0, 1, 1, 1), (1, 1, 1, 1, 1)):
            weights[dims] = None
            continue
        active.append(dims)
        pair_shapes = ((4, 6), (6,), (6, 4), (4,))
        tensors = [torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
                   for shape in pair_shapes]
        weights[dims] = [tensors[:2], tensors[2:]]
        targets.extend([inputs[dims], *tensors])
    masks = torch.randint(0, 2, (2, 3, 5, 2), generator=generator).double()
    flags = dict(pre_norm=pre_norm, post_norm=post_norm, use_bias=use_bias)
    expected, actual = expected_layer(inputs, weights, masks, **flags), actual_layer(inputs, weights, masks, **flags)
    expected_terms, actual_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[dims], rtol=1e-9, atol=1e-9)
        if dims not in active:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn(actual[dims].shape, generator=generator, dtype=torch.float64)
            expected_terms.append((expected[dims] * upstream).sum())
            actual_terms.append((actual[dims] * upstream).sum())
    wanted = torch.autograd.grad(sum(expected_terms), [*targets, *expected_layer.parameters()], allow_unused=True)
    got = torch.autograd.grad(sum(actual_terms), [*targets, *actual_layer.parameters()], allow_unused=True)
    for observed, reference in zip(got, wanted):
        if reference is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, reference, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.usefixtures("triton_shift_runtime")
def test_full_projected_layer_matches_conv2d(post_norm):
    from compressarc.model.layer_factory import LayerFactory
    from compressarc.model.multitensor_systems import MultiTensorSystem, multify

    torch.manual_seed(241)
    system = MultiTensorSystem(2, 2, 3, 5, task=None)
    factory = LayerFactory()
    reference = factory.create_shift("tied_conv", multify=multify).cuda()
    actual_layer = factory.create_shift("triton", multify=multify).cuda()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.2)
    actual_layer.load_state_dict(reference.state_dict(), strict=True)
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
    with torch.backends.cudnn.flags(allow_tf32=False):
        expected = reference(inputs, weights, masks, pre_norm=False, post_norm=post_norm)
    actual = actual_layer(inputs, weights, masks, pre_norm=False, post_norm=post_norm)
    wanted_terms, got_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[dims], rtol=3e-5, atol=5e-6)
        if dims not in active:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn_like(actual[dims])
            wanted_terms.append((expected[dims] * upstream).sum())
            got_terms.append((actual[dims] * upstream).sum())
    with torch.backends.cudnn.flags(allow_tf32=False):
        wanted = torch.autograd.grad(sum(wanted_terms), [*targets, *reference.parameters()])
    got = torch.autograd.grad(sum(got_terms), [*targets, *actual_layer.parameters()])
    for observed, reference_grad in zip(got, wanted):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference_grad, rtol=5e-4, atol=1e-4)


# ============================================================================
# Triton input validation
# ============================================================================

@pytest.mark.usefixtures("triton_shift_runtime")
def test_validation():
    from compressarc.layers.shift.triton_op import DepthwiseConvFunction

    kernel = torch.zeros(8, 1, 3, 3, device="cuda")
    with pytest.raises(ValueError, match="nonempty"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 0, 3, device="cuda"), kernel)
    with pytest.raises(ValueError, match="Kernel"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 3, 5, device="cuda"), kernel[:, :, :2])
    with pytest.raises(ValueError, match="CUDA"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 3, 5, device="cpu"), kernel)
    with pytest.raises(TypeError, match="float32"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 3, 5, device="cuda", dtype=torch.float64), kernel)
