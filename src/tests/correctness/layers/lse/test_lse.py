"""Direction checks and regression checks for both LSE implementations."""

import pytest
import torch

from compressarc.layers.cummax.optimisation import MorphologicalMax
from compressarc.layers.cummax.morphological import MorphologicalMax as MorphologicalMaxPreoptimized
from compressarc.layers.cummax.layer import LSELayer
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.multitensor_systems import MultiTensorSystem, multify

SHAPES = [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3), (4, 4)]
DIRECTIONS = [(0, 1), (1, 1), (-1, 0), (-1, 1),
              (0, -1), (-1, -1), (1, 0), (1, -1)]


# ============================================================================
# Reference helpers
# ============================================================================

def reference_direction(x, kernel, tau, delta):
    # Explicit predecessor coordinates, independent of rotations and strided views.
    dr, dc = delta
    result = torch.empty_like(x)
    for row in range(x.shape[-2]):
        for col in range(x.shape[-1]):
            scores = []
            r, c, offset = row, col, 0
            while 0 <= r < x.shape[-2] and 0 <= c < x.shape[-1]:
                scores.append(x[:, r, c] + kernel[offset])
                r, c, offset = r - dr, c - dc, offset + 1
            result[:, row, col] = tau * torch.logsumexp(torch.stack(scores, -1) / tau, -1)
    return result


def initialize(model):
    with torch.no_grad():
        model.kernel.copy_(torch.linspace(-0.2, 0.4, model.kernel.numel()))
        model.diagonal_kernel.copy_(torch.linspace(0.3, -0.1, model.diagonal_kernel.numel()))
    return model


def reference_lse_adapter(dims, x, weights, masks, model):
    """Build one projected/masked layer output without using the LSE adapter."""
    valid = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
    for _ in range(sum(dims[1:3])):
        valid = valid[:, None]
    valid = valid[..., None]
    projected = x @ weights[0][0] + weights[0][1]
    masked = projected * valid
    direction_axis = sum(dims[:2])
    mappings = ((6, 7, 0, 1, 2, 3, 4, 5), (2, 3, 4, 5, 6, 7, 0, 1))
    halves = []
    for half, mapping in enumerate(mappings):
        outputs = []
        for direction, output_direction in enumerate(mapping):
            grid = torch.select(masked, direction_axis, direction)[..., half::2]
            moved = grid.movedim(-1, -3)
            shape = moved.shape
            kernel = model.kernel if output_direction % 2 == 0 else model.diagonal_kernel
            transformed = reference_direction(
                moved.reshape(-1, shape[-2], shape[-1]),
                kernel,
                model.tau,
                DIRECTIONS[output_direction],
            )
            outputs.append(transformed.reshape(shape).movedim(-3, -1))
        halves.append(torch.stack(outputs, dim=direction_axis))
    communicated = torch.cat(halves, dim=-1) * valid
    return x + communicated @ weights[1][0] + weights[1][1]


# ============================================================================
# Correctness
# ============================================================================
# LSE outputs are checked against the coordinate-based LSE reference.
# Hard cummax primitives compute a different quantity.

@pytest.mark.parametrize('implementation', [MorphologicalMaxPreoptimized, MorphologicalMax],
                         ids=['preoptimized', 'current'])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('tau', [0.1, 0.7])
def test_lse_all_directions(implementation, shape, tau):
    height, width = shape
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(2, height, width, generator=generator, dtype=torch.float64)
    model = initialize(implementation(height, width, tau=tau).double())
    actual = model(x)
    assert actual.shape == (2, 8, height, width)
    for direction, delta in enumerate(DIRECTIONS):
        kernel = model.kernel if direction % 2 == 0 else model.diagonal_kernel
        expected = reference_direction(x, kernel, tau, delta)
        torch.testing.assert_close(actual[:, direction], expected)
        torch.testing.assert_close(model.direction(x, direction), expected)


# ============================================================================
# Equivariance
# ============================================================================

@pytest.mark.parametrize("implementation", [MorphologicalMaxPreoptimized, MorphologicalMax],
                         ids=["preoptimized", "optimized"])
@pytest.mark.parametrize("rotation", range(4))
def test_lse_class_is_equivariant_to_quarter_turns(implementation, rotation):
    """Rotating the grid rotates both each direction's map and its direction slot."""
    height, width, tau = 3, 5, 0.3
    generator = torch.Generator().manual_seed(211)
    x = torch.randn(2, height, width, generator=generator, dtype=torch.float64)
    model = initialize(implementation(height, width, tau=tau).double())
    original = model(x)
    rotated_input = torch.rot90(x, rotation, (-2, -1))
    rotated_output = model(rotated_input)

    for direction, (dr, dc) in enumerate(DIRECTIONS):
        for _ in range(rotation):
            dr, dc = -dc, dr
        rotated_direction = DIRECTIONS.index((dr, dc))
        torch.testing.assert_close(
            rotated_output[:, rotated_direction],
            torch.rot90(original[:, direction], rotation, (-2, -1)),
            rtol=1e-10,
            atol=1e-10,
        )


# ============================================================================
# Group weight correctness
# ============================================================================

@pytest.mark.parametrize("implementation", [MorphologicalMaxPreoptimized, MorphologicalMax],
                         ids=["preoptimized", "optimized"])
@pytest.mark.parametrize("kernel_name,parity", [("kernel", 0), ("diagonal_kernel", 1)])
def test_lse_directions_share_their_group_kernel(implementation, kernel_name, parity):
    """Changing one shared kernel shifts all four outputs in its direction group."""
    generator = torch.Generator().manual_seed(313)
    x = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    model = initialize(implementation(3, 5, tau=0.3).double())
    parameters = dict(model.named_parameters())
    assert set(parameters) == {"kernel", "diagonal_kernel"}
    assert parameters["kernel"] is model.kernel
    assert parameters["diagonal_kernel"] is model.diagonal_kernel
    assert model.kernel is not model.diagonal_kernel
    before = model(x).detach()
    # LSE(scores + offset) = LSE(scores) + offset, for every causal window.
    with torch.no_grad():
        parameters[kernel_name].add_(0.25)
    after = model(x).detach()
    for direction in range(8):
        expected = before[:, direction] + (0.25 if direction % 2 == parity else 0.)
        torch.testing.assert_close(after[:, direction], expected, rtol=1e-10, atol=1e-10)
    assert getattr(model, kernel_name) is parameters[kernel_name]


@pytest.mark.parametrize("implementation", [MorphologicalMaxPreoptimized, MorphologicalMax],
                         ids=["preoptimized", "optimized"])
def test_lse_direction_gradients_accumulate_into_shared_kernels(implementation):
    """Each direction connects to its group kernel, and their gradients add up."""
    generator = torch.Generator().manual_seed(317)
    x = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    model = initialize(implementation(3, 5, tau=0.3).double())
    parameters = (model.kernel, model.diagonal_kernel)
    group_gradients = [[], []]
    for direction, delta in enumerate(DIRECTIONS):
        group = direction % 2
        output = model.direction(x, direction)
        grads = torch.autograd.grad(output.sum(), parameters, allow_unused=True)
        assert grads[group] is not None
        assert grads[1 - group] is None
        assert torch.isfinite(grads[group]).all()
        assert torch.count_nonzero(grads[group]) > 0
        reference = reference_direction(x, parameters[group], model.tau, delta)
        expected_grad, = torch.autograd.grad(reference.sum(), parameters[group])
        torch.testing.assert_close(grads[group], expected_grad, rtol=1e-10, atol=1e-10)
        group_gradients[group].append(grads[group])
    # One forward over all eight directions must accumulate all contributions.
    model.zero_grad(set_to_none=True)
    model(x).sum().backward()
    for parameter, contributions in zip(parameters, group_gradients):
        assert len(contributions) == 4
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, torch.stack(contributions).sum(dim=0),
            rtol=1e-10, atol=1e-10,
        )


@pytest.mark.parametrize("implementation", [MorphologicalMaxPreoptimized, MorphologicalMax],
                         ids=["preoptimized", "optimized"])
def test_lse_optimizer_preserves_shared_group_parameters(implementation):
    """SGD updates the two shared parameters used by all eight directions."""
    generator = torch.Generator().manual_seed(331)
    x = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    model = initialize(implementation(3, 5, tau=0.3).double())
    parameters = (model.kernel, model.diagonal_kernel)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(x).mean().backward()
    gradients = [parameter.grad.detach().clone() for parameter in parameters]
    optimizer.step()
    assert model.kernel is parameters[0]
    assert model.diagonal_kernel is parameters[1]
    assert list(model.parameters()) == list(parameters)
    for parameter, original, grad in zip(parameters, before, gradients):
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0
        torch.testing.assert_close(parameter, original - 0.01 * grad)
        assert not torch.equal(parameter, original)
    actual = model(x)
    for direction, delta in enumerate(DIRECTIONS):
        kernel = parameters[direction % 2]
        expected = reference_direction(x, kernel, model.tau, delta)
        torch.testing.assert_close(actual[:, direction], expected, rtol=1e-10, atol=1e-10)


# ============================================================================
# Outputs and gradients
# ============================================================================

@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('rotation', range(4))
@pytest.mark.parametrize('sliced', [False, True], ids=['offset', 'strided'])
def test_morphological_max_core_matches(shape, rotation, sliced):
    height, width = shape
    generator = torch.Generator().manual_seed(42)
    base = torch.randn(3, 2 * height + 2, 2 * width + 2,
                       generator=generator, dtype=torch.float64, requires_grad=True)
    if sliced:
        x = base[1:, 1:1 + 2 * height:2, 1:1 + 2 * width:2]
    else:
        x = base[1:, 1:1 + height, 1:1 + width]
    x = torch.rot90(x, rotation, (-2, -1))
    reference = initialize(MorphologicalMaxPreoptimized(height, width, tau=0.3).double())
    current = MorphologicalMax(height, width, tau=0.3).double()
    current.load_state_dict(reference.state_dict(), strict=True)
    expected, actual = reference(x), current(x)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    expected_grad = torch.autograd.grad((expected * upstream).sum(),
                                       (base, reference.kernel, reference.diagonal_kernel))
    actual_grad = torch.autograd.grad((actual * upstream).sum(),
                                     (base, current.kernel, current.diagonal_kernel))
    for got, want in zip(actual_grad, expected_grad):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want)


# ============================================================================
# Adapter logic and integration
# ============================================================================

@pytest.mark.parametrize("variant,core_type", [
    pytest.param("d4", MorphologicalMaxPreoptimized, id="d4"),
    pytest.param("optimised", MorphologicalMax, id="optimised"),
])
def test_layer_correctness_on_outputs_and_gradients(variant, core_type):
    """Compare complete projected LSE layers, including masked adapter and residual paths."""
    torch.manual_seed(107)
    height, width, features = 3, 5, 4
    system = MultiTensorSystem(2, 3, height, width, task=None)
    factory = LayerFactory()
    reference_layer = factory.create_cummax(
        "d4", multify=multify, height=height, width=width, tau=0.3,
    ).double()
    current_layer = factory.create_cummax(
        variant, multify=multify, height=height, width=width, tau=0.3,
    ).double()
    assert isinstance(current_layer, LSELayer)
    assert type(current_layer.models[0]) is core_type
    initialize(reference_layer.models[0])
    current_layer.load_state_dict(reference_layer.state_dict(), strict=True)

    reference_x, current_x = system.make_multitensor(), system.make_multitensor()
    reference_weights, current_weights = system.make_multitensor(), system.make_multitensor()
    active_dims = []
    reference_inputs, current_inputs = [], []
    reference_weight_tensors, current_weight_tensors = [], []
    for dims in system:
        value = torch.randn(*system.shape(dims, extra_dim=features), dtype=torch.float64)
        reference_x[dims] = value.clone().requires_grad_()
        current_x[dims] = value.clone().requires_grad_()
        if tuple(dims) not in (
            (1, 1, 1, 1, 1),
            (1, 0, 1, 1, 1),
        ):
            reference_weights[dims] = current_weights[dims] = None
            continue
        active_dims.append(dims)
        reference_inputs.append(reference_x[dims])
        current_inputs.append(current_x[dims])
        components = [
            [torch.randn(features, features, dtype=torch.float64), torch.randn(features, dtype=torch.float64)],
            [torch.randn(features, features, dtype=torch.float64), torch.randn(features, dtype=torch.float64)],
        ]
        reference_weights[dims] = [
            [tensor.clone().requires_grad_() for tensor in pair] for pair in components
        ]
        current_weights[dims] = [
            [tensor.clone().requires_grad_() for tensor in pair] for pair in components
        ]
        reference_weight_tensors.extend(tensor for pair in reference_weights[dims] for tensor in pair)
        current_weight_tensors.extend(tensor for pair in current_weights[dims] for tensor in pair)

    masks = torch.randint(0, 2, (2, height, width, 2)).double()
    masks[0, 1, 2] = 0
    masks[1, 0, 1, 0] = 0
    masks[1, 2, 3, 1] = 0
    preoptimized_outputs = reference_layer(
        reference_x, reference_weights, masks, use_bias=True
    )
    actual = current_layer(current_x, current_weights, masks, use_bias=True)
    expected = system.make_multitensor()
    for dims in system:
        if dims in active_dims:
            expected[dims] = reference_lse_adapter(
                dims, reference_x[dims], reference_weights[dims], masks,
                reference_layer.models[0],
            )
        else:
            expected[dims] = reference_x[dims]
        torch.testing.assert_close(preoptimized_outputs[dims], expected[dims], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(actual[dims], expected[dims], rtol=1e-10, atol=1e-10)
        if dims not in active_dims:
            assert actual[dims] is current_x[dims]

    reference_terms, preoptimized_terms, current_terms = [], [], []
    for dims in active_dims:
        grad = torch.randn_like(expected[dims])
        reference_terms.append((expected[dims] * grad).sum())
        preoptimized_terms.append((preoptimized_outputs[dims] * grad).sum())
        current_terms.append((actual[dims] * grad).sum())
    expected_grads = torch.autograd.grad(
        sum(reference_terms),
        [*reference_inputs, *reference_weight_tensors, *reference_layer.parameters()],
    )
    preoptimized_grads = torch.autograd.grad(
        sum(preoptimized_terms),
        [*reference_inputs, *reference_weight_tensors, *reference_layer.parameters()],
    )
    actual_grads = torch.autograd.grad(
        sum(current_terms),
        [*current_inputs, *current_weight_tensors, *current_layer.parameters()],
    )
    for got, want, preoptimized in zip(actual_grads, expected_grads, preoptimized_grads):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(preoptimized, want, rtol=1e-9, atol=1e-9)
    for kernel_grad in actual_grads[-2:]:
        assert torch.count_nonzero(kernel_grad) > 0


# ============================================================================
# Timing extensions
# ============================================================================

@pytest.mark.parametrize('implementation', [MorphologicalMaxPreoptimized, MorphologicalMax])
def test_timing_disabled_by_default(implementation, monkeypatch):
    from compressarc.layers import helper

    def unexpected_timer():
        raise AssertionError('Timing disabled: clock must not be read')

    monkeypatch.setattr(helper.time, 'perf_counter', unexpected_timer)
    model = implementation(3, 4)
    model(torch.zeros(1, 3, 4))
    assert model.timings == {}


@pytest.mark.parametrize('implementation', [MorphologicalMaxPreoptimized, MorphologicalMax])
def test_timing_flag_preserves_outputs_and_records_calls(implementation):
    model = implementation(3, 4, timing=True)
    plain = implementation(3, 4)
    plain.load_state_dict(model.state_dict())
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    torch.testing.assert_close(model(x), plain(x))
    for name in ['phi', 'diagonal_phi']:
        assert model.timings[f'timing/{name}_calls'] == 4
        assert model.timings[f'timing/{name}_forward_seconds'] >= 0
    model.timings.clear()
    model.timing = False
    model(x)
    assert model.timings == {}
