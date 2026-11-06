"""D4 core and projected-layer correctness, gradients, and integration checks."""

from itertools import product

import pytest
import torch

from config import load_config
from preprocessing import Task
from compressarc.layers.direction_share.morphological import (
    D4DirectionShare, compressarc_direction_share,
)
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.model import ARCCompressor
from compressarc.model.multitensor_systems import MultiTensorSystem, multify
from compressarc.train import take_step


# ============================================================================
# Reference helpers
# ============================================================================

def model_with_distinct_orbits():
    model = D4DirectionShare().double()
    with torch.no_grad():
        model.theta.copy_(torch.arange(1, 11, dtype=torch.float64) / 10)
    return model


# ============================================================================
# Correctness
# ============================================================================

# The original primitive reproduces D4 under factorized pair weights.
# Compare outputs and gradients in the shared parameter coordinates.

@pytest.mark.parametrize("dims", list(product([0, 1], repeat=5)))
@pytest.mark.parametrize("pre_norm", [False, True])
def test_projected_d4_matches_primitive_special_case_outputs_and_gradients(dims, pre_norm):
    """The primitive contains projected D4 when pair matrices share its factors.

    Compare gradients in the common (x, W_up, W_down, theta) coordinates,
    rather than comparing independent primitive pair parameters with theta.
    """
    from compressarc.layers.direction_share.primitives import DirectionSharePrimitives
    from compressarc.layers.direction_share.morphological import D4DirectionShareLayer

    # Bind one layout so even layouts excluded by the task system are exercised.
    def bind_dims(operation):
        def apply(*args, **kwargs):
            return operation(dims, *args, **kwargs)
        return apply

    generator = torch.Generator().manual_seed(83)
    shape = [length for flag, length in zip(dims, [2, 3, 8, 4, 5]) if flag] + [3]
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    up = torch.randn(3, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    down = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    projected = D4DirectionShareLayer(bind_dims).double()
    with torch.no_grad():
        projected.models[0].theta.copy_(
            torch.linspace(-0.7, 0.9, 10, dtype=torch.float64)
        )
    primitive = DirectionSharePrimitives(bind_dims)
    theta = projected.models[0].theta
    projections = [[up, None], [down, None]] if dims[2] else None
    pair_weights = None
    if dims[2]:
        coefficients = (1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2)
        mixing = projected.models[0].weight()
        channel_matrix = up @ down
        pair_weights = [
            [
                [mixing[d_out, d_in] * channel_matrix
                 / coefficients[(d_in - d_out) % 8], None]
                for d_in in range(8)
            ]
            for d_out in range(8)
        ]

    actual = projected(x, projections, pre_norm=pre_norm, use_bias=False)
    expected = primitive(x, pair_weights, pre_norm=pre_norm, use_bias=False)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    if not dims[2]:
        assert actual is x and expected is x

    upstream = torch.randn(shape, generator=generator, dtype=torch.float64)
    targets = (x, up, down, theta)
    actual_gradients = torch.autograd.grad((actual * upstream).sum(), targets, allow_unused=True)
    expected_gradients = torch.autograd.grad((expected * upstream).sum(), targets, allow_unused=True)
    for index, (observed, reference) in enumerate(zip(actual_gradients, expected_gradients)):
        if not dims[2] and index > 0:
            assert observed is None and reference is None
        else:
            assert observed is not None and reference is not None
            assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
            torch.testing.assert_close(observed, reference, rtol=1e-9, atol=1e-9)


# ============================================================================
# Equivariance
# ============================================================================

@pytest.mark.parametrize('rotation,reflection', list(product(range(4), [False, True])))
def test_d4_direction_share_equivariance(rotation, reflection):
    model = model_with_distinct_orbits()
    x = torch.arange(2 * 8 * 3 * 5, dtype=torch.float64).reshape(2, 8, 3, 5)
    directions = torch.arange(8)
    permutation = ((-directions if reflection else directions) + 2 * rotation) % 8
    actual = model(x[:, permutation])
    expected = model(x)[:, permutation]
    torch.testing.assert_close(actual, expected)


# ============================================================================
# Group weight correctness
# ============================================================================

@pytest.mark.parametrize('input_direction', range(8))
def test_d4_direction_share_orbit_weights(input_direction):
    model = model_with_distinct_orbits()
    x = torch.zeros(1, 8, 2, 3, dtype=torch.float64)
    x[:, input_direction] = 1
    expected = torch.zeros_like(x)
    for output_direction in range(8):
        distance = min((input_direction - output_direction) % 8,
                       (output_direction - input_direction) % 8)
        if output_direction % 2 == input_direction % 2:
            orbit = (3 if output_direction % 2 else 0) + distance // 2
        else:
            orbit = (8 if output_direction % 2 else 6) + (distance - 1) // 2
        expected[:, output_direction] = (orbit + 1) / 10
    torch.testing.assert_close(model(x), expected)


# ============================================================================
# Outputs and gradients
# ============================================================================

def test_d4_direction_share_parameter_gradients():
    model = model_with_distinct_orbits()
    x = torch.ones(2, 8, 3, 5, dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert model.theta.numel() == 10
    assert model.theta.grad.shape == (10,)
    assert torch.isfinite(model.theta.grad).all()
    assert (model.theta.grad > 0).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("pre_norm", [False, True])
def test_projected_d4_matches_pilot_equation_and_gradients(pre_norm):
    generator = torch.Generator().manual_seed(53)
    system = MultiTensorSystem(2, 2, 2, 3, task=None)
    layer = LayerFactory().create_direction_share("d4", multify=multify, n_layers=2).double()
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    targets, directional_dims = [], []
    expected = {}
    matrix = layer.models[1].weight()
    for dims in system:
        x = torch.randn(system.shape(dims, extra_dim=3), generator=generator,
                        dtype=torch.float64, requires_grad=True)
        inputs[dims] = x
        if not dims[2]:
            weights[dims] = None
            expected[tuple(dims)] = x
            continue
        directional_dims.append(dims)
        w1 = torch.randn(3, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        w2 = torch.randn(8, 3, generator=generator, dtype=torch.float64, requires_grad=True)
        weights[dims] = [[w1, None], [w2, None]]
        targets.extend([x, w1, w2])
        z = x
        if pre_norm:
            axes = tuple(range(x.ndim - 1))
            centered = x - x.mean(dim=axes)
            z = centered / torch.sqrt(1e-8 + centered.square().mean(dim=axes))
        projected = z @ w1
        axis = sum(dims[:2])
        mixed = torch.einsum("ij,j...->i...", matrix, projected.movedim(axis, 0)).movedim(0, axis)
        expected[tuple(dims)] = x + mixed @ w2
    actual = layer(inputs, weights, pre_norm=pre_norm, layer_index=1)
    actual_terms, expected_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[tuple(dims)], rtol=1e-10, atol=1e-10)
        if not dims[2]:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn(actual[dims].shape, generator=generator, dtype=torch.float64)
            actual_terms.append((actual[dims] * upstream).sum())
            expected_terms.append((expected[tuple(dims)] * upstream).sum())
    targets.append(layer.models[1].theta)
    wanted = torch.autograd.grad(sum(expected_terms), targets)
    got = torch.autograd.grad(sum(actual_terms), [*targets, layer.models[0].theta], allow_unused=True)
    assert got[-1] is None
    for observed, reference in zip(got[:-1], wanted):
        assert observed is not None and torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=1e-9, atol=1e-9)
    assert layer.models[0].theta is not layer.models[1].theta


# ============================================================================
# Adapter logic and integration
# ============================================================================

@pytest.mark.parametrize('dims', list(product([0, 1], repeat=5)))
def test_d4_adapter_axes_residual_and_passthrough(dims):
    model = model_with_distinct_orbits()
    lengths = [2, 3, 8, 4, 5]
    shape = [length for flag, length in zip(dims, lengths) if flag] + [2]
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape) / 100
    actual = compressarc_direction_share(dims, x, model)
    if not dims[2]:
        assert actual is x
        return
    axis = sum(dims[:2])
    directions_first = x.movedim(axis, 0)
    # Contract over the direction dimension directly; no adapter axis manipulation.
    mixed = torch.tensordot(model.weight(), directions_first, dims=([1], [0]))
    expected = x + mixed.movedim(0, axis)
    torch.testing.assert_close(actual, expected)



def test_factory_d4_multitensor_wrapper_and_optimizer():
    from compressarc.model.layer_factory import LayerFactory
    from compressarc.model.multitensor_systems import MultiTensorSystem, multify
    from compressarc.layers.helper import normalize

    system = MultiTensorSystem(2, 2, 3, 4, task=None)
    inputs = system.make_multitensor()
    for dims in system:
        inputs[dims] = torch.randn(system.shape(dims, extra_dim=2), dtype=torch.float64)
    weights = system.make_multitensor()
    for dims in system:
        weights[dims] = (
            [[torch.eye(2, dtype=torch.float64), None],
             [torch.eye(2, dtype=torch.float64), None]] if dims[2] else None
        )
    layer = LayerFactory().create_direction_share('d4', multify=multify, n_layers=2).double()
    assert layer.models[0].theta is not layer.models[1].theta
    for index in range(2):
        actual = layer(inputs, weights, pre_norm=True, use_bias=False, layer_index=index)
        for dims in system:
            if not dims[2]:
                assert actual[dims] is inputs[dims]
            else:
                z = normalize(inputs[dims])
                mixed = compressarc_direction_share(dims, z, layer.models[index]) - z
                torch.testing.assert_close(actual[dims], inputs[dims] + mixed)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = [m.theta.detach().clone() for m in layer.models]
    loss = sum(layer(inputs, weights, layer_index=i, pre_norm=False)[[1, 0, 1, 1, 1]].square().sum()
               for i in range(2))
    loss.backward()
    optimizer.step()
    assert all(not torch.equal(old, m.theta) for old, m in zip(before, layer.models))


def test_projected_d4_model_registers_and_updates_projections_and_mixer():
    config = load_config(overrides=[
        "model.direction_share_implementation=d4", "model.n_layers=1",
        "model.channel_dim_shared=2", "model.channel_dim_grid=2",
        "model.share_up_dim=2", "model.share_down_dim=2", "model.decoding_dim=2",
        "model.softmax_dim=2", "model.cummax_dim=2", "model.shift_dim=2", "model.nonlinear_dim=2",
    ])
    task = Task("projection_test", {
        "train": [{"input": [[0, 1], [1, 0]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 0], [0, 1]]}],
    }, None)
    model = ARCCompressor(task, config.model)
    dims = (1, 0, 1, 1, 1)
    projection = model.direction_projection_weights[0]
    assert model.layer_weights[0]["direction_share"] is projection
    w1, w2 = projection[dims][0][0], projection[dims][1][0]
    theta = model.direction_share.models[0].theta
    assert w1.shape == (2, 8) and w2.shape == (8, 2)
    registered = {id(weight) for weight in model.weights_list}
    assert len(registered) == len(model.weights_list)
    assert {id(w1), id(w2), id(theta)} <= registered
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01)
    metrics = take_step(task, model, optimizer, 0, config.training)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for value in (w1, w2, theta):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    # A task can give a component zero gradient. Use a controlled layer loss
    # to verify the registered projection matrices and mixer really update.
    generator = torch.Generator().manual_seed(71)
    inputs = task.multitensor_system.make_multitensor()
    for component in task.multitensor_system:
        inputs[component] = torch.randn(
            task.multitensor_system.shape(component, extra_dim=model.channel_dim_fn(component)),
            generator=generator,
        )
    optimizer.zero_grad()
    before = [value.detach().clone() for value in (w1, w2, theta)]
    output = model.direction_share(inputs, projection, pre_norm=True, layer_index=0)
    output[dims].square().sum().backward()
    optimizer.step()
    for old, value in zip(before, (w1, w2, theta)):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
        assert not torch.equal(old, value)
