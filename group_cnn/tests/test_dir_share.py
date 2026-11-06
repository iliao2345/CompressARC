import torch
from src import D4DirectionShare, compressarc_direction_share


from src.native_reference import reference


def test_original_direction_share_tying():
    model = D4DirectionShare()

    orbit_id = model.orbit_id
    W_old = model.weight()

    for orbit in range(10):
        values = W_old[orbit_id == orbit]

        assert torch.allclose(
            values,
            values[0].expand_as(values),
            atol=1e-6,
        )


def _reference_weights(system, model, feature_count):
    """Express the D4 scalar map through CompressARC's affine pair weights."""
    weights = system.make_multitensor()
    coefficients = (1.0, 0.2, 0.4, 0.2, 1.0, 0.2, 0.4, 0.2)
    identity = torch.eye(feature_count, dtype=model.theta.dtype)
    matrix = model.weight()
    for dims in system:
        if dims[2]:
            weights[dims] = [
                [
                    [matrix[d_out, d_in] / coefficients[(d_in - d_out) % 8] * identity,
                     torch.zeros(feature_count, dtype=model.theta.dtype)]
                    for d_in in range(8)
                ]
                for d_out in range(8)
            ]
    return weights


def test_matches_compressarc_direction_share_outputs_and_gradients():
    generator = torch.Generator().manual_seed(42)
    system = reference.multitensor_systems.MultiTensorSystem(2, 3, 3, 5, task=None)
    feature_count = 4
    expected_x = system.make_multitensor()
    actual_x = system.make_multitensor()
    expected_model = D4DirectionShare().double()
    actual_model = D4DirectionShare().double()
    with torch.no_grad():
        expected_model.theta.copy_(torch.randn(10, generator=generator, dtype=torch.float64))
        actual_model.theta.copy_(expected_model.theta)

    expected_inputs = []
    actual_inputs = []
    for dims in system:
        value = torch.randn(
            *system.shape(dims, extra_dim=feature_count),
            generator=generator,
            dtype=torch.float64,
        )
        expected_x[dims] = value.clone().requires_grad_(True)
        actual_x[dims] = value.clone().requires_grad_(True)
        if dims[2]:
            expected_inputs.append(expected_x[dims])
            actual_inputs.append(actual_x[dims])

    expected_weights = _reference_weights(system, expected_model, feature_count)
    expected = reference.direction_share(
        expected_x, expected_weights, pre_norm=False, use_bias=False
    )
    actual = system.make_multitensor()
    for dims in system:
        actual[dims] = compressarc_direction_share(dims, actual_x[dims], actual_model)
        torch.testing.assert_close(actual[dims], expected[dims], rtol=1e-10, atol=1e-10)
        if not dims[2]:
            assert actual[dims] is actual_x[dims]

    direction_dims = [dims for dims in system if dims[2]]
    expected_loss = sum(expected[dims].square().sum() for dims in direction_dims)
    actual_loss = sum(actual[dims].square().sum() for dims in direction_dims)
    expected_grads = torch.autograd.grad(
        expected_loss, [*expected_inputs, expected_model.theta]
    )
    actual_grads = torch.autograd.grad(
        actual_loss, [*actual_inputs, actual_model.theta]
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)
