"""Numerical parity of DirectionSharePrimitives against original main code."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest
import torch

from compressarc.layers.direction_share.primitives import DirectionSharePrimitives
from compressarc.layers.direction_share.vectorized import DirectionShareVectorized

_spec = importlib.util.spec_from_file_location(
    "direction_share_main_reference", Path(__file__).with_name("reference_direction_share_main.py"),
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


@pytest.mark.parametrize("backend", [DirectionSharePrimitives, DirectionShareVectorized])
@pytest.mark.parametrize("dims", list(product([0, 1], repeat=5)))
@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("share_pairs", [False, True], ids=["independent-weights", "shared-weights"])
def test_direction_share_primitives_match_main_outputs_and_gradients(
    backend, dims, pre_norm, use_bias, share_pairs,
):
    def multify(operation):
        def apply(*args, **kwargs):
            return operation(dims, *args, **kwargs)
        return apply

    generator = torch.Generator().manual_seed(607)
    shape = [size for flag, size in zip(dims, [2, 3, 8, 3, 5]) if flag] + [4]
    values = torch.randn(shape, generator=generator, dtype=torch.float64)
    current_x = values.clone().requires_grad_()
    main_x = values.clone().requires_grad_()
    current_parameters, main_parameters = [], []
    current_weights = main_weights = None
    if dims[2]:
        # Reusing parameter objects checks accumulation from multiple pairs,
        # rather than treating every direction pair as an independent leaf.
        count = 8 if share_pairs else 64
        current_pool, main_pool = [], []
        for _ in range(count):
            pair_values = [torch.randn(size, generator=generator, dtype=torch.float64) * 0.2
                           for size in ((4, 4), (4,))]
            current_pair = [value.clone().requires_grad_() for value in pair_values]
            main_pair = [value.clone().requires_grad_() for value in pair_values]
            current_pool.append(current_pair)
            main_pool.append(main_pair)
            current_parameters.extend(current_pair)
            main_parameters.extend(main_pair)
        indices = [[(out_direction * 8 + in_direction) % count
                    for in_direction in range(8)] for out_direction in range(8)]
        current_weights = [[current_pool[index] for index in row] for row in indices]
        main_weights = [[main_pool[index] for index in row] for row in indices]

    layer = backend(multify)
    actual = layer(current_x, current_weights, pre_norm=pre_norm, use_bias=use_bias)
    if dims[2]:
        expected = reference.direction_share(dims, main_x, main_weights,
                                             pre_norm=pre_norm, use_bias=use_bias)
    else:
        # Original wrapper skips components without a direction axis.
        expected = main_x
        assert actual is current_x
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn(shape, generator=generator, dtype=torch.float64)
    got = torch.autograd.grad((actual * upstream).sum(), [current_x, *current_parameters], allow_unused=True)
    wanted = torch.autograd.grad((expected * upstream).sum(), [main_x, *main_parameters], allow_unused=True)
    for observed, target in zip(got, wanted):
        if target is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, target, rtol=1e-10, atol=1e-10)
