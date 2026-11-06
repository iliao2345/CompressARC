"""End-to-end original-model parity with frozen main-branch source."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step


@pytest.fixture(scope="module")
def main_modules():
    folder = Path(__file__).with_name("reference_main_model")
    names = ("multitensor_systems", "initializers", "layers", "preprocessing", "arc_compressor", "objective")
    saved = {name: sys.modules.get(name) for name in names}
    original_device, original_dtype = torch.get_default_device(), torch.get_default_dtype()
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"audit_main_{name}", folder / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        torch.set_default_device(original_device)
        torch.set_default_dtype(original_dtype)
    return SimpleNamespace(**loaded)


def build_pair(main, seed, variable_shapes):
    problem = {
        "train": [{"input": [[0, 1, 0], [1, 0, 1]], "output": [[1, 0, 1], [0, 1, 0]]}],
        "test": [{"input": [[1, 0, 1], [0, 1, 0]]}],
    }
    if variable_shapes:
        problem["train"].append({"input": [[1, 0]], "output": [[0], [1]]})
    config = load_config(overrides=["training.device=cpu", f"training.seed={seed}"])
    original_task = main.preprocessing.Task("full_parity", problem, None)
    current_task = Task("full_parity", problem, None)
    np.random.seed(seed)
    torch.manual_seed(seed)
    original = main.arc_compressor.ARCCompressor(original_task)
    np.random.seed(seed)
    torch.manual_seed(seed)
    current = ARCCompressor(current_task, config.model)
    return original_task, current_task, original, current, config


def equal_initial_state(original_task, current_task, original, current):
    assert original_task.colors == current_task.colors
    assert original_task.shapes == current_task.shapes
    torch.testing.assert_close(original_task.problem, current_task.problem, rtol=0, atol=0)
    torch.testing.assert_close(original_task.masks, current_task.masks, rtol=0, atol=0)
    assert len(original.weights_list) == len(current.weights_list)
    for index, (left, right) in enumerate(zip(original.weights_list, current.weights_list)):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=lambda message: f"weight {index}: {message}")


@pytest.fixture(autouse=True)
def cpu_runtime():
    device, dtype, threads = torch.get_default_device(), torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    yield
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("variable_shapes", [False, True])
def test_full_model_initialization_matches_main(main_modules, seed, variable_shapes):
    old_task, new_task, old, new, _ = build_pair(main_modules, seed, variable_shapes)
    equal_initial_state(old_task, new_task, old, new)


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("variable_shapes", [False, True])
def test_full_model_forward_matches_main(main_modules, seed, variable_shapes):
    old_task, new_task, old, new, _ = build_pair(main_modules, seed, variable_shapes)
    equal_initial_state(old_task, new_task, old, new)
    torch.manual_seed(811)
    expected = old.forward()
    torch.manual_seed(811)
    actual = new.forward()
    for label, got, wanted in zip(("color logits", "x mask", "y mask"), actual[:3], expected[:3]):
        torch.testing.assert_close(got, wanted, rtol=1e-5, atol=1e-5, msg=lambda message: f"{label}: {message}")
    assert [tuple(ast.literal_eval(name)) for name in actual[4]] == [tuple(ast.literal_eval(name)) for name in expected[4]]
    for got, wanted in zip(actual[3], expected[3]):
        torch.testing.assert_close(got, wanted, rtol=1e-6, atol=1e-6)


class RecordingAdam(torch.optim.Adam):
    def step(self, closure=None):
        self.captured_gradients = [None if p.grad is None else p.grad.detach().clone()
                                  for group in self.param_groups for p in group["params"]]
        return super().step(closure)


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("variable_shapes", [False, True])
def test_full_training_step_matches_main(main_modules, seed, variable_shapes):
    old_task, new_task, old, new, config = build_pair(main_modules, seed, variable_shapes)
    equal_initial_state(old_task, new_task, old, new)
    old_optimizer = RecordingAdam(old.weights_list, lr=0.01, betas=(0.5, 0.9))
    new_optimizer = RecordingAdam(new.weights_list, lr=0.01, betas=(0.5, 0.9))
    saved_metrics = {}

    def log(step, logits, x_mask, y_mask, kl_amounts, kl_names, total_kl, reconstruction, loss):
        saved_metrics.update(loss=float(loss.detach()), kl=float(total_kl.detach()), reconstruction_error=float(reconstruction.detach()))

    torch.manual_seed(823)
    main_modules.objective.take_step(old_task, old, old_optimizer, 0, SimpleNamespace(log=log))
    torch.manual_seed(823)
    actual = take_step(new_task, new, new_optimizer, 0, config.training)
    for name, wanted in saved_metrics.items():
        assert actual[name] == pytest.approx(wanted, rel=1e-5, abs=1e-5), name
    for index, (got, wanted) in enumerate(zip(new_optimizer.captured_gradients, old_optimizer.captured_gradients)):
        if wanted is None:
            assert got is None
        else:
            assert got is not None and torch.isfinite(got).all()
            torch.testing.assert_close(got, wanted, rtol=1e-4, atol=1e-5, msg=lambda message: f"gradient {index}: {message}")
    for index, (got, wanted) in enumerate(zip(new.weights_list, old.weights_list)):
        torch.testing.assert_close(got, wanted, rtol=1e-5, atol=1e-5, msg=lambda message: f"updated weight {index}: {message}")

def test_locate_first_full_model_divergence(main_modules):
    import compressarc.model.model as current_model_module
    old_task, new_task, old, new, _ = build_pair(main_modules, 0, False)
    equal_initial_state(old_task, new_task, old, new)
    torch.manual_seed(811)
    old_x = main_modules.layers.decode_latents(old.target_capacities, old.decode_weights, old.multiposteriors)[0]
    torch.manual_seed(811)
    new_x = new.latent_decoder(new.target_capacities, new.decode_weights, new.multiposteriors)[0]

    def difference(left, right):
        return max(float((left[dims] - right[dims]).abs().max()) for dims in old_task.multitensor_system)

    latent_difference = difference(old_x, new_x)
    assert latent_difference < 1e-6
    old_x = main_modules.layers.share_up(old_x, old.share_up_weights[0])
    new_x = new.share.share_up(new_x, new.share_up_weights[0])
    share_difference = difference(old_x, new_x)
    assert share_difference < 1e-6
    old_softmax = main_modules.layers.softmax(old_x, old.softmax_weights[0], pre_norm=True, post_norm=False, use_bias=False)
    new_softmax = current_model_module.softmax(new_x, new.softmax_weights[0], pre_norm=True, post_norm=False, use_bias=False)
    difference_with_normalization = difference(old_softmax, new_softmax)
    assert difference_with_normalization > 1e-4
    new_without_normalization = current_model_module.softmax(new_x, new.softmax_weights[0], pre_norm=False, post_norm=False, use_bias=False)
    difference_without_normalization = difference(old_softmax, new_without_normalization)
    assert difference_without_normalization < 2e-6
    print({"latent_max_difference": latent_difference,
           "share_up_max_difference": share_difference,
           "softmax_max_difference": difference_with_normalization,
           "softmax_without_prenorm_difference": difference_without_normalization})


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("variable_shapes", [False, True])
def test_full_model_matches_when_legacy_prenorm_and_latent_math_are_reproduced(main_modules, monkeypatch, seed, variable_shapes):
    from compressarc.layers import helper
    original_apply = helper.apply_residual

    def legacy_prenorm(x, weights, operation, *, pre_norm=False, post_norm=False, use_bias=False):
        # Diagnostic only: legacy add_residual computed normalize(x), then
        # overwrote that result with affine(x), so pre_norm had no effect.
        return original_apply(x, weights, operation, pre_norm=False,
                              post_norm=post_norm, use_bias=use_bias)

    monkeypatch.setattr(helper, "apply_residual", legacy_prenorm)
    from compressarc.layers import latent_decoder
    # Control a mathematically equivalent floating-point ordering difference:
    # main averages (mean**2 + eps), current adds eps after averaging.
    monkeypatch.setattr(latent_decoder, "channel_layer", main_modules.layers.channel_layer)
    test_full_model_forward_matches_main(main_modules, seed, variable_shapes)
    test_full_training_step_matches_main(main_modules, seed, variable_shapes)
