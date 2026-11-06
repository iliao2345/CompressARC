"""Verify the real final-pilot YAML selects tested layers at every model depth."""

from pathlib import Path

import pytest
import torch

from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.layers.shift.layer import TiedShiftLayer
from compressarc.layers.shift.morphological import TiedDirectionalConv
from compressarc.layers.direction_share.morphological import D4DirectionShare, D4DirectionShareLayer
from compressarc.layers.cummax.layer import LSELayer
from compressarc.train import take_step


def test_final_pilot_model_uses_tested_layers_and_options():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("The final pilot selects CUDA Triton LSE")
    from compressarc.layers.cummax.triton_op import TritonMorphologicalMax

    source = Path(__file__).resolve().parents[2] / "config/models/final_pilot_8.yaml"
    config = load_config(source)
    assert (config.model.shift_implementation, config.model.direction_share_implementation,
            config.model.cummax_implementation) == ("tied_conv", "d4", "triton")
    assert config.model.n_layers == 4
    assert config.model.shift_dim == config.model.cummax_dim == 4
    assert config.model.direction_share_dim == 8
    assert config.model.lse_tau == 0.1
    assert config.model.multitensor_constraints == "strict"

    original_device = torch.get_default_device()
    handles = []
    try:
        torch.set_default_device("cuda")
        torch.manual_seed(config.training.seed)
        task = Task("final_pilot_wiring", {
            "train": [{"input": [[0, 1, 0], [1, 0, 1]],
                       "output": [[1, 0, 1], [0, 1, 0]]}],
            "test": [{"input": [[1, 0, 1], [0, 1, 0]]}],
        }, None)
        model = ARCCompressor(task, config.model)
        assert type(model.shift) is TiedShiftLayer
        assert type(model.direction_share) is D4DirectionShareLayer
        assert type(model.cummax) is LSELayer
        assert all(type(core) is TiedDirectionalConv for core in model.shift.models)
        assert all(type(core) is D4DirectionShare for core in model.direction_share.models)
        assert all(type(core) is TritonMorphologicalMax for core in model.cummax.models)
        assert all(core.tau == 0.1 for core in model.cummax.models)
        calls = {name: [] for name in ("shift", "direction_share", "cummax")}
        parameter_ids = [id(value) for value in model.weights_list]
        assert len(parameter_ids) == len(set(parameter_ids))
        for name in calls:
            layer = getattr(model, name)
            assert len(layer.models) == 4
            assert len({id(core) for core in layer.models}) == 4
            for parameter in layer.parameters():
                assert parameter_ids.count(id(parameter)) == 1
                assert parameter.dtype == torch.float32 and parameter.is_cuda
            def capture(module, args, kwargs, name=name):
                calls[name].append(dict(kwargs))
            handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
        for depth in range(4):
            for dims in task.multitensor_system:
                if dims[2]:
                    up, down = model.direction_projection_weights[depth][dims]
                    assert up[0].shape == (8, 8) and down[0].shape == (8, 8)
            assert model.layer_weights[depth]["direction_share"] is model.direction_projection_weights[depth]

        optimizer = torch.optim.Adam(model.weights_list, lr=config.training.learning_rate,
                                     betas=tuple(config.training.optimizer_betas))
        result = take_step(task, model, optimizer, 0, config.training)
        assert torch.isfinite(torch.tensor(result["loss"]))
        for name in calls:
            assert [call["layer_index"] for call in calls[name]] == list(range(4))
            for call in calls[name]:
                assert call["use_bias"] is False
                assert call["pre_norm"] is (name == "direction_share")
                if name != "direction_share":
                    assert call["post_norm"] is True
            for parameter in getattr(model, name).parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    finally:
        for handle in handles:
            handle.remove()
        torch.set_default_device(original_device)
