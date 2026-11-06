"""Versioned final-model snapshots for inference, including selection and RNG state."""
from dataclasses import dataclass
from pathlib import Path
import platform
import random

import numpy as np
from omegaconf import OmegaConf
import torch

SELECTION_SETTINGS = ("ema_decay", "uncertainty_weight", "early_step_penalty",
                      "early_step_cutoff", "ema_penalty")
LAYERS = ("shift", "cummax", "direction_share")


def save_inference_checkpoint(path, task, model, tracker, config, completed_steps, metrics):
    device = model.weights_list[0].device
    numpy_rng = np.random.get_state()
    rng = {"torch_cpu": torch.random.get_rng_state(),
           "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
           "python": random.getstate(),
           "numpy": {"algorithm": numpy_rng[0], "keys": numpy_rng[1].tolist(),
                     "position": numpy_rng[2], "has_gauss": numpy_rng[3], "cached_gaussian": numpy_rng[4]}}
    selection = {
        "settings": {name: getattr(tracker, name) for name in SELECTION_SETTINGS},
        "ema_logits": tracker.ema_logits.detach().cpu().clone(),
        "ema_x_mask": tracker.ema_x_mask.detach().cpu().clone(),
        "ema_y_mask": tracker.ema_y_mask.detach().cpu().clone(),
        "candidates": [{"grids": grid, "record": dict(record), "score": tracker.scores[hash(grid)]}
                       for grid, record in tracker.candidate_records.items()],
        "top_1": tracker.solution_most_frequent,
        "top_2": tracker.solution_second_most_frequent,
        "first_correct_generation_step": tracker.first_correct_generation_step,
        "promotion_rule": "greater_or_equal_replaces_incumbent",
        "ranking_tie_rule": "first_appearance",
        "decode_rule": "argmax_colour_and_best_mask_crop",
    }
    modules = {}
    for name in LAYERS:
        layer = getattr(model, name, None)
        if isinstance(layer, torch.nn.Module):
            modules[name] = {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
    payload = {
        "schema_version": 1, "purpose": "inference", "task": task.task_name,
        "split": config.training.split, "task_problem": task.unprocessed_problem,
        "task_metadata": {"colors": task.colors, "shapes": task.shapes,
                          "multitensor_constraints": task.multitensor_constraints},
        "config": {"model": OmegaConf.to_container(config.model, resolve=True),
                   "training": {"iterations": completed_steps, "seed": config.training.seed,
                                "split": config.training.split}},
        "weights": [weight.detach().cpu().clone() for weight in model.weights_list],
        "module_states": modules, "selection": selection, "rng": rng,
        "dtype": str(model.weights_list[0].dtype).split(".")[-1],
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": str(np.__version__), "training_device": str(device)},
        "completed_steps": completed_steps,
        "prediction_phase": "selected_guesses_from_pre_final_optimizer_forward",
        "weight_phase": "post_final_optimizer_update",
        "metrics": {key: value for key, value in metrics.items() if key not in ("outputs", "kl_components")},
    }
    torch.save(payload, Path(path))


@dataclass
class InferenceCheckpoint:
    task: object
    model: object
    tracker: object
    config: object
    completed_steps: int


def load_inference_checkpoint(path, device="cpu", restore_rng=True):
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from .metrics import SolutionTracker
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported inference checkpoint schema")
    target = torch.device(device)
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    try:
        torch.set_default_dtype(getattr(torch, payload["dtype"]))
        torch.set_default_device(target)
        with torch.device(target):
            task = Task(payload["task"], payload["task_problem"], None,
                        multitensor_constraints=payload["task_metadata"]["multitensor_constraints"])
            config = OmegaConf.create(payload["config"])
            if task.colors != payload["task_metadata"]["colors"] or task.shapes != payload["task_metadata"]["shapes"]:
                raise ValueError("Checkpoint task preprocessing differs from saved colors/shapes")
            model = ARCCompressor(task, config.model)
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)
    if len(model.weights_list) != len(payload["weights"]):
        raise ValueError("Checkpoint parameter count does not match reconstructed model")
    with torch.no_grad():
        for parameter, saved in zip(model.weights_list, payload["weights"]):
            if parameter.shape != saved.shape:
                raise ValueError("Checkpoint parameter shape mismatch")
            parameter.copy_(saved.to(device=target, dtype=parameter.dtype))
    for name, state in payload["module_states"].items():
        getattr(model, name).load_state_dict(state, strict=True)
    tracker = SolutionTracker(task)
    selection = payload["selection"]
    for name, value in selection["settings"].items():
        setattr(tracker, name, value)
    for name in ("ema_logits", "ema_x_mask", "ema_y_mask"):
        setattr(tracker, name, selection[name])
    for entry in selection["candidates"]:
        grid = tuple(tuple(tuple(row) for row in output) for output in entry["grids"])
        record = dict(entry["record"])
        record["hash"] = str(hash(grid))
        tracker.candidate_records[grid] = record
        tracker.scores[hash(grid)] = entry["score"]
    tracker.solution_most_frequent = selection["top_1"]
    tracker.solution_second_most_frequent = selection["top_2"]
    tracker.first_correct_generation_step = selection["first_correct_generation_step"]
    # Construction consumes random numbers; restore saved sampling state afterwards.
    if restore_rng:
        torch.random.set_rng_state(payload["rng"]["torch_cpu"])
        if target.type == "cuda" and payload["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state(payload["rng"]["torch_cuda"], target)
        random.setstate(payload["rng"]["python"])
        state = payload["rng"]["numpy"]
        np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                             state["position"], state["has_gauss"], state["cached_gaussian"]))
    return InferenceCheckpoint(task, model, tracker, config, payload["completed_steps"])
