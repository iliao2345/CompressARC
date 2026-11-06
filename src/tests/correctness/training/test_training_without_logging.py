"""Training keeps per-step guesses and local checkpoints without W&B."""
import builtins
import sys
from types import SimpleNamespace

import pytest
import torch

import train
from config import load_config
from preprocessing import Task
from compressarc.train.metrics import SolutionTracker


@pytest.mark.parametrize("save_model", [False, True])
@pytest.mark.parametrize("save_checkpoints", [False, True])
@pytest.mark.parametrize("wandb_enabled", [False, True])
@pytest.mark.parametrize("has_labels", [False, True])
def test_training_without_wandb_tracks_every_step_and_saves_checkpoint(monkeypatch, tmp_path, has_labels, wandb_enabled, save_checkpoints, save_model):
    config = load_config(overrides=["training.iterations=101", "training.device=cpu"])
    config.logging.wandb = wandb_enabled
    config.logging.save_model = save_model
    records, finished = [], []
    run = SimpleNamespace(log=records.append, summary={}, define_metric=lambda *a, **k: None,
                          finish=lambda **kwargs: finished.append(kwargs))
    uploads = []
    monkeypatch.setattr(train, "log_model_checkpoint", lambda logged_run, path: uploads.append(
        (logged_run, path, path.exists(), bool(finished))) if logged_run is not None else None)
    init_calls = []
    def initialize(**kwargs):
        init_calls.append(kwargs)
        return run
    if wandb_enabled:
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize, Settings=lambda **k: k))
    task = Task("no_logging", {
        "train": [{"input": [[0]], "output": [[0]]}],
        "test": [{"input": [[0]]}],
    }, [[[0]]] if has_labels else None)
    updates, requests = [], []
    tracker = SolutionTracker(task)
    tracker.solution_most_frequent = tracker.solution_second_most_frequent = (((0,),),)
    def update(step, outputs):
        assert outputs == (step,)
        updates.append(step)
        tracker.solution_most_frequent = (((1,),),) if step % 2 == 0 else (((0,),),)
        tracker.solution_second_most_frequent = (((0,),),)
    tracker.update = update
    tracker.first_correct_generation_step = None
    tracker.candidate_metrics = lambda: {}
    monkeypatch.setattr(train, "log_final_candidates", lambda *a: None)
    def step(task, model, optimizer, index, config, return_outputs=False, diagnostic_callback=None):
        requests.append(return_outputs)
        return {"loss": 3.0, "reconstruction_error": 2.0, "kl": 1.0,
                "kl_components": {}, "outputs": (index,)}
    original_import = builtins.__import__
    def block_wandb(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            if not wandb_enabled:
                raise AssertionError("Disabled logging must not import W&B")
            return sys.modules["wandb"]
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", block_wandb)
    monkeypatch.setattr(train.preprocessing, "preprocess_tasks", lambda *a, **k: [task])
    monkeypatch.setattr(train, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(train, "ARCCompressor", lambda *a: SimpleNamespace(weights_list=[torch.nn.Parameter(torch.zeros(1))]))
    monkeypatch.setattr(train, "SolutionTracker", lambda *a: tracker)
    monkeypatch.setattr(train, "take_step", step)
    original_device = torch.get_default_device()
    try:
        arguments = ["--task", "no_logging", "--output-dir", str(tmp_path)]
        if save_checkpoints:
            arguments.append("--save-checkpoints")
        train.main(arguments)
    finally:
        torch.set_default_device(original_device)
    assert updates == list(range(101))
    assert requests == [True] * 101
    if save_checkpoints or (wandb_enabled and save_model):
        checkpoint = torch.load(tmp_path / "no_logging.pt", weights_only=True)
        assert checkpoint["task"] == "no_logging"
        assert checkpoint["metrics"]["loss"] == 3.0
        assert "outputs" not in checkpoint["metrics"]
        assert "logging" not in checkpoint["config"]
        assert checkpoint["config"]["model"]
        assert list(tmp_path.iterdir()) == [tmp_path / "no_logging.pt"]
    else:
        assert list(tmp_path.iterdir()) == []

    if wandb_enabled:
        assert len(init_calls) == 1 and "config" not in init_calls[0]
        assert finished == [{"exit_code": 0}]
        assert uploads == ([(run, tmp_path / "no_logging.pt", True, False)]
                           if save_checkpoints or save_model else [])
    if wandb_enabled:
        expected = []
        for step in range(101):
            record = {"train_step": step, "train/reconstruction_error": 2.0, "train/total_KL": 1.0}
            if has_labels:
                record.update({"predictions/top_1_correct": step % 2, "predictions/pass_2_correct": 1})
            expected.append(record)
        assert records == expected
        assert run.summary["run/status"] == "finished"
        assert run.summary["run/wall_clock_seconds"] >= 0
        assert "run/peak_gpu_allocated_mib" not in run.summary
        if has_labels:
            assert run.summary["top_1_correct"] is False
            assert run.summary["pass_2_correct"] is True
        else:
            assert "top_1_correct" not in run.summary
    else:
        assert records == [] and run.summary == {}
