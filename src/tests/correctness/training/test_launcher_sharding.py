"""Ensure independent Slurm workers cover the pilot exactly once."""

from pathlib import Path

import pytest

import run_training_timing as launcher
from config import load_config

SRC = Path(__file__).resolve().parents[3]
CONFIGS = [SRC / "config/models/conv_d4_cummax_pilot_8_corrected.yaml",
           SRC / "config/models/all_projected_triton_pilot_8_corrected.yaml"]


def argument(command, name):
    return command[command.index(name) + 1]


def test_sixteen_shards_cover_two_eight_task_variants_without_overlap(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launcher.train, "main", lambda command: calls.append(command))
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    common = ["--mode", "train", "--save-checkpoints", "--output-dir", str(tmp_path)]
    for config in CONFIGS:
        common.extend(["--config", str(config)])
    for shard in range(16):
        before = len(calls)
        launcher.main([*common, "--shard-count", "16", "--shard-index", str(shard)])
        assert len(calls) == before + 1
    pairs = [(argument(command, "--task"), argument(command, "--config")) for command in calls]
    assert len(set(pairs)) == 16
    expected = {(task, str(path)) for path in CONFIGS
                for task in load_config(path).experiment.task_ids}
    assert set(pairs) == expected
    for command in calls:
        path = Path(argument(command, "--config"))
        assert Path(argument(command, "--output-dir")) == tmp_path / path.stem
        assert "--save-checkpoints" in command
        config = load_config(path)
        assert config.training.seed == 0 and config.training.iterations == 1500
        assert config.model.shift_implementation == "tied_conv"
        assert config.model.direction_share_implementation == "d4"
    assert load_config(CONFIGS[0]).model.cummax_implementation == "primitives"
    assert load_config(CONFIGS[1]).model.cummax_implementation == "triton"


@pytest.mark.parametrize("count,index", [(0, 0), (-1, 0), (16, -1), (16, 16)])
def test_invalid_shard_coordinates_fail_before_training(monkeypatch, count, index):
    calls = []
    monkeypatch.setattr(launcher.train, "main", lambda command: calls.append(command))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--mode", "train", "--shard-count", str(count), "--shard-index", str(index)])
    assert error.value.code == 2
    assert calls == []


def test_single_worker_preserves_original_output_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launcher.train, "main", lambda command: calls.append(command))
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    launcher.main(["--mode", "train", "--config", str(CONFIGS[0]), "--task", "54d82841",
                   "--save-checkpoints", "--output-dir", str(tmp_path)])
    assert len(calls) == 1
    assert Path(argument(calls[0], "--output-dir")) == tmp_path


def test_dry_run_does_not_execute_training(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(launcher.train, "main", lambda command: calls.append(command))
    launcher.main(["--mode", "train", "--config", str(CONFIGS[0]), "--config", str(CONFIGS[1]),
                   "--shard-count", "16", "--shard-index", "1", "--dry-run"])
    output = capsys.readouterr().out
    assert calls == []
    assert "--task 0dfd9992" in output
    assert CONFIGS[1].name in output
    assert CONFIGS[0].name not in output
