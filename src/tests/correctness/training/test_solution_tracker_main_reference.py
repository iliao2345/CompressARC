"""Compare candidate decoding, scoring and ranking with original CompressARC."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compressarc.train.metrics import SolutionTracker
from preprocessing import preprocess_tasks

_spec = importlib.util.spec_from_file_location(
    "tracker_main_reference", Path(__file__).resolve().parents[1] / "reference_main_model/solution_selection.py",
)
SCORE_CALLS = []
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    device, dtype, threads = torch.get_default_device(), torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    original_logaddexp = np.logaddexp
    def record_score(previous, contribution):
        SCORE_CALLS.append((previous, contribution))
        return original_logaddexp(previous, contribution)
    monkeypatch.setattr(np, "logaddexp", record_score)
    yield
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def attach_decoding_records(old, current):
    old_results, current_results = [], []
    old_decode, current_decode = old._postprocess_solution, current.decode

    def record_old(*args):
        result = old_decode(*args)
        old_results.append(result)
        return result

    def record_current(*args):
        result = current_decode(*args)
        current_results.append(result)
        return result

    old._postprocess_solution = record_old
    current.decode = record_current
    return old_results, current_results


def compare_update(old, current, step, outputs, records):
    old_records, current_records = records
    SCORE_CALLS.clear()
    old_records.clear()
    current_records.clear()
    old._track_solution(step, *outputs)
    current.update(step, outputs)
    assert len(old_records) == len(current_records) == 2
    assert len(SCORE_CALLS) == 4
    for candidate, (old_call, current_call) in enumerate(zip(SCORE_CALLS[:2], SCORE_CALLS[2:])):
        assert current_call[1] == pytest.approx(old_call[1], abs=3e-6, rel=2e-6), (step, candidate, "score contribution")
    for candidate, ((wanted_grid, wanted_uncertainty), (got_grid, got_uncertainty)) in enumerate(zip(old_records, current_records)):
        assert got_grid == wanted_grid, (step, candidate, "decoded candidate")
        assert got_uncertainty == pytest.approx(wanted_uncertainty, abs=2e-7, rel=2e-6), (step, candidate, "uncertainty")
    for name in ("ema_logits", "ema_x_mask", "ema_y_mask"):
        torch.testing.assert_close(getattr(current, name), getattr(old, name), rtol=0, atol=0)
    assert current.scores.keys() == old.solution_hashes_count.keys(), (step, "candidate keys")
    for key, wanted in old.solution_hashes_count.items():
        assert current.scores[key] == pytest.approx(wanted, abs=3e-6, rel=2e-6), (step, key, "accumulated score")
    assert current.solution_most_frequent == old.solution_most_frequent, (step, "guess 1")
    assert current.solution_second_most_frequent == old.solution_second_most_frequent, (step, "guess 2")


@pytest.mark.parametrize("fixed_shape", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tied_logits", [False, True])
def test_identical_history_matches_main_at_every_update(fixed_shape, dtype, tied_logits):
    task = SimpleNamespace(n_train=1, n_test=2, n_colors=2, n_x=3, n_y=4,
                           in_out_same_size=fixed_shape, all_out_same_size=False,
                           shapes=[((3, 4), (3, 4)), ((3, 4), (2, 3)), ((3, 4), (1, 2))],
                           colors=[0, 2, 7])
    old, current = reference.Logger(task), SolutionTracker(task)
    records = attach_decoding_records(old, current)
    generator = torch.Generator().manual_seed(911)
    # Covers the early penalty boundary and the user's area of interest.
    steps = [*range(171), *range(1045, 1081), 1499]
    for step in steps:
        logits = torch.zeros(3, 3, 3, 4, 2, dtype=dtype) if tied_logits else torch.randn(3, 3, 3, 4, 2, generator=generator, dtype=dtype)
        masks = [torch.zeros(3, length, 2, dtype=dtype) if tied_logits else
                 torch.randn(3, length, 2, generator=generator, dtype=dtype)
                 for length in (3, 4)]
        compare_update(old, current, step, (logits, *masks), records)


def test_full_1500_step_history_on_run_task_matches_main():
    task = preprocess_tasks("training", ["50cb2852"])[0]
    old, current = reference.Logger(task), SolutionTracker(task)
    records = attach_decoding_records(old, current)
    generator = torch.Generator().manual_seed(919)
    for step in range(1500):
        # Controlled competing patterns; this is not a replay of saved run logits.
        logits = torch.randn(task.n_examples, task.n_colors + 1, task.n_x, task.n_y, 2,
                             generator=generator) * 0.02
        winner = 1 if step < 750 else min(2, task.n_colors)
        logits[task.n_train:, :, :, :, 1] = 0
        logits[task.n_train:, winner, :, :, 1] = 3.35 if step < 750 else 10
        x_mask = torch.ones(task.n_examples, task.n_x, 2)
        y_mask = torch.ones(task.n_examples, task.n_y, 2)
        compare_update(old, current, step, (logits, x_mask, y_mask), records)


def test_exact_score_ties_and_promotions_match_main():
    task = SimpleNamespace(n_test=1, n_colors=2, n_x=1, n_y=1)
    old, current = reference.Logger(task), SolutionTracker(task)
    solutions = [(((color,),),) for color in (0, 2, 7)]
    for index, contribution in ((0, 0), (1, 0), (2, 0), (2, 0), (0, 0), (1, -4), (1, 0)):
        solution = solutions[index]
        key = hash(solution)
        old.solution_hashes_count[key] = float(np.logaddexp(old.solution_hashes_count.get(key, -np.inf), contribution))
        current.scores[key] = float(np.logaddexp(current.scores.get(key, -np.inf), contribution))
        old._update_most_frequent_solutions(key, solution)
        current._update_top_two(key, solution)
        assert current.scores == old.solution_hashes_count
        assert current.solution_most_frequent == old.solution_most_frequent
        assert current.solution_second_most_frequent == old.solution_second_most_frequent
