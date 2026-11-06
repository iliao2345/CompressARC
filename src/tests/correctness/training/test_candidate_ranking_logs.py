import sys
from types import SimpleNamespace
import pytest
import torch
from compressarc.train.metrics import SolutionTracker
from compressarc.train.logging import log_training_metrics, log_final_candidates


def tracker_with_reference(known=True):
    correct=(((1,),),)
    task=SimpleNamespace(n_train=1,n_test=1,n_colors=1,n_x=1,n_y=1,
                         solution_hash=hash(correct) if known else None,
                         reference_solutions=correct if known else None)
    return SolutionTracker(task),correct


def feed(tracker,monkeypatch,step,raw,ema):
    sequence=iter([(raw,0.0),(ema,0.0)])
    monkeypatch.setattr(tracker,"decode",lambda *a:next(sequence))
    tracker.update(step,(torch.zeros(2,2,1,1,2),torch.zeros(2,1,2),torch.zeros(2,1,2)))


def test_unique_correct_counts_first_generation_and_rank(monkeypatch):
    tracker,correct=tracker_with_reference()
    wrong=(((0,),),)
    feed(tracker,monkeypatch,0,wrong,wrong)
    before=tracker.candidate_metrics()
    assert before["candidates/cumulative_unique_count"] == 1
    assert before["candidates/cumulative_correct_count"] == 0
    assert before["candidates/best_correct_rank"] is None
    assert before["candidates/best_correct_score"] is None
    assert tracker.first_correct_generation_step is None
    feed(tracker,monkeypatch,1,wrong,correct)
    assert tracker.first_correct_generation_step == 2
    stats=tracker.candidate_metrics()
    assert stats["candidates/cumulative_unique_count"] == 2
    assert stats["candidates/cumulative_correct_count"] == 1
    assert stats["candidates/best_correct_rank"] == 2
    assert stats["candidates/best_correct_score"] == tracker.scores[hash(correct)]
    assert stats["candidates/top_ranked_score"] == tracker.scores[hash(wrong)]
    feed(tracker,monkeypatch,150,correct,correct)
    assert tracker.first_correct_generation_step == 2
    assert tracker.candidate_metrics()["candidates/best_correct_rank"] == 1
    assert len(tracker.candidate_records) == 2


def test_correct_rank_can_worsen_after_other_candidate_overtakes(monkeypatch):
    tracker,correct=tracker_with_reference()
    wrong=(((0,),),)
    feed(tracker,monkeypatch,150,correct,correct)
    feed(tracker,monkeypatch,151,wrong,wrong)
    # Give the correct grid another appearance to make its lead unambiguous.
    feed(tracker,monkeypatch,152,correct,correct)
    leading=tracker.candidate_metrics()
    assert leading["candidates/best_correct_rank"] == 1
    current_correct_score=tracker.scores[hash(correct)]
    feed(tracker,monkeypatch,153,wrong,wrong)
    feed(tracker,monkeypatch,154,wrong,wrong)
    current=tracker.candidate_metrics()
    assert current["candidates/best_correct_rank"] == 2
    assert current["candidates/best_correct_score"] == current_correct_score
    assert current["candidates/top_ranked_score"] == tracker.scores[hash(wrong)]
    # The logged snapshot must reflect rank 2, rather than the earlier rank 1.
    records=[]
    run=SimpleNamespace(log=records.append,summary={})
    log_training_metrics(run,tracker.task,tracker,199,{"reconstruction_error":1.0,"kl":2.0})
    assert records[-1]["candidates/best_correct_rank"] == 2
    assert records[-1]["candidates/best_correct_score"] == tracker.scores[hash(correct)]


@pytest.mark.parametrize("known",[False,True])
def test_metric_cadence_and_final_table(monkeypatch,known):
    tracker,correct=tracker_with_reference(known)
    feed(tracker,monkeypatch,0,correct,correct)
    feed(tracker,monkeypatch,1,correct,correct)
    records=[]
    run=SimpleNamespace(log=records.append,summary={})
    for step in (0,48,49,50,99):
        log_training_metrics(run,tracker.task,tracker,step,{"reconstruction_error":2.0,"kl":3.0})
    assert [r["train_step"] for r in records if "candidates/cumulative_unique_count" in r] == [49,99]
    assert all("train/total_KL" in r for r in records)
    if known:
        assert run.summary["candidates/first_correct_generation_step"] == 1
    else:
        assert "candidates/first_correct_generation_step" not in run.summary
        assert all("candidates/cumulative_correct_count" not in r for r in records)
    class Table:
        def __init__(self,columns): self.columns,self.rows=columns,[]
        def add_data(self,*row): self.rows.append(row)
    monkeypatch.setitem(sys.modules,"wandb",SimpleNamespace(Table=Table))
    log_final_candidates(run,tracker)
    table=records[-1]["candidates/all_unique"]
    assert len(table.rows) == 1
    row=dict(zip(table.columns,table.rows[0]))
    assert row["hash"] == str(hash(correct))
    assert table.columns == ["hash", "first_step", "occurrences", "is_correct"]
    assert row["is_correct"] is (True if known else None)
    assert row["first_step"] == 1
    assert row["occurrences"] == 2


def test_occurrences_deduplicate_within_step_and_accumulate_across_steps(monkeypatch):
    tracker,correct=tracker_with_reference()
    wrong=(((0,),),)
    feed(tracker,monkeypatch,0,correct,correct)
    feed(tracker,monkeypatch,1,wrong,correct)
    feed(tracker,monkeypatch,2,wrong,wrong)
    assert len(tracker.candidate_records)==2
    assert tracker.candidate_records[correct]["occurrences"]==2
    assert tracker.candidate_records[wrong]["occurrences"]==2
    assert tracker.candidate_records[correct]["first_generation_step"]==1
    assert tracker.candidate_records[wrong]["first_generation_step"]==2
