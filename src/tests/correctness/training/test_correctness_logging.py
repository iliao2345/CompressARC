from types import SimpleNamespace
import pytest
from compressarc.train.logging import log_training_metrics

@pytest.mark.parametrize("first,second,expected", [(0,0,(0,0)),(1,0,(1,1)),(0,1,(0,1)),(1,1,(1,1))])
def test_correctness_truth_table(first, second, expected):
    correct, wrong = (((1,),),), (((0,),),)
    task = SimpleNamespace(solution_hash=hash(correct))
    tracker = SimpleNamespace(solution_most_frequent=correct if first else wrong,
                              solution_second_most_frequent=correct if second else wrong,
                              first_correct_generation_step=None, candidate_metrics=lambda: {})
    records=[]
    run=SimpleNamespace(log=records.append,summary={})
    log_training_metrics(run,task,tracker,1499,{"reconstruction_error":2.5,"kl":1.5})
    assert records == [{"train_step":1499,"predictions/top_1_correct":expected[0],
                        "predictions/pass_2_correct":expected[1],
                        "train/reconstruction_error":2.5,"train/total_KL":1.5}]

@pytest.mark.parametrize("step", [0, 48, 49, 50, 98, 99, 100, 1498, 1499, 1500])
def test_training_metrics_every_step_without_candidate_logging(step):
    records=[]
    run=SimpleNamespace(log=records.append,summary={})
    task=SimpleNamespace(solution_hash=None)
    tracker=SimpleNamespace(first_correct_generation_step=None, candidate_metrics=lambda: {})
    log_training_metrics(run,task,tracker,step,{"reconstruction_error":2.5,"kl":1.5})
    assert len(records) == 1
    assert records[0]["train/reconstruction_error"] == 2.5
    assert records[0]["train/total_KL"] == 1.5
    assert not any(key.startswith("candidates/") for key in records[0])
    assert "predictions/top_1_correct" not in records[0]
