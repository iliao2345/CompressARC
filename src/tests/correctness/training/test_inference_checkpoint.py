import random
import numpy as np
import pytest
import torch
from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step
from compressarc.train.metrics import SolutionTracker
from compressarc.train.checkpoint import save_inference_checkpoint,load_inference_checkpoint


@pytest.mark.parametrize("projected",[False,True])
def test_inference_roundtrip_preserves_next_sample_and_candidate_selection(tmp_path,projected):
    config=load_config(overrides=["model.n_layers=1","model.channel_dim_shared=2","model.channel_dim_grid=2",
        "model.share_up_dim=2","model.share_down_dim=2","model.decoding_dim=2","model.softmax_dim=2",
        "model.cummax_dim=2","model.shift_dim=2","model.nonlinear_dim=2"] + (
        ["model.shift_implementation=tied_conv","model.direction_share_implementation=d4",
         "model.cummax_implementation=optimised"] if projected else []))
    task=Task("checkpoint_test",{"train":[{"input":[[0,1],[1,0]],"output":[[1,0],[0,1]]}],
              "test":[{"input":[[1,0],[0,1]]}]},None)
    torch.manual_seed(303)
    model=ARCCompressor(task,config.model)
    optimizer=torch.optim.Adam(model.weights_list,lr=.01,betas=(.5,.9))
    tracker=SolutionTracker(task)
    # Nondefault settings must survive serialization, rather than being implied by code defaults.
    tracker.ema_decay=.9
    tracker.ema_penalty=3
    tracker.early_step_cutoff=20
    tracker.uncertainty_weight=8
    metrics=take_step(task,model,optimizer,0,config.training,return_outputs=True)
    tracker.update(0,metrics["outputs"])
    path=tmp_path/"model.pt"
    save_inference_checkpoint(path,task,model,tracker,config,1,metrics)
    payload=torch.load(path,weights_only=True)
    assert "optimizer" not in payload and "optimizer_state_dict" not in payload
    assert "logging" not in payload["config"]
    assert payload["task_problem"] == task.unprocessed_problem
    with torch.no_grad(): expected=model.forward()
    expected_numpy=np.random.random()
    expected_python=random.random()
    # Loading restores the RNG after model construction has consumed initialization draws.
    restored=load_inference_checkpoint(path)
    assert restored.completed_steps==1
    assert restored.tracker.solution_most_frequent==tracker.solution_most_frequent
    assert restored.tracker.solution_second_most_frequent==tracker.solution_second_most_frequent
    assert restored.tracker.scores==tracker.scores
    assert restored.tracker.candidate_records==tracker.candidate_records
    assert restored.tracker.ema_decay==.9 and restored.tracker.ema_penalty==3
    assert restored.tracker.early_step_cutoff==20 and restored.tracker.uncertainty_weight==8
    torch.testing.assert_close(restored.tracker.ema_logits,tracker.ema_logits,rtol=0,atol=0)
    with torch.no_grad(): actual=restored.model.forward()
    for got,wanted in zip(actual[:3],expected[:3]):
        torch.testing.assert_close(got,wanted,rtol=0,atol=0)
    for got,wanted in zip(actual[3],expected[3]):
        torch.testing.assert_close(got,wanted,rtol=0,atol=0)
    assert np.random.random()==expected_numpy
    assert random.random()==expected_python
    def output_for_tracker(result):
        logits,x,y=result[:3]
        return torch.cat([torch.zeros_like(logits[:,:1]),logits],dim=1).detach().cpu(),x.detach().cpu(),y.detach().cpu()
    tracker.update(1,output_for_tracker(expected))
    restored.tracker.update(1,output_for_tracker(actual))
    assert restored.tracker.scores==tracker.scores
    assert restored.tracker.solution_most_frequent==tracker.solution_most_frequent
    assert restored.tracker.solution_second_most_frequent==tracker.solution_second_most_frequent


def test_gradient_observer_runs_after_backward_before_optimizer():
    config=load_config(overrides=["model.n_layers=1","model.channel_dim_shared=2","model.channel_dim_grid=2",
        "model.share_up_dim=2","model.share_down_dim=2","model.decoding_dim=2","model.softmax_dim=2",
        "model.cummax_dim=2","model.shift_dim=2","model.nonlinear_dim=2"])
    task=Task("gradient_timing",{"train":[{"input":[[0,1]],"output":[[1,0]]}],
                                 "test":[{"input":[[0,1]]}]},None)
    model=ARCCompressor(task,config.model)
    optimizer=torch.optim.Adam(model.weights_list,lr=.01)
    events=[]
    norms=[]
    original_step=optimizer.step
    def step(*args,**kwargs):
        assert events[-1]=="gradients"
        for parameter,wanted in norms:
            torch.testing.assert_close(parameter.grad.norm(),wanted,rtol=0,atol=0)
        events.append("optimizer")
        return original_step(*args,**kwargs)
    optimizer.step=step
    def observe(category,tensors):
        events.append(category)
        if category=="gradients":
            norms.extend((p,p.grad.norm().detach().clone()) for p in model.weights_list if p.grad is not None)
            assert norms
    take_step(task,model,optimizer,0,config.training,diagnostic_callback=observe)
    assert events==["outputs","loss","gradients","optimizer","parameters"]
