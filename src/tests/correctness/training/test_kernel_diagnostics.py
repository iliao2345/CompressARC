from types import SimpleNamespace
import pytest
import torch
from compressarc.train.diagnostics import TrainingDiagnostics
from compressarc.layers.shift.morphological import TiedDirectionalConv
from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step


def recording_run():
    records=[]
    return SimpleNamespace(log=records.append,summary={}),records


def test_kernel_norms_collected_only_every_fifty_steps():
    core=TiedDirectionalConv()
    model=SimpleNamespace(shift=SimpleNamespace(models=[core]))
    run,records=recording_run()
    with TrainingDiagnostics(run,model,SimpleNamespace(n_x=3,n_y=5)) as diag:
        with torch.no_grad(): core.cardinal_kernel.add_(1)
        core.cardinal_kernel.grad=torch.full_like(core.cardinal_kernel,2)
        diag.step=48
        diag.observe("gradients",[core.cardinal_kernel.grad])
        diag.after_step()
        assert not any("train_step" in r for r in records)
        diag.step=49
        diag.observe("gradients",[core.cardinal_kernel.grad])
        diag.after_step()
    record=next(r for r in records if "train_step" in r)
    prefix="kernels/shift/depth_0/cardinal_kernel"
    assert record[prefix+"/gradient_norm"] == 6
    assert record[prefix+"/distance_from_initialization"] == 3
    assert record[prefix+"/parameter_norm"] == pytest.approx(float(core.cardinal_kernel.norm()))
    assert record["kernels/shift/depth_0/diagonal_kernel/gradient_norm"] is None
    assert not any(key.startswith("equivariance/") for r in records for key in r)
    assert run.summary["numerics/nonfinite_detected"] is False


def test_nonfinite_checks_only_emit_events_and_always_summarize():
    run,records=recording_run()
    with pytest.raises(ValueError):
        with TrainingDiagnostics(run,SimpleNamespace(),None) as diag:
            for step in range(3):
                diag.step=step
                diag.observe("outputs",[torch.ones(3)])
            assert records == []
            diag.step=3
            diag.observe("outputs",[torch.tensor([float("nan"),float("inf"),float("-inf")])])
            diag.observe("gradients",[torch.tensor([float("nan")])])
            raise ValueError("stop")
    assert len(records)==2
    assert records[0]["numerics/nan_count"]==1
    assert records[0]["numerics/inf_count"]==2
    assert run.summary["numerics/nonfinite_event_count"]==2
    assert run.summary["numerics/nonfinite_step_count"]==1
    assert run.summary["numerics/first_nonfinite_step"]==4
    assert run.summary["numerics/nan_detected"] and run.summary["numerics/inf_detected"]


@pytest.mark.parametrize("implementation", ["reference", "triton"])
def test_diagnostics_preserve_loss_gradients_and_optimizer_update(implementation):
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config=load_config(overrides=["model.n_layers=1","model.channel_dim_shared=2", "model.channel_dim_grid=2",
            "model.share_up_dim=2","model.share_down_dim=2","model.decoding_dim=2","model.softmax_dim=2",
            "model.cummax_dim=2","model.shift_dim=2","model.nonlinear_dim=2",
            "model.shift_implementation=tied_conv","model.direction_share_implementation=d4",
            "model.cummax_implementation=optimised"])
        problem={"train":[{"input":[[0,1],[1,0]],"output":[[1,0],[0,1]]}],"test":[{"input":[[1,0],[0,1]]}]}
        task=Task("diagnostic_parity",problem,None)
        models=[]
        for _ in range(2):
            torch.manual_seed(22)
            models.append(ARCCompressor(task,config.model))
        optimizers=[torch.optim.Adam(m.weights_list,lr=.01,betas=(.5,.9)) for m in models]
        torch.manual_seed(33)
        expected=take_step(task,models[0],optimizers[0],49,config.training)
        torch.manual_seed(33)
        run,records=recording_run()
        with TrainingDiagnostics(run,models[1],task,implementation=implementation) as diag:
            diag.step=49
            actual=take_step(task,models[1],optimizers[1],49,config.training,diagnostic_callback=diag.observe)
            diag.after_step()
        assert actual==expected
        for a,b in zip(models[0].weights_list,models[1].weights_list):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        assert run.summary["numerics/nonfinite_detected"] is False
    finally:
        torch.set_num_threads(previous_threads)
