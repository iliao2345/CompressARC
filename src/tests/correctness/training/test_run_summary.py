from types import SimpleNamespace
import pytest
import torch
from compressarc.train.logging import RunSummary
from compressarc.train import logging

@pytest.mark.parametrize("failure,status", [(None,"finished"),(ValueError,"failed"),(KeyboardInterrupt,"interrupted")])
def test_run_summary_tracks_each_run_and_failure(monkeypatch, failure, status):
    ticks=iter([10.0,14.5,20.0,22.0])
    monkeypatch.setattr(logging.time,"perf_counter",lambda:next(ticks))
    events=[]
    monkeypatch.setattr(torch.cuda,"synchronize",lambda device:events.append("sync"))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda device:events.append("reset"))
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda device:3*2**20)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda device:5*2**20)
    for expected_elapsed in (4.5,2.0):
        codes=[]
        run=SimpleNamespace(summary={},finish=lambda **kwargs:codes.append(kwargs["exit_code"]))
        def execute():
            with RunSummary(run,torch.device("cuda:0")):
                assert run.summary["run/status"] == "running"
                if failure:
                    raise failure("test")
        if failure:
            with pytest.raises(failure): execute()
        else: execute()
        assert run.summary == {"run/status":status,"run/wall_clock_seconds":expected_elapsed,
                               "run/peak_gpu_allocated_mib":3.0,"run/peak_gpu_reserved_mib":5.0}
        assert codes == [0 if failure is None else 1]
    assert events == ["sync","reset","sync"]*2


def test_disabled_run_does_not_touch_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *a:pytest.fail("unexpected CUDA call"))
    with RunSummary(None,torch.device("cuda:0")):
        pass


def test_cuda_synchronization_failure_still_finishes_with_status(monkeypatch):
    calls=[]
    run=SimpleNamespace(summary={},finish=lambda **kw:calls.append(kw))
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *a:(_ for _ in ()).throw(RuntimeError("CUDA failure")))
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda *a:0)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda *a:0)
    with pytest.raises(RuntimeError,match="CUDA failure"):
        with RunSummary(run,torch.device("cuda:0")):
            pass
    assert run.summary["run/status"] == "failed"
    assert calls == [{"exit_code":1}]
