import json
from types import SimpleNamespace

import pytest
import torch

from compressarc.train.step_timing import StepTimer


def test_disabled_timer_does_not_touch_cuda_or_write_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled timing must not synchronize or allocate events')
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    monkeypatch.setattr(torch.cuda, 'Event', forbidden)
    path = tmp_path/'off.jsonl'
    with StepTimer('cuda:0', 'off', path) as timer:
        timer.start(0)
        timer.mark('forward')
        assert timer.finish() == {}
        timer.publish({})
    assert not path.exists()


def test_phase_records_and_local_summary(tmp_path):
    ticks = iter([0., 1., 2., 3., 4.])
    path = tmp_path/'task.jsonl'
    logs = []
    with StepTimer('cpu', 'phases', path, clock=lambda:next(ticks)) as timer:
        timer.start(12)
        timer.mark('forward')
        timer.mark('loss')
        result = timer.finish()
        timer.publish(result, SimpleNamespace(log=logs.append))
    assert result == {'train_step':12, 'timing/step_seconds':4.,
                      'timing/host_forward_seconds':1., 'timing/host_loss_seconds':1.}
    assert json.loads(path.read_text()) == result == logs[0]
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    assert summary['completed_timed_steps'] == 1
    assert summary['warmup_steps_omitted'] == 0
    assert summary['steady_mean_step_seconds'] == 4


def test_cuda_phase_markers_do_not_add_per_phase_synchronizations(monkeypatch):
    syncs=[]
    class Event:
        def __init__(self, **kwargs): pass
        def record(self, stream): assert stream == 'test-stream'
        def elapsed_time(self, end): return 125.
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device:syncs.append(str(device)))
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device:'test-stream')
    ticks=iter([0.,1.,2.,3.,4.])
    timer=StepTimer('cuda:0','phases',clock=lambda:next(ticks))
    timer.start(0)
    timer.mark('forward')
    timer.mark('loss')
    assert len(syncs)==1
    result=timer.finish()
    assert len(syncs)==2
    assert result['timing/stream_forward_seconds']==.125
    assert result['timing/stream_loss_seconds']==.125


def test_wall_mode_reports_whole_step_without_phase_noise():
    ticks=iter([4.,4.15])
    timer=StepTimer('cpu','wall',clock=lambda:next(ticks))
    timer.start(3)
    timer.mark('forward')
    result=timer.finish()
    assert set(result)=={'train_step','timing/step_seconds'}
    assert result['timing/step_seconds']==pytest.approx(.15)


def test_timer_closes_file_after_failure(tmp_path):
    timer=StepTimer('cpu','wall',tmp_path/'failed.jsonl')
    with pytest.raises(RuntimeError):
        with timer:
            raise RuntimeError('training interrupted')
    assert timer.file.closed
