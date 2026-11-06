import json
import subprocess
import sys
from pathlib import Path

import torch

from compressarc.train.layer_profiling import active_layer_profile, profile_layer
from compressarc.train.operator_profiling import OperatorProfiler
from compressarc.train.profile_analysis import analyze_trace


def event(cat, name, ts, dur, tid=1, **args):
    return dict(ph='X', cat=cat, name=name, ts=ts, dur=dur, pid=10, tid=tid, args=args)


def test_cross_thread_backward_attribution_and_launch_counts(tmp_path):
    es = [event('user_annotation', 'ttt/forward', 0, 100),
          event('user_annotation', 'layer/depth_0/direction_share', 1, 90),
          event('cpu_op', 'aten::mm', 2, 10, **{'Sequence number':7,'External id':3}),
          event('cuda_runtime', 'cudaLaunchKernel', 3, 1, correlation=1, **{'External id':3}),
          event('kernel', 'gemm', 10, 2, correlation=1, **{'External id':3}),
          event('kernel', 'splitKreduce', 13, 1, correlation=1, **{'External id':3}),
          event('user_annotation', 'ttt/backward', 100, 100),
          event('cpu_op', 'autograd::engine::evaluate_function: MmBackward0', 110, 40, tid=2,
                **{'Sequence number':7,'External id':4}),
          event('cpu_op', 'aten::mm', 111, 30, tid=2, **{'External id':5}),
          event('cuda_runtime', 'cudaLaunchKernel', 112, 1, tid=2, correlation=2, **{'External id':5}),
          event('kernel', 'backward_gemm', 115, 3, correlation=2, **{'External id':5}),
          event('user_annotation', 'ttt/optimizer', 200, 100),
          event('cpu_op', 'aten::_local_scalar_dense', 210, 20, **{'External id':6}),
          event('cuda_runtime', 'cudaStreamSynchronize', 215, 4, **{'External id':6}),
          event('kernel', 'orphan', 310, 2, correlation=99, **{'External id':99})]
    p = tmp_path/'trace.json'
    p.write_text(json.dumps({'traceEvents':es}))
    result = analyze_trace(p, tmp_path/'result')
    assert result['kernel_launches'] == 4
    assert result['unmatched_kernel_launches'] == 1
    assert result['layer_attributed_kernel_launches'] == 3
    phases = {r['phase']:r for r in result['phases']}
    assert phases['forward']['kernel_launches'] == 2
    assert phases['backward']['kernel_launches'] == 1
    assert phases['optimizer']['scalar_reads'] == 1
    assert phases['optimizer']['synchronization_calls'] == 1
    assert phases['optimizer']['synchronization_host_us'] == 4
    layers = {(r['phase'],r['layer']):r for r in result['layers']}
    assert layers['backward','depth_0/direction_share']['gpu_kernel_us'] == 3
    assert (tmp_path/'result.layers.csv').is_file()


def test_actual_autograd_sequence_mapping_and_no_math_change(tmp_path):
    def run(enabled):
        torch.manual_seed(17)
        w = torch.randn(3, 3, requires_grad=True)
        with OperatorProfiler('cpu', tmp_path/('on' if enabled else 'off'), enabled=enabled,
                              start_step=0, steps=1) as profiler:
            profiler.before_step(0)
            profiler.mark('forward')
            with profile_layer('depth_0/direction_share'):
                y = w @ w
            profiler.mark('loss')
            loss = y.square().sum()
            profiler.mark('backward')
            loss.backward()
            profiler.after_step(0)
        assert not active_layer_profile.get()
        return loss.detach(), w.grad
    loss, grad = run(True)
    expected_loss, expected_grad = run(False)
    assert torch.equal(loss, expected_loss)
    assert torch.equal(grad, expected_grad)
    data = json.loads((tmp_path/'on.layers.json').read_text())
    assert any(r['phase']=='backward' and r['layer']=='depth_0/direction_share'
               and r['cpu_operator_self_us'] > 0 for r in data['layers'])


def test_two_pass_dry_run(tmp_path):
    script = Path(__file__).resolve().parents[3]/'profile_layers.py'
    result = subprocess.run([sys.executable, str(script), '--config', str(tmp_path/'config.json'),
                             '--task','00576224','--dry-run','--capture-steps','2','--no-shapes','--sharing-stages',
                             '--output-dir',str(tmp_path/'report')], capture_output=True, text=True, check=True)
    assert 'training.operator_profile=false' in result.stdout
    assert 'training.operator_profile=true' in result.stdout
    assert 'training.operator_profile_steps=2' in result.stdout
    assert 'training.sharing_stage_timing=true' in result.stdout
    assert 'logging.save_model=false' in result.stdout
    assert not (tmp_path/'report').exists()


def test_artifact_upload_targets_reports_and_uses_online_mode(tmp_path, monkeypatch):
    import profile_layers
    calls = []
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setattr(profile_layers.subprocess, 'run', lambda cmd, **kw:calls.append((cmd,kw)))
    profile_layers.upload_profile(tmp_path, 'arc_agi', 'compressarc-final-ablation', 'd4-conv-layer-profile')
    cmd, kwargs = calls[0]
    assert cmd == [sys.executable, '-m', 'wandb', 'artifact', 'put', str(tmp_path),
                   '--name', 'compressarc-final-ablation/d4-conv-layer-profile', '--type', 'profile']
    assert kwargs['check'] is True
    assert kwargs['env']['WANDB_ENTITY'] == 'arc_agi'
    assert kwargs['env']['WANDB_MODE'] == 'online'


def test_profile_upload_refuses_stale_model_weights(tmp_path, monkeypatch):
    import pytest
    import profile_layers
    stale=tmp_path/'clean'/'old.pt'
    stale.parent.mkdir()
    stale.write_bytes(b'checkpoint')
    calls=[]
    monkeypatch.setattr(profile_layers.subprocess,'run',lambda *a,**kw:calls.append(a))
    with pytest.raises(ValueError,match='checkpoint files'):
        profile_layers.upload_profile(tmp_path,'arc_agi','compressarc-final-ablation','profile')
    assert calls==[]
