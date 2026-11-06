"""Variant benchmarks preserve each variable layer and disable model storage."""
import json
from pathlib import Path
import subprocess
import pytest
import torch
import benchmark_variants as bench
from config import load_config

EXPECTED={
 'original':('primitives','primitives','primitives'),
 'projected_conv_c4':('tied_conv','primitives','primitives'),
 'projected_conv_d4':('d4_tied_conv','primitives','primitives'),
 'projected_d4':('primitives','primitives','d4'),
 'projected_lse':('primitives','optimised','primitives'),
}


@pytest.mark.parametrize('variant',list(bench.VARIANTS))
def test_compatible_settings_preserve_the_variable_layer(variant):
    path=bench.REPO/bench.VARIANTS[variant]
    original=load_config(path)
    accelerated=load_config(path,bench.compatible_overrides(original,'accelerated'))
    shift,cummax,direction=EXPECTED[variant]
    assert original.model.shift_implementation==shift
    assert original.model.cummax_implementation==cummax
    assert original.model.direction_share_implementation==direction
    assert accelerated.model.shift_implementation==shift
    assert accelerated.model.cummax_implementation==('triton_hard_grouped' if cummax=='primitives' else cummax)
    assert accelerated.model.direction_share_implementation==('vectorized' if direction=='primitives' else direction)
    allowed={'cummax_implementation','direction_share_implementation','share_implementation','softmax_implementation','layer_timing'}
    for key in original.model:
        if key not in allowed:assert accelerated.model[key]==original.model[key]
    assert accelerated.logging.save_model is False


def test_clean_only_driver_never_captures_or_saves_weights(tmp_path,monkeypatch):
    import profile_layers
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        out=Path(command[command.index('--output-dir')+1]);out.mkdir(parents=True)
        (out/'tiny.step_timing.jsonl').write_text('\n'.join(json.dumps({'timing/step_seconds':.2}) for _ in range(3)))
    monkeypatch.setattr(profile_layers.subprocess,'run',run)
    profile_layers.main(['--config',str(bench.REPO/bench.VARIANTS['original']),
                         '--task','tiny','--clean-only','--clean-steps','3','--warmup-steps','1',
                         '--output-dir',str(tmp_path)])
    assert len(calls)==1
    assert 'training.operator_profile=false' in calls[0]
    assert 'logging.save_model=false' in calls[0]
    report=json.loads((tmp_path/'comparison.json').read_text())
    assert report['clean_only'] and report['profile']=={}
    assert not (tmp_path/'profile').exists()


def test_comparison_runner_writes_reports_and_reference_speedup(tmp_path,monkeypatch):
    root=tmp_path/'comparison'
    def run(command,**kwargs):
        if command[1]=='-c':
            return subprocess.CompletedProcess(command,0,stdout=json.dumps({'gpu':'test GPU','torch':'test'}))
        assert '--clean-only' in command
        assert 'logging.save_model=false' in command
        out=Path(command[command.index('--output-dir')+1]);out.mkdir(parents=True)
        fast='model.share_implementation=reuse_fused_norm' in command
        data=dict(clean_mean_seconds=.2 if fast else .4,clean_median_seconds=.2 if fast else .4,
                  clean_phase_means={})
        (out/'comparison.json').write_text(json.dumps(data))
    monkeypatch.setattr(bench.subprocess,'run',run)
    report=bench.main(['--variants','original','projected_lse','--output-dir',str(root)])
    assert len(report['results'])==4
    assert all(r['speedup_vs_reference']==2 for r in report['results'] if r['mode']=='accelerated')
    assert (root/'variant_latency.csv').exists()
    assert not list(root.rglob('*.pt'))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('variant',list(bench.VARIANTS))
def test_reference_accelerated_model_outputs_gradients_and_parameters(variant,device,monkeypatch):
    from compressarc.model.model import ARCCompressor
    from compressarc.model import layer_factory
    from compressarc.layers.cummax.primitives import CummaxPrimitives
    from preprocessing import Task
    from compressarc.train import take_step
    from compressarc.train.optimizer import create_optimizer
    from compressarc.train.device_dispatch import runtime_device_dispatch
    if device=='cuda':
        if not torch.cuda.is_available():pytest.skip('Real accelerated backend parity requires M3 CUDA')
        pytest.importorskip('triton')
    else:
        # CPU checks only native equivalent paths. CUDA tests exercise real Triton.
        monkeypatch.setattr(layer_factory,'create_triton_hard_grouped_cummax',lambda **kw:CummaxPrimitives(**kw))
    path=bench.REPO/bench.VARIANTS[variant]
    base=load_config(path)
    tiny=['model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2',
          'model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2',
          'model.softmax_dim=2','model.cummax_dim=2','model.shift_dim=2',
          'model.nonlinear_dim=2','model.direction_share_dim=2','training.learning_rate=0']
    results=[];previous=torch.get_default_device()
    try:
        torch.set_default_device(device)
        for mode in ['reference','accelerated']:
            cfg=load_config(path,bench.compatible_overrides(base,mode)+tiny+
                            (['training.optimizer_implementation=default'] if device=='cpu' else []))
            torch.manual_seed(44)
            task=Task('parity',{'train':[{'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]}],
                               'test':[{'input':[[1,0],[0,1]]}]},None)
            model=ARCCompressor(task,cfg.model)
            initial=[p.detach().cpu().clone() for p in model.weights_list]
            optimizer=create_optimizer(model.weights_list,cfg.training)
            torch.manual_seed(128)
            with runtime_device_dispatch(cfg.training.runtime_device_dispatch):
                metrics=take_step(task,model,optimizer,0,cfg.training,return_outputs=True)
            grads=[None if p.grad is None else p.grad.detach().cpu().clone() for p in model.weights_list]
            for p,x in zip(model.weights_list,initial):torch.testing.assert_close(p.detach().cpu(),x,rtol=0,atol=0)
            results.append((initial,metrics,grads))
    finally:torch.set_default_device(previous)
    a,b=results
    assert len(a[0])==len(b[0])
    for x,y in zip(a[0],b[0]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert a[1]['loss']==pytest.approx(b[1]['loss'],rel=3e-4,abs=3e-4)
    for x,y in zip(a[1]['outputs'],b[1]['outputs']):torch.testing.assert_close(x,y,rtol=5e-4,atol=5e-4)
    for x,y in zip(a[2],b[2]):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=3e-3,atol=3e-3)
