from contextlib import contextmanager
from types import SimpleNamespace
import json
import pytest
import torch

from config import load_config
from compressarc.train.optimizer import create_optimizer
from compressarc.train.step_timing import StepTimer


def test_default_optimizer_preserves_original_update():
    config=load_config().training
    a=torch.tensor([1.,-2.,3.],requires_grad=True)
    b=a.detach().clone().requires_grad_()
    opt=create_optimizer([a],config)
    ref=torch.optim.Adam([b],lr=config.learning_rate,betas=tuple(config.optimizer_betas))
    assert opt.defaults['fused'] is None
    for _ in range(3):
        opt.zero_grad();ref.zero_grad()
        a.square().sum().backward();b.square().sum().backward()
        opt.step();ref.step()
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_fused_requires_cuda_and_rejects_unknown_modes():
    p=torch.ones(1,requires_grad=True)
    cfg=load_config(overrides=['training.optimizer_implementation=fused']).training
    with pytest.raises(ValueError,match='CUDA'):
        create_optimizer([p],cfg)
    cfg.optimizer_implementation='typo'
    with pytest.raises(ValueError,match='default or fused'):
        create_optimizer([p],cfg)


def test_fused_constructor_passes_only_original_settings_and_fused_flag(monkeypatch):
    captured=[]
    monkeypatch.setattr(torch.optim,'Adam',lambda params,**kw:captured.append((params,kw)))
    cfg=load_config(overrides=['training.optimizer_implementation=fused']).training
    param=SimpleNamespace(device=torch.device('cuda:0'))
    create_optimizer([param],cfg)
    assert captured==[([param],dict(lr=cfg.learning_rate,betas=tuple(cfg.optimizer_betas),fused=True))]


def test_subtimings_are_nested_in_optimizer_without_cuda_barriers(monkeypatch):
    syncs=[]
    class Event:
        def __init__(self,**kwargs):pass
        def record(self,stream):pass
        def elapsed_time(self,other):return 1.
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:syncs.append(str(device)))
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:None)
    ticks=iter(float(x) for x in range(20))
    with StepTimer('cuda:0','phases',clock=lambda:next(ticks)) as timer:
        timer.start(0);timer.mark('optimizer')
        for name in ['gradient_checks','adam','parameter_checks']:
            with timer.optimizer_scope(name):pass
        assert len(syncs)==1
        result=timer.finish()
    assert len(syncs)==2
    assert result['timing/host_optimizer_seconds']==7.
    for name in ['gradient_checks','adam','parameter_checks']:
        assert result[f'timing/host_optimizer_{name}_seconds']==1.
        assert result[f'timing/stream_optimizer_{name}_seconds']==.001


def test_disabled_nested_timing_does_not_read_clock_or_cuda(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('disabled timing touched clock/CUDA')
    monkeypatch.setattr(torch.cuda,'Event',forbidden)
    timer=StepTimer('cuda:0','off',clock=forbidden)
    with timer.optimizer_scope('adam'):pass


def test_observers_and_scopes_preserve_backward_update_order_and_weights():
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.train.objective import take_step
    cfg=load_config(overrides=['model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2',
        'model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2','model.softmax_dim=2',
        'model.cummax_dim=2','model.shift_dim=2','model.nonlinear_dim=2'])
    task=Task('optimizer_scopes',{'train':[{'input':[[0,1]],'output':[[1,0]]}],
                                 'test':[{'input':[[1,0]]}]},None)
    models=[]
    for _ in range(2):
        torch.manual_seed(17);models.append(ARCCompressor(task,cfg.model))
    events=[]
    @contextmanager
    def scope(name):
        events.append('enter:'+name)
        yield
        events.append('exit:'+name)
    for i,model in enumerate(models):
        torch.manual_seed(19)
        opt=create_optimizer(model.weights_list,cfg.training)
        take_step(task,model,opt,0,cfg.training,**({'optimizer_scope':scope} if i else {}))
    assert events==['enter:gradient_checks','exit:gradient_checks','enter:adam','exit:adam',
                    'enter:parameter_checks','exit:parameter_checks']
    for a,b in zip(models[0].weights_list,models[1].weights_list):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for fused Adam parity')
def test_cuda_fused_adam_parameter_and_state_parity():
    cfg=load_config().training
    torch.manual_seed(92)
    a=[torch.randn(s,device='cuda').requires_grad_() for s in [(8,8),(4,),(3,5),()]]
    b=[p.detach().clone().requires_grad_() for p in a]
    opt=create_optimizer(a,cfg)
    cfg.optimizer_implementation='fused'
    fused=create_optimizer(b,cfg)
    for step in range(8):
        opt.zero_grad();fused.zero_grad()
        for i,(pa,pb) in enumerate(zip(a,b)):
            if i==2 and step%2==0:continue  # unused-gradient behavior must match.
            grad=torch.randn_like(pa) if step%3 else torch.zeros_like(pa)
            pa.grad=grad;pb.grad=grad.clone()
        opt.step();fused.step()
        for pa,pb in zip(a,b):
            torch.testing.assert_close(pa,pb,rtol=3e-5,atol=3e-6)
            if pa in opt.state:
                for key in ['step','exp_avg','exp_avg_sq']:
                    torch.testing.assert_close(opt.state[pa][key].cpu(),fused.state[pb][key].cpu(),rtol=3e-5,atol=3e-6)


def test_optimizer_trace_attributes_launches_syncs_and_scalar_reads(tmp_path):
    from compressarc.train.profile_analysis import analyze_trace
    def e(cat,name,start,duration,**args):
        return dict(ph='X',cat=cat,name=name,ts=start,dur=duration,pid=10,tid=1,args=args)
    events=[e('user_annotation','ttt/optimizer',0,100),
            e('user_annotation','ttt/optimizer/gradient_checks',1,10),
            e('user_annotation','ttt/optimizer/adam',20,30),
            e('cpu_op','aten::_local_scalar_dense',21,10,**{'External id':4}),
            e('cuda_runtime','cudaStreamSynchronize',22,1,**{'External id':4}),
            e('cpu_op','aten::_fused_adam_',40,5,**{'External id':5}),
            e('cuda_runtime','cudaLaunchKernel',41,1,correlation=2,**{'External id':5}),
            e('kernel','fused_adam',42,1,correlation=2,**{'External id':5}),
            e('user_annotation','ttt/optimizer/parameter_checks',60,10)]
    path=tmp_path/'trace.json';path.write_text(json.dumps({'traceEvents':events}))
    result=analyze_trace(path)
    parts={r['component']:r for r in result['optimizer_components']}
    assert parts['gradient_checks']['host_range_us']==10
    assert parts['parameter_checks']['host_range_us']==10
    assert parts['adam']['kernel_launches']==1
    assert parts['adam']['synchronization_calls']==1
    assert parts['adam']['scalar_reads']==1
