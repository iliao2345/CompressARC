"""Optimized sharing contracts against frozen original share_direction."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from compressarc.layers.share.reuse import ReusedMultitensorShare
from compressarc.model.initializers import Initializer
from compressarc.model import multitensor_systems


@pytest.fixture(scope='module')
def original():
    source=Path(__file__).resolve().parents[2]/'reference_main_model/layers.py'
    spec=importlib.util.spec_from_file_location('frozen_sharing_reference',source)
    module=importlib.util.module_from_spec(spec)
    saved=sys.modules.get('multitensor_systems')
    sys.modules['multitensor_systems']=multitensor_systems
    try:spec.loader.exec_module(module)
    finally:
        if saved is None:sys.modules.pop('multitensor_systems',None)
        else:sys.modules['multitensor_systems']=saved
    return module


def make_case(shape=(3,5),known=True,policy='strict',mask_mode='holes',dtype=torch.float64,device='cpu',learn_masks=False):
    torch.manual_seed(322)
    h,w=shape
    mask=torch.ones(2,h,w,2,dtype=dtype,device=device)
    if mask_mode=='empty':mask.zero_()
    elif mask_mode=='holes':
        mask[0,h//2,w//2,:]=0
        mask[1,0,0,0]=0
        mask[0,-1,-1,1]=0
    if learn_masks:mask.requires_grad_()
    task=SimpleNamespace(in_out_same_size=known,all_out_same_size=False,masks=mask)
    system=multitensor_systems.MultiTensorSystem(2,2,h,w,task,constraint_policy=policy)
    old=torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        initializer=Initializer(system,lambda dims:4 if dims[2] else 2)
        weights=initializer.initialize_multiresidual(3,3)
        initializer.symmetrize_xy(weights)
    finally:torch.set_default_dtype(old)
    # Preserve aliases while moving leaves to the requested device.
    moved={}
    for dims in system:
        for pair in weights[dims]:
            for i,value in enumerate(pair):
                if id(value) not in moved:
                    moved[id(value)]=value.detach().to(device=device).requires_grad_()
                pair[i]=moved[id(value)]
    # Deduplicate the actual leaves in their current aliased structures.
    parameters={id(v):v for dims in system for pair in weights[dims] for v in pair}
    inputs=system.make_multitensor()
    for dims in system:
        inputs[dims]=torch.randn(system.shape(dims,initializer.channel_dim_fn(dims)),dtype=dtype,device=device,requires_grad=True)
    return system,inputs,weights,list(parameters.values())


def assert_parity(original,backend,direction,**kwargs):
    system,inputs,weights,params=make_case(**kwargs)
    layer=ReusedMultitensorShare(batch=backend=='batched')
    expected=original.share_direction(inputs,weights,direction)
    actual=layer.share_direction(inputs,weights,direction)
    dtype=next(iter(inputs[d] for d in system)).dtype
    tolerance=dict(rtol=2e-4,atol=3e-5) if dtype==torch.float32 else dict(rtol=1e-10,atol=1e-10)
    loss_a=loss_b=0
    for dims in system:
        torch.testing.assert_close(actual[dims],expected[dims],**tolerance)
        upstream=torch.randn_like(actual[dims])
        loss_a=loss_a+(actual[dims]*upstream).sum()
        loss_b=loss_b+(expected[dims]*upstream).sum()
    leaves=[inputs[d] for d in system]+params
    if system.task.masks.requires_grad:leaves.append(system.task.masks)
    ga=torch.autograd.grad(loss_a,leaves,allow_unused=True)
    gb=torch.autograd.grad(loss_b,leaves,allow_unused=True)
    for a,b in zip(ga,gb):
        if b is None:assert a is None
        else:
            assert a is not None and torch.isfinite(a).all()
            torch.testing.assert_close(a,b,rtol=1e-3 if dtype==torch.float32 else 1e-9,atol=1e-4 if dtype==torch.float32 else 1e-9)
    return layer


@pytest.mark.parametrize('backend',['reuse','batched'])
@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('known',[False,True])
@pytest.mark.parametrize('policy',['strict','relaxed'])
@pytest.mark.parametrize('mask_mode',['full','holes','empty'])
def test_all_layouts_tied_weights_masks_and_gradients(original,backend,direction,known,policy,mask_mode):
    assert_parity(original,backend,direction,known=known,policy=policy,mask_mode=mask_mode)


@pytest.mark.parametrize('backend',['reuse','batched'])
@pytest.mark.parametrize('shape',[(1,1),(1,5),(5,1),(5,3)])
def test_rectangular_singleton_and_mask_gradients(original,backend,shape):
    assert_parity(original,backend,-1,shape=shape,learn_masks=True)


@pytest.mark.parametrize('backend',['reuse','batched'])
@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('known',[False,True])
def test_float32_contract(original,backend,direction,known):
    assert_parity(original,backend,direction,known=known,dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('backend',['reuse','batched'])
@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('known',[False,True])
@pytest.mark.parametrize('shape',[(3,5),(6,6),(1,5)])
def test_cuda_float32_contract(original,backend,direction,known,shape):
    assert_parity(original,backend,direction,known=known,shape=shape,dtype=torch.float32,device='cuda')


def test_reuse_and_batching_reduce_repeated_work_and_bound_stacks():
    system,inputs,weights,params=make_case(shape=(3,3))
    reuse=ReusedMultitensorShare()
    batched=ReusedMultitensorShare(batch=True)
    limited=ReusedMultitensorShare(batch=True,max_batch_elements=1)
    for layer in [reuse,batched,limited]:layer.share_down(inputs,weights)
    a,b,c=[l.last_stats['share_down'] for l in [reuse,batched,limited]]
    assert a['unique_steps']<a['requested_steps']
    assert b['batch_groups']>0
    assert b['reduction_calls']<a['reduction_calls']
    assert c['batch_groups']==0
    assert c['reduction_calls']==a['reduction_calls']


def test_plan_reuse_does_not_retain_values_masks_or_autograd_graphs(original):
    layer=ReusedMultitensorShare(batch=True)
    for step in range(3):
        system,inputs,weights,params=make_case()
        system.task.masks.fill_(0 if step==1 else 1)
        result=layer.share_down(inputs,weights)
        expected=original.share_direction(inputs,weights,-1)
        for dims in system:torch.testing.assert_close(result[dims],expected[dims],rtol=1e-10,atol=1e-10)
        sum(result[d].square().sum() for d in system).backward()
    assert len(layer._plans)==1
    def has_tensor(value):
        if isinstance(value,torch.Tensor):return True
        if isinstance(value,dict):return any(has_tensor(v) for v in value.values())
        if isinstance(value,(list,tuple)):return any(has_tensor(v) for v in value)
        return False
    assert not has_tensor(layer._plans)


@pytest.mark.parametrize('backend',['reuse','batched'])
def test_model_parameters_training_and_inference_checkpoint(tmp_path,backend):
    from config import load_config
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.train import take_step
    from compressarc.train.metrics import SolutionTracker
    from compressarc.train.checkpoint import save_inference_checkpoint,load_inference_checkpoint
    overrides=['model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2',
        'model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2','model.softmax_dim=2',
        'model.cummax_dim=2','model.shift_dim=2','model.nonlinear_dim=2',
        'model.direction_share_implementation=vectorized','model.softmax_implementation=fused',
        'model.shift_implementation=d4_tied_conv']
    task=Task('reuse_sharing',{'train':[{'input':[[0,1,0],[1,0,1]],'output':[[1,0,1],[0,1,0]]}],
                              'test':[{'input':[[0,1,0],[1,0,1]]}]},None)
    models=[]
    for name in ['primitives',backend]:
        cfg=load_config(overrides=overrides+[f'model.share_implementation={name}'])
        torch.manual_seed(210)
        models.append((ARCCompressor(task,cfg.model),cfg))
    assert len(models[0][0].weights_list)==len(models[1][0].weights_list)
    for a,b in zip(models[0][0].weights_list,models[1][0].weights_list):torch.testing.assert_close(a,b,rtol=0,atol=0)
    outputs=[]
    for model,cfg in models:
        torch.manual_seed(23);outputs.append(model.forward())
    for a,b in zip(outputs[0][:3],outputs[1][:3]):torch.testing.assert_close(a,b,rtol=2e-4,atol=3e-5)
    model,cfg=models[1]
    optimizer=torch.optim.Adam(model.weights_list,lr=.01)
    metrics=take_step(task,model,optimizer,0,cfg.training,return_outputs=True)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    tracker=SolutionTracker(task);tracker.update(0,metrics['outputs'])
    checkpoint=tmp_path/'shared.pt'
    save_inference_checkpoint(checkpoint,task,model,tracker,cfg,1,metrics)
    with torch.no_grad():expected=model.forward()
    restored=load_inference_checkpoint(checkpoint)
    assert isinstance(restored.model.share,ReusedMultitensorShare)
    assert restored.model.share.batch==(backend=='batched')
    with torch.no_grad():actual=restored.model.forward()
    for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('backend',['reuse','batched'])
def test_wandb_tag_identifies_selected_sharing_backend(monkeypatch,backend):
    from config import load_config
    from compressarc.train.logging import initialize_wandb
    calls=[]
    run=SimpleNamespace(define_metric=lambda *a,**k:None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(
        init=lambda **kw:(calls.append(kw) or run),Settings=lambda **kw:kw))
    cfg=load_config(overrides=['logging.wandb=true',f'model.share_implementation={backend}'])
    initialize_wandb(cfg,SimpleNamespace(task_name='test'))
    assert 'share-'+backend in calls[0]['tags']
