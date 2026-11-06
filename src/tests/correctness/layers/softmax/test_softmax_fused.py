"""Fused native softmax contracts against frozen original subset equations."""
import ast
import itertools
from pathlib import Path

import pytest
import torch

from compressarc.layers.helper import apply_residual
from compressarc.layers.softmax_fused import softmax_fused, softmax_operation_fused

# Extract only the operation from the frozen original; its old residual wrapper
# had a known pre-norm bug, so wrapper parity uses the current apply_residual.
_source=Path(__file__).resolve().parents[2]/'reference_main_model/layers.py'
_tree=ast.parse(_source.read_text())
_fn=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='softmax')
_fn.decorator_list=[]
_namespace={'torch':torch,'itertools':itertools}
exec(compile(ast.fix_missing_locations(ast.Module(body=[_fn],type_ignores=[])),str(_source),'exec'),_namespace)
original_operation=_namespace['softmax']


def compare(dims,dtype,device,noncontiguous,values='random'):
    torch.manual_seed(402)
    shape=[n for present,n in zip(dims,[2,3,8,3,5]) if present]+[2]
    if noncontiguous:
        x=torch.randn([*shape[:-1],4],dtype=dtype,device=device)[...,::2]
    else:x=torch.randn(shape,dtype=dtype,device=device)
    if values=='ties':x=torch.zeros_like(x)
    if values=='extreme':x=x*1000
    x=x.detach().requires_grad_();xr=x.detach().clone().requires_grad_()
    got=softmax_operation_fused(dims,x);wanted=original_operation(dims,xr)
    tol=dict(rtol=2e-5,atol=2e-6) if dtype==torch.float32 else dict(rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(got,wanted,**tol)
    upstream=torch.randn_like(got)
    ga,=torch.autograd.grad((got*upstream).sum(),x)
    gb,=torch.autograd.grad((wanted*upstream).sum(),xr)
    assert torch.isfinite(ga).all()
    torch.testing.assert_close(ga,gb,rtol=2e-4 if dtype==torch.float32 else 1e-10,atol=2e-5 if dtype==torch.float32 else 1e-11)


@pytest.mark.parametrize('dims',list(itertools.product([0,1],repeat=5)))
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('noncontiguous',[False,True])
def test_all_layouts_values_order_and_gradients(dims,dtype,noncontiguous):
    compare(dims,dtype,'cpu',noncontiguous)


@pytest.mark.parametrize('dims',[(1,0,0,0,0),(0,0,1,0,0),(1,1,1,1,1)])
@pytest.mark.parametrize('values',['ties','extreme'])
def test_ties_extremes_and_empty_subset(dims,values):
    compare(dims,torch.float32,'cpu',True,values)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dims',list(itertools.product([0,1],repeat=5)))
@pytest.mark.parametrize('values',['random','ties','extreme'])
def test_cuda_outputs_and_gradients(dims,values):
    compare(dims,torch.float32,'cuda',True,values)


@pytest.mark.parametrize('dims',[(1,0,0,0,0),(1,0,1,0,0),(1,1,1,1,1)])
@pytest.mark.parametrize('pre_norm',[False,True])
@pytest.mark.parametrize('post_norm',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_residual_input_and_projection_gradients(dims,pre_norm,post_norm,bias):
    torch.manual_seed(67)
    shape=[n for present,n in zip(dims,[2,2,8,2,3]) if present]+[3]
    x=torch.randn(shape,dtype=torch.float64,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    out=2*(2**sum(dims[1:])-1)
    p=[torch.randn(s,dtype=torch.float64).mul(.1).requires_grad_() for s in [(3,2),(2,),(out,3),(3,)]]
    pr=[v.detach().clone().requires_grad_() for v in p]
    # Direct undecorated wrapped functions are not assumed; test the same residual formula.
    actual=apply_residual(x,[p[:2],p[2:]],lambda z:softmax_operation_fused(dims,z),
                          pre_norm=pre_norm,post_norm=post_norm,use_bias=bias)
    expected=apply_residual(xr,[pr[:2],pr[2:]],lambda z:original_operation(dims,z),
                            pre_norm=pre_norm,post_norm=post_norm,use_bias=bias)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    upstream=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*upstream).sum(),[x,*p],allow_unused=True)
    gb=torch.autograd.grad((expected*upstream).sum(),[xr,*pr],allow_unused=True)
    for a,b in zip(ga,gb):
        if b is None:assert a is None
        else:torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)


def test_model_initialization_training_and_checkpoint(tmp_path):
    from config import load_config
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.train import take_step
    from compressarc.train.metrics import SolutionTracker
    from compressarc.train.checkpoint import save_inference_checkpoint,load_inference_checkpoint
    overrides=['model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2',
               'model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2',
               'model.softmax_dim=2','model.cummax_dim=2','model.shift_dim=2','model.nonlinear_dim=2',
               'model.shift_implementation=d4_tied_conv','model.direction_share_implementation=vectorized']
    task=Task('softmax_contract',{'train':[{'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]}],
                                 'test':[{'input':[[0,1],[0,1]]}]},None)
    models=[]
    for backend in ['primitives','fused']:
        config=load_config(overrides=overrides+[f'model.softmax_implementation={backend}'])
        torch.manual_seed(210)
        models.append((ARCCompressor(task,config.model),config))
    assert len(models[0][0].weights_list)==len(models[1][0].weights_list)
    for a,b in zip(models[0][0].weights_list,models[1][0].weights_list):torch.testing.assert_close(a,b,rtol=0,atol=0)
    outputs=[]
    for model,cfg in models:
        torch.manual_seed(82);outputs.append(model.forward())
    for a,b in zip(outputs[0][:3],outputs[1][:3]):torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    model,cfg=models[1]
    optimizer=torch.optim.Adam(model.weights_list,lr=.01)
    metrics=take_step(task,model,optimizer,0,cfg.training,return_outputs=True)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    tracker=SolutionTracker(task);tracker.update(0,metrics['outputs'])
    path=tmp_path/'fused_softmax.pt'
    save_inference_checkpoint(path,task,model,tracker,cfg,1,metrics)
    with torch.no_grad():expected=model.forward()
    restored=load_inference_checkpoint(path)
    assert restored.model.softmax is softmax_fused
    with torch.no_grad():actual=restored.model.forward()
    for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
