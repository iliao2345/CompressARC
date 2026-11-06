"""CUDA contracts against frozen original hard-max equations (not LSE)."""
import importlib.util
from pathlib import Path
import pytest
from types import SimpleNamespace
import torch

_spec=importlib.util.spec_from_file_location('frozen_hard_cummax',Path(__file__).with_name('reference_cummax_main.py'))
reference=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


@pytest.fixture(params=["triton_hard", "triton_hard_grouped"])
def backend(request):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required to compile and validate Triton hard cummax')
    pytest.importorskip('triton')
    from compressarc.layers.cummax import triton_hard
    if request.param == "triton_hard_grouped":
        from compressarc.layers.cummax import triton_hard_grouped as grouped
        return SimpleNamespace(hard_cummax=grouped.hard_cummax_grouped,
                               hard_diagonal_cummax=triton_hard.hard_diagonal_cummax,
                               TritonHardCummax=grouped.TritonHardCummaxGrouped,
                               implementation=request.param)
    return SimpleNamespace(hard_cummax=triton_hard.hard_cummax,
                           hard_diagonal_cummax=triton_hard.hard_diagonal_cummax,
                           TritonHardCummax=triton_hard.TritonHardCummax,
                           implementation=request.param)


def tensors(shape,mode,values):
    torch.manual_seed(82)
    x=torch.randn(shape,device='cuda',dtype=torch.float32)
    if values=='ties':
        x=torch.zeros_like(x)
    elif values=='padding_ties':
        # Exercise ties with the original +/-1000 constant scan padding.
        x=(x.sign()*1000.)
    m=torch.ones_like(x)
    if mode=='empty':m.zero_()
    elif mode=='holes':m[...,shape[-2]//2,shape[-1]//2]=0
    return x.requires_grad_(),m


def assert_parity(actual,expected,x,xref):
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
    upstream=torch.linspace(-.7,1.3,actual.numel(),device=actual.device).reshape(actual.shape)
    got,=torch.autograd.grad((actual*upstream).sum(),x)
    wanted,=torch.autograd.grad((expected*upstream).sum(),xref)
    assert torch.isfinite(got).all() and torch.isfinite(wanted).all()
    torch.testing.assert_close(got,wanted,rtol=1e-3,atol=1e-3)


@pytest.mark.parametrize('shape',[(1,1),(1,5),(5,1),(3,5),(5,3),(16,16),(30,30)])
@pytest.mark.parametrize('mode',['full','holes','empty'])
@pytest.mark.parametrize('values',['random','ties','padding_ties'])
@pytest.mark.parametrize('axis',[1,2])
def test_axis_output_and_gradient(backend,shape,mode,values,axis):
    x,m=tensors((2,*shape),mode,values)
    xr=x.detach().clone().requires_grad_()
    assert_parity(backend.hard_cummax(x,axis,m),reference.cummax_(xr,axis,m),x,xr)


@pytest.mark.parametrize('shape',[(1,1),(1,5),(5,1),(3,5),(5,3),(16,16),(30,30)])
@pytest.mark.parametrize('mode',['full','holes','empty'])
@pytest.mark.parametrize('values',['random','ties','padding_ties'])
def test_diagonal_tree_output_and_gradient(backend,shape,mode,values):
    x,m=tensors((2,*shape),mode,values)
    xr=x.detach().clone().requires_grad_()
    assert_parity(backend.hard_diagonal_cummax(x,1,2,m),reference.diagonal_cummax_(xr,1,2,m),x,xr)


@pytest.mark.parametrize('colors',[False,True])
@pytest.mark.parametrize('mode',['full','holes','empty'])
@pytest.mark.parametrize('post_norm',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_all_direction_layer_input_and_projection_gradients(backend,colors,mode,post_norm,bias):
    from compressarc.layers.cummax.primitives import CummaxPrimitives
    dims=(1,int(colors),1,1,1)
    bind=lambda fn:lambda *a,**kw:fn(dims,*a,**kw)
    torch.manual_seed(91)
    shape=[2]+([3] if colors else [])+[8,3,5,8]
    x=torch.randn(shape,device='cuda',requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    sizes=[(8,4),(4,),(4,8),(8,)]
    p=[torch.randn(s,device='cuda').mul(.2).requires_grad_() for s in sizes]
    pr=[v.detach().clone().requires_grad_() for v in p]
    mask=torch.ones(2,3,5,2,device='cuda')
    if mode=='empty':mask.zero_()
    elif mode=='holes':
        mask[:,1,2,:]=0
        mask[0,0,0,0]=0
        mask[1,2,3,1]=0
    kwargs=dict(pre_norm=False,post_norm=post_norm,use_bias=bias)
    a=backend.TritonHardCummax(bind)(x,[p[:2],p[2:]],mask,**kwargs)
    b=CummaxPrimitives(bind)(xr,[pr[:2],pr[2:]],mask,**kwargs)
    torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-4)
    upstream=torch.randn_like(a)
    ga=torch.autograd.grad((a*upstream).sum(),[x,*p],allow_unused=True)
    gb=torch.autograd.grad((b*upstream).sum(),[xr,*pr],allow_unused=True)
    for got,wanted in zip(ga,gb):
        if wanted is None:assert got is None
        else:torch.testing.assert_close(got,wanted,rtol=2e-3,atol=2e-3)


def test_noncontiguous_broadcast_masks(backend):
    x=torch.randn(2,4,3,5,device='cuda').transpose(1,2).requires_grad_()
    xr=x.detach().clone().requires_grad_()
    mask=torch.ones(2,3,1,5,device='cuda')
    assert_parity(backend.hard_diagonal_cummax(x,1,3,mask),
                  reference.diagonal_cummax_(xr,1,3,mask),x,xr)


def test_no_learned_scan_parameters_and_factory_integration(backend):
    from config import load_config
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.train import take_step
    torch.set_default_device('cuda')
    try:
        config=load_config(overrides=['model.n_layers=1','model.channel_dim_shared=2',
            'model.channel_dim_grid=2','model.share_up_dim=2','model.share_down_dim=2',
            'model.decoding_dim=2','model.softmax_dim=2','model.cummax_dim=2',
            'model.shift_dim=2','model.nonlinear_dim=2','model.shift_implementation=d4_tied_conv',
            'model.direction_share_implementation=vectorized',f'model.cummax_implementation={backend.implementation}'])
        task=Task('hard_scan',{'train':[{'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]}],
                              'test':[{'input':[[0,1],[0,1]]}]},None)
        model=ARCCompressor(task,config.model)
        assert not model._layer_call_options['cummax']
        assert not hasattr(model.cummax,'kernel')
        opt=torch.optim.Adam(model.weights_list,lr=.01)
        metrics=take_step(task,model,opt,0,config.training)
        assert torch.isfinite(torch.tensor(metrics['loss']))
    finally:
        torch.set_default_device('cpu')


def test_reference_tie_conventions_used_by_triton_design():
    # This is a CPU check of the assumed PyTorch conventions, not a CUDA test.
    x=torch.zeros(4,requires_grad=True)
    assert torch.max(x,dim=0).indices.item()==0
    assert torch.min(x,dim=0).indices.item()==0
    assert torch.cummax(x,dim=0).indices.tolist()==[0,1,2,3]
    y=torch.zeros(4,requires_grad=True)
    torch.maximum(x,y).sum().backward()
    torch.testing.assert_close(x.grad,torch.full_like(x,.5))
    torch.testing.assert_close(y.grad,torch.full_like(y,.5))
