"""Contracts an equivalent vectorised primitive must keep; frozen main is the oracle.

The vectorized backend must match the frozen original equations and preserve
tied-leaf gradients, including after optimizer updates.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from compressarc.layers.direction_share.vectorized import DirectionShareVectorized
from compressarc.model.initializers import Initializer
from compressarc.model.multitensor_systems import MultiTensorSystem, multify

_spec = importlib.util.spec_from_file_location(
    "frozen_direction_share_contract", Path(__file__).with_name("reference_direction_share_main.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def candidate(dims):
    return DirectionShareVectorized(lambda fn: lambda *a, **kw: fn(dims, *a, **kw))


def clone_pairs(weights):
    # Preserve repeated leaf objects exactly; never replace a tied parameter
    # with independent copies when comparing its accumulated gradient.
    clones = {}
    rows = []
    for row in weights:
        output = []
        for pair in row:
            result = []
            for value in pair:
                if id(value) not in clones:
                    clones[id(value)] = value.detach().clone().requires_grad_()
                result.append(clones[id(value)])
            output.append(result)
        rows.append(output)
    return rows, list(clones.values())


def make_case(dims, dtype, device, noncontiguous):
    torch.manual_seed(902)
    shape = [n for present,n in zip(dims, [2,3,8,3,5]) if present] + [4]
    if noncontiguous:
        x = torch.randn([*shape[:-1],8], dtype=dtype, device=device)[...,::2]
        assert not x.is_contiguous()
    else:
        x = torch.randn(shape, dtype=dtype, device=device)
    # Nontrivial tying: shared across output and input directions.
    pool = [[torch.randn(4,4,dtype=dtype,device=device)*.1,
             torch.randn(4,dtype=dtype,device=device)*.1] for _ in range(10)]
    weights = [[pool[(3*out+inp)%10] for inp in range(8)] for out in range(8)]
    actual, ap = clone_pairs(weights)
    expected, ep = clone_pairs(weights)
    return x.detach().requires_grad_(), x.detach().clone().requires_grad_(), actual, expected, ap, ep


def compare(dims, dtype, device, noncontiguous, pre_norm, use_bias):
    x, xr, w, wr, ap, ep = make_case(dims,dtype,device,noncontiguous)
    actual = candidate(dims)(x,w,pre_norm=pre_norm,use_bias=use_bias)
    expected = reference.direction_share(dims,xr,wr,pre_norm=pre_norm,use_bias=use_bias)
    tol = dict(rtol=3e-5,atol=3e-5) if dtype==torch.float32 else dict(rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(actual,expected,**tol)
    upstream = torch.randn_like(actual)
    got = torch.autograd.grad((actual*upstream).sum(),[x,*ap],allow_unused=True)
    wanted = torch.autograd.grad((expected*upstream).sum(),[xr,*ep],allow_unused=True)
    for a,b in zip(got,wanted):
        if b is None:
            assert a is None
        else:
            assert a is not None and torch.isfinite(a).all()
            torch.testing.assert_close(a,b,**tol)


@pytest.mark.parametrize("dims",[(0,0,1,0,0),(1,1,1,1,0),(1,1,1,0,1),(1,1,1,1,1)])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("noncontiguous",[False,True])
@pytest.mark.parametrize("pre_norm",[False,True])
@pytest.mark.parametrize("use_bias",[False,True])
def test_outputs_and_tied_leaf_gradients(dims,dtype,noncontiguous,pre_norm,use_bias):
    compare(dims,dtype,'cpu',noncontiguous,pre_norm,use_bias)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("dims",[(0,0,1,0,0),(1,1,1,1,0),(1,1,1,0,1),(1,1,1,1,1)])
@pytest.mark.parametrize("pre_norm",[False,True])
@pytest.mark.parametrize("use_bias",[False,True])
def test_cuda_float32_outputs_and_gradients(dims,pre_norm,use_bias):
    # Isolate normal float32 error from TF32 policies on different GPUs.
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        compare(dims,torch.float32,'cuda',True,pre_norm,use_bias)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_actual_initializer_tying_survives_three_adam_updates():
    torch.manual_seed(314)
    system = MultiTensorSystem(2,2,3,5,task=None)
    initializer = Initializer(system,lambda dims:3)
    weights = initializer.initialize_multidirection_share()
    initializer.symmetrize_xy(weights)
    initializer.symmetrize_direction_sharing(weights)
    dims = [1,1,1,1,1]
    actual,ap = clone_pairs(weights[dims])
    expected,ep = clone_pairs(weights[dims])
    assert len(ap) < 128, "Test must exercise actual shared leaves"
    identities = [[id(pair[0]) for pair in row] for row in actual]
    x = torch.randn(system.shape(dims,3),dtype=torch.float32)
    upstream = torch.randn_like(x)
    opt = torch.optim.Adam(ap,lr=.01,betas=(.5,.9))
    refopt = torch.optim.Adam(ep,lr=.01,betas=(.5,.9))
    layer = candidate(dims)
    for _ in range(3):
        opt.zero_grad(); refopt.zero_grad()
        a = layer(x,actual,pre_norm=True,use_bias=False)
        b = reference.direction_share(dims,x,expected,pre_norm=True,use_bias=False)
        torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-5)
        (a*upstream).mean().backward(); (b*upstream).mean().backward()
        for pa,pb in zip(ap,ep):
            assert (pa.grad is None)==(pb.grad is None)
            if pa.grad is not None:
                torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-5,atol=3e-5)
        opt.step(); refopt.step()
        for pa,pb in zip(ap,ep):
            torch.testing.assert_close(pa,pb,rtol=3e-5,atol=3e-5)
    assert identities==[[id(pair[0]) for pair in row] for row in actual]


def test_d4_conv_model_factory_parameter_parity_and_checkpoint(tmp_path):
    from config import load_config
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.train import take_step
    from compressarc.train.metrics import SolutionTracker
    from compressarc.train.checkpoint import save_inference_checkpoint, load_inference_checkpoint
    overrides = ['model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2',
                 'model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2',
                 'model.softmax_dim=2','model.cummax_dim=2','model.shift_dim=2',
                 'model.nonlinear_dim=2','model.shift_implementation=d4_tied_conv']
    problem = {'train':[{'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]}],
               'test':[{'input':[[0,1],[0,1]]}]}
    task = Task('vectorized_contract',problem,None)
    models = []
    for impl in ['primitives','vectorized']:
        config = load_config(overrides=overrides+[f'model.direction_share_implementation={impl}'])
        torch.manual_seed(210)
        model = ARCCompressor(task,config.model)
        models.append((model,config))
    original, candidate_model = [m for m,c in models]
    assert len(original.weights_list)==len(candidate_model.weights_list)
    assert not hasattr(candidate_model,'direction_projection_weights')
    assert candidate_model._layer_call_options['direction_share'] is False
    for a,b in zip(original.weights_list,candidate_model.weights_list):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    outputs=[]
    for model,config in models:
        torch.manual_seed(51)
        outputs.append(model.forward())
    for a,b in zip(outputs[0][:3],outputs[1][:3]):
        torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-5)
    optimizer=torch.optim.Adam(candidate_model.weights_list,lr=.01,betas=(.5,.9))
    metrics=take_step(task,candidate_model,optimizer,0,models[1][1].training,return_outputs=True)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    tracker=SolutionTracker(task);tracker.update(0,metrics['outputs'])
    checkpoint=tmp_path/'vectorized.pt'
    save_inference_checkpoint(checkpoint,task,candidate_model,tracker,models[1][1],1,metrics)
    with torch.no_grad():expected=candidate_model.forward()
    restored=load_inference_checkpoint(checkpoint)
    assert isinstance(restored.model.direction_share,DirectionShareVectorized)
    with torch.no_grad():actual=restored.model.forward()
    for a,b in zip(actual[:3],expected[:3]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
