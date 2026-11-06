import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from config import load_config
from compressarc.train.cuda_graph import CudaGraphStep, validate_cuda_graph_options


@pytest.mark.parametrize('override,error',[
    ('training.cuda_graph_start=99','annealing_steps'),
    ('training.cuda_graph_warmup=0','positive'),
    ('training.step_timing=phases','phase timings'),
    ('training.operator_profile=true','Disable'),
    ('training.sharing_stage_timing=true','Disable'),
    ('model.layer_timing=true','Disable')])
def test_reject_capture_that_would_freeze_annealing_or_misreport_timing(override,error):
    cfg=load_config(overrides=['training.cuda_graph=true',override])
    with pytest.raises(ValueError,match=error):
        validate_cuda_graph_options(cfg.training,cfg.model,'train','cuda:0')


def test_device_mode_and_default_eager_guards():
    cfg=load_config()
    validate_cuda_graph_options(cfg.training,cfg.model,'benchmark','cpu')
    cfg.training.cuda_graph=True
    with pytest.raises(ValueError,match='CUDA device'):
        validate_cuda_graph_options(cfg.training,cfg.model,'train','cpu')
    with pytest.raises(ValueError,match='train mode'):
        validate_cuda_graph_options(cfg.training,cfg.model,'benchmark','cuda:0')
    validate_cuda_graph_options(cfg.training,cfg.model,'train','cuda:0')
    with pytest.raises(ValueError,match='one CUDA device'):
        CudaGraphStep(None,SimpleNamespace(weights_list=[torch.zeros(1)]),None,cfg.training)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_graph_safe_scalar_cleanup_preserves_eager_outputs_gradients_and_rng(dtype):
    from compressarc.layers.latent_decoder import channel_layer
    code=textwrap.dedent(inspect.getsource(channel_layer))
    code=code.replace('min_capacity = 0.5','min_capacity = mean.new_tensor(0.5)')
    code=code.replace('init_capacity = 10000.0','init_capacity = mean.new_tensor(10000.0)')
    namespace={'torch':torch};exec(code,namespace)
    original=namespace['channel_layer']
    torch.manual_seed(3)
    initial=[torch.randn(3,dtype=dtype)*.01,torch.randn(2,4,3,dtype=dtype),torch.randn(2,4,3,dtype=dtype)*.01]
    results=[]
    for fn in [original,channel_layer]:
        capacity,mean,adjust=[value.clone().requires_grad_() for value in initial]
        torch.manual_seed(41)
        z,kl=fn(capacity,[mean,adjust])
        grad=torch.autograd.grad(z.square().sum()+kl.sum(),[capacity,mean,adjust])
        results.append((z,kl,*grad,torch.get_rng_state()))
    for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA capture requires a GPU')
def test_real_d4_capture_updates_gradients_rng_and_exact_update_count():
    pytest.importorskip('triton')
    from cuda_graph_preflight import real_model_check
    report=real_model_check()
    assert report['optimizer_updates']==10
    assert report['optimizer_captured'] is False
