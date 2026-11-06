"""Fused sharing normalization must preserve values, VJPs and tied gradients."""
import importlib.util
from pathlib import Path
import pytest
import torch
from compressarc.layers.helper import normalize


def share_fixture():
    path=Path(__file__).resolve().parents[1]/'layers/share/test_share_reuse.py'
    spec=importlib.util.spec_from_file_location('sharing_test_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('direction',['share_up','share_down'])
@pytest.mark.parametrize('known',[False,True])
def test_cpu_fused_option_preserves_full_sharing_and_mask_gradients(direction,known):
    from compressarc.layers.share.reuse import ReusedMultitensorShare
    system,inputs,weights,params=share_fixture().make_case(known=known,learn_masks=True)
    plain=getattr(ReusedMultitensorShare(),direction)(inputs,weights)
    fused=getattr(ReusedMultitensorShare(fused_normalization=True),direction)(inputs,weights)
    old_loss=new_loss=0
    for dims in system:
        torch.testing.assert_close(fused[dims],plain[dims],rtol=0,atol=0)
        upstream=torch.randn_like(plain[dims])
        old_loss=old_loss+(plain[dims]*upstream).sum()
        new_loss=new_loss+(fused[dims]*upstream).sum()
    leaves=[inputs[d] for d in system]+params+[system.task.masks]
    a=torch.autograd.grad(old_loss,leaves,allow_unused=True)
    b=torch.autograd.grad(new_loss,leaves,allow_unused=True)
    for x,y in zip(a,b):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=0,atol=0)


@pytest.mark.parametrize('shape',[(1,2),(3,8),(30,16),(256,8),(6048,16),(8192,2)])
@pytest.mark.parametrize('pattern',['random','constant','small_variance'])
def test_cuda_normalization_output_and_input_gradients(shape,pattern):
    if not torch.cuda.is_available():pytest.skip('CUDA/Triton required')
    pytest.importorskip('triton')
    from compressarc.layers.share.normalization_triton import normalize_tensor,eligible
    torch.manual_seed(173)
    x=torch.randn(shape,device='cuda')
    if pattern=='constant':x.zero_()
    elif pattern=='small_variance':x.mul_(1e-5)
    x.requires_grad_();reference=x.detach().clone().requires_grad_()
    assert eligible(x)
    y=normalize_tensor(x);expected=normalize(reference)
    torch.testing.assert_close(y,expected,rtol=3e-4,atol=3e-4)
    upstream=torch.randn_like(y)
    dx,=torch.autograd.grad(y,x,upstream)
    wanted,=torch.autograd.grad(expected,reference,upstream)
    torch.testing.assert_close(dx,wanted,rtol=5e-4,atol=5e-4)


@pytest.mark.parametrize('direction',['share_up','share_down'])
@pytest.mark.parametrize('known',[False,True])
def test_cuda_full_sharing_with_projection_and_mask_gradients(direction,known):
    if not torch.cuda.is_available():pytest.skip('CUDA/Triton required')
    pytest.importorskip('triton')
    from compressarc.layers.share.reuse import ReusedMultitensorShare
    system,inputs,weights,params=share_fixture().make_case(known=known,device='cuda',dtype=torch.float32,learn_masks=True)
    plain=getattr(ReusedMultitensorShare(),direction)(inputs,weights)
    fused=getattr(ReusedMultitensorShare(fused_normalization=True),direction)(inputs,weights)
    a=b=0
    for dims in system:
        torch.testing.assert_close(fused[dims],plain[dims],rtol=5e-4,atol=5e-4)
        up=torch.randn_like(plain[dims]);a=a+(plain[dims]*up).sum();b=b+(fused[dims]*up).sum()
    leaves=[inputs[d] for d in system]+params+[system.task.masks]
    ga=torch.autograd.grad(a,leaves,allow_unused=True);gb=torch.autograd.grad(b,leaves,allow_unused=True)
    for x,y in zip(ga,gb):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=2e-3,atol=2e-3)


def test_model_factory_selects_fused_reuse_without_new_parameters():
    from config import load_config
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    parameters=[]
    for backend in ('reuse','reuse_fused_norm'):
        torch.manual_seed(17)
        task=Task('tiny',{'train':[{'input':[[0,1]],'output':[[1,0]]}],
                          'test':[{'input':[[0,1]]}]},None)
        cfg=load_config(overrides=['model.n_layers=1',f'model.share_implementation={backend}'])
        model=ARCCompressor(task,cfg.model)
        assert model.share.fused_normalization==(backend=='reuse_fused_norm')
        parameters.append([p.detach().clone() for p in model.weights_list])
    assert len(parameters[0])==len(parameters[1])
    for x,y in zip(*parameters):torch.testing.assert_close(x,y,rtol=0,atol=0)

@pytest.mark.parametrize('mode',['strided','float64','large'])
def test_unsupported_cuda_layouts_use_exact_reference_fallback(mode):
    if not torch.cuda.is_available():pytest.skip('CUDA/Triton required')
    pytest.importorskip('triton')
    from compressarc.layers.share.normalization_triton import normalize_tensor,eligible
    if mode=='strided':x=torch.randn(7,16,device='cuda')[:,::2]
    elif mode=='float64':x=torch.randn(7,8,device='cuda',dtype=torch.float64)
    else:x=torch.randn(8193,2,device='cuda')
    x=x.detach().requires_grad_()
    expected_x=x.detach().clone().requires_grad_()
    assert not eligible(x)
    actual=normalize_tensor(x);expected=normalize(expected_x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    upstream=torch.randn_like(actual)
    ga,=torch.autograd.grad(actual,x,upstream)
    gb,=torch.autograd.grad(expected,expected_x,upstream)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
