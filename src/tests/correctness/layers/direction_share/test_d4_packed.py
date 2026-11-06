"""Packed learned D4 maps preserve outputs and all parameter gradients."""
import pytest
import torch
from compressarc.layers.direction_share.morphological import D4DirectionShareLayer

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("dims", [(0,0,1,0,0), (1,0,1,1,0), (1,1,1,1,1)])
@pytest.mark.parametrize("pre_norm", [False, True])
def test_packed_matches_reference(device, dtype, dims, pre_norm):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(12)
    identity = lambda fn: fn
    reference = D4DirectionShareLayer(identity, n_layers=2).to(device=device, dtype=dtype)
    packed = D4DirectionShareLayer(identity, n_layers=2, packed=True).to(device=device, dtype=dtype)
    packed.load_state_dict(reference.state_dict())
    lengths = [2, 3, 8, 3, 4]
    shape = [length for length, enabled in zip(lengths, dims) if enabled] + [3]
    # Noncontiguous spatial/channel storage exercises layout copies too.
    x = torch.randn(*shape[:-1], 6, device=device, dtype=dtype)[..., ::2].detach().requires_grad_()
    a = torch.randn(3, 2, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(2, 3, device=device, dtype=dtype, requires_grad=True)
    w = [(a,), (b,)]
    y = reference._apply_multitensor(dims, x, w, pre_norm=pre_norm, layer_index=1)
    z = packed._apply_multitensor(dims, x, w, pre_norm=pre_norm, layer_index=1)
    upstream = torch.randn_like(y)
    grads_y = torch.autograd.grad(y, (x,a,b,reference.models[1].theta), upstream)
    grads_z = torch.autograd.grad(z, (x,a,b,packed.models[1].theta), upstream)
    tolerance = dict(atol=2e-12, rtol=2e-11) if dtype == torch.float64 else dict(atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(y, z, **tolerance)
    for expected, actual in zip(grads_y, grads_z):
        torch.testing.assert_close(expected, actual, **tolerance)
    assert packed.state_dict().keys() == reference.state_dict().keys()

@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("turns", range(4))
def test_packed_commutes_with_d4_direction_action(reflection, turns):
    layer = D4DirectionShareLayer(lambda f: f, packed=True).double()
    torch.manual_seed(7)
    x = torch.randn(2,8,3,dtype=torch.float64)
    w = [(torch.randn(3,2,dtype=torch.float64),), (torch.randn(2,3,dtype=torch.float64),)]
    permutation = ((-torch.arange(8) if reflection else torch.arange(8)) + 2*turns) % 8
    dims = (1,0,1,0,0)
    y = layer._apply_multitensor(dims,x,w)
    transformed = layer._apply_multitensor(dims,x[:,permutation],w)
    torch.testing.assert_close(transformed,y[:,permutation],atol=1e-12,rtol=1e-11)
