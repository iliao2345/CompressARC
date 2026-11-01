from types import SimpleNamespace
import pytest
import torch
from cuda_graph_lse_validation import (synchronize_state, verify_adam_update,
    output_difference, gradient_difference, parameter_difference)

def case():
    model=SimpleNamespace(weights_list=[
        torch.nn.Parameter(torch.tensor([.2,-.1],dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor([.7],dtype=torch.float64))])
    optimizer=torch.optim.Adam(model.weights_list,lr=.01,betas=(.5,.9),eps=1e-8)
    return model,optimizer

def test_snapshot_copy_preserves_parameter_and_static_gradient_addresses():
    source,source_opt=case();target,target_opt=case()
    source.weights_list[0].grad=torch.tensor([.1,-.2],dtype=torch.float64)
    source_opt.step()
    target.weights_list[0].grad=torch.ones(2,dtype=torch.float64)
    identity=id(target.weights_list[0]);gradient=target.weights_list[0].grad
    synchronize_state(source,source_opt,target,target_opt)
    assert id(target.weights_list[0])==identity
    assert target.weights_list[0].grad is gradient
    torch.testing.assert_close(source.weights_list[0],target.weights_list[0],rtol=0,atol=0)
    target_opt.state[target.weights_list[0]]['exp_avg'].add_(10)
    assert not torch.equal(source_opt.state[source.weights_list[0]]['exp_avg'],
                           target_opt.state[target.weights_list[0]]['exp_avg'])

def test_adam_verification_uses_exact_candidate_gradient_and_preserves_unused_none():
    candidate,opt=case();expected,expected_opt=case()
    candidate.weights_list[0].grad=torch.tensor([1e-8,-1e-8],dtype=torch.float64)
    opt.step()
    verify_adam_update(candidate,expected,expected_opt)
    assert expected.weights_list[1].grad is None
    assert len(expected_opt.state)==len(opt.state)==1
    for a,b in zip(candidate.weights_list,expected.weights_list):
        torch.testing.assert_close(a,b,rtol=0,atol=0)

def test_adam_verification_rejects_missing_actual_update():
    candidate,opt=case();expected,expected_opt=case()
    candidate.weights_list[0].grad=torch.tensor([.2,-.3],dtype=torch.float64)
    with pytest.raises(AssertionError):verify_adam_update(candidate,expected,expected_opt)

def test_tiny_gradient_sign_changes_can_create_large_adam_parameter_changes():
    a,oa=case();b,ob=case()
    a.weights_list[0].grad=torch.tensor([1e-8,1e-8],dtype=torch.float64)
    b.weights_list[0].grad=-a.weights_list[0].grad.clone()
    torch.testing.assert_close(a.weights_list[0].grad,b.weights_list[0].grad,rtol=1e-4,atol=1e-4)
    oa.step();ob.step()
    assert float((a.weights_list[0]-b.weights_list[0]).abs().max().detach())>.009


def outputs(*values):
    return dict(outputs=tuple(torch.tensor(value,dtype=torch.float64) for value in values))


def single(model):
    return SimpleNamespace(weights_list=[torch.nn.Parameter(torch.tensor(model,dtype=torch.float64))])


def test_step_difference_helpers_report_largest_absolute_gap():
    assert output_difference(outputs([0.,1.],[2.,3.]),outputs([0.,1.5],[2.,2.5]))==0.5
    assert output_difference(outputs([0.]),outputs([0.]))==0.
    left,right=single([1.,-2.]),single([1.,-2.])
    assert parameter_difference(left,right)==0.
    left.weights_list[0].grad=torch.tensor([.1,.2],dtype=torch.float64)
    right.weights_list[0].grad=torch.tensor([.1,-.2],dtype=torch.float64)
    assert gradient_difference(left,right)==0.4
    right.weights_list[0].grad=None
    with pytest.raises(AssertionError):
        gradient_difference(left,right)
