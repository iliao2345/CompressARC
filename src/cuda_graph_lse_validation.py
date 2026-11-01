"""Same-state CUDA Graph checks for numerically non-repeatable LSE training.

The LSE cummax backward accumulates the shared kernel gradient with
``tl.atomic_add`` (``axis_lse_bwd_dk_kernel`` and ``diagonal_lse_bwd_dk_kernel``),
so the accumulation order, and therefore the last bits of that gradient, depend on
kernel scheduling. Two eager runs from bit-identical state are already not
bit-identical. Adam then amplifies the difference: while |g| is near Adam's eps,
|dtheta| = lr*|g|/(|g|+eps) ~ lr/2 regardless of |g|, so a gradient change far
below the 1e-4 gradient tolerance (even a full sign flip, |dg| = 2|g|) produces a
parameter change of order lr. Comparing two independently evolving trajectories is
therefore not a valid replay check.

Instead this gate compares one update at a time from synchronized parameters and
optimizer state, keeps the original output/gradient tolerances, verifies Adam
against the candidate's exact gradients, and measures the eager/eager roundoff
floor in the same run so a failure can be classified as noise or as a real bug.
It does not claim multi-update trajectory identity.
"""
import copy
import torch

from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step
from compressarc.train.cuda_graph import CudaGraphStep
from compressarc.train.optimizer import create_optimizer
from compressarc.train.device_dispatch import runtime_device_dispatch


def synchronize_state(source_model, source_optimizer, target_model, target_optimizer):
    source=list(source_model.weights_list)
    target=list(target_model.weights_list)
    if len(source)!=len(target):
        raise ValueError('Parameter counts differ')
    with torch.no_grad():
        for a,b in zip(source,target):
            if a.shape!=b.shape:raise ValueError('Parameter shapes differ')
            b.copy_(a)
    # Preserve parameter objects/addresses and graph-owned gradient buffers.
    # Adam is outside capture, so replacing its state tensors is safe here.
    target_optimizer.load_state_dict(copy.deepcopy(source_optimizer.state_dict()))


def verify_adam_update(candidate_model, expected_model, expected_optimizer):
    for actual,expected in zip(candidate_model.weights_list,expected_model.weights_list):
        expected.grad=None if actual.grad is None else actual.grad.detach().clone()
    expected_optimizer.step()
    for actual,expected in zip(candidate_model.weights_list,expected_model.weights_list):
        torch.testing.assert_close(actual,expected,rtol=5e-5,atol=5e-5)


def _assert_close(actual, expected, label, step, rtol, atol):
    try:
        torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
    except AssertionError as error:
        raise AssertionError(f'{label} mismatch at step {step}: {error}') from error


def _max_pair_difference(left, right):
    left,right=list(left),list(right)
    if not left:return 0.
    return max(float((a-b).abs().max()) for a,b in zip(left,right))


def output_difference(first, second):
    """Largest absolute difference between two steps' returned model outputs."""
    return _max_pair_difference(first['outputs'],second['outputs'])


def gradient_difference(left_model, right_model):
    """Largest absolute parameter-gradient difference; None if no gradient exists."""
    left=list(left_model.weights_list);right=list(right_model.weights_list)
    pairs=[]
    for a,b in zip(left,right):
        if (a.grad is None)!=(b.grad is None):
            raise AssertionError('Gradient-presence mismatch')
        if a.grad is not None and a.grad.numel():pairs.append((a.grad,b.grad))
    return _max_pair_difference([a for a,_ in pairs],[b for _,b in pairs])


def parameter_difference(left_model, right_model):
    """Largest absolute difference between two models' current parameters."""
    return _max_pair_difference(list(left_model.weights_list),list(right_model.weights_list))


def _build_case(problem, cfg):
    torch.manual_seed(33)
    task=Task('lse_same_state',problem,None)
    model=ARCCompressor(task,cfg.model)
    return task,model,create_optimizer(model.weights_list,cfg.training)


def _load_config():
    return load_config(overrides=['model.n_layers=1',
        *[f'model.{name}=2' for name in ['channel_dim_shared','channel_dim_grid','share_up_dim','share_down_dim',
            'decoding_dim','softmax_dim','cummax_dim','shift_dim','nonlinear_dim','direction_share_dim']],
        'model.shift_implementation=d4_tied_conv','model.direction_share_implementation=d4',
        'model.direction_share_packed=true','model.cummax_implementation=triton','model.lse_tau=0.1',
        'model.share_implementation=reuse_fused_norm','model.softmax_implementation=fused',
        'training.optimizer_implementation=fused','training.runtime_device_dispatch=explicit',
        'training.annealing_steps=2','training.cuda_graph_start=2','training.cuda_graph_warmup=2'])


def check_combined_lse():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    previous=torch.get_default_device()
    tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        torch.set_default_device('cuda:0')
        cfg=_load_config()
        problem={'train':[{'input':[[0,1,0],[1,0,1],[0,1,0]],'output':[[1,0,1],[0,1,0],[1,0,1]]}],
                 'test':[{'input':[[1,0,1],[0,1,0],[1,0,1]]}]}
        # reference: eager trajectory. candidate: graph runner. expected: Adam
        # reference for candidate's own gradients. control: second eager run used
        # only to measure the roundoff floor this variant cannot avoid.
        cases=[_build_case(problem,cfg) for _ in range(4)]
        task,reference,opt_reference=cases[0]
        _,candidate,opt_candidate=cases[1]
        _,expected,opt_expected=cases[2]
        _,control,opt_control=cases[3]
        for case in cases[1:]:
            for a,b in zip(reference.weights_list,case[1].weights_list):
                if not torch.equal(a,b):raise AssertionError('Initial parameters differ; seeding is not reproducible')
        graph=CudaGraphStep(*cases[1],cfg.training)
        max_output_difference=0.
        max_gradient_difference=0.
        noise=dict(outputs=0.,gradients=0.,parameters=0.)
        steps=[]
        with runtime_device_dispatch('explicit'):
            for step in range(8):
                seed=100+step
                # Every comparison starts at exactly the same state; copying
                # values does not move graph parameters or gradient buffers.
                for model,optimizer in ((candidate,opt_candidate),(control,opt_control),(expected,opt_expected)):
                    synchronize_state(reference,opt_reference,model,optimizer)
                torch.manual_seed(seed)
                eager=take_step(task,reference,opt_reference,step,cfg.training,return_outputs=True)
                torch.manual_seed(seed)
                repeated=take_step(task,control,opt_control,step,cfg.training,return_outputs=True)
                torch.manual_seed(seed)
                replayed=graph(cases[1][0],candidate,opt_candidate,step,cfg.training,return_outputs=True)
                # Roundoff floor: identical code, identical state, identical seed.
                step_noise=dict(outputs=output_difference(eager,repeated),
                                gradients=gradient_difference(reference,control),
                                parameters=parameter_difference(reference,control))
                for key in noise:noise[key]=max(noise[key],step_noise[key])
                for name in ['loss','kl','reconstruction_error']:
                    _assert_close(torch.tensor(replayed[name]),torch.tensor(eager[name]),
                                  'graph '+name,step,5e-5,5e-5)
                    assert torch.isfinite(torch.tensor(replayed[name])),'Nonfinite '+name+' at step '+str(step)
                observed_outputs=output_difference(eager,replayed)
                observed_gradients=gradient_difference(reference,candidate)
                max_output_difference=max(max_output_difference,observed_outputs)
                max_gradient_difference=max(max_gradient_difference,observed_gradients)
                for left,right in zip(reference.weights_list,candidate.weights_list):
                    if left.grad is not None:
                        assert torch.isfinite(right.grad).all(),'Nonfinite gradient at step '+str(step)
                assert observed_outputs<=5e-5,(
                    f'graph outputs mismatch at step {step}: {observed_outputs:.2e} > 5e-5 '
                    f'(eager/eager noise floor {noise["outputs"]:.2e})')
                assert observed_gradients<=1e-4,(
                    f'graph gradients mismatch at step {step}: {observed_gradients:.2e} > 1e-4 '
                    f'(eager/eager noise floor {noise["gradients"]:.2e})')
                # Do not confuse tiny gradient roundoff amplified by Adam with
                # an incorrect optimizer update. Check each actual update using
                # precisely the gradients that the candidate computed.
                verify_adam_update(candidate,expected,opt_expected)
                assert len(opt_candidate.state)==len(opt_expected.state)
                for optimizer in [opt_reference,opt_candidate,opt_expected]:
                    assert all(int(state['step'].item())==step+1 for state in optimizer.state.values())
                for left,right in zip(opt_candidate.state.values(),opt_expected.state.values()):
                    for key in ['exp_avg','exp_avg_sq']:
                        _assert_close(left[key],right[key],'Adam '+key,step,5e-5,5e-5)
                steps.append(dict(step=step,graph_active=step>=cfg.training.cuda_graph_start,
                                  graph_output_difference=observed_outputs,
                                  graph_gradient_difference=observed_gradients,
                                  eager_noise=step_noise))
        summary=graph.summary()
        assert summary['optimizer_updates']==8 and summary['eager_steps']==2 and summary['replayed_steps']==6
        before=torch.cuda.get_rng_state()
        graph(cases[1][0],candidate,opt_candidate,8,cfg.training)
        middle=torch.cuda.get_rng_state()
        graph(cases[1][0],candidate,opt_candidate,9,cfg.training)
        after=torch.cuda.get_rng_state()
        assert not torch.equal(before,middle) and not torch.equal(middle,after),'Replay did not advance CUDA RNG'
        summary=graph.summary()
        summary.update(validation='same-state step parity plus exact-gradient eager Adam reference',
                       synchronized_comparisons=8,max_output_difference=max_output_difference,
                       max_gradient_difference=max_gradient_difference,
                       eager_control_noise_floor=noise,
                       steps=steps,
                       independent_trajectory_identity_claimed=False)
        return summary
    finally:
        torch.set_default_device(previous)
        torch.backends.cuda.matmul.allow_tf32=tf32
