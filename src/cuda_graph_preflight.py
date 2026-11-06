"""Small CUDA integration gate: update counts, gradients, RNG and real D4 capture.

Variants are ``conv``, ``combined`` and ``combined_lse``. The LSE variant cannot
assert multi-update trajectory equality (its Triton backward accumulates a shared
kernel gradient with atomics), so it is gated from synchronized state by
``cuda_graph_lse_validation`` instead of the strict loop below.
"""
import json
from types import SimpleNamespace

import torch

from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step
from compressarc.train.cuda_graph import CudaGraphStep
from compressarc.train.optimizer import create_optimizer
from compressarc.train.device_dispatch import runtime_device_dispatch


def preflight_model_options(variant):
    if variant not in ('conv','combined','combined_lse'):
        raise ValueError('Unknown CUDA preflight variant: '+variant)
    options = dict(shift_implementation='d4_tied_conv',
                   direction_share_implementation='vectorized',
                   cummax_implementation='triton_hard_grouped')
    if variant != 'conv':
        options.update(direction_share_implementation='d4', direction_share_packed=True)
    if variant == 'combined_lse':
        options.update(cummax_implementation='triton', lse_tau=0.1)
    return options


def real_model_check(variant="conv"):
    if variant == 'combined_lse':
        from cuda_graph_lse_validation import check_combined_lse
        return check_combined_lse()
    if not torch.cuda.is_available():raise RuntimeError('CUDA preflight requires a CUDA device')
    previous=torch.get_default_device()
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        torch.set_default_device('cuda:0')
        cfg=load_config(overrides=['model.n_layers=1',
            *[f'model.{name}=2' for name in ['channel_dim_shared','channel_dim_grid','share_up_dim','share_down_dim',
                'decoding_dim','softmax_dim','cummax_dim','shift_dim','nonlinear_dim','direction_share_dim']],
            'model.shift_implementation=d4_tied_conv','model.direction_share_implementation=vectorized',
            'model.cummax_implementation=triton_hard_grouped','model.share_implementation=reuse_fused_norm',
            'model.softmax_implementation=fused','training.optimizer_implementation=fused',
            'training.runtime_device_dispatch=explicit','training.annealing_steps=2',
            'training.cuda_graph_start=2','training.cuda_graph_warmup=2','training.iterations=8'])
        cfg.model.update(preflight_model_options(variant))
        problem={'train':[{'input':[[0,1,0],[1,0,1],[0,1,0]],'output':[[1,0,1],[0,1,0],[1,0,1]]}],
                 'test':[{'input':[[1,0,1],[0,1,0],[1,0,1]]}]}
        pairs=[]
        for _ in range(2):
            torch.manual_seed(33)
            task=Task('cuda_graph_preflight',problem,None)
            model=ARCCompressor(task,cfg.model)
            optimizer=create_optimizer(model.weights_list,cfg.training)
            pairs.append((task,model,optimizer))
        runner=CudaGraphStep(*pairs[1],cfg.training)
        with runtime_device_dispatch('explicit'):
            for step in range(8):
                outputs=[]
                for index,(task,model,optimizer) in enumerate(pairs):
                    torch.manual_seed(100+step)
                    fn=take_step if index==0 else runner
                    metrics=fn(task,model,optimizer,step,cfg.training,return_outputs=True)
                    outputs.append(metrics)
                for name in ['loss','kl','reconstruction_error']:
                    torch.testing.assert_close(torch.tensor(outputs[0][name]),torch.tensor(outputs[1][name]),rtol=5e-5,atol=5e-5)
                for a,b in zip(outputs[0]['outputs'],outputs[1]['outputs']):
                    torch.testing.assert_close(a,b,rtol=5e-5,atol=5e-5)
                for a,b in zip(pairs[0][1].weights_list,pairs[1][1].weights_list):
                    assert (a.grad is None)==(b.grad is None)
                    torch.testing.assert_close(a,b,rtol=5e-5,atol=5e-5)
                    if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-4)
                assert len(pairs[0][2].state)==len(pairs[1][2].state)
                for optimizer in [pairs[0][2],pairs[1][2]]:
                    assert all(int(state['step'].item())==step+1 for state in optimizer.state.values())
        assert runner.summary()['optimizer_updates']==8
        assert runner.summary()['eager_steps']==2 and runner.summary()['replayed_steps']==6
        # No reseeding between these replays. Inspect generator state rather
        # than changing outputs: tiny models can have almost zero latent noise.
        task,model,optimizer=pairs[1]
        rng_before=torch.cuda.get_rng_state()
        runner(task,model,optimizer,8,cfg.training,return_outputs=True)
        rng_middle=torch.cuda.get_rng_state()
        runner(task,model,optimizer,9,cfg.training,return_outputs=True)
        rng_after=torch.cuda.get_rng_state()
        assert not torch.equal(rng_before,rng_middle), 'First replay failed to advance CUDA RNG'
        assert not torch.equal(rng_middle,rng_after), 'Second replay failed to advance CUDA RNG'
        return runner.summary()
    finally:
        torch.set_default_device(previous)
        torch.backends.cuda.matmul.allow_tf32=old_tf32


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['conv','combined','combined_lse'],default='conv')
    args=parser.parse_args()
    print(json.dumps(dict(cuda_graph_preflight='passed',variant=args.variant,report=real_model_check(args.variant)),indent=2),flush=True)
