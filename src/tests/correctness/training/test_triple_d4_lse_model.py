"""Exercise all three learned operators together, on CPU and the deployed CUDA path."""
import json
from pathlib import Path
import pytest
import torch
from config import load_config

ROOT=Path(__file__).resolve().parents[4]
FOLDER=ROOT/'experiments/seed0_triple_d4_lse'

def test_frozen_manifest():
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    frozen=json.loads((ROOT/'experiments/seed0_combined_d4/manifest.json').read_text())
    assert manifest['task_ids']==frozen['task_ids']
    assert manifest['cohorts']==frozen['cohorts']
    assert len(manifest['jobs'])==40
    assert len({job['task'] for job in manifest['jobs']})==40
    assert {job['seed'] for job in manifest['jobs']}=={0}
    cfg=load_config(FOLDER/'configs/triple_d4_lse.json')
    assert cfg.model.cummax_implementation=='triton'
    assert cfg.model.direction_share_packed
    assert cfg.training.iterations==1500 and cfg.training.seed==0
    assert not cfg.logging.save_model

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all_three_operators_receive_gradients_and_update(device):
    if device=='cuda':
        if not torch.cuda.is_available(): pytest.skip('Requires M3 CUDA GPU')
        pytest.importorskip('triton')
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.train import take_step
    from compressarc.train.optimizer import create_optimizer
    from compressarc.train.device_dispatch import runtime_device_dispatch
    overrides=['model.n_layers=1']+[f'model.{key}=2' for key in ['channel_dim_shared','channel_dim_grid','share_up_dim','share_down_dim','decoding_dim','softmax_dim','cummax_dim','shift_dim','nonlinear_dim','direction_share_dim']]
    if device=='cpu': overrides+=['model.cummax_implementation=optimised','training.optimizer_implementation=default']
    cfg=load_config(FOLDER/'configs/triple_d4_lse.json',overrides)
    previous=torch.get_default_device()
    try:
        torch.set_default_device(device)
        torch.manual_seed(0)
        task=Task('triple',{'train':[{'input':[[0,1,0],[1,0,1],[0,1,0]],'output':[[1,0,1],[0,1,0],[1,0,1]]}],'test':[{'input':[[1,0,1],[0,1,0],[1,0,1]]}]},None)
        model=ARCCompressor(task,cfg.model)
        groups=[list(module.parameters()) for module in [model.shift,model.direction_share,model.cummax]]
        assert all(groups)
        assert len({id(p) for group in groups for p in group})==sum(map(len,groups))
        assert {id(p) for group in groups for p in group}<={id(p) for p in model.weights_list}
        before=[[p.detach().clone() for p in group] for group in groups]
        optimizer=create_optimizer(model.weights_list,cfg.training)
        with runtime_device_dispatch(cfg.training.runtime_device_dispatch):
            for step in range(2):
                metrics=take_step(task,model,optimizer,step,cfg.training)
                assert torch.isfinite(torch.tensor(metrics['loss']))
                for group in groups:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group)
                    assert any(torch.count_nonzero(p.grad)>0 for p in group)
        for group,old in zip(groups,before):
            assert any(not torch.equal(p,q) for p,q in zip(group,old))
    finally:
        torch.set_default_device(previous)
