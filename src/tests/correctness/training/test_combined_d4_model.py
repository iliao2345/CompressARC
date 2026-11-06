"""Combined D4 ablation is one model with both learned operators enabled."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from config import load_config

REPO=Path(__file__).resolve().parents[4]
FOLDER=REPO/'experiments/seed0_combined_d4'


def test_combined_config_and_manifest_are_one_distinct_40_task_variant():
    cfg=load_config(FOLDER/'configs/combined_d4.json')
    frozen=json.loads((REPO/'experiments/seed2_40/manifest.json').read_text())
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    assert manifest['task_ids']==frozen['task_ids'] and manifest['cohorts']==frozen['cohorts']
    assert len(manifest['jobs'])==40 and len({j['task'] for j in manifest['jobs']})==40
    assert {j['variant'] for j in manifest['jobs']}=={'combined_d4_conv_sharing'}
    assert cfg.model.shift_implementation=='d4_tied_conv'
    assert cfg.model.direction_share_implementation=='d4'
    assert cfg.model.cummax_implementation=='triton_hard_grouped'
    assert cfg.training.seed==0 and cfg.training.iterations==1500
    assert not cfg.logging.save_model and cfg.logging.log_config
    assert cfg.experiment.variant=='combined-d4-conv-sharing'
    assert cfg.experiment.comparison_role=='combined_ablation'
    assert '--array=0-39%2' in (FOLDER/'submit.slurm').read_text()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_both_d4_operators_receive_gradients_and_update_in_one_training_step(device, packed):
    if device=='cuda':
        if not torch.cuda.is_available():pytest.skip('Combined accelerated CUDA path requires M3')
        pytest.importorskip('triton')
    from preprocessing import Task
    from compressarc.model.model import ARCCompressor
    from compressarc.layers.shift.D4_morphological import D4DirectionalConv
    from compressarc.layers.direction_share.morphological import D4DirectionShare
    from compressarc.train.optimizer import create_optimizer
    from compressarc.train import take_step
    from compressarc.train.device_dispatch import runtime_device_dispatch
    overrides=[f'model.direction_share_packed={str(packed).lower()}', 'model.n_layers=1','model.channel_dim_shared=2','model.channel_dim_grid=2',
        'model.share_up_dim=2','model.share_down_dim=2','model.decoding_dim=2',
        'model.softmax_dim=2','model.cummax_dim=2','model.shift_dim=2','model.nonlinear_dim=2',
        'model.direction_share_dim=2']
    if device=='cpu':overrides+=['model.cummax_implementation=primitives','training.optimizer_implementation=default']
    cfg=load_config(FOLDER/'configs/combined_d4.json',overrides)
    previous=torch.get_default_device()
    try:
        torch.set_default_device(device);torch.manual_seed(0)
        task=Task('combined',{'train':[{'input':[[0,1,0],[1,0,1],[0,1,0]],
                                       'output':[[1,0,1],[0,1,0],[1,0,1]]}],
                             'test':[{'input':[[1,0,1],[0,1,0],[1,0,1]]}]},None)
        model=ARCCompressor(task,cfg.model)
        assert isinstance(model.shift.models[0],D4DirectionalConv)
        assert isinstance(model.direction_share.models[0],D4DirectionShare)
        conv=list(model.shift.parameters());sharing=list(model.direction_share.parameters())
        assert conv and sharing and {id(p) for p in conv}.isdisjoint(id(p) for p in sharing)
        assert {id(p) for p in conv+sharing}<={id(p) for p in model.weights_list}
        before=[p.detach().clone() for p in conv+sharing]
        optimizer=create_optimizer(model.weights_list,cfg.training)
        with runtime_device_dispatch(cfg.training.runtime_device_dispatch):
            for step in range(2):
                metrics=take_step(task,model,optimizer,step,cfg.training)
                assert torch.isfinite(torch.tensor(metrics['loss'],device=device))
                for group in (conv,sharing):
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group)
                    assert any(torch.count_nonzero(p.grad)>0 for p in group)
        assert any(not torch.equal(p,old) for p,old in zip(conv,before[:len(conv)]))
        assert any(not torch.equal(p,old) for p,old in zip(sharing,before[len(conv):]))
    finally:torch.set_default_device(previous)


def test_combined_wandb_label_cannot_be_counted_as_single_operator_variant(monkeypatch):
    from compressarc.train.logging import initialize_wandb
    calls=[]
    run=SimpleNamespace(define_metric=lambda *a,**kw:None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **kw:calls.append(kw) or run,Settings=lambda **kw:kw))
    cfg=load_config(FOLDER/'configs/combined_d4.json')
    initialize_wandb(cfg,SimpleNamespace(task_name='3eda0437'))
    assert 'combined-d4-conv-sharing' in calls[0]['tags']
    assert 'D4-conv' not in calls[0]['tags'] and 'projected-d4' not in calls[0]['tags']
    assert calls[0]['config']['model']['shift_implementation']=='d4_tied_conv'
    assert calls[0]['config']['model']['direction_share_implementation']=='d4'


def test_combined_worker_launches_one_model_without_weights():
    spec=importlib.util.spec_from_file_location('combined_worker',FOLDER/'run_job.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    cmd,job,out=worker.command(FOLDER,0)
    assert job['variant']=='combined_d4_conv_sharing' and out.name=='seed_0'
    assert cmd.count('--config')==1 and '--save-checkpoints' not in cmd
    assert 'logging.save_model=false' in cmd
