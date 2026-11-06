"""Fresh seed0 retests cover all tasks and retain controlled optimised settings."""
import importlib.util
import json
from pathlib import Path
import pytest
from config import load_config

REPO=Path(__file__).resolve().parents[4]
CASES=[('seed0_d4conv_retest','projected_conv_d4','seed2_d4conv_priority'),
       ('seed0_d4sharing_retest','projected_d4','seed2_original_d4_remaining')]


@pytest.mark.parametrize('folder,variant,source',CASES)
def test_seed0_retest_keeps_frozen_tasks_and_optimised_model(folder,variant,source):
    p=REPO/'experiments'/folder
    manifest=json.loads((p/'manifest.json').read_text())
    frozen=json.loads((REPO/'experiments/seed2_40/manifest.json').read_text())
    assert manifest['task_ids']==frozen['task_ids'] and manifest['cohorts']==frozen['cohorts']
    assert len(manifest['jobs'])==len({j['task'] for j in manifest['jobs']})==40
    assert all(j['seed']==0 and j['variant']==variant for j in manifest['jobs'])
    cfg=load_config(p/f'configs/{variant}.json')
    before=load_config(REPO/'experiments'/source/f'configs/{variant}.json')
    assert cfg.model==before.model
    assert cfg.training.seed==0 and cfg.training.iterations==1500
    for key in cfg.training:
        if key!='seed':assert cfg.training[key]==before.training[key]
    assert cfg.logging.save_model is False and cfg.logging.log_config
    assert 'seed0-optimised-retest' in cfg.logging.wandb_tags
    assert cfg.experiment.comparison_role=='backend_retest'
    assert cfg.experiment.rerun_completed_tasks
    assert cfg.experiment.name!=before.experiment.name
    assert cfg.model.cummax_implementation=='triton_hard_grouped'
    assert cfg.model.direction_share_implementation==('d4' if variant=='projected_d4' else 'vectorized')


@pytest.mark.parametrize('folder,variant,source',CASES)
def test_seed0_worker_and_slurm_do_not_save_or_resume_checkpoints(folder,variant,source):
    p=REPO/'experiments'/folder
    spec=importlib.util.spec_from_file_location('seed0_worker',p/'run_job.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    for index in [0,39]:
        cmd,job,out=worker.command(p,index)
        assert job['seed']==0 and out.name=='seed_0'
        assert '--save-checkpoints' not in cmd and 'logging.save_model=false' in cmd
        assert '--resume' not in cmd
    with pytest.raises(ValueError):worker.command(p,40)
    slurm=(p/'submit.slurm').read_text()
    assert '--array=0-39%2' in slurm and '--gres=gpu:1' in slurm
    assert f'experiments/{folder}/run_job.py' in slurm
    assert 'seed2_' not in slurm
