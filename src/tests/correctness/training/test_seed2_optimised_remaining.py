"""Pending seed2 jobs preserve frozen cohorts and skip completion, never accuracy."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from config import load_config
from compressarc.train.logging import initialize_wandb

REPO=Path(__file__).resolve().parents[4]
PACKAGE=REPO/'experiments/seed2_40_optimised_remaining'
spec=importlib.util.spec_from_file_location('prepare_optimised',REPO/'experiments/prepare_seed2_40_optimised.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def test_completion_selection_deduplicates_and_ignores_accuracy():
    def run(rid,state='finished',step=1499,tag='projected-d4',name='3eda0437-training-run-seed2'):
        return dict(id=rid,name=name,state=state,tags=[tag],summary={'train_step':step,'top_1_correct':False})
    rows=[run('first'),run('duplicate'),run('short',step=39,tag='projected-lse'),
          run('wrong_seed',name='3eda0437-training-run-seed0'),
          run('wrong_split',name='3eda0437-evaluation-run-seed2'),
          run('running',state='running',step=100,tag='D4-conv'),run('c4',tag='C4')]
    done,active=prepare.completed(rows,{'3eda0437'})
    assert done['projected_d4']=={'3eda0437':['first','duplicate']}
    assert done['projected_lse']=={} and done['projected_conv_d4']=={}
    assert active['projected_conv_d4']=={'3eda0437':['running']}


def test_frozen_remaining_manifest_has_expected_unique_pairs_and_cohorts():
    manifest=json.loads((PACKAGE/'manifest.json').read_text())
    original=json.loads((REPO/'experiments/seed2_40/manifest.json').read_text())
    assert manifest['task_ids']==original['task_ids'] and manifest['cohorts']==original['cohorts']
    assert len(manifest['jobs'])==86
    assert len({(j['task'],j['variant'],j['seed']) for j in manifest['jobs']})==86
    assert {v:s['remaining'] for v,s in manifest['counts'].items()}=={
        'projected_conv_d4':40,'projected_d4':23,'projected_lse':23}
    for j in manifest['jobs']:
        assert j['seed']==2 and j['task'] not in manifest['completed'][j['variant']]
        cfg=load_config(PACKAGE/j['config'])
        assert cfg.training.iterations==1500 and cfg.training.seed==2
        assert cfg.training.split=='training' and not cfg.logging.save_model
        assert cfg.logging.log_config and cfg.logging.wandb
        assert cfg.model.share_implementation=='reuse_fused_norm'
        if j['variant']=='projected_lse':assert cfg.model.cummax_implementation=='optimised'
        else:assert cfg.model.cummax_implementation=='triton_hard_grouped'
        if j['variant']=='projected_d4':assert cfg.model.direction_share_implementation=='d4'
        if j['variant']=='projected_conv_d4':assert cfg.model.shift_implementation=='d4_tied_conv'


def test_worker_never_requests_checkpoints_or_changes_seed():
    spec=importlib.util.spec_from_file_location('remaining_worker',PACKAGE/'run_job.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    cmd,job,out=worker.command(PACKAGE,0)
    assert '--save-checkpoints' not in cmd and 'logging.save_model=false' in cmd
    assert '--config' in cmd and '--task' in cmd and job['seed']==2
    assert out.name=='seed_2'
    with pytest.raises(ValueError):worker.command(PACKAGE,86)
    sbatch=(PACKAGE/'submit.slurm').read_text()
    assert '--array=0-85%2' in sbatch and '--gres=gpu:1' in sbatch


def test_opt_in_wandb_configuration_contains_effective_backends_and_commit(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    fake=SimpleNamespace(define_metric=lambda *a,**kw:None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **kw:calls.append(kw) or fake,Settings=lambda **kw:kw))
    cfg=load_config(PACKAGE/'configs/projected_lse.json',['experiment.code_commit=test-commit'])
    initialize_wandb(cfg,SimpleNamespace(task_name='3eda0437'))
    saved=calls[0]['config']
    assert saved['task_id']=='3eda0437' and saved['training']['seed']==2
    assert saved['model']['cummax_implementation']=='optimised'
    assert saved['experiment']['code_commit']=='test-commit'
    assert saved['logging']['save_model'] is False


def test_priority_packages_exclude_lse_and_run_d4conv_before_controls():
    root=REPO/'experiments'
    a=json.loads((root/'seed2_d4conv_priority/manifest.json').read_text())
    b=json.loads((root/'seed2_original_d4_remaining/manifest.json').read_text())
    assert len(a['jobs'])==40 and {j['variant'] for j in a['jobs']}=={'projected_conv_d4'}
    assert len(b['jobs'])==45 and {j['variant'] for j in b['jobs']}=={'original','projected_d4'}
    assert b['counts']['original']['remaining']==22
    assert b['counts']['projected_d4']['remaining']==23
    for manifest in (a,b):
        assert manifest['task_ids']==json.loads((PACKAGE/'manifest.json').read_text())['task_ids']
        for job in manifest['jobs']:
            assert job['task'] not in manifest['completed'][job['variant']]
    assert '--array=0-39%2' in (root/'seed2_d4conv_priority/submit.slurm').read_text()
    assert '--array=0-44%2' in (root/'seed2_original_d4_remaining/submit.slurm').read_text()
    helper=(root/'submit_seed2_priority.sh').read_text()
    assert '--dependency="afterany:${d4_job}"' in helper
    assert 'projected_lse' not in helper and 'seed2_40_optimised_remaining/submit.slurm' not in helper
