"""The LSE first test uses learned Triton and excludes finished/duplicate jobs."""
import importlib.util
import json
from pathlib import Path
from config import load_config

REPO=Path(__file__).resolve().parents[4]
FOLDER=REPO/'experiments/seed2_lse_triton_remaining'


def test_lse_package_preserves_learned_math_and_pending_tasks():
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    cfg=load_config(FOLDER/'configs/projected_lse.json')
    old=load_config(REPO/'experiments/seed2_40_optimised_remaining/configs/projected_lse.json')
    assert len(manifest['jobs'])==23 and len(manifest['completed']['projected_lse'])==17
    assert manifest['jobs'][0]['task']=='6d75e8bb'
    assert all(j['seed']==2 and j['task'] not in manifest['completed']['projected_lse'] for j in manifest['jobs'])
    assert cfg.training==old.training
    assert cfg.model.cummax_implementation=='triton' and cfg.model.lse_tau==.1
    for key in cfg.model:
        if key!='cummax_implementation':assert cfg.model[key]==old.model[key]
    assert cfg.logging.save_model is False and cfg.logging.log_config
    assert 'seed2-lse-triton-remaining' in cfg.logging.wandb_tags


def test_first_job_and_remainder_do_not_overlap_or_save_weights():
    first=(FOLDER/'submit_first.slurm').read_text()
    rest=(FOLDER/'submit_remaining.slurm').read_text()
    assert '#SBATCH --array=' not in first and '--index 0' in first
    assert '--array=1-22%2' in rest
    spec=importlib.util.spec_from_file_location('lse_worker',FOLDER/'run_job.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    command,job,out=worker.command(FOLDER,0)
    assert job['task']=='6d75e8bb' and out.name=='seed_2'
    assert '--save-checkpoints' not in command and 'logging.save_model=false' in command
