"""Prepare unfinished selected seed-2 runs from the frozen 40-task manifest."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from benchmark_variants import VARIANTS,compatible_overrides
from config import load_config
from omegaconf import OmegaConf
SELECTED=('projected_conv_d4','projected_d4','projected_lse')
TAGS={'original':'original','projected_conv_d4':'D4-conv','projected_d4':'projected-d4','projected_lse':'projected-lse'}


def completed(rows,tasks,selected=SELECTED):
    found={variant:{} for variant in selected}
    active={variant:{} for variant in selected}
    for row in rows:
        match=re.match(r'^([0-9a-f]{8})-training-.*-seed2$',row.get('name',''))
        if not match or match[1] not in tasks:continue
        if any(t in row.get('tags',[]) for t in ('variant-latency','operator-profile')):continue
        for variant in selected:
            tag=TAGS[variant]
            if tag not in row.get('tags',[]):continue
            if row['state']=='finished' and (row.get('summary',{}).get('train_step') or -1)>=1499:
                found[variant].setdefault(match[1],[]).append(row['id'])
            elif row['state']=='running':
                active[variant].setdefault(match[1],[]).append(row['id'])
    return found,active


def fetch_audit():
    import wandb
    rows=[]
    for run in wandb.Api().runs('arc_agi/compressarc-final-ablation',
              filters={'$or':[{'tags':'seed2'},{'displayName':{'$regex':'seed2'}}]},per_page=100):
        rows.append(dict(id=run.id,name=run.name,state=run.state,tags=run.tags,
                         summary={key:run.summary.get(key) for key in ['train_step','run/status']}))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lse-backend',choices=['optimised','triton'],default='optimised')
    p.add_argument('--variants',nargs='+',choices=['original',*SELECTED],default=list(SELECTED))
    p.add_argument('--audit',type=Path,help='Use a saved W&B audit; otherwise query live W&B')
    p.add_argument('--output-dir',type=Path,default=REPO/'experiments/seed2_40_optimised_remaining')
    args=p.parse_args(argv)
    selected=list(dict.fromkeys(args.variants))
    if args.output_dir.exists():raise FileExistsError('Use a new output directory: '+str(args.output_dir))
    source=REPO/'experiments/seed2_40/manifest.json'
    original=json.loads(source.read_text())
    tasks=original['task_ids']
    assert len(tasks)==len(set(tasks))==40 and original['training_seed']==2 and original['iterations']==1500
    rows=json.loads(args.audit.read_text()) if args.audit else fetch_audit()
    rows=[dict(id=r['id'],name=r['name'],state=r['state'],tags=r.get('tags',[]),
               summary={k:r.get('summary',{}).get(k) for k in ('train_step','run/status')}) for r in rows]
    done,active=completed(rows,set(tasks),selected)
    folder=args.output_dir;folder.mkdir(parents=True);(folder/'configs').mkdir()
    configs={}
    for variant in selected:
        path=REPO/VARIANTS[variant];base=load_config(path)
        cfg=load_config(path,compatible_overrides(base,'accelerated')+[
            'training.seed=2','training.iterations=1500','training.split=training',
            'training.step_timing=wall','logging.save_model=false'])
        if variant=='projected_lse':
            cfg.model.cummax_implementation=args.lse_backend
        cfg=OmegaConf.to_container(cfg,resolve=True)
        cfg['experiment'].update(name=f'{variant}_seed2_optimised40',task_ids=tasks,seeds=[2],
                                 task_cohorts=original['cohorts'],backend_regime='optimised',
                                 resumed_from_model_checkpoint=False)
        if variant=='projected_lse' and args.lse_backend=='triton':
            cfg['experiment']['name']='projected_lse_seed2_triton_optimised40'
        cfg['experiment'].pop('task_sample_seed',None)
        cfg['logging']['log_config']=True
        cfg['logging']['wandb_tags']=['final-ablation','sample40','seed2-rerun','seed2-optimised-remaining',
                                       'optimised-backends','steps1500','strict',variant]
        if variant=='projected_lse' and args.lse_backend=='triton':
            cfg['logging']['wandb_tags']+=['seed2-lse-triton-remaining','lse-triton']
        configs[variant]=cfg
        (folder/f'configs/{variant}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    jobs=[]
    pending={variant:[task for task in tasks if task not in done[variant] and task not in active[variant]]
             for variant in selected}
    # Interleave selected pending queues while retaining each frozen task order.
    for offset in range(max((len(q) for q in pending.values()),default=0)):
        for variant in selected:
            if offset>=len(pending[variant]):continue
            task=pending[variant][offset]
            jobs.append(dict(index=len(jobs),task=task,variant=variant,seed=2,
                             cohort='original20' if task in original['cohorts']['original20'] else 'additional20',
                             config=f'configs/{variant}.json'))
    manifest=dict(generated_utc=datetime.now(timezone.utc).isoformat(),training_seed=2,iterations=1500,
                  split='training',task_ids=tasks,cohorts=original['cohorts'],
                  dataset_sha256=original['dataset_sha256'],
                  source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  task_selection='Same frozen 40 tasks; selected variants only; skip verified completed/active pairs, never filter by accuracy',
                  completed=done,active=active,jobs=jobs,
                  counts={v:dict(completed=len(done[v]),active=len(active[v]),
                                 remaining=sum(j['variant']==v for j in jobs)) for v in selected})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'completion_evidence.json').write_text(json.dumps(rows,indent=2)+'\n')
    worker='''"""Run one frozen pending job; no model/checkpoint weights are saved."""
import argparse,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path

def command(folder,index,device='cuda:0'):
    manifest=json.loads((folder/'manifest.json').read_text())
    if not 0<=index<len(manifest['jobs']):raise ValueError('Index outside remaining manifest')
    job=manifest['jobs'][index]
    repo=folder.parents[1]
    output=repo/'outputs'/folder.name/os.environ.get('SLURM_ARRAY_JOB_ID','manual')/job['task']/job['variant']/'seed_2'
    cmd=[sys.executable,str(repo/'src/train.py'),'--config',str(folder/job['config']),
         '--task',job['task'],'--set','training.device='+device,
         '--set','logging.save_model=false','--output-dir',str(output)]
    return cmd,job,output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);folder=Path(__file__).resolve().parent
    cmd,job,output=command(folder,a.index,a.device)
    repo=folder.parents[1]
    print(shlex.join(cmd),flush=True)
    if a.dry_run:return cmd
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
    output.mkdir(parents=True,exist_ok=True)
    provenance=dict(job=job,code_commit=commit,config_sha256=hashlib.sha256((folder/job['config']).read_bytes()).hexdigest(),
                    model_saving=False,checkpoint_resume=False,command=cmd)
    (output/'execution.json').write_text(json.dumps(provenance,indent=2))
    print('Code commit: '+commit,flush=True)
    cmd+=['--set','experiment.code_commit='+commit]
    subprocess.run(cmd,cwd=repo,check=True)

if __name__=='__main__':main()
'''
    (folder/'run_job.py').write_text(worker)
    if jobs:
        sbatch=f'''#!/bin/bash -l
#SBATCH --job-name=arc-seed2-opt
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-{len(jobs)-1}%2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --output=slurm-seed2-opt-%A_%a.out
#SBATCH --error=slurm-seed2-opt-%A_%a.err
set -euo pipefail
[[ -f src/train.py ]] || {{ printf 'Submit from ARC-fafo root.\\n' >&2; exit 1; }}
module load miniforge3
mamba activate SymARC
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONUNBUFFERED=1
srun python experiments/{folder.name}/run_job.py --index "$SLURM_ARRAY_TASK_ID"
'''
        (folder/'submit.slurm').write_text(sbatch,newline='\n')
    readme=f'''# Remaining optimised seed-2 runs

{len(jobs)} pending runs over the same fixed 40 tasks. Counts:

{json.dumps(manifest['counts'],indent=2)}

Complete means W&B finished and final logged train_step >=1499, matched by task,
training split, seed2 and variant tag. Duplicate run IDs are retained as evidence
but each task is counted once. Accuracy is not used. Only the variants listed in the counts above are scheduled. Active runs are recorded separately and excluded.

New runs use compatible optimised settings, 1500 steps, seed2, ordinary metrics
and candidate logging. LSE retains learned LSE math with the selected backend recorded in config; projected D4
sharing retains its d4 operator; hard-cummax variants use grouped hard Triton.
No weights/checkpoints are saved or uploaded. This restarts unfinished runs from
initialisation, not deleted checkpoints. Legacy completed runs keep their older
backends; report that provenance when combining results for the paper.

Submit from repository root: `sbatch experiments/{folder.name}/submit.slurm`.
One GPU per array element; at most two elements run concurrently. Any GPU in the
gpu partition; normal/vf38, 4 CPUs, 16 GB RAM, 12 hours per run, existing email.
Keep the old seed2_40 array held to avoid launching overlapping pending jobs.
No Slurm submission or cancellation was performed by the preparation script.

Dry-run one command: `python experiments/{folder.name}/run_job.py --index 0 --dry-run`.
Inside an existing GPU allocation, omit `--dry-run` to run that one pair.
Use a fresh package via prepare_seed2_40_optimised.py if the completion state changes;
the manifest is a snapshot, not a live scheduler. Execution JSON stores the code
commit and config digest locally. Source configs and both cohorts are preserved.
'''
    (folder/'README.md').write_text(readme)
    print(json.dumps(dict(output=str(folder),pending=len(jobs),counts=manifest['counts']),indent=2))
    return manifest

if __name__=='__main__':main()
