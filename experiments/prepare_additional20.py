"""Freeze a uniform additional ARC training sample; never reads model results.

Usage: python prepare_additional20.py --repo-root /path/to/ARC-fafo
Creates experiments/additional20. Does not submit jobs.
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

PREVIOUS = '3eda0437 0dfd9992 95990924 7f4411dc 6f8cd79b 445eab21 3bdb4ada 36fdfd69 d43fd935 1c786137 1bfc4729 3aa6fb7a 6cdd2623 74dd1130 f8c80d96 623ea044 d406998b 6d75e8bb d9fac9be 97999447'.split()
VARIANTS = ('original', 'projected_conv', 'projected_d4', 'projected_lse')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sample-seed', type=int, default=43)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = (args.output_dir or Path(__file__).parent/'additional20').resolve()
    if out.exists():
        raise SystemExit(f'Refusing to overwrite an existing sample: {out}')
    dataset = repo/'src/dataset/arc-agi_training_challenges.json'
    raw = dataset.read_bytes()
    tasks = json.loads(raw)
    assert set(PREVIOUS) <= set(tasks), 'Expected previous task IDs absent from dataset'
    pool = sorted(set(tasks)-set(PREVIOUS))
    sample = random.Random(args.sample_seed).sample(pool, 20)
    # Freeze resolved copies, so later edits to base configs cannot change this batch.
    import yaml
    default = yaml.safe_load((repo/'src/config/default.yaml').read_text())
    def merge(a,b):
        result = dict(a)
        for k,v in b.items():
            result[k] = merge(result[k],v) if isinstance(v,dict) and isinstance(result.get(k),dict) else v
        return result
    configs = {}
    for variant in VARIANTS:
        base = yaml.safe_load((repo/f'src/config/models/final_{variant}_20.yaml').read_text())
        config = merge(default,base)
        config['training'].update(seed=0,iterations=1500,split='training')
        config['experiment'].update(name=f'{variant}_additional20',task_ids=sample,seeds=[0],task_sample_seed=args.sample_seed)
        config['logging']['wandb_tags'] = ['final-ablation','additional20','uniform-sample','steps1500','strict']
        configs[variant] = config
    for variant in VARIANTS:
        assert configs[variant]['training'] == configs['original']['training']
    jobs = [dict(index=i*4+j,task=t,variant=v,seed=0,config=f'configs/{v}.json') for i,t in enumerate(sample) for j,v in enumerate(VARIANTS)]
    manifest = dict(split='training',sampling='uniform without replacement from sorted task IDs; no model outcomes read',sampling_seed=args.sample_seed,training_seed=0,iterations=1500,dataset_sha256=hashlib.sha256(raw).hexdigest(),dataset_tasks=len(tasks),eligible_tasks=len(pool),excluded_task_ids=PREVIOUS,task_ids=sample,jobs=jobs)
    out.mkdir(parents=True)
    (out/'configs').mkdir()
    for variant,config in configs.items():
        (out/f'configs/{variant}.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'run_job.py').write_text(WORKER)
    (out/'submit.slurm').write_text(SBATCH, newline='\n')
    (out/'README.md').write_text('''# Additional 20 training tasks

Sampling seed 43 by default; uniform sampling without replacement from all training tasks excluding the previous 20. No performance, difficulty, shape, or runtime filters. All four variants use the same tasks, training seed 0, and 1500 steps. This sample should be frozen before viewing its model results.

This directory is tracked at `experiments/additional20` in ARC-fafo. From that repository root:

```bash
sbatch experiments/additional20/submit.slurm
```

This is an 80-element array with one concurrent run and a 12-hour limit per run. Existing A40/gpu/normal/vf38 settings and email alerts are retained. W&B uses the existing project, with an `additional20` tag and distinct run names. No jobs are submitted by the preparation script.

For one selected job in an allocated GPU session:

```bash
python experiments/additional20/run_job.py --index 0
```

Print a command without training:

```bash
python experiments/additional20/run_job.py --index 0 --dry-run
```

Do not substitute tasks after inspecting outcomes. This samples training tasks, not an evaluation split. GPU and numeric runtime differences can still affect results.
''')
    print(json.dumps({'output':str(out),'pool':len(pool),'tasks':sample,'runs':len(jobs)},indent=2))

WORKER = '''import argparse,json,os,subprocess,sys,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
folder=Path(__file__).resolve().parent
manifest=json.loads((folder/'manifest.json').read_text())
if not 0<=a.index<len(manifest['jobs']):p.error('index must be 0..79')
roots=[Path.cwd(),*folder.parents]
repo=next((r for r in roots if (r/'src/train.py').is_file()),None)
if repo is None:raise SystemExit('Run from ARC-fafo root, or place this folder inside ARC-fafo')
j=manifest['jobs'][a.index]
out=repo/'outputs'/folder.name/'checkpoints'/os.environ.get('SLURM_ARRAY_JOB_ID','manual')/j['task']/j['variant']/'seed_0'
cmd=[sys.executable,str(repo/'src/train.py'),'--config',str(folder/j['config']),'--task',j['task'],'--set','training.device=cuda:0','--save-checkpoints','--output-dir',str(out)]
print(shlex.join(cmd),flush=True)
if not a.dry_run:subprocess.run(cmd,cwd=repo,check=True)
'''

SBATCH = '''#!/bin/bash -l
#SBATCH --job-name=arc-additional20
#SBATCH --partition=gpu
#SBATCH --account=vf38
#SBATCH --qos=normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-79%1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teedsingyau@gmail.com
#SBATCH --output=slurm-additional20-%A_%a.out
#SBATCH --error=slurm-additional20-%A_%a.err
set -euo pipefail
# Submit from the ARC-fafo root; ignore any inherited stale submit directory.
if [[ ! -f src/train.py ]]; then
    printf 'Submit from the ARC-fafo repository root.\\n' >&2
    exit 1
fi
module load miniforge3
mamba activate SymARC
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
srun python experiments/additional20/run_job.py --index "$SLURM_ARRAY_TASK_ID"
'''

if __name__ == '__main__':
    main()
