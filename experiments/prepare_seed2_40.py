"""Prepare the fixed original + additional 40-task set with training seed 2."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_additional20 import WORKER, SBATCH, VARIANTS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--additional-dir',type=Path,default=Path(__file__).parent/'additional20')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'seed2_40')
    args=parser.parse_args()
    source=args.additional_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise SystemExit(f'Refusing to overwrite {out}')
    base=json.loads((source/'manifest.json').read_text())
    tasks=base['excluded_task_ids']+base['task_ids']
    assert len(tasks)==len(set(tasks))==40
    configs={}
    for variant in VARIANTS:
        cfg=json.loads((source/f'configs/{variant}.json').read_text())
        cfg['training']['seed']=2
        cfg['experiment'].update(name=f'{variant}_seed2_sample40',task_ids=tasks,seeds=[2])
        # This experiment reuses two frozen cohorts; there is no new 40-task draw.
        cfg['experiment'].pop('task_sample_seed',None)
        cfg['experiment']['task_cohorts']={'original20':base['excluded_task_ids'],'additional20':base['task_ids']}
        cfg['logging']['wandb_tags']=['final-ablation','sample40','seed2-rerun','steps1500','strict']
        configs[variant]=cfg
    jobs=[dict(index=i*4+j,task=t,variant=v,seed=2,config=f'configs/{v}.json') for i,t in enumerate(tasks) for j,v in enumerate(VARIANTS)]
    manifest=dict(training_seed=2,iterations=1500,split='training',task_selection='Reuse original20 and independently sampled additional20; no resampling or outcome filters',dataset_sha256=base['dataset_sha256'],source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),cohorts={'original20':base['excluded_task_ids'],'additional20':base['task_ids']},task_ids=tasks,jobs=jobs)
    out.mkdir(parents=True);(out/'configs').mkdir()
    for variant,cfg in configs.items():(out/f'configs/{variant}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    worker=WORKER.replace("'index must be 0..79'","'index must be 0..159'").replace("/'seed_0'","/f\"seed_{j['seed']}\"")
    (out/'run_job.py').write_text(worker)
    sbatch=SBATCH.replace('arc-additional20','arc-seed2-40').replace('--array=0-79%1','--array=0-159%1').replace('slurm-additional20','slurm-seed2-40').replace('experiments/additional20/run_job.py','experiments/seed2_40/run_job.py')
    (out/'submit.slurm').write_text(sbatch,newline='\n')
    (out/'README.md').write_text('''# Same 40 tasks, training seed 2

40 fixed tasks = the original 20 plus the independently sampled additional 20. No new sampling. Four variants, 1500 steps, seed 2 only: 160 runs. Same model/training controls as the additional20 package, apart from the training seed. Existing W&B project; `sample40` and `seed2-rerun` tags and distinct names.

Copy this directory to `experiments/seed2_40` in ARC-fafo. From its repository root:

```bash
sbatch experiments/seed2_40/submit.slurm
```

Array 0..159, one concurrent run, 12-hour limit per run, A40/gpu/normal/vf38, existing email alerts. Do not submit the older additional20 array as well unless you also intend to run that seed-0 batch.

Allocated GPU session, one run:

```bash
module load miniforge3
mamba activate SymARC
python experiments/seed2_40/run_job.py --index 0
```

Add `--dry-run` to print the command without training. The manifest maps all indices to task, variant and seed. Configurations and the task cohorts are frozen in this folder. Checkpoint directories use seed_2. No job is submitted by the preparation script.
''')
    print(json.dumps({'output':str(out),'tasks':len(tasks),'runs':len(jobs),'training_seed':2}))

if __name__=='__main__':main()
