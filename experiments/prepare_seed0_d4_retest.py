"""Prepare two fresh optimised seed-0 rerun arrays on the frozen 40 tasks."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PAIRS=[('projected_conv_d4','seed0_d4conv_retest','seed2_d4conv_priority','arc-d4conv-s0'),
       ('projected_d4','seed0_d4sharing_retest','seed2_original_d4_remaining','arc-d4share-s0')]


def main():
    source=REPO/'experiments/seed2_40/manifest.json'
    frozen=json.loads(source.read_text());tasks=frozen['task_ids']
    assert len(tasks)==len(set(tasks))==40
    for _,name,_,_ in PAIRS:
        if (REPO/'experiments'/name).exists():
            raise FileExistsError('Refusing to overwrite existing rerun package '+name)
    template=REPO/'experiments/seed2_d4conv_priority'
    worker=(template/'run_job.py').read_text().replace(
        '"""Run one frozen pending job; no model/checkpoint weights are saved."""',
        '"""Run one fresh seed-0 backend retest; no model/checkpoint weights saved."""').replace(
        "/'seed_2'", "/f\"seed_{job['seed']}\"")
    for variant,name,source_name,label in PAIRS:
        folder=REPO/'experiments'/name;folder.mkdir();(folder/'configs').mkdir()
        base_path=REPO/'experiments'/source_name/f'configs/{variant}.json'
        cfg=json.loads(base_path.read_text())
        cfg['training']['seed']=0
        cfg['training']['iterations']=1500
        cfg['logging'].update(save_model=False,log_config=True)
        cfg['logging']['wandb_tags']=['final-ablation','sample40','seed0-optimised-retest',
                                    'optimised-backends','steps1500','strict',variant]
        cfg['experiment'].update(name=f'{variant}_seed0_optimised_retest40',task_ids=tasks,
                                 seeds=[0],task_cohorts=frozen['cohorts'],
                                 backend_regime='optimised',comparison_role='backend_retest',
                                 rerun_completed_tasks=True,resumed_from_model_checkpoint=False)
        jobs=[dict(index=i,task=task,variant=variant,seed=0,config=f'configs/{variant}.json',
                   cohort='original20' if task in frozen['cohorts']['original20'] else 'additional20')
              for i,task in enumerate(tasks)]
        manifest=dict(training_seed=0,iterations=1500,split='training',task_ids=tasks,
                      cohorts=frozen['cohorts'],dataset_sha256=frozen['dataset_sha256'],
                      source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      source_optimised_config=str(base_path.relative_to(REPO)).replace('\\','/'),
                      source_optimised_config_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                      purpose='Fresh optimised-backend retest on all fixed 40 tasks, including already-completed legacy seed0 tasks; no accuracy filtering',
                      comparison_role='backend_retest',jobs=jobs)
        (folder/f'configs/{variant}.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (folder/'run_job.py').write_text(worker)
        slurm=(template/'submit.slurm').read_text().replace('arc-d4conv-s2',label).replace('seed2_d4conv_priority',name)
        (folder/'submit.slurm').write_text(slurm,newline='\n')
        (folder/'README.md').write_text(f'''# Optimised seed-0 retest: {variant}

40 fresh runs, 1500 steps each, seed0. The same frozen pilot20 + additional20
cohorts are retained. Already completed legacy seed0 tasks are deliberately
rerun to test the optimised implementation. No task is selected by accuracy.

Submission: `sbatch experiments/{name}/submit.slurm` from ARC-fafo root.
Array 0-39%2; one GPU per element, at most two running concurrently.
Submitting both packages permits four concurrent GPU runs in total.

W&B name suffix: `{variant}_seed0_optimised_retest40-seed0`.
W&B tag: `seed0-optimised-retest`. Original completed runs are not changed or
removed. Keep legacy and retest results separate when assessing backend effects;
do not select the best run by outcome. No model/checkpoint weights are saved.
Effective config and code commit are logged; execution JSON is saved locally.

Dry-run: `python experiments/{name}/run_job.py --index 0 --dry-run`.
Omit `--dry-run` only inside an allocated GPU session to run that one task.
''')
        print(json.dumps(dict(package=name,variant=variant,runs=len(jobs),seed=0)))

if __name__=='__main__':main()
