"""Prepare one combined D4-convolution + D4-direction-sharing seed0 model."""
import hashlib,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]


def main():
    folder=REPO/'experiments/seed0_combined_d4'
    if folder.exists():raise FileExistsError('Refusing to overwrite '+str(folder))
    source=REPO/'experiments/seed2_40/manifest.json'
    frozen=json.loads(source.read_text());tasks=frozen['task_ids']
    base=REPO/'experiments/seed0_d4conv_retest'
    cfg=json.loads((base/'configs/projected_conv_d4.json').read_text())
    cfg['model']['direction_share_implementation']='d4'
    assert cfg['model']['shift_implementation']=='d4_tied_conv'
    cfg['logging']['wandb_tags']=['final-ablation','sample40','seed0-combined-d4',
        'combined-d4-conv-sharing','optimised-backends','steps1500','strict']
    cfg['experiment'].update(name='combined_d4_conv_sharing_seed0_sample40',
        variant='combined-d4-conv-sharing',comparison_role='combined_ablation',
        parent_variants=['projected_conv_d4','projected_d4'],
        variable_operators=['shift','direction_share'])
    cfg['experiment'].pop('rerun_completed_tasks',None)
    assert cfg['training']['seed']==0 and cfg['training']['iterations']==1500
    assert not cfg['logging']['save_model']
    folder.mkdir();(folder/'configs').mkdir()
    (folder/'configs/combined_d4.json').write_text(json.dumps(cfg,indent=2)+'\n')
    jobs=[dict(index=i,task=task,variant='combined_d4_conv_sharing',seed=0,
               config='configs/combined_d4.json',
               cohort='original20' if task in frozen['cohorts']['original20'] else 'additional20')
          for i,task in enumerate(tasks)]
    manifest=dict(training_seed=0,iterations=1500,split='training',task_ids=tasks,
        cohorts=frozen['cohorts'],dataset_sha256=frozen['dataset_sha256'],
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        architecture='Both D4 convolution and projected D4 direction sharing in the same model',
        selection='Same frozen 40 tasks; no accuracy filtering',jobs=jobs)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    worker=(base/'run_job.py').read_text().replace('Run one fresh seed-0 backend retest;',
        'Run one combined D4 seed-0 model;')
    (folder/'run_job.py').write_text(worker)
    sbatch=(base/'submit.slurm').read_text().replace('arc-d4conv-s0','arc-combined-d4-s0').replace('seed0_d4conv_retest','seed0_combined_d4')
    (folder/'submit.slurm').write_text(sbatch,newline='\n')
    (folder/'README.md').write_text('''# Combined D4-conv + D4 direction sharing, seed0

ONE model uses shift_implementation=d4_tied_conv AND
 direction_share_implementation=d4. This is a new combined ablation, not an
ensemble, not two independent variants, and not fused up/down communication.
Both operators are learned jointly at all four depths. Hard cummax, fused
softmax, reused/fused-normalisation sharing, fused Adam, direct checks and explicit
runtime dispatch retain the established optimised settings.

40 fixed training tasks, pilot20 + additional20, 1500 steps, seed0. One array
0-39%2, one GPU per element, at most two concurrent. No model weights/checkpoints
are saved. Normal W&B metrics, candidate tables, effective config and code commit
are retained under variant combined-d4-conv-sharing and tag seed0-combined-d4.
This is separate from the single-operator seed0 results.

From repository root:

    sbatch experiments/seed0_combined_d4/submit.slurm

Dry-run one task:

    python experiments/seed0_combined_d4/run_job.py --index 0 --dry-run

Inside an allocated GPU, run the combined architecture correctness test first:

    python -m pytest src/tests/correctness/training/test_combined_d4_model.py -q -x

Then omit --dry-run to execute just index0 if a one-task check is desired.
Do not also submit the two independent retest arrays unless those additional
experiments are intended. No full-model D4-equivariance claim is established by
selecting these two operators. Accuracy and runtime of the combination must be
measured; they cannot be inferred by adding the separate variants' results.
''')
    print(json.dumps(dict(package=str(folder),runs=len(jobs),seed=0,combined_model=True)))

if __name__=='__main__':main()
