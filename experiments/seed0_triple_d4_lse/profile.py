"""Paired full-training latency and a separate operator trace; never launches the 40-task batch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys


def plan(folder, index, device, output):
    manifest=json.loads((folder/'profile_manifest.json').read_text())
    if not 0 <= index < len(manifest['tasks']):
        raise ValueError('Profile index must be 0, 1 or 2')
    task=manifest['tasks'][index]['task']
    repo=folder.parents[1]
    jobs=[]
    for repeat in range(manifest['repeats']):
        variants=manifest['variants'] if repeat % 2 == 0 else list(reversed(manifest['variants']))
        for variant in variants:
            dest=output/task/variant/f'repeat_{repeat}'
            overrides=[f'training.device={device}','training.seed=0',
                f"training.iterations={manifest['warmup_steps']+manifest['measured_steps']}",
                'training.step_timing=wall','training.operator_profile=false',
                'training.sharing_stage_timing=false','logging.save_model=false',
                'logging.wandb_tags=[latency-only,triple-d4-lse-profile]',
                f'experiment.name=triple_latency_{variant}_repeat{repeat}']
            cmd=[sys.executable,str(repo/'src/train.py'),'--task',task,'--config',str(folder/f'configs/{variant}.json'),'--output-dir',str(dest)]
            for value in overrides: cmd.extend(['--set',value])
            jobs.append(dict(kind='wall',variant=variant,repeat=repeat,output=str(dest),command=cmd))
    dest=output/task/'operator_trace'
    cmd=[sys.executable,str(repo/'src/train.py'),'--task',task,'--config',str(folder/'configs/triple_d4_lse.json'),'--output-dir',str(dest)]
    for value in [f'training.device={device}','training.seed=0',f"training.iterations={manifest['warmup_steps']+1}",
        'training.step_timing=phases','training.operator_profile=true',f"training.operator_profile_start={manifest['warmup_steps']}",
        'training.operator_profile_steps=1','training.sharing_stage_timing=true','logging.save_model=false',
        'logging.wandb_tags=[latency-only,triple-d4-lse-profile,operator-profile]',
        'experiment.name=triple_latency_operator_trace']:
        cmd.extend(['--set',value])
    jobs.append(dict(kind='trace',output=str(dest),command=cmd))
    return manifest,task,jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    folder=Path(__file__).resolve().parent
    repo=folder.parents[1]
    output=args.output_dir or repo/'outputs'/folder.name/'latency'/os.environ.get('SLURM_ARRAY_JOB_ID','manual')
    manifest,task,jobs=plan(folder,args.index,args.device,output)
    for job in jobs: print(shlex.join(job['command']),flush=True)
    if args.dry_run: return
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('This latency experiment requires a CUDA GPU; CPU timings are not comparable.')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
    props=torch.cuda.get_device_properties(args.device)
    metadata=dict(task=task,seed=0,code_commit=commit,host=platform.node(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=props.name,gpu_memory_bytes=props.total_memory,device=args.device,manifest=manifest,jobs=jobs,
        config_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'configs').glob('*.json')})
    dest=output/task
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'execution.json').write_text(json.dumps(metadata,indent=2))
    results=[]
    for job in jobs:
        command=job['command']+['--set','experiment.code_commit='+commit]
        subprocess.run(command,cwd=repo,check=True)
        if job['kind']!='wall': continue
        rows=[json.loads(line) for line in (Path(job['output'])/f'{task}.step_timing.jsonl').read_text().splitlines()]
        values=[row['timing/step_seconds'] for row in rows[manifest['warmup_steps']:]]
        if len(values)!=manifest['measured_steps'] or any(not 0 < value < float('inf') for value in values):
            raise RuntimeError('Incomplete or invalid latency measurement')
        results.append(dict(variant=job['variant'],repeat=job['repeat'],steps=len(values),
            mean_ms=1000*statistics.mean(values),median_ms=1000*statistics.median(values),
            p90_ms=1000*sorted(values)[int(.9*(len(values)-1))]))
        (dest/'wall_results.json').write_text(json.dumps(results,indent=2))
    means={variant:statistics.mean(r['mean_ms'] for r in results if r['variant']==variant) for variant in manifest['variants']}
    report=dict(task=task,metadata=metadata,repeats=results,mean_ms=means,
        triple_over_hard_ratio=means['triple_d4_lse']/means['hard_reference'],
        estimated_1500_step_seconds={key:value*1.5 for key,value in means.items()},
        note='Full training loop includes candidates, numerical diagnostics and W&B logging; excludes timing export. First 20 steps omitted. No operator instrumentation in wall passes. One separate instrumented step for attribution. 1500-step extrapolation excludes setup and may change later in training; not a 40-task runtime estimate.')
    (dest/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'},indent=2),flush=True)


if __name__=='__main__': main()
