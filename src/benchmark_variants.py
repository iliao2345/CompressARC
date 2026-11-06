"""Clean reference/accelerated latency comparison without changing variant operators."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from config import load_config

REPO=Path(__file__).resolve().parents[1]
VARIANTS={
    'original':'src/config/models/final_original_20.yaml',
    'projected_conv_c4':'src/config/models/final_projected_conv_20.yaml',
    'projected_conv_d4':'experiments/d4_conv_pilot20/configs/projected_conv_d4.json',
    'projected_d4':'src/config/models/final_projected_d4_20.yaml',
    'projected_lse':'src/config/models/final_projected_lse_20.yaml',
}


def compatible_overrides(config, mode):
    if mode not in ('reference','accelerated'):
        raise ValueError('mode must be reference or accelerated')
    result=['logging.save_model=false','model.layer_timing=false',
            'training.operator_profile=false','training.sharing_stage_timing=false']
    if mode=='reference':
        return result+['model.share_implementation=primitives',
                       'model.softmax_implementation=primitives',
                       'training.optimizer_implementation=default',
                       'training.diagnostics_implementation=reference',
                       'training.runtime_device_dispatch=global']
    result += ['model.share_implementation=reuse_fused_norm',
               'model.softmax_implementation=fused',
               'training.optimizer_implementation=fused',
               'training.diagnostics_implementation=triton',
               'training.runtime_device_dispatch=explicit']
    if config.model.direction_share_implementation=='primitives':
        result.append('model.direction_share_implementation=vectorized')
    if config.model.cummax_implementation=='primitives':
        result.append('model.cummax_implementation=triton_hard_grouped')
    # Shift always stays unchanged. Learned LSE and projected D4 sharing stay unchanged.
    return result


def make_plan(variants,modes,tasks,split,seed,device,root,clean_steps,warmup):
    plan=[]
    for task in tasks:
        for variant in variants:
            path=REPO/VARIANTS[variant]
            config=load_config(path)
            for mode in modes:
                overrides=compatible_overrides(config,mode)+[
                    f'training.split={split}',f'training.seed={seed}']
                tags=list(config.logging.get('wandb_tags',[]))+['variant-latency',variant,mode]
                overrides.append('logging.wandb_tags='+json.dumps(tags))
                output=root/task/variant/mode
                command=[sys.executable,str(REPO/'src/profile_layers.py'),'--config',str(path),
                         '--task',task,'--device',device,'--clean-only',
                         '--run-label',f'latency_{variant}_{mode}',
                         '--clean-steps',str(clean_steps),'--warmup-steps',str(warmup),
                         '--output-dir',str(output)]
                for override in overrides:command+=['--set',override]
                effective=load_config(path,overrides)
                from omegaconf import OmegaConf
                plan.append(dict(variant=variant,mode=mode,task=task,split=split,seed=seed,
                                 command=command,output=str(output),
                                 model=OmegaConf.to_container(effective.model,resolve=True)))
    return plan


def comparison_rows(results):
    reference={(r['variant'],r['task']):r['mean_seconds'] for r in results
               if r['mode']=='reference' and r['status']=='finished'}
    rows=[]
    for result in results:
        row=dict(result)
        baseline=reference.get((row['variant'],row['task']))
        row['speedup_vs_reference']=baseline/row['mean_seconds'] if baseline and row['status']=='finished' else None
        rows.append(row)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--modes',nargs='+',choices=['reference','accelerated'],default=['reference','accelerated'])
    p.add_argument('--tasks',nargs='+',default=['00576224'])
    p.add_argument('--split',choices=['training','evaluation'],default='evaluation')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--clean-steps',type=int,default=40)
    p.add_argument('--warmup-steps',type=int,default=10)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/variant_latency_comparison'))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--upload-wandb',action='store_true')
    args=p.parse_args(argv)
    if args.warmup_steps<1 or args.clean_steps<=args.warmup_steps:
        p.error('clean-steps must exceed positive warmup-steps')
    root=args.output_dir.resolve()
    plan=make_plan(list(dict.fromkeys(args.variants)),list(dict.fromkeys(args.modes)),
                   list(dict.fromkeys(args.tasks)),args.split,args.seed,args.device,root,
                   args.clean_steps,args.warmup_steps)
    if args.dry_run:
        for item in plan:
            print(item['variant'],item['mode'],subprocess.list2cmdline(item['command']))
        return dict(plan=plan)
    if root.exists():
        raise FileExistsError('Use a fresh output directory: '+str(root))
    # Hardware probe exits before any timed worker starts; no parent CUDA context.
    code='''import torch,json,sys
x=torch.device(sys.argv[1])
if x.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA required for compatible accelerated comparison')
print(json.dumps(dict(device=str(x),gpu=torch.cuda.get_device_name(x),torch=str(torch.__version__),cuda=torch.version.cuda)))'''
    hardware=json.loads(subprocess.run([sys.executable,'-c',code,args.device],capture_output=True,text=True,check=True).stdout)
    root.mkdir(parents=True)
    report=dict(hardware=hardware,plan=plan,results=[],
                caveats=['Reference and accelerated modes preserve each variant architecture; floating-point reduction/optimiser order can differ.',
                         'Short timing runs do not establish identical training trajectories or task accuracy.',
                         'Same task/seed/GPU, sequential independent workers; setup/compilation excluded via warmup.',
                         'Operator traces and checkpoints are disabled. Timed training retains metrics, diagnostics and candidate tracking.',
                         'Each run is one measurement; repeat and use larger tasks before extrapolating total runtime.'])
    results=[]
    for item in plan:
        print(f"Benchmarking {item['task']} {item['variant']} {item['mode']}",flush=True)
        row={k:item[k] for k in ('variant','mode','task')}
        try:
            subprocess.run(item['command'],cwd=REPO,check=True)
            data=json.loads((Path(item['output'])/'comparison.json').read_text())
            phases=data['clean_phase_means']
            row.update(status='finished',mean_seconds=data['clean_mean_seconds'],
                       median_seconds=data['clean_median_seconds'],
                       forward_seconds=phases.get('timing/host_forward_seconds'),
                       backward_seconds=phases.get('timing/host_backward_seconds'),
                       optimizer_seconds=phases.get('timing/host_optimizer_seconds'))
        except (subprocess.CalledProcessError,OSError,ValueError,KeyError) as error:
            row.update(status='failed',error=str(error))
        results.append(row)
        report['results']=comparison_rows(results)
        (root/'variant_latency.json').write_text(json.dumps(report,indent=2))
    fields=['variant','task','mode','status','mean_seconds','median_seconds','forward_seconds',
            'backward_seconds','optimizer_seconds','speedup_vs_reference','error']
    with (root/'variant_latency.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fields);writer.writeheader();writer.writerows(report['results'])
    print(json.dumps(report['results'],indent=2))
    if args.upload_wandb:
        from profile_layers import upload_profile
        upload_profile(root,'arc_agi','compressarc-final-ablation','variant-latency-comparison')
    if any(r['status']=='failed' for r in results):
        raise RuntimeError('Some benchmarks failed; inspect variant_latency.json')
    return report


if __name__=='__main__':
    main()
