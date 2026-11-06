"""Sweep hard-cummax launch layouts; validate outputs/gradients before timing.

Preallocated contiguous kernel pipelines only. Not a model training benchmark.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics
import time
import torch


def cases(args):
    if args.kind in ('axis','both'):
        for rows,length,positions,warps in itertools.product(args.rows,args.lengths,args.positions,args.warps):
            yield dict(kind='axis',shape=[rows,length],positions=positions,warps=warps)
    if args.kind in ('diagonal','both'):
        for batch,size,block,warps in itertools.product(args.batches,args.spatial_sizes,args.blocks,args.warps):
            h,w = map(int,size.lower().split('x'))
            yield dict(kind='diagonal',shape=[batch,h,w],block=block,warps=warps)


def make_data(shape, pattern, device, seed=812):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape,device=device,generator=generator)
    mask = torch.ones_like(x)
    if pattern == 'holes':
        mask.reshape(-1)[::5] = 0
    elif pattern == 'ties':
        x.zero_()
    elif pattern == 'padding_empty':
        x = x.sign()*1000.
        mask.zero_()
    dy = torch.randn(shape,device=device,generator=generator)
    return x,mask,dy


def make_plan(case, data):
    from compressarc.layers.cummax.launch_benchmark import axis_plan,diagonal_plan
    x,mask,dy = data
    if case['kind'] == 'axis':
        return axis_plan(x,mask,dy,positions=case['positions'],warps=case['warps'])
    return diagonal_plan(x,mask,dy,block=case['block'],warps=case['warps'])


def reference(case, data):
    from compressarc.layers.cummax.primitives import cummax_,diagonal_cummax_
    x,mask,dy = data
    x = x.detach().requires_grad_()
    y = cummax_(x,-1,mask) if case['kind']=='axis' else diagonal_cummax_(x,-2,-1,mask)
    gradient, = torch.autograd.grad(y,x,dy)
    return y.detach(),gradient.detach()


def validate(plan, wanted):
    plan['forward'](); plan['backward']()
    torch.testing.assert_close(plan['output'],wanted[0],rtol=3e-4,atol=3e-4)
    torch.testing.assert_close(plan['gradient'],wanted[1],rtol=2e-3,atol=2e-3)


def measure(fn, device, warmup, samples, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    wall, stream = [], []
    for _ in range(samples):
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize(device)
        start.record()
        began = time.perf_counter()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-began)*1e6/repeats)
        stream.append(start.elapsed_time(end)*1000/repeats)
    return dict(wall_median_us=statistics.median(wall),wall_mean_us=statistics.mean(wall),
                stream_median_us=statistics.median(stream),samples=samples,repeats_per_sample=repeats)


def winners(rows):
    grouped = {}
    for row in rows:
        if row.get('status') != 'valid':
            continue
        key = (row['kind'],tuple(row['shape']))
        group = grouped.setdefault(key,[])
        group.append(row)
    result = []
    for (kind,shape),group in grouped.items():
        baseline = next((r for r in group if r['warps']==4 and
                         (r.get('positions')==1 if kind=='axis' else r.get('block')==256)),None)
        best = min(group,key=lambda r:r['backward']['wall_median_us'])
        result.append(dict(kind=kind,shape=shape,best_config={k:best[k] for k in ('positions','block','warps') if k in best},
                           backward_wall_median_us=best['backward']['wall_median_us'],
                           baseline_backward_us=baseline['backward']['wall_median_us'] if baseline else None,
                           backward_speedup=baseline['backward']['wall_median_us']/best['backward']['wall_median_us'] if baseline else None))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('axis','diagonal','both'),default='both')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--lengths',type=int,nargs='+',default=[6,16,30,64])
    p.add_argument('--rows',type=int,nargs='+',default=[32,256,2048])
    p.add_argument('--positions',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--warps',type=int,nargs='+',default=[4,8])
    p.add_argument('--spatial-sizes',nargs='+',default=['6x6','16x16','30x30'])
    p.add_argument('--batches',type=int,nargs='+',default=[8,64])
    p.add_argument('--blocks',type=int,nargs='+',default=[128,256,512])
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--samples',type=int,default=10)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/hard_cummax_launch_sweep'))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--upload-wandb',action='store_true')
    args = p.parse_args(argv)
    try:
        all_cases = list(cases(args))
    except (ValueError,TypeError):
        p.error('Spatial sizes must be positive HxW values, e.g. 16x30')
    if any(min(c['shape'])<1 for c in all_cases) or any(x<1 or x>128 for x in args.lengths):
        p.error('Shapes must be positive; cardinal lengths must be <=128')
    if any(x not in (1,2,4,8,16,32) for x in args.positions) or any(x not in (1,2,4,8) for x in args.warps):
        p.error('Unsupported positions or warp counts')
    if any(x not in (64,128,256,512,1024) for x in args.blocks):
        p.error('Unsupported diagonal blocks')
    if args.warmup<0 or args.samples<1 or args.repeats<1:
        p.error('Warmup must be nonnegative; samples/repeats positive')
    if args.dry_run:
        print(json.dumps(dict(configurations=len(all_cases),cases=all_cases),indent=2))
        return
    device = torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU with Triton is required for measurements')
    torch.cuda.set_device(device.index or 0)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows = []
    report = dict(gpu=torch.cuda.get_device_name(device),torch=str(torch.__version__),
                  cases=rows,winners=[],
                  notes=[
                      'Experimental plans only: training backend and defaults are unchanged.',
                      'Shapes are synthetic representative cases, not captured model activations.',
                      'Correctness covers random masks, equal-value ties and +/-1000 padding ties before timing.',
                      'Preallocated contiguous pipelines exclude allocation, layout transforms and autograd scheduling.',
                      'Wall timing is amortized dispatch-to-completion per call; CUDA stream spans include dispatch gaps, not kernel-active time.',
                      'Programs count workgroups within a kernel launch; varying them does not mean one kernel launch per value.',
                      'Diagonal program counts are per stage kernel; native final gradient additions are included.',
                      'A best configuration on one shape/GPU is not a whole-training speedup.'])
    cache = {}
    for case in all_cases:
        key = (case['kind'],tuple(case['shape']))
        if key not in cache:
            # Retain only this shape's references to bound benchmark memory.
            cache.clear()
            data = [make_data(case['shape'],pattern,device) for pattern in ('holes','ties','padding_empty')]
            cache[key] = [(d,reference(case,d)) for d in data]
        row = dict(case)
        try:
            plan = None
            for data,wanted in cache[key]:
                candidate = make_plan(case,data)
                validate(candidate,wanted)
                if plan is None:
                    plan = candidate
            row.update({k:v for k,v in plan.items() if k not in ('forward','backward','output','gradient')})
            row['forward'] = measure(plan['forward'],device,args.warmup,args.samples,args.repeats)
            # Restore forward state once before repeated backward launches.
            plan['forward']()
            row['backward'] = measure(plan['backward'],device,args.warmup,args.samples,args.repeats)
            row['status'] = 'valid'
            print(f"{case}: backward {row['backward']['wall_median_us']:.2f} us",flush=True)
        except Exception as error:
            row.update(status='invalid',error=f'{type(error).__name__}: {error}')
            print(f'{case}: INVALID {error}',flush=True)
            if 'illegal memory access' in str(error).lower():
                raise
        rows.append(row)
        report['winners'] = winners(rows)
        (args.output_dir/'launch_sweep.json').write_text(json.dumps(report,indent=2))
    with (args.output_dir/'launch_sweep.csv').open('w',newline='') as f:
        fields=['kind','shape','positions','block','warps','status','forward_wall_us','backward_wall_us','error']
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r.get(k,'') for k in fields[:6]},
                             'forward_wall_us':r.get('forward',{}).get('wall_median_us',''),
                             'backward_wall_us':r.get('backward',{}).get('wall_median_us',''),
                             'error':r.get('error','')})
    print(json.dumps(report['winners'],indent=2))
    valid_count = sum(r.get('status') == 'valid' for r in rows)
    print(f'Validated {valid_count}/{len(rows)} layouts; invalid layouts have no timings.')
    if valid_count == 0:
        raise RuntimeError('No layout passed correctness; inspect launch_sweep.json')
    if args.upload_wandb:
        from profile_layers import upload_profile
        upload_profile(args.output_dir,'arc_agi','compressarc-final-ablation','hard-cummax-launch-sweep')
    return report


if __name__=='__main__':
    main()
