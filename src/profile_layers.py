"""Run clean latency and detailed layer profiling as separate training passes."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def upload_profile(root, entity, project, name):
    # Refuse stale checkpoints from a reused output directory.
    weights = [p for p in Path(root).rglob('*') if p.is_file() and
               p.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors')]
    if weights:
        raise ValueError('Profile upload contains checkpoint files; use a fresh output directory: '+
                         ', '.join(str(p) for p in weights))
    # Use the installed CLI so upload progress and the final artifact link are visible.
    cmd = [sys.executable, '-m', 'wandb', 'artifact', 'put', str(root),
           '--name', f'{project}/{name}', '--type', 'profile']
    env = dict(os.environ, WANDB_ENTITY=entity, WANDB_MODE='online')
    subprocess.run(cmd, env=env, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--task', required=True)
    p.add_argument('--run-label', default='layer_profile')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--set', dest='overrides', action='append', default=[])
    p.add_argument('--clean-only', action='store_true', help='Measure latency without an operator trace')
    p.add_argument('--clean-steps', type=int, default=40)
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--capture-steps', type=int, default=1)
    p.add_argument('--sharing-stages', action='store_true', help='Record clean share-up/down stage spans')
    p.add_argument('--no-shapes', action='store_true')
    p.add_argument('--profile-memory', action='store_true')
    p.add_argument('--with-stack', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/d4_conv_layer_profile'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--upload-wandb', action='store_true', help='Upload reports and traces after both passes succeed')
    p.add_argument('--wandb-entity', default='arc_agi')
    p.add_argument('--wandb-project', default='compressarc-final-ablation')
    p.add_argument('--artifact-name', default='d4-conv-layer-profile')
    args = p.parse_args(argv)
    if args.clean_steps <= args.warmup_steps or args.warmup_steps < 1 or args.capture_steps < 1:
        p.error('clean-steps must exceed positive warmup-steps; capture-steps must be positive')
    repo = Path(__file__).resolve().parents[1]
    root = args.output_dir.resolve()
    for capture in ((False,) if args.clean_only else (False, True)):
        name = 'profile' if capture else 'clean'
        iterations = args.warmup_steps + args.capture_steps if capture else args.clean_steps
        cmd = [sys.executable, str(repo/'src/train.py'), '--config', str(args.config.resolve()),
               '--task', args.task, '--output-dir', str(root/name)]
        overrides = [*args.overrides, 'logging.save_model=false', f'training.device={args.device}',
                     f'training.iterations={iterations}', 'training.step_timing=phases',
                     f'training.operator_profile={str(capture).lower()}',
                     f'training.operator_profile_start={args.warmup_steps}',
                     f'training.operator_profile_steps={args.capture_steps}',
                     f'training.operator_profile_shapes={str(not args.no_shapes).lower()}',
                     f'training.operator_profile_memory={str(args.profile_memory).lower()}',
                     f'training.operator_profile_stack={str(args.with_stack).lower()}',
                     f'experiment.name={args.run_label}_{name}']
        if args.sharing_stages:
            overrides.append('training.sharing_stage_timing=true')
        for override in overrides:
            cmd.extend(['--set', override])
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo, check=True)
    if args.dry_run:
        return
    rows = [json.loads(line) for line in (root/'clean'/f'{args.task}.step_timing.jsonl').read_text().splitlines()]
    clean = rows[args.warmup_steps:]
    keys = sorted({k for r in clean for k in r if k.startswith('timing/')})
    means = {k:sum(r.get(k, 0) for r in clean)/len(clean) for k in keys}
    prefix = root/'profile'/f'{args.task}.operators'
    analysis = {} if args.clean_only else json.loads(Path(str(prefix)+'.layers.json').read_text())
    import statistics
    latencies = [r['timing/step_seconds'] for r in clean]
    from config import load_config
    effective_config = load_config(args.config, args.overrides)
    report = dict(task=args.task, cummax_implementation=effective_config.model.get('cummax_implementation', 'primitives'), optimizer_implementation=effective_config.training.get('optimizer_implementation', 'default'),
                  diagnostics_implementation=effective_config.training.get('diagnostics_implementation', 'reference'),
                  share_implementation=effective_config.model.get('share_implementation', 'primitives'),
                  runtime_device_dispatch=effective_config.training.get('runtime_device_dispatch', 'global'),
                  clean_only=args.clean_only, warmup_steps=args.warmup_steps,
                  measured_clean_steps=len(clean), clean_mean_seconds=statistics.mean(latencies),
                  clean_median_seconds=statistics.median(latencies), clean_phase_means=means,
                  profile=analysis,
                  optimization_options=[
                      'Batch existing direction-sharing affine maps if measured launch counts dominate; preserve tied weights and coefficients.',
                      'Fuse pointwise/normalization operations or evaluate compilation if dispatch gaps dominate.',
                      'Evaluate compatible optimizer modes if scalar-read/synchronization counts dominate optimizer host time.',
                      'Consider a custom D4-conv kernel only if shift GPU execution or launch count is substantial.',
                      'Preserve math and validate outputs/gradients, then repeat clean timing on the same task and GPU.'])
    (root/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    phase_keys = ['forward','loss','backward','optimizer',
                  'optimizer_gradient_checks','optimizer_adam','optimizer_parameter_checks',
                  'metrics','candidates','logging']
    text = [f"Clean full-step mean: {report['clean_mean_seconds']:.6f} s",
            f"Clean full-step median: {report['clean_median_seconds']:.6f} s",
            f"Measured steps: {len(clean)} after {args.warmup_steps} warmup steps", '',
            'Optimizer subspans are included in the optimizer total; do not add both.',
            'Clean phase | host seconds | CUDA-stream seconds (includes dispatch gaps)']
    for name in phase_keys:
        text.append(f"{name} | {means.get('timing/host_'+name+'_seconds',0):.6f} | {means.get('timing/stream_'+name+'_seconds',0):.6f}")
    text += ['', 'Sharing forward stage detail (already included in forward) | host seconds | CUDA-stream seconds']
    for key in sorted(k for k in means if k.startswith('timing/host_sharing_')):
        name = key.removeprefix('timing/host_').removesuffix('_seconds')
        text.append(f"{name} | {means[key]:.6f} | {means.get('timing/stream_'+name+'_seconds',0):.6f}")
    text += ['', 'Diagnostic detail (already included in parent phases) | host seconds | CUDA-stream seconds']
    for key in sorted(k for k in means if k.startswith('timing/host_numerics_')):
        name = key.removeprefix('timing/host_').removesuffix('_seconds')
        text.append(f"{name} | {means[key]:.6f} | {means.get('timing/stream_'+name+'_seconds',0):.6f}")
    if not args.clean_only:
        text += ['', Path(str(prefix)+'.layers.txt').read_text(), '', 'Options to test, selected using the measured bottleneck:', *report['optimization_options']]
    (root/'comparison.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('\n'.join(text), flush=True)
    if args.upload_wandb:
        upload_profile(root, args.wandb_entity, args.wandb_project, args.artifact_name)


if __name__ == '__main__':
    main()
