"""Sweep LSE forward timings; write one JSON record per method and shape."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from lse import MorphologicalMax


class StageRecorder:
    """CUDA stream elapsed time and CPU submission time between checkpoints."""

    def __init__(self):
        self.points = []

    def __call__(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.points.append((name, event, time.perf_counter()))

    def results(self):
        self.points[-1][1].synchronize()
        return {
            name: {
                'cuda_elapsed_ms': previous[1].elapsed_time(event),
                'cpu_submission_ms': (host - previous[2]) * 1000,
            }
            for previous, (name, event, host) in zip(self.points, self.points[1:])
        }


def measure(fn, x, warmup, runs, stage_runs):
    for _ in range(warmup):
        result = fn(x)
        del result
    torch.cuda.synchronize(x.device)
    torch.cuda.reset_peak_memory_stats(x.device)
    baseline_bytes = torch.cuda.memory_allocated(x.device)
    elapsed, wall = [], []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        result = fn(x)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
        elapsed.append(start.elapsed_time(end))
        del result
    peak_bytes = torch.cuda.max_memory_allocated(x.device) - baseline_bytes

    samples = []
    for _ in range(stage_runs):
        recorder = StageRecorder()
        result = fn(x, mark=recorder)
        samples.append(recorder.results())
        del result
    stages = {
        name: {
            metric: statistics.median(sample[name][metric] for sample in samples)
            for metric in samples[0][name]
        }
        for name in samples[0]
    }
    return {
        'cuda_median_ms': statistics.median(elapsed),
        'cuda_min_ms': min(elapsed),
        'cuda_max_ms': max(elapsed),
        'wall_median_ms': statistics.median(wall),
        'peak_extra_allocated_bytes': peak_bytes,
        'stages': stages,
        'largest_cuda_stage': max(stages, key=lambda name: stages[name]['cuda_elapsed_ms']),
        'largest_cpu_stage': max(stages, key=lambda name: stages[name]['cpu_submission_ms']),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-exponent', type=int, default=0)
    parser.add_argument('--max-exponent', type=int, default=9, help='Inclusive; 9 means 512.')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--stage-runs', type=int, default=3)
    parser.add_argument('--mode', choices=['training-forward', 'inference'], default='training-forward')
    parser.add_argument('--output', type=Path, default=Path('lse_profile.jsonl'))
    args = parser.parse_args()
    if not 0 <= args.min_exponent <= args.max_exponent:
        parser.error('Require 0 <= min-exponent <= max-exponent')
    if min(args.batches) < 1 or min(args.runs, args.stage_runs) < 1 or args.warmup < 0:
        parser.error('Batches and run counts must be positive; warmup must be nonnegative')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark.')
    torch.manual_seed(42)
    metadata = {
        'type': 'metadata', 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'dtype': 'float32', 'mode': args.mode,
        'min_exponent': args.min_exponent, 'max_exponent': args.max_exponent,
        'batches': args.batches, 'warmup': args.warmup,
        'runs': args.runs, 'stage_runs': args.stage_runs,
        'scope': 'Forward only, including exact max and smooth LSE; no backward.',
        'timing_note': 'Stage CUDA elapsed includes stream idle gaps from CPU dispatch; '
                       'CPU submission overlaps GPU work. Do not add the two. '
                       'Stage instrumentation is excluded from total timings.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting a previous long sweep.
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(metadata) + '\n')
        output.flush()
        for h_exp in range(args.min_exponent, args.max_exponent + 1):
            for w_exp in range(args.min_exponent, args.max_exponent + 1):
                for batch in args.batches:
                    height, width = 2 ** h_exp, 2 ** w_exp
                    methods = ['phi', 'diagonal_phi']
                    if (h_exp + w_exp + batch // 2) % 2:
                        methods.reverse()
                    for method in methods:
                        row = {'type': 'measurement', 'B': batch, 'H': height,
                               'W': width, 'method': method}
                        print(f'Starting B={batch} H={height} W={width} {method}', flush=True)
                        model = x = None
                        try:
                            model = MorphologicalMax(height, width).cuda()
                            x = torch.randn(batch, height, width, device='cuda', dtype=torch.float32,
                                            requires_grad=args.mode == 'training-forward')
                            context = torch.no_grad() if args.mode == 'inference' else nullcontext()
                            with context:
                                row.update(measure(getattr(model, method), x, args.warmup,
                                                   args.runs, args.stage_runs))
                            row['status'] = 'ok'
                            print(f"  {row['cuda_median_ms']:.3f} ms; largest CUDA stage: "
                                  f"{row['largest_cuda_stage']}", flush=True)
                        except torch.cuda.OutOfMemoryError:
                            row['status'] = 'out_of_memory'
                            print('  Out of memory; continuing.', flush=True)
                        finally:
                            del model, x
                            gc.collect()
                            torch.cuda.empty_cache()
                        output.write(json.dumps(row) + '\n')
                        output.flush()


if __name__ == '__main__':
    main()
