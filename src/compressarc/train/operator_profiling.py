"""Opt-in, short-window operator profiling of the real training loop."""
import csv
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from .layer_profiling import active_layer_profile


PHASES = {'forward', 'loss', 'backward', 'optimizer', 'metrics',
          'progress', 'diagnostics', 'candidates', 'logging'}


def device_time(event, self_only=False):
    name = 'self_device_time_total' if self_only else 'device_time_total'
    if hasattr(event, name):
        return float(getattr(event, name))
    return float(getattr(event, 'self_cuda_time_total' if self_only else 'cuda_time_total', 0))


def attribution(event):
    node, scope, phase = event.cpu_parent, None, None
    while node is not None:
        if node.name.startswith('ttt/'):
            name = node.name[4:]
            if scope is None:
                scope = name
            if name in PHASES:
                phase = name
                break
        node = node.cpu_parent
    return phase or 'unscoped', scope or phase or 'unscoped'


class OperatorProfiler:
    def __init__(self, device, prefix, enabled=False, start_step=10, steps=1,
                 record_shapes=True, profile_memory=False, with_stack=False):
        self.device, self.prefix = torch.device(device), Path(prefix)
        self.enabled = bool(enabled)
        self.start_step, self.steps = int(start_step), int(steps)
        if self.start_step < 0 or self.steps < 1:
            raise ValueError('operator_profile_start must be >=0 and operator_profile_steps >=1')
        self.record_shapes, self.profile_memory, self.with_stack = record_shapes, profile_memory, with_stack
        self.active, self.phase_scope, self.profiler = False, None, None
        self.captured, self.exported = [], False

    def __enter__(self):
        if self.enabled:
            self.prefix.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *error):
        if self.active:
            self._stop(error)
        elif self.enabled and not self.exported:
            self._manifest('not_started')
        return False

    def before_step(self, step):
        if not self.enabled or step < self.start_step or step >= self.start_step+self.steps:
            return
        if self.profiler is None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities, record_shapes=self.record_shapes,
                                                   profile_memory=self.profile_memory, with_stack=self.with_stack)
            self.profiler.__enter__()
            self.active = True
            self.layer_token = active_layer_profile.set(True)
        self.current_step = step

    def mark(self, phase):
        if not self.active:
            return
        self._close_phase()
        if phase != 'end':
            self.phase_scope = torch.profiler.record_function('ttt/'+phase)
            self.phase_scope.__enter__()

    def scope(self, name):
        return torch.profiler.record_function('ttt/'+name) if self.active else nullcontext()

    def _close_phase(self):
        if self.phase_scope is not None:
            self.phase_scope.__exit__(None, None, None)
            self.phase_scope = None

    def after_step(self, step):
        if not self.active:
            return
        self._close_phase()
        self.captured.append(step)
        self.profiler.step()
        if step+1 >= self.start_step+self.steps:
            self._stop((None, None, None))

    def _path(self, suffix):
        return Path(str(self.prefix)+suffix)

    def _manifest(self, status):
        metadata = dict(status=status, device=str(self.device), torch=str(torch.__version__),
                        cuda=torch.version.cuda, start_step_zero_based=self.start_step,
                        requested_steps=self.steps, captured_steps_zero_based=self.captured,
                        captured_iterations_one_based=[x+1 for x in self.captured],
                        record_shapes=self.record_shapes, profile_memory=self.profile_memory,
                        with_stack=self.with_stack,
                        warning='Instrumented run: do not use profiler durations as production latency. Inclusive totals overlap; do not sum parent and child totals.')
        if self.device.type == 'cuda':
            metadata['gpu'] = torch.cuda.get_device_name(self.device)
        self._path('.manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def _stop(self, error):
        self._close_phase()
        self.profiler.__exit__(*error)
        self.active = False
        if hasattr(self, "layer_token"):
            active_layer_profile.reset(self.layer_token)
        self.profiler.export_chrome_trace(str(self._path('.trace.json')))
        averages = self.profiler.key_averages(group_by_input_shape=self.record_shapes)
        self._path('.cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=50), encoding='utf-8')
        if self.device.type == 'cuda':
            self._path('.cuda.txt').write_text(averages.table(sort_by='self_device_time_total', row_limit=50), encoding='utf-8')
        grouped = defaultdict(lambda:dict(calls=0, self_cpu_us=0., self_device_us=0.))
        kernels = defaultdict(lambda:dict(calls=0, duration_us=0.))
        for event in self.profiler.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                item = kernels[event.name]
                item['calls'] += 1
                item['duration_us'] += float(event.time_range.end - event.time_range.start)
                continue
            if event.device_type != torch.autograd.DeviceType.CPU or event.name.startswith('ttt/'):
                continue
            phase, scope = attribution(event)
            key = (phase, scope, event.name, str(event.input_shapes) if self.record_shapes else '')
            item = grouped[key]
            item['calls'] += 1
            item['self_cpu_us'] += float(event.self_cpu_time_total)
            item['self_device_us'] += device_time(event, self_only=True)
        rows = [dict(phase=k[0], scope=k[1], operator=k[2], input_shapes=k[3], **v) for k,v in grouped.items()]
        rows.sort(key=lambda r:r['self_device_us'] if self.device.type=='cuda' else r['self_cpu_us'], reverse=True)
        fields = ['phase','scope','operator','input_shapes','calls','self_cpu_us','self_device_us']
        with self._path('.operators.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
        by_phase = {phase:[r for r in rows if r['phase']==phase][:30] for phase in sorted(PHASES)}
        self._path('.by_phase.json').write_text(json.dumps(by_phase, indent=2), encoding='utf-8')
        kernel_rows = sorted([dict(kernel=k, **v) for k,v in kernels.items()], key=lambda r:r['duration_us'], reverse=True)
        self._path('.kernels.json').write_text(json.dumps(kernel_rows, indent=2), encoding='utf-8')
        from .profile_analysis import analyze_trace
        analyze_trace(self._path('.trace.json'), self.prefix)
        self._manifest('complete' if len(self.captured)==self.steps else 'partial')
        self.exported = True
        print(f'Operator profile saved: {self.prefix}.*', flush=True)
