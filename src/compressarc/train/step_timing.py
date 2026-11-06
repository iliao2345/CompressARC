"""Optional training-step latency measurements without per-phase CUDA barriers."""
from contextlib import contextmanager
import json
import math
import statistics
import time
from pathlib import Path

import torch


class StepTimer:
    """Wall timings include normal scalar logging; timing export is excluded.

    phases additionally records host phase times and CUDA stream event times.
    Stream elapsed includes CPU-dispatch gaps; it is NOT GPU kernel-active time.
    Host/stream measurements overlap and must not be added together.
    Only step boundaries synchronize CUDA, not individual phase boundaries.
    """

    def __init__(self, device, mode='off', path=None, clock=time.perf_counter):
        if mode not in ('off', 'wall', 'phases'):
            raise ValueError('training.step_timing must be off, wall or phases')
        self.device = torch.device(device)
        self.mode, self.clock = mode, clock
        self.path = Path(path) if path is not None else None
        self.rows, self.file = [], None
        self.export_seconds = 0.

    @property
    def enabled(self):
        return self.mode != 'off'

    def __enter__(self):
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open('w', encoding='utf-8', buffering=1)
        return self

    def __exit__(self, *error):
        if self.file is not None:
            self.file.close()
        if self.enabled and self.path is not None:
            values = [r['timing/step_seconds'] for r in self.rows]
            steady = values[10:] or values
            report = dict(mode=self.mode, device=str(self.device), completed_timed_steps=len(values),
                          warmup_steps_omitted=10 if len(values)>10 else 0,
                          note='CUDA stream times include dispatch gaps. Host/stream overlap. Optimizer subspans are included in the optimizer total; do not add both. Timing export excluded.',
                          mean_step_seconds=statistics.mean(values) if values else None,
                          steady_mean_step_seconds=statistics.mean(steady) if steady else None,
                          steady_median_step_seconds=statistics.median(steady) if steady else None,
                          steady_p90_step_seconds=sorted(steady)[math.ceil(.9*len(steady))-1] if steady else None)
            report['timing_export_seconds_total'] = self.export_seconds
            report['mean_step_including_timing_export_seconds'] = ((sum(values)+self.export_seconds)/len(values)
                                                                  if values else None)
            if self.mode == 'phases' and self.rows:
                keys = sorted({k for r in self.rows for k in r if k.startswith('timing/host_')})
                selected = self.rows[10:] or self.rows
                report['steady_mean_host_phase_seconds'] = {
                    k:statistics.mean(r.get(k, 0) for r in selected) for k in keys}
            captured = [r for r in self.rows if r.get('timing/operator_profile_active', False)]
            if any('timing/operator_profile_active' in r for r in self.rows):
                report['contains_operator_profile'] = bool(captured)
                report['instrumented_step_count'] = len(captured)
                report['instrumented_mean_step_seconds'] = (statistics.mean(r['timing/step_seconds'] for r in captured) if captured else None)
                report['note'] += ' This run contains an operator capture; use the separate clean pass for baseline latency.'
            summary = self.path.with_suffix('.summary.json')
            summary.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def _event(self):
        if self.mode == 'phases' and self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return None

    def start(self, step):
        if not self.enabled:
            return
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.step, self.started = step, self.clock()
        self.phase, self.host, self.events = None, {}, []

    def mark(self, phase):
        if self.mode != 'phases':
            return
        now, event = self.clock(), self._event()
        if self.phase is not None:
            self.host[self.phase] = self.host.get(self.phase, 0) + now-self.phase_started
            if event is not None:
                self.events.append((self.phase, self.phase_event, event))
        self.phase = None if phase == 'end' else phase
        self.phase_started, self.phase_event = now, event

    @contextmanager
    def scope(self, name):
        """Nested span; adds CUDA events but no synchronization barriers."""
        if self.mode != 'phases':
            yield
            return
        key = name
        started, event = self.clock(), self._event()
        try:
            yield
        finally:
            ended, end_event = self.clock(), self._event()
            self.host[key] = self.host.get(key, 0) + ended-started
            if event is not None:
                self.events.append((key, event, end_event))

    def optimizer_scope(self, name):
        return self.scope('optimizer_' + name)

    def finish(self):
        if not self.enabled:
            return {}
        self.mark('end')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        metrics = {'train_step':self.step, 'timing/step_seconds':self.clock()-self.started}
        metrics.update({f'timing/host_{k}_seconds':v for k,v in self.host.items()})
        for phase, start, end in self.events:
            key = f'timing/stream_{phase}_seconds'
            metrics[key] = metrics.get(key, 0) + start.elapsed_time(end)/1000
        return metrics

    def publish(self, metrics, run=None):
        if not metrics:
            return
        export_started = time.perf_counter()
        self.rows.append(metrics)
        if self.file is not None:
            self.file.write(json.dumps(metrics)+'\n')
        if run is not None:
            run.log(metrics)
        self.export_seconds += time.perf_counter()-export_started
