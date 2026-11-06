"""Optional wrappers for the agreed train.py timing and memory metrics."""
import math
import time

import torch

from training_metrics_spec import OPERATIONS


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class OperationTimer:
    """Nonoverlapping event spans; no synchronization between operations."""

    def __init__(self, device, iterations):
        self.device = device
        self.names = [name.removeprefix("timing/").removesuffix("_ms") for name in OPERATIONS]
        self.boundaries = [*self.names, "end"]
        self.capacity = iterations * len(self.boundaries)
        self.times = []
        self.events = []
        if device.type == "cuda":
            self.events = [torch.cuda.Event(enable_timing=True) for _ in range(self.capacity)]
            for event in self.events:
                event.record()
            synchronize(device)

    def mark(self, name):
        index = len(self.times)
        if index >= self.capacity or name != self.boundaries[index % len(self.boundaries)]:
            raise RuntimeError(f"Unexpected profiling boundary: {name}")
        self.times.append(time.perf_counter())
        if self.events:
            self.events[index].record()

    def records(self, first_step=0):
        if len(self.times) != self.capacity:
            raise RuntimeError("Incomplete profiling pass")
        records = []
        for index, start in enumerate(range(0, self.capacity, len(self.boundaries))):
            record = {"train_step": first_step + index}
            for offset, name in enumerate(self.names):
                boundary = start + offset
                if self.events:
                    elapsed = self.events[boundary].elapsed_time(self.events[boundary + 1])
                else:
                    elapsed = 1000 * (self.times[boundary + 1] - self.times[boundary])
                record[f"timing/{name}_ms"] = elapsed
            records.append(record)
        return records

    def summary(self):
        records = self.records()
        return {
            key: sum(record[key] for record in records) / len(records)
            for key in OPERATIONS
        }


class TrainingTiming:
    def __init__(self, device, pass_name, warmup, measured):
        self.device = device
        self.pass_name = pass_name
        self.warmup = warmup
        self.measured = measured
        self.history = []
        self.metrics = {
            "timing/warmup_iterations": warmup,
            "timing/measured_iterations": measured,
        }
        synchronize(device)
        self.setup_started = time.perf_counter()

    def wall(self, fn):
        synchronize(self.device)
        started = time.perf_counter()
        result = fn()
        synchronize(self.device)
        return result, 1000 * (time.perf_counter() - started)

    def setup_call(self, metric_name, fn):
        if self.pass_name != "clean":
            return fn()
        result, elapsed_ms = self.wall(fn)
        self.metrics[metric_name] = elapsed_ms
        return result

    def finish_setup(self):
        if self.pass_name == "clean":
            synchronize(self.device)
            self.metrics["setup/total_ms"] = 1000 * (time.perf_counter() - self.setup_started)

    def train(self, task, model, optimizer, config, take_step):
        # Warmup includes the first update and lazy Adam state allocation.
        first = lambda: take_step(task, model, optimizer, 0, config)
        if self.pass_name == "clean":
            _, first_ms = self.wall(first)
            self.metrics["timing/first_step_ms"] = first_ms
            self.history.append({"train_step": 0, "timing/first_step_ms": first_ms})
        else:
            first()
        for step in range(1, self.warmup):
            take_step(task, model, optimizer, step, config)
        synchronize(self.device)

        if self.pass_name == "clean":
            # Do not attach operation markers to the clean measured block.
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            for step in range(self.warmup, self.warmup + self.measured):
                last = take_step(task, model, optimizer, step, config)
            synchronize(self.device)
            block_ms = 1000 * (time.perf_counter() - started)
            self.metrics["timing/block_ms"] = block_ms
            self.metrics["timing/mean_step_ms"] = block_ms / self.measured
            if self.device.type == "cuda":
                self.metrics["memory/peak_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
                self.metrics["memory/peak_reserved_mib"] = torch.cuda.max_memory_reserved(self.device) / 2**20
        else:
            timer = OperationTimer(self.device, self.measured)
            for step in range(self.warmup, self.warmup + self.measured):
                last = take_step(task, model, optimizer, step, config, phase_callback=timer.mark)
            synchronize(self.device)
            self.history = timer.records(first_step=self.warmup)
            self.metrics.update(timer.summary())
        if not math.isfinite(last["loss"]):
            raise RuntimeError("Non-finite loss after timing pass")
        return dict(self.metrics)
