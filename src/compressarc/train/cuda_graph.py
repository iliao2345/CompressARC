"""Opt-in per-task forward/loss/backward replay, with eager Adam and CPU metrics.

Capture happens only after loss annealing is finished. Preparatory passes never
call optimizer.step, and restore RNG state before the first real replay.
"""
from contextlib import nullcontext
import time

import torch

from .objective import compute_objective, take_step


def validate_cuda_graph_options(training, model, mode, device):
    if not training.get('cuda_graph', False):
        return
    if mode != 'train':
        raise ValueError('CUDA Graph replay currently supports train mode only')
    if torch.device(device).type != 'cuda':
        raise ValueError('training.cuda_graph requires a CUDA device')
    if training.get('step_timing', 'off') not in ('off', 'wall'):
        raise ValueError('CUDA Graph replay requires step_timing=off or wall; phase timings would be misleading')
    if training.get('operator_profile', False) or training.get('sharing_stage_timing', False) or model.get('layer_timing', False):
        raise ValueError('Disable operator_profile, sharing_stage_timing and layer_timing for CUDA Graph replay')
    start = training.get('cuda_graph_start', 100)
    warmup = training.get('cuda_graph_warmup', 3)
    if type(start) is not int or start < training.annealing_steps:
        raise ValueError('cuda_graph_start must be an integer >= annealing_steps')
    if type(warmup) is not int or warmup < 1:
        raise ValueError('cuda_graph_warmup must be a positive integer')


class CudaGraphStep:
    def __init__(self, task, model, optimizer, config):
        self.task, self.model, self.optimizer, self.config = task, model, optimizer, config
        self.parameters = list(model.weights_list)
        self.device = self.parameters[0].device
        if self.device.type != 'cuda' or any(p.device != self.device for p in self.parameters):
            raise ValueError('CUDA Graph replay requires all model parameters on one CUDA device')
        self.start = config.get('cuda_graph_start', 100)
        if type(self.start) is not int or self.start < config.annealing_steps:
            raise ValueError('cuda_graph_start must be >= annealing_steps')
        self.warmup = config.get('cuda_graph_warmup', 3)
        if type(self.warmup) is not int or self.warmup < 1:
            raise ValueError('cuda_graph_warmup must be positive')
        self.graph = None
        self.capture_seconds = 0.
        self.replayed_steps = 0
        self.eager_steps = 0
        self.next_step = 0

    def _forward_backward(self):
        self.raw_outputs = self.model.forward()
        self.objective = compute_objective(self.task, self.raw_outputs, self.start, self.config)
        self.objective[0].backward()

    def _capture(self):
        started = time.perf_counter()
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        graph = torch.cuda.CUDAGraph()
        try:
            # Warm kernels/caches on a side stream. These are NOT training updates.
            with torch.cuda.stream(stream):
                for _ in range(self.warmup):
                    self.optimizer.zero_grad(set_to_none=True)
                    self._forward_backward()
            stream.synchronize()
            # Keep only buffers for parameters that actually received gradients.
            # Unused biases must retain grad=None, just as in eager Adam.
            self.gradients = [p.grad for p in self.parameters if p.grad is not None]
            if not self.gradients:
                raise RuntimeError('No gradients to capture')
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
            with torch.cuda.graph(graph, stream=stream):
                # Stable buffers, cleared inside every replay; no accumulation
                # across optimizer updates and no extra updates during capture.
                torch._foreach_zero_(self.gradients)
                self._forward_backward()
            current.wait_stream(stream)
            stream.synchronize()
            self.graph = graph
        except Exception as error:
            raise RuntimeError('CUDA Graph capture failed; no optimizer update was performed for this step. '
                               'Inspect the underlying error or rerun with training.cuda_graph=false.') from error
        finally:
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
        self.capture_seconds = time.perf_counter() - started

    def __call__(self, task, model, optimizer, train_step, config=None, return_outputs=False,
                 phase_callback=None, diagnostic_callback=None, optimizer_scope=None):
        if task is not self.task or model is not self.model or optimizer is not self.optimizer:
            raise ValueError('A CUDA Graph step is bound to exactly one task/model/optimizer')
        if train_step != self.next_step:
            raise ValueError(f'Expected training step {self.next_step}, got {train_step}')
        if phase_callback is not None:
            raise ValueError('Per-phase callbacks are not supported by the CUDA Graph path')
        if train_step < self.start:
            metrics = take_step(task, model, optimizer, train_step, self.config,
                return_outputs=return_outputs, diagnostic_callback=diagnostic_callback,
                optimizer_scope=optimizer_scope)
            self.eager_steps += 1
        else:
            if self.graph is None:
                self._capture()
            self.graph.replay()
            loss, total_kl, reconstruction_error, kl_sums, kl_names, outputs = self.objective
            logits, x_mask, y_mask, kl_amounts, _ = self.raw_outputs
            # Checks, Adam, tracking, scalar transfers and logging stay eager.
            # Diagnostic callbacks see the same pre-update outputs/gradients as before.
            if diagnostic_callback is not None:
                diagnostic_callback('outputs', (logits, x_mask, y_mask, *kl_amounts))
                diagnostic_callback('loss', (loss, total_kl, reconstruction_error))
            scope = optimizer_scope or (lambda name: nullcontext())
            with scope('gradient_checks'):
                if diagnostic_callback is not None:
                    diagnostic_callback('gradients', (p.grad for p in self.parameters))
            with scope('adam'):
                optimizer.step()
            with scope('parameter_checks'):
                if diagnostic_callback is not None:
                    diagnostic_callback('parameters', self.parameters)
            scalars = torch.stack([total_kl, reconstruction_error, loss, *kl_sums]).detach().cpu().tolist()
            metrics = dict(kl=scalars[0], reconstruction_error=scalars[1], loss=scalars[2],
                           kl_components=dict(zip(kl_names, scalars[3:])))
            if return_outputs:
                metrics['outputs'] = tuple(value.detach().cpu() for value in outputs)
            self.replayed_steps += 1
        self.next_step += 1
        return metrics

    def summary(self):
        return dict(enabled=True, captured=self.graph is not None, start_step=self.start,
                    eager_steps=self.eager_steps, replayed_steps=self.replayed_steps,
                    optimizer_updates=self.next_step, capture_seconds=self.capture_seconds,
                    warmup_backward_passes=self.warmup if self.graph is not None else 0,
                    optimizer_captured=False)
