# Frozen main-branch reference: master 8282287, source/lse.py.
"""Smooth D4-tied morphological max adapted from ARC-fafo/group_cnn."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

import torch
import torch.nn as nn


_timings = ContextVar('morphology_timings', default=None)


@contextmanager
def measure_morphology(enabled=False):
    """Collect synchronized forward wall time, excluding rotations and backward."""
    totals = {}
    token = _timings.set(totals if enabled else None)
    try:
        yield totals
    finally:
        _timings.reset(token)


def _measure_call(fn, x, name):
    totals = _timings.get()
    if totals is None:
        return fn(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    result = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    seconds = time.perf_counter() - started
    key = f'timing/{name}_forward_seconds'
    totals[key] = totals.get(key, 0.0) + seconds
    key = f'timing/{name}_calls'
    totals[key] = totals.get(key, 0) + 1
    return result


class MorphologicalMax(nn.Module):
    def __init__(self, height, width, tau=0.1):
        super().__init__()
        self.tau = tau
        self.kernel = nn.Parameter(torch.zeros(max(height, width)))
        self.diagonal_kernel = nn.Parameter(torch.zeros(min(height, width)))

    def _axis_lse(self, x):
        length = x.shape[-1]
        shifted = x.new_full((*x.shape, length), float('-inf'))
        for index in range(length):
            for offset in range(index + 1):
                shifted[..., index, offset] = x[..., index - offset]
        scores = shifted + self.kernel[:length]
        return self.tau * torch.logsumexp(scores / self.tau, dim=-1)

    def _diagonal_lse(self, x):
        batch, height, width = x.shape
        result = torch.empty_like(x)
        for diagonal in range(-(height - 1), width):
            coordinates = [(row, row + diagonal) for row in range(height)
                           if 0 <= row + diagonal < width]
            values = torch.stack([x[:, row, column] for row, column in coordinates], dim=-1)
            length = values.shape[-1]
            shifted = x.new_full((batch, length, length), float('-inf'))
            for index in range(length):
                for offset in range(index + 1):
                    shifted[:, index, offset] = values[:, index - offset]
            scores = shifted + self.diagonal_kernel[:length]
            output = self.tau * torch.logsumexp(scores / self.tau, dim=-1)
            for index, (row, column) in enumerate(coordinates):
                result[:, row, column] = output[:, index]
        return result

    def forward(self, x):
        """Return [right, NE, up, NW, left, SW, down, SE]."""
        return torch.stack([self.direction(x, index) for index in range(8)], dim=1)

    def direction(self, x, index):
        """Compute one direction without materializing the other seven."""
        rotations = (0, 0, -1, -1, 2, 2, 1, 1)
        rotation = rotations[index]
        rotated = torch.rot90(x, rotation, (-2, -1)) if rotation else x
        fn, name = (self._axis_lse, 'phi') if index % 2 == 0 else (self._diagonal_lse, 'diagonal_phi')
        transformed = _measure_call(fn, rotated, name)
        return torch.rot90(transformed, -rotation, (-2, -1)) if rotation else transformed


def directional_lse(dims, x, masks, model):
    """Apply the matching LSE direction to each CompressARC direction/feature half."""
    if x.shape[-1] % 2:
        raise ValueError('LSE directional features must have even width')
    direction_axis = sum(dims[:2])
    height_axis = direction_axis + 1
    width_axis = direction_axis + 2
    valid = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
    for _ in range(sum(dims[1:3])):
        valid = valid[:, None]
    valid = valid[..., None]
    masked = x * valid
    mappings = ((6, 7, 0, 1, 2, 3, 4, 5), (2, 3, 4, 5, 6, 7, 0, 1))
    halves = []
    for half, mapping in enumerate(mappings):
        outputs = []
        for direction, output_direction in enumerate(mapping):
            grid = torch.select(masked, direction_axis, direction)[..., half::2]
            grid = grid.movedim(-1, -3)
            shape = grid.shape
            selected = model.direction(
                grid.reshape(-1, shape[-2], shape[-1]), output_direction
            ).reshape(shape).movedim(-3, -1)
            outputs.append(selected)
        halves.append(torch.stack(outputs, dim=direction_axis))
    result = torch.cat(halves, dim=-1)
    return result * valid
