"""Smooth D4-tied morphological max adapted from ARC-fafo/group_cnn."""
import torch
import torch.nn as nn

from ..helper import _measure_call, measure_morphology
from .layer import directional_lse


class MorphologicalMax(nn.Module):
    def __init__(self, height, width, tau=0.1, *, timing=False):
        super().__init__()
        self.tau = tau
        self.timing = timing
        # Accumulated forward timings; clear() resets the measurements.
        self.timings = {}
        self.kernel = nn.Parameter(torch.zeros(max(height, width)))
        self.diagonal_kernel = nn.Parameter(torch.zeros(min(height, width)))

    def _axis_lse(self, x):
        """Reduce flipped causal windows without a position-by-offset matrix."""
        length = x.shape[-1]
        padded = torch.nn.functional.pad(x, (length - 1, 0), value=float('-inf'))
        kernel = self.kernel[:length]
        outputs = []
        for index in range(length):
            # Window ends at x[index]; flipping aligns x[index-offset] with kernel[offset].
            window = padded[..., index:index + length].flip(-1)
            scores = window + kernel
            outputs.append(self.tau * torch.logsumexp(scores / self.tau, dim=-1))
        return torch.stack(outputs, dim=-1)


    def _diagonal_lse(self, x):
        """Apply causal phi to diagonal views without a length-by-length buffer."""
        batch, height, width = x.shape
        batch_stride, row_stride, column_stride = x.stride()
        result = torch.empty_like(x)
        starts = [(0, column) for column in range(width)]
        starts.extend((row, 0) for row in range(1, height))
        for row, column in starts:
            length = min(height - row, width - column)
            diagonal = torch.as_strided(
                x, size=(batch, length),
                stride=(batch_stride, row_stride + column_stride),
                storage_offset=x.storage_offset() + row * row_stride + column * column_stride,
            )
            # At position i, offset k pairs x[i-k] with kernel[k].
            # Only one prefix is materialized at a time, never an L x L matrix.
            values = []
            for index in range(length):
                scores = diagonal[:, :index + 1].flip(-1) + self.diagonal_kernel[:index + 1]
                values.append(self.tau * torch.logsumexp(scores / self.tau, dim=-1))
            output = torch.stack(values, dim=-1)
            rows = torch.arange(row, row + length, device=x.device)
            columns = torch.arange(column, column + length, device=x.device)
            result[:, rows, columns] = output
        return result

    def forward(self, x):
        """Return [right, down-right, up, up-right, left, up-left, down, down-left]."""
        return torch.stack([self.direction(x, index) for index in range(8)], dim=1)

    def direction(self, x, index):
        """Compute one direction without materializing the other seven."""
        rotations = (0, 0, -1, -1, 2, 2, 1, 1)
        rotation = rotations[index]
        rotated = torch.rot90(x, rotation, (-2, -1)) if rotation else x
        fn, name = (self._axis_lse, 'phi') if index % 2 == 0 else (self._diagonal_lse, 'diagonal_phi')
        transformed = _measure_call(
            fn, rotated, name, self.timings if self.timing else None
        )
        return torch.rot90(transformed, -rotation, (-2, -1)) if rotation else transformed




