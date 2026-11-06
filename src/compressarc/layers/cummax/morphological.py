"""Standalone LSE with the original materialized diagonal implementation."""
import torch
import torch.nn as nn

from ..helper import _measure_call


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



