"""D4-constrained directional convolution, separate from the deployed C4 core."""
import torch

from .morphological import C4DirectionalConv
from ...train.layer_profiling import profile_operation_stage


class D4DirectionalConv(C4DirectionalConv):
    """Enforce the stabilizer symmetry of each canonical movement every forward.

    The rightward cardinal kernel must be invariant under up/down reflection.
    The down-right diagonal kernel must be invariant under row/column exchange.
    Their rotated copies then respect all D4 spatial/direction transformations.
    Raw parameters remain learnable; differentiable projections enforce these
    constraints even after arbitrary optimizer updates.
    """

    def canonical_kernels(self):
        with profile_operation_stage("shift", "d4_projection"):
            cardinal = (self.cardinal_kernel + self.cardinal_kernel.flip(0)) * .5
            diagonal = (self.diagonal_kernel + self.diagonal_kernel.transpose(0, 1)) * .5
        return cardinal, diagonal

    def directional_kernels(self):
        cardinal, diagonal = self.canonical_kernels()
        with profile_operation_stage("shift", "kernel_expansion"):
            cardinals = torch.stack([torch.rot90(cardinal, k, (0, 1)) for k in (1, 2, 3, 0)])
            diagonals = torch.stack([torch.rot90(diagonal, k, (0, 1)) for k in range(4)])
            return torch.stack([cardinals, diagonals]).reshape(8, 1, 3, 3)
