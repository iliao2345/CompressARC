
import torch
import torch.nn as nn

import torch.nn.functional as F
from ...train.layer_profiling import profile_operation_stage

class C4DirectionalConv(nn.Module):
    """Rotation-tied directional convolution; learned reflections are not constrained."""
    def __init__(self):
        super().__init__()

        # canonical cardinal kernel
        self.cardinal_kernel = nn.Parameter(torch.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ]))

        # canonical diagonal kernel
        self.diagonal_kernel = nn.Parameter(torch.tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]))

    def directional_kernels(self):
        """Expand the two learned canonical kernels into their rotated C4 orbits."""
        # --------------------------------
        # Cardinal orbit
        # --------------------------------
        right = self.cardinal_kernel
        up = torch.rot90(
            self.cardinal_kernel,
            1,
            dims=(0, 1)
        )
        left = torch.rot90(
            self.cardinal_kernel,
            2,
            dims=(0, 1)
        )
        down = torch.rot90(
            self.cardinal_kernel,
            3,
            dims=(0, 1)
        )

        cardinal = torch.stack([
            up,
            left,
            down,
            right
        ])                          # [4, 3, 3]

        # --------------------------------
        # Diagonal orbit
        # --------------------------------
        diag0 = self.diagonal_kernel
        diag1 = torch.rot90(
            self.diagonal_kernel,
            1,
            dims=(0, 1)
        )
        diag2 = torch.rot90(
            self.diagonal_kernel,
            2,
            dims=(0, 1)
        )
        diag3 = torch.rot90(
            self.diagonal_kernel,
            3,
            dims=(0, 1)
        )

        diagonal = torch.stack([
            diag0,
            diag1,
            diag2,
            diag3,
        ])                          # [4, 3, 3]

        # --------------------------------
        # Two C4 orbits
        # --------------------------------
        kernels = torch.stack([
            cardinal,
            diagonal
        ])                          # [2, 4, 3, 3]

        return kernels.reshape(8, 1, 3, 3)

    def forward(self, x):
        """Depthwise conv2d on [B, 2, 4, H, W] with shared rotated weights."""
        B, _, _, H, W = x.shape

        # conv2d wants channels rather than [orbit, direction]
        x_flat = x.reshape(B, 8, H, W)

        # [2,4,3,3] -> [8,1,3,3]
        kernels_flat = self.directional_kernels()

        with profile_operation_stage("shift", "convolution"):
            y = F.conv2d(
                x_flat,
                kernels_flat,
                padding=1,
                groups=8,
            )

        # restore [cardinal/diagonal, direction]
        y = y.reshape(B, 2, 4, H, W)

        return y





# Keep old imports/checkpoint class lookup compatible. This is the SAME class,
# not a replacement with different parameters or numerical behavior.
TiedDirectionalConv = C4DirectionalConv


class UnfoldTiedDirectionalConv(C4DirectionalConv):
    """Same tied convolution through explicit im2col and weighted patch reduction."""

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:3] != (2, 4) or min(x.shape) < 1:
            raise ValueError("Expected nonempty [B, 2, 4, H, W]")
        batch, _, _, height, width = x.shape
        patches = F.unfold(x.reshape(batch, 8, height, width), kernel_size=3, padding=1)
        patches = patches.reshape(batch, 8, 9, height * width)
        kernels = self.directional_kernels().reshape(8, 9)
        output = torch.einsum("nckl,ck->ncl", patches, kernels)
        return output.reshape(batch, 2, 4, height, width)
