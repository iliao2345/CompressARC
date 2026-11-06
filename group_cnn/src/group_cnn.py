import torch
import torch.nn as nn
import torch.nn.functional as F

def shift_(x, dim, masks=None):
    padding = torch.zeros_like(torch.narrow(x, dim, 0, 1))
    narrowed = torch.narrow(x, dim, 0, x.shape[dim]-1)
    return torch.cat([padding, narrowed], dim=dim)

def diagonal_shift_(x, dim1, dim2, masks=None, shift_amount=1, pad_value=0):
    for dim in (dim1, dim2):
        padding = pad_value+torch.zeros_like(torch.narrow(x, dim, 0, abs(shift_amount)))
        if shift_amount >= 0:
            narrowed = torch.narrow(x, dim, 0, x.shape[dim]-shift_amount)
            x = torch.cat([padding, narrowed], dim=dim)
        else:
            narrowed = torch.narrow(x, dim, -shift_amount, x.shape[dim]+shift_amount)
            x = torch.cat([narrowed, padding], dim=dim)
    return x


class TiedDirectionalConv(nn.Module):
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

    def forward(self, x):
        """
        x: [B, 2, 4, H, W]

        x[:, 0] = cardinal directions
        x[:, 1] = diagonal directions
        """

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
        # Two D4 orbits
        # --------------------------------
        kernels = torch.stack([
            cardinal,
            diagonal
        ])                          # [2, 4, 3, 3]

        B, _, _, H, W = x.shape

        # conv2d wants channels rather than [orbit, direction]
        x_flat = x.reshape(B, 8, H, W)

        # [2,4,3,3] -> [8,1,3,3]
        kernels_flat = kernels.reshape(8, 1, 3, 3)

        y = F.conv2d(
            x_flat,
            kernels_flat,
            padding=1,
            groups=8,
        )

        # restore [cardinal/diagonal, direction]
        y = y.reshape(B, 2, 4, H, W)

        return y




def ConvN():

    kernel = torch.tensor([
        [0. , 0. ,0.],
        [1. , 0. ,0.],
        [0. , 0. ,0.],        
    ])

    kernel = kernel.unsqueeze(0).unsqueeze(0)

    return kernel

def conv_shift(x, kernel):
    x = x.unsqueeze(0).unsqueeze(0)

    print("x shape:", x.shape)
    print("kernel shape:", kernel.shape)

    y = F.conv2d(x, kernel, padding=1)

    return y[0, 0]

if __name__ == "__main__":
    # --------------------------------------------------
    # CHOOSE X
    # --------------------------------------------------

    x = torch.zeros((5, 5))
    x[2, 2] = 1.0

    print("x")
    print(x)


    k = ConvN()

    y = conv_shift(x, k )

    print(y)