import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from static_op import diagonal_shift_

def cummax_(x, dim, masks):
    masks = 1e3*(1-masks)
    max_ = torch.max(x-masks, dim=dim, keepdim=True)[0] + masks + 1e-3
    min_ = torch.min(x+masks, dim=dim, keepdim=True)[0] - masks - 1e-3
    print(max_, min_)
    x = torch.cummax(x-masks, dim=dim)[0] + masks
    return (x - min_) / (max_-min_) * 2 - 1

def diagonal_cummax_(x, dim1, dim2, masks):
    masks_ = 1e3*(1-masks)
    min_dim = min(x.shape[dim1], x.shape[dim2])
    n_iters = int(np.ceil(np.log2(min_dim)))
    # compute the cummax and max via forward+backward associative scan
    max_x = x - masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(max_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=-1e3)
            max_x = torch.max(max_x, shifted_x)
        if sign == 1:  # save the cummax after the forward associative scan
            cummax_x = max_x + masks_
    max_x = max_x + masks_
    # compute the min via forward+backward associative scan
    min_x = x + masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(min_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=1e3)
            min_x = torch.min(min_x, shifted_x)
    min_x = min_x - masks_
    return ((cummax_x - min_x) / (max_x-min_x+1e-5) * 2 - 1)*masks  # rescale the cummax to fit the max and min


class MorphologicalMax(nn.Module):
    def __init__(self, H, W, tau=0.1):
        super().__init__()
        L_min = min(H, W)
        L_max = max(H, W)

        self.tau = tau
        self.kernel = nn.Parameter(torch.zeros(L_max))
        self.diagonal_kernel = nn.Parameter(torch.zeros(L_min))
        
        support = torch.ones(L_max, dtype=torch.bool)
        self.register_buffer("support", support)



    def diagonal_phi(self, x, mark=None):
        """
        Canonical diagonal morphological cummax.

        x: [B, H, W]

        Returns:
            out:   [B, H, W]  LSE version
            exact: [B, H, W]  exact max version
        """
        B, H, W = x.shape
        if mark:
            mark('start')

        D = H + W - 1
        L = min(H, W)

        # --------------------------------------------------
        # 1. Pack NW-SE diagonals into rows
        #    [B, D, L]
        # --------------------------------------------------

        diagonals = torch.full(
            (B, D, L),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )
        
        if mark:
            mark('allocate_diagonals')

        offsets = range(-(H - 1), W)
        lengths = [min(H + min(d, 0), W - max(d, 0)) for d in offsets]
        if mark:
            mark('build_coordinates')
        # diagonal() preserves the batch axis and supports rotated/strided inputs.
        for index, (offset, length) in enumerate(zip(offsets, lengths)):
            diagonals[:, index, :length] = x.diagonal(offset, -2, -1)
        if mark:
            mark('pack_diagonals')

        
        # --------------------------------------------------
        # 2. Build f(x-u)
        #    [B, D, output_position, offset]
        # --------------------------------------------------

        padded = F.pad(diagonals, (L - 1, 0), value=float('-inf'))
        shifted = padded.unfold(-1, L, 1).flip(-1)
        if mark:
            mark('build_shifted')
        # --------------------------------------------------
        # 3. Morphological kernel
        # --------------------------------------------------

        scores = shifted + self.diagonal_kernel.view(1, 1, 1, L)
        if mark:
            mark('add_kernel')

        # Unused packed positions must have finite reductions to avoid NaN gradients.
        valid = torch.arange(L, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        scores = scores.masked_fill(~valid[None, :, :, None], 0)
        if mark:
            mark('mask_unused_positions')

        # Exact morphological max
        exact_diag = scores.max(dim=-1).values
        if mark:
            mark('exact_max')

        # Smooth max
        out_diag = self.tau * torch.logsumexp(
            scores / self.tau,
            dim=-1,
        )
        if mark:
            mark('smooth_logsumexp')

        # --------------------------------------------------
        # 4. Scatter diagonals back into [B, H, W]
        # --------------------------------------------------

        out = torch.empty_like(x)
        exact = torch.empty_like(x)

        for index, (offset, length) in enumerate(zip(offsets, lengths)):
            out.diagonal(offset, -2, -1).copy_(out_diag[:, index, :length])
            exact.diagonal(offset, -2, -1).copy_(exact_diag[:, index, :length])
        if mark:
            mark('allocate_and_scatter')

        return out, exact





    def phi(self, x, mark=None):
        if mark:
            mark('start')
        # canonical direction is always last dim
        original_shape = x.shape
        W = x.shape[-1]

        x_flat = x.reshape(-1, W)
        if mark:
            mark('reshape_input')

        # Reverse windows so offset u still selects x[i - u].
        x_pad = F.pad(x_flat, (W - 1, 0), value=float("-inf"))
        x_shifted = x_pad.unfold(-1, W, 1).flip(-1)

        if mark:
            mark('build_shifted')
        
        kernel = self.kernel[:W]
        support = self.support[:W]
        scores = x_shifted + kernel.view(1, 1, W)
        if mark:
            mark('add_kernel')

        scores = scores.masked_fill(
            ~support.view(1, 1, W),
            float("-inf")
        )
        if mark:
            mark('apply_support')

        exact = scores.max(dim=-1).values
        if mark:
            mark('exact_max')

        out = self.tau * torch.logsumexp(
            scores / self.tau,
            dim=-1
        )
        if mark:
            mark('smooth_logsumexp')

        out = out.reshape(original_shape)
        exact = exact.reshape(original_shape)
        if mark:
            mark('reshape_output')

        return out, exact

    def forward(self, x):
        # x: [B, H, W]

        # ==================================================
        # Cardinal directions
        # ==================================================
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True)


        start.record()
        right, exact_right = self.phi(x)
        end.record()

        x_rot = torch.rot90(x, -1, dims=(1, 2))
        up, _ = self.phi(x_rot)
        up = torch.rot90(up, 1, dims=(1, 2))

        x_rot = torch.rot90(x, 2, dims=(1, 2))
        left, _ = self.phi(x_rot)
        left = torch.rot90(left, 2, dims=(1, 2))

        x_rot = torch.rot90(x, 1, dims=(1, 2))
        down, _ = self.phi(x_rot)
        down = torch.rot90(down, -1, dims=(1, 2))

        torch.cuda.synchronize()
      

        # ==================================================
        # Diagonal directions
        # ==================================================
        time_start = torch.cuda.Event(enable_timing=True) 
        time_end = torch.cuda.Event(enable_timing=True)

        time_start.record()
        top_right, exact_top_right = self.diagonal_phi(x)
        time_end.record()
        
        x_rot = torch.rot90(x, -1, dims=(1, 2))
        top_left, _ = self.diagonal_phi(x_rot)
        top_left = torch.rot90(
            top_left,
            1,
            dims=(1, 2)
        )

        x_rot = torch.rot90(x, 2, dims=(1, 2))
        bottom_left, _ = self.diagonal_phi(x_rot)
        bottom_left = torch.rot90(
            bottom_left,
            2,
            dims=(1, 2)
        )

        x_rot = torch.rot90(x, 1, dims=(1, 2))
        bottom_right, _ = self.diagonal_phi(x_rot)
        bottom_right = torch.rot90(
            bottom_right,
            -1,
            dims=(1, 2)
        )
       

        torch.cuda.synchronize()

        print(f"phi: {start.elapsed_time(end):.3f} ms")
        print(f"diagonal_phi: {time_start.elapsed_time(time_end):.3f} ms")
        # ==================================================
        # Stack all 8 directions
        # ==================================================

        out = torch.stack(
            [
                right,
                top_right,
                up,
                top_left,
                left,
                bottom_left,
                down,
                bottom_right,
            ],
            dim=1
        )

        return out, exact_right, exact_top_right

def benchmark(fn, x, warmup=20, runs=100):
    # warm up
    for _ in range(warmup):
        fn(x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(runs):
        fn(x)

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / runs

if __name__ == "__main__":

    x = torch.tensor([[
        [1., 4., 2., 7., 3.],
        [5., 1., 8., 2., 6.],
        [0., 9., 3., 4., 2.],
        [6., 2., 1., 5., 4.],
        [3., 7., 2., 1., 8.],
    ]])

    model = MorphologicalMax(
        H=x.shape[-2],
        W=x.shape[-1],
        tau=0.1
    )
    x = x.cuda()
    model = model.cuda()
    # actual, exact_right, exact_top_right = model(x)

    # right = torch.cummax(x, dim=-1).values

    # print("input:\n", x)
    # print("output shape:", actual.shape)

    # print("\nexpected right:\n", right)
    # print("\nexact right:\n", exact_right)

    # print("\nexact diagonal:\n", exact_top_right)

    # assert torch.allclose(exact_right, right)

    # assert actual.shape == (
    #     x.shape[0],
    #     8,
    #     x.shape[1],
    #     x.shape[2],
    # )
    phi_ms = benchmark(model.phi, x)
    diag_ms = benchmark(model.diagonal_phi, x)

    print("phi:", phi_ms, "ms")
    print("diagonal:", diag_ms, "ms")


    diag_ms = benchmark(model.diagonal_phi, x)
    phi_ms = benchmark(model.phi, x)

    print("phi:", phi_ms, "ms")
    print("diagonal:", diag_ms, "ms")

