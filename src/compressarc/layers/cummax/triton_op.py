import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable

import triton
import triton.language as tl

from .optimisation import MorphologicalMax as OptimisedMorphologicalMax

@triton.jit
def axis_lse_kernel(
    x_ptr,
    kernel_ptr,
    out_ptr,
    W: tl.constexpr,
    TAU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    row = pid // W
    i = pid % W

    u = tl.arange(0, BLOCK) # [0, ... , BLOCK - 1]

    valid = (u <= i) & (u < W) # still  [ 0 ,... , BLOCK - 1] be in 0 and 1 depedningon the position 

    x_pos = i - u  

    x_vals = tl.load(
        x_ptr + row * W + x_pos,
        mask=valid,
        other=-float("inf"),
    )

    k_vals = tl.load(
        kernel_ptr + u,
        mask=valid,
        other=0.0,
    )

    scores = x_vals + k_vals

    m = tl.max(scores, axis=0)

    # Convert natural exponentials to base 2 while preserving the LSE temperature.
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    exp_vals = tl.exp2((scores - m) * (LOG2E / TAU))

    z = tl.sum(exp_vals, axis=0)

    y = m + (TAU * LN2) * tl.log2(z)

    tl.store(
        out_ptr + row * W + i,
        y,
    )


@triton.jit
def axis_lse_bwd_dx_kernel(
    x_ptr, kernel_ptr, y_ptr, dy_ptr, dx_ptr,
    W: tl.constexpr, TAU: tl.constexpr, BLOCK: tl.constexpr,
):
    # One program owns an input element and gathers future outputs in its row.
    pid = tl.program_id(0)
    position = pid % W
    u = tl.arange(0, BLOCK)
    valid = (u < W - position) & (u < W)
    future_y = tl.load(y_ptr + pid + u, mask=valid, other=0.0)
    future_dy = tl.load(dy_ptr + pid + u, mask=valid, other=0.0)
    k_vals = tl.load(kernel_ptr + u, mask=valid, other=0.0)
    x_value = tl.load(x_ptr + pid)
    LOG2E: tl.constexpr = 1.4426950408889634
    log_probability = tl.where(
        valid, (x_value + k_vals - future_y) * (LOG2E / TAU), -float("inf")
    )
    dx = tl.sum(future_dy * tl.exp2(log_probability), axis=0)
    tl.store(dx_ptr + pid, dx)


@triton.jit
def axis_lse_bwd_dk_kernel(
    x_ptr, kernel_ptr, y_ptr, dy_ptr, dk_ptr,
    W: tl.constexpr, TAU: tl.constexpr, BLOCK: tl.constexpr,
):
    # Every output contributes to all causal offsets; the tail remains unused.
    pid = tl.program_id(0)
    position = pid % W
    u = tl.arange(0, BLOCK)
    valid = (u <= position) & (u < W)
    x_vals = tl.load(x_ptr + pid - u, mask=valid, other=0.0)
    k_vals = tl.load(kernel_ptr + u, mask=valid, other=0.0)
    y = tl.load(y_ptr + pid)
    dy = tl.load(dy_ptr + pid)
    LOG2E: tl.constexpr = 1.4426950408889634
    log_probability = tl.where(
        valid, (x_vals + k_vals - y) * (LOG2E / TAU), -float("inf")
    )
    tl.atomic_add(dk_ptr + u, dy * tl.exp2(log_probability), mask=valid)


class AxisLSEFunction(torch.autograd.Function):
    """First-order FP32 CUDA LSE along the last axis; tau is constant."""

    @staticmethod
    def forward(ctx, x, kernel, tau):
        if x.ndim < 1 or min(x.shape) < 1:
            raise ValueError("Expected nonempty x with at least one axis")
        width = x.shape[-1]
        if kernel.ndim != 1 or kernel.numel() < width:
            raise ValueError("Kernel must be 1D with at least W entries")
        if not x.is_cuda or not kernel.is_cuda or x.device != kernel.device:
            raise ValueError("x and kernel must be on the same CUDA device")
        if x.dtype != torch.float32 or kernel.dtype != torch.float32:
            raise TypeError("V1 supports float32 x and kernel")
        if isinstance(tau, torch.Tensor):
            raise TypeError("tau must be a constant Python number")
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be positive and finite")
        x, kernel = x.contiguous(), kernel.contiguous()
        block = triton.next_power_of_2(width)
        out = torch.empty_like(x)
        with torch.cuda.device(x.device):
            axis_lse_kernel[(x.numel(),)](
                x, kernel, out, W=width, TAU=tau, BLOCK=block,
            )
        ctx.save_for_backward(x, kernel, out)
        ctx.width, ctx.tau, ctx.block = width, tau, block
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, kernel, y = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grid = (x.numel(),)
        dx, dk = None, None
        with torch.cuda.device(x.device):
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(x)
                axis_lse_bwd_dx_kernel[grid](
                    x, kernel, y, grad_out, dx,
                    W=ctx.width, TAU=ctx.tau, BLOCK=ctx.block,
                )
            if ctx.needs_input_grad[1]:
                dk = torch.zeros_like(kernel)
                axis_lse_bwd_dk_kernel[grid](
                    x, kernel, y, grad_out, dk,
                    W=ctx.width, TAU=ctx.tau, BLOCK=ctx.block,
                )
        return dx, dk, None


class FusedAxisLSE(nn.Module):
    """Trainable last-axis LSE backed by the Triton axis operation."""

    def __init__(self, max_length, tau=0.1):
        super().__init__()
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if isinstance(tau, torch.Tensor):
            raise TypeError("tau must be a constant Python number")
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be positive and finite")
        self.tau = tau
        self.kernel = nn.Parameter(torch.zeros(max_length))

    def forward(self, x):
        if x.ndim < 1:
            raise ValueError("Expected x with at least one axis")
        length = x.shape[-1]
        if length > self.kernel.numel():
            raise ValueError("Input axis exceeds max_length")
        return AxisLSEFunction.apply(x, self.kernel[:length], self.tau)

@triton.jit
def diagonal_lse_kernel(
    x_ptr,
    kernel_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    TAU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Contiguous [B, H, W]; launch B*H*W programs with BLOCK >= min(H, W)."""
    pid = tl.program_id(0)

    batch = pid // (H * W)

    spatial = pid % (H * W)
    row = spatial // W
    col = spatial % W

    u = tl.arange(0, BLOCK)
    max_u = tl.minimum(row, col)
    max_len = tl.minimum(H, W)
    valid = (u <= max_u) & (u < max_len)

    center = batch * H * W + row * W + col
    x_vals = tl.load(
        x_ptr + center - u * (W + 1),
        mask=valid,
        other=-float("inf"),
    )
    k_vals = tl.load(
        kernel_ptr + u,
        mask=valid,
        other=0.0,
    )
    scores = x_vals + k_vals
    m = tl.max(scores, axis=0)

    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    exp_vals = tl.exp2((scores - m) * (LOG2E / TAU))
    z = tl.sum(exp_vals, axis=0)
    y = m + (TAU * LN2) * tl.log2(z)

    tl.store(out_ptr + center, y)



@triton.jit
def diagonal_lse_bwd_dx_kernel(
    x_ptr, kernel_ptr, y_ptr, dy_ptr, dx_ptr,
    H: tl.constexpr, W: tl.constexpr, TAU: tl.constexpr, BLOCK: tl.constexpr,
):
    # One program owns one input pixel: gather all future outputs, without atomics.
    pid = tl.program_id(0)
    batch = pid // (H * W)
    spatial = pid % (H * W)
    row = spatial // W
    col = spatial % W
    u = tl.arange(0, BLOCK)
    max_u = tl.minimum(H - 1 - row, W - 1 - col)
    valid = (u <= max_u) & (u < H) & (u < W)
    center = batch * H * W + row * W + col
    future_offset = center + u * (W + 1)
    future_y = tl.load(y_ptr + future_offset, mask=valid, other=0.0)
    future_dy = tl.load(dy_ptr + future_offset, mask=valid, other=0.0)
    k_vals = tl.load(kernel_ptr + u, mask=valid, other=0.0)
    x_value = tl.load(x_ptr + center)
    LOG2E: tl.constexpr = 1.4426950408889634
    # Mask before exp2: padded lanes must not create 0 * inf or NaNs.
    log_probability = tl.where(
        valid, (x_value + k_vals - future_y) * (LOG2E / TAU), -float("inf")
    )
    contribution = future_dy * tl.exp2(log_probability)
    dx = tl.sum(contribution, axis=0)
    tl.store(dx_ptr + center, dx)


@triton.jit
def diagonal_lse_bwd_dk_kernel(
    x_ptr, kernel_ptr, y_ptr, dy_ptr, dk_ptr,
    H: tl.constexpr, W: tl.constexpr, TAU: tl.constexpr, BLOCK: tl.constexpr,
):
    # Each output contributes to all causal kernel offsets; V1 uses atomic adds.
    pid = tl.program_id(0)
    batch = pid // (H * W)
    spatial = pid % (H * W)
    row = spatial // W
    col = spatial % W
    u = tl.arange(0, BLOCK)
    max_u = tl.minimum(row, col)
    valid = (u <= max_u) & (u < H) & (u < W)
    center = batch * H * W + row * W + col
    x_vals = tl.load(x_ptr + center - u * (W + 1), mask=valid, other=0.0)
    k_vals = tl.load(kernel_ptr + u, mask=valid, other=0.0)
    y = tl.load(y_ptr + center)
    dy = tl.load(dy_ptr + center)
    LOG2E: tl.constexpr = 1.4426950408889634
    log_probability = tl.where(
        valid, (x_vals + k_vals - y) * (LOG2E / TAU), -float("inf")
    )
    tl.atomic_add(dk_ptr + u, dy * tl.exp2(log_probability), mask=valid)


class DiagonalLSEFunction(torch.autograd.Function):
    """First-order FP32 CUDA diagonal LSE; tau is a positive constant."""

    @staticmethod
    def forward(ctx, x, kernel, tau):
        if x.ndim != 3 or min(x.shape) < 1:
            raise ValueError('Expected nonempty x shaped [B, H, W]')
        if kernel.ndim != 1 or kernel.numel() < min(x.shape[-2:]):
            raise ValueError('Kernel must be 1D with at least min(H, W) entries')
        if not x.is_cuda or not kernel.is_cuda or x.device != kernel.device:
            raise ValueError('x and kernel must be on the same CUDA device')
        if x.dtype != torch.float32 or kernel.dtype != torch.float32:
            raise TypeError('V1 supports float32 x and kernel')
        if isinstance(tau, torch.Tensor):
            raise TypeError('tau must be a constant Python number')
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be positive and finite')
        x, kernel = x.contiguous(), kernel.contiguous()
        batch, height, width = x.shape
        block = triton.next_power_of_2(min(height, width))
        out = torch.empty_like(x)
        with torch.cuda.device(x.device):
            diagonal_lse_kernel[(batch * height * width,)](
                x, kernel, out, H=height, W=width, TAU=tau, BLOCK=block,
            )
        ctx.save_for_backward(x, kernel, out)
        ctx.height, ctx.width, ctx.tau, ctx.block = height, width, tau, block
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, kernel, y = ctx.saved_tensors
        height, width, tau, block = ctx.height, ctx.width, ctx.tau, ctx.block
        grad_out = grad_out.contiguous()
        grid = (x.shape[0] * height * width,)
        dx, dk = None, None
        with torch.cuda.device(x.device):
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(x)
                diagonal_lse_bwd_dx_kernel[grid](
                    x, kernel, y, grad_out, dx,
                    H=height, W=width, TAU=tau, BLOCK=block,
                )
            if ctx.needs_input_grad[1]:
                # Every launch starts with zeros; unused kernel tail stays zero.
                dk = torch.zeros_like(kernel)
                diagonal_lse_bwd_dk_kernel[grid](
                    x, kernel, y, grad_out, dk,
                    H=height, W=width, TAU=tau, BLOCK=block,
                )
        return dx, dk, None


class FusedDiagonalLSE(nn.Module):
    """A trainable diagonal LSE kernel backed by the Triton V1 operation."""

    def __init__(self, max_length, tau=0.1):
        super().__init__()
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError('max_length must be a positive integer')
        if not math.isfinite(float(tau)) or tau <= 0:
            raise ValueError('tau must be positive and finite')
        self.tau = float(tau)
        self.kernel = nn.Parameter(torch.zeros(max_length))

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected x shaped [B, H, W]')
        length = min(x.shape[-2:])
        if length > self.kernel.numel():
            raise ValueError('Input diagonal exceeds max_length')
        return DiagonalLSEFunction.apply(x, self.kernel[:length], self.tau)


# Keep the same directional interface and parameter/state-dict layout; both
# canonical axis and diagonal operations now use Triton forward/backward.


class TritonMorphologicalMax(OptimisedMorphologicalMax):
    """Full directional LSE interface with Triton cardinal and diagonal scans."""

    def _axis_lse(self, x):
        return AxisLSEFunction.apply(x, self.kernel, self.tau)

    def _diagonal_lse(self, x):
        return DiagonalLSEFunction.apply(x, self.diagonal_kernel, self.tau)
