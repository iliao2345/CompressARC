"""Direct FP32 CUDA depthwise 3x3 convolution with input/filter autograd."""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .morphological import C4DirectionalConv


@triton.jit
def depthwise_forward(x_ptr, k_ptr, y_ptr, N: tl.constexpr, C: tl.constexpr,
                      H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < N * C * H * W
    column = indices % W
    row = (indices // W) % H
    channel = (indices // (H * W)) % C
    base = (indices // (H * W)) * H * W
    result = tl.full((BLOCK,), 0., tl.float32)
    for kr in tl.static_range(3):
        for kc in tl.static_range(3):
            rr = row + kr - 1
            cc = column + kc - 1
            inside = valid & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            value = tl.load(x_ptr + base + rr * W + cc, mask=inside, other=0.)
            weight = tl.load(k_ptr + channel * 9 + kr * 3 + kc, mask=valid, other=0.)
            result += value * weight
    tl.store(y_ptr + indices, result, mask=valid)


@triton.jit
def depthwise_backward_input(dy_ptr, k_ptr, dx_ptr, N: tl.constexpr, C: tl.constexpr,
                             H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < N * C * H * W
    column = indices % W
    row = (indices // W) % H
    channel = (indices // (H * W)) % C
    base = (indices // (H * W)) * H * W
    result = tl.full((BLOCK,), 0., tl.float32)
    for kr in tl.static_range(3):
        for kc in tl.static_range(3):
            rr = row - kr + 1
            cc = column - kc + 1
            inside = valid & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            value = tl.load(dy_ptr + base + rr * W + cc, mask=inside, other=0.)
            weight = tl.load(k_ptr + channel * 9 + kr * 3 + kc, mask=valid, other=0.)
            result += value * weight
    tl.store(dx_ptr + indices, result, mask=valid)


@triton.jit
def depthwise_backward_kernel(x_ptr, dy_ptr, dk_ptr, N: tl.constexpr, C: tl.constexpr,
                              H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    # A program reduces a tile for one channel/filter tap; only its partial sum
    # is added atomically, rather than one atomic add per pixel.
    tap = tl.program_id(1)
    channel = tap // 9
    kr = (tap % 9) // 3
    kc = tap % 3
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < N * H * W
    batch = indices // (H * W)
    row = (indices // W) % H
    column = indices % W
    rr = row + kr - 1
    cc = column + kc - 1
    base = (batch * C + channel) * H * W
    inside = valid & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    value = tl.load(x_ptr + base + rr * W + cc, mask=inside, other=0.)
    upstream = tl.load(dy_ptr + base + row * W + column, mask=valid, other=0.)
    contribution = tl.sum(value * upstream, axis=0)
    tl.atomic_add(dk_ptr + tap, contribution)


class DepthwiseConvFunction(torch.autograd.Function):
    """First-order FP32 CUDA correlation, stride 1, padding 1, groups=C."""

    @staticmethod
    def forward(ctx, x, kernel):
        if x.ndim != 4 or min(x.shape) < 1:
            raise ValueError("Expected nonempty [N, C, H, W]")
        n, c, h, w = x.shape
        if kernel.shape != (c, 1, 3, 3):
            raise ValueError("Kernel must have shape [C, 1, 3, 3]")
        if not x.is_cuda or not kernel.is_cuda or x.device != kernel.device:
            raise ValueError("Input and kernel must share a CUDA device")
        if x.dtype != torch.float32 or kernel.dtype != torch.float32:
            raise TypeError("Triton shift supports float32 input and kernel")
        x, kernel = x.contiguous(), kernel.contiguous()
        output = torch.empty_like(x)
        block = 256
        with torch.cuda.device(x.device):
            depthwise_forward[(triton.cdiv(x.numel(), block),)](
                x, kernel, output, N=n, C=c, H=h, W=w, BLOCK=block,
            )
        ctx.save_for_backward(x, kernel)
        ctx.shape, ctx.block = (n, c, h, w), block
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, kernel = ctx.saved_tensors
        n, c, h, w = ctx.shape
        grad_out = grad_out.contiguous()
        dx, dk = None, None
        with torch.cuda.device(x.device):
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(x)
                depthwise_backward_input[(triton.cdiv(x.numel(), ctx.block),)](
                    grad_out, kernel, dx, N=n, C=c, H=h, W=w, BLOCK=ctx.block,
                )
            if ctx.needs_input_grad[1]:
                dk = torch.zeros_like(kernel)
                depthwise_backward_kernel[(triton.cdiv(n * h * w, ctx.block), c * 9)](
                    x, grad_out, dk, N=n, C=c, H=h, W=w, BLOCK=ctx.block,
                )
        return dx, dk


class TritonTiedDirectionalConv(C4DirectionalConv):
    """Direct Triton core; canonical weight tying/rotations remain differentiable."""

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:3] != (2, 4) or min(x.shape) < 1:
            raise ValueError("Expected nonempty [B, 2, 4, H, W]")
        batch, _, _, height, width = x.shape
        output = DepthwiseConvFunction.apply(
            x.reshape(batch, 8, height, width), self.directional_kernels(),
        )
        return output.reshape(batch, 2, 4, height, width)
