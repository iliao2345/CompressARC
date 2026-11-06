"""Optional FP32 CUDA sharing normalization with two kernels per forward/backward.

The population variance is computed from explicitly centered inputs, with the
original epsilon 1e-8. Unsupported layouts use the primitive implementation.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _statistics(X, S, N:tl.constexpr, C:tl.constexpr, B:tl.constexpr):
    c=tl.program_id(0)
    i=tl.arange(0,B)
    x=tl.load(X+i*C+c,i<N,other=0.)
    mean=tl.sum(tl.where(i<N,x,0.),0)/N
    centered=x-mean
    variance=tl.sum(tl.where(i<N,centered*centered,0.),0)/N
    rstd=tl.rsqrt(variance+1.e-8)
    tl.store(S+c,mean)
    tl.store(S+C+c,rstd)


@triton.jit
def _apply(X,S,Y,N:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    p=tl.program_id(0)*B+tl.arange(0,B)
    c=p%C
    x=tl.load(X+p,p<N*C,other=0.)
    mean=tl.load(S+c)
    rstd=tl.load(S+C+c)
    tl.store(Y+p,(x-mean)*rstd,p<N*C)


@triton.jit
def _gradient_statistics(X,S,DY,G,N:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    c=tl.program_id(0)
    i=tl.arange(0,B)
    mean=tl.load(S+c)
    rstd=tl.load(S+C+c)
    centered=tl.load(X+i*C+c,i<N,other=0.)-mean
    grad=tl.load(DY+i*C+c,i<N,other=0.)
    mean_gc=tl.sum(tl.where(i<N,grad*centered,0.),0)/N
    correction=rstd*rstd*rstd*mean_gc
    gc=grad*rstd-centered*correction
    # Keep the centering derivative explicit rather than assuming mean(centered)==0.
    mean_gradient=tl.sum(tl.where(i<N,gc,0.),0)/N
    tl.store(G+c,correction)
    tl.store(G+C+c,mean_gradient)


@triton.jit
def _gradient_apply(X,S,DY,G,DX,N:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    p=tl.program_id(0)*B+tl.arange(0,B)
    c=p%C
    mean=tl.load(S+c)
    rstd=tl.load(S+C+c)
    correction=tl.load(G+c)
    mean_gradient=tl.load(G+C+c)
    centered=tl.load(X+p,p<N*C,other=0.)-mean
    grad=tl.load(DY+p,p<N*C,other=0.)
    dx=grad*rstd-centered*correction-mean_gradient
    tl.store(DX+p,dx,p<N*C)


class _Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        channels=x.shape[-1]
        n=x.numel()//channels
        stats=torch.empty(2*channels,device=x.device,dtype=torch.float32)
        y=torch.empty_like(x)
        with torch.cuda.device(x.device):
            _statistics[(channels,)](x,stats,N=n,C=channels,B=triton.next_power_of_2(n),num_warps=8 if n>=4096 else 4,enable_fp_fusion=False)
            _apply[(triton.cdiv(x.numel(),256),)](x,stats,y,N=n,C=channels,B=256,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,stats)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,stats=ctx.saved_tensors
        channels=x.shape[-1]
        n=x.numel()//channels
        grad=grad.contiguous()
        gradient_stats=torch.empty_like(stats)
        dx=torch.empty_like(x)
        with torch.cuda.device(x.device):
            _gradient_statistics[(channels,)](x,stats,grad,gradient_stats,N=n,C=channels,
                                             B=triton.next_power_of_2(n),num_warps=8 if n>=4096 else 4,enable_fp_fusion=False)
            _gradient_apply[(triton.cdiv(x.numel(),256),)](x,stats,grad,gradient_stats,dx,
                N=n,C=channels,B=256,num_warps=4,enable_fp_fusion=False)
        return dx


def normalize_tensor(x):
    """First-order fused path; fallback keeps CPU/FP64/strided/large cases intact."""
    from ..helper import normalize
    if not eligible(x):
        return normalize(x)
    return _Normalize.apply(x)


def eligible(x):
    return (x.is_cuda and x.dtype==torch.float32 and x.ndim>=2 and
            x.is_contiguous() and x.shape[-1]>0 and
            0<x.numel()//x.shape[-1]<=8192)
