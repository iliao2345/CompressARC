"""FP32 CUDA hard cummax, preserving primitive scan topology and tie rules.

V1 accepts finite FP32 inputs and fixed masks, with first-order gradients only.
No learned scan weights. Cardinal scans use last-axis row programs; diagonal
scans fuse max/min at each original associative stage. Diagonal history retains
the original tie-split backward rather than substituting an argmax derivative.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from ..helper import apply_residual, make_directional_layer, only_do_for_certain_shapes
from .primitives import CummaxPrimitives


@triton.jit
def _latest_max(a, ai, b, bi):
    take_b = (b >= a)
    return tl.where(take_b, b, a), tl.where(take_b, bi, ai)


@triton.jit
def _axis_forward(X, M, Y, INDEX, HI, LO, L:tl.constexpr, B:tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    valid = j < L
    x = tl.load(X + row*L+j, valid, other=0.)
    mask = tl.load(M + row*L+j, valid, other=0.)
    penalty = 1000.*(1.-mask)
    a = tl.where(valid, x-penalty, -float('inf'))
    b = tl.where(valid, x+penalty, float('inf'))
    high, low = tl.max(a, 0), tl.min(b, 0)
    # max/min(dim) choose the first equal extremum; cummax chooses the latest.
    hi = tl.min(tl.where(valid & (a == high), j, 2147483647), 0)
    lo = tl.min(tl.where(valid & (b == low), j, 2147483647), 0)
    prefix, index = tl.associative_scan((a, j), 0, _latest_max)
    numerator = (prefix+penalty) - (low-penalty-0.001)
    denominator = (high+penalty+0.001) - (low-penalty-0.001)
    y = numerator/denominator*2.-1.
    tl.store(Y+row*L+j, y, valid)
    tl.store(INDEX+row*L+j, index, valid)
    tl.store(HI+row, hi)
    tl.store(LO+row, lo)


@triton.jit
def _axis_backward(X, M, INDEX, HI, LO, DY, DX, L:tl.constexpr, B:tl.constexpr):
    pid = tl.program_id(0)
    row, position = pid//L, pid%L
    j = tl.arange(0, B)
    valid = j<L
    idx = tl.load(INDEX+row*L+j, valid, other=0)
    mask = tl.load(M+row*L+j, valid, other=0.)
    penalty = 1000.*(1.-mask)
    selected = tl.load(X+row*L+idx, valid, other=0.)
    selected_mask = tl.load(M+row*L+idx, valid, other=0.)
    prefix = selected-1000.*(1.-selected_mask)
    hi, lo = tl.load(HI+row), tl.load(LO+row)
    high = tl.load(X+row*L+hi)-1000.*(1.-tl.load(M+row*L+hi))
    low = tl.load(X+row*L+lo)+1000.*(1.-tl.load(M+row*L+lo))
    numerator = (prefix+penalty)-(low-penalty-0.001)
    denominator = (high+penalty+0.001)-(low-penalty-0.001)
    grad = tl.load(DY+row*L+j, valid, other=0.)*2.
    gn = grad/denominator
    gd = -grad*numerator/(denominator*denominator)
    contribution = tl.where(idx==position, gn, 0.)
    contribution += tl.where(position==hi, gd, 0.)
    contribution += tl.where(position==lo, -gn-gd, 0.)
    dx = tl.sum(tl.where(valid, contribution, 0.), 0)
    tl.store(DX+pid, dx)


@triton.jit
def _diag_init(X, M, MAX, MIN, N:tl.constexpr, B:tl.constexpr):
    p = tl.program_id(0)*B+tl.arange(0,B)
    x = tl.load(X+p,p<N,other=0.)
    penalty = 1000.*(1.-tl.load(M+p,p<N,other=0.))
    tl.store(MAX+p,x-penalty,p<N)
    tl.store(MIN+p,x+penalty,p<N)


@triton.jit
def _diag_stage(MAX, MIN, NEXTMAX, NEXTMIN, N:tl.constexpr, H:tl.constexpr,
                W:tl.constexpr, SHIFT:tl.constexpr, B:tl.constexpr):
    p = tl.program_id(0)*B+tl.arange(0,B)
    r, c = (p//W)%H, p%W
    valid = (p<N)&(r-SHIFT>=0)&(r-SHIFT<H)&(c-SHIFT>=0)&(c-SHIFT<W)
    q = p-SHIFT*(W+1)
    a = tl.load(MAX+p,p<N,other=0.)
    b = tl.load(MAX+q,valid,other=-1000.)
    u = tl.load(MIN+p,p<N,other=0.)
    v = tl.load(MIN+q,valid,other=1000.)
    tl.store(NEXTMAX+p,tl.maximum(a,b),p<N)
    tl.store(NEXTMIN+p,tl.minimum(u,v),p<N)


@triton.jit
def _diag_output(PREFIX, MAX, MIN, M, Y, N:tl.constexpr, B:tl.constexpr):
    p = tl.program_id(0)*B+tl.arange(0,B)
    mask = tl.load(M+p,p<N,other=0.)
    penalty = 1000.*(1.-mask)
    a = tl.load(PREFIX+p,p<N,other=0.)+penalty
    high = tl.load(MAX+p,p<N,other=0.)+penalty
    low = tl.load(MIN+p,p<N,other=0.)-penalty
    y = ((a-low)/(high-low+0.00001)*2.-1.)*mask
    tl.store(Y+p,y,p<N)


@triton.jit
def _diag_grad_output(PREFIX, MAX, MIN, M, DY, GP, GH, GL, N:tl.constexpr, B:tl.constexpr):
    p = tl.program_id(0)*B+tl.arange(0,B)
    mask = tl.load(M+p,p<N,other=0.)
    penalty = 1000.*(1.-mask)
    a = tl.load(PREFIX+p,p<N,other=0.)+penalty
    high = tl.load(MAX+p,p<N,other=0.)+penalty
    low = tl.load(MIN+p,p<N,other=0.)-penalty
    denominator = high-low+0.00001
    grad = tl.load(DY+p,p<N,other=0.)*mask*2.
    gn = grad/denominator
    gd = -grad*(a-low)/(denominator*denominator)
    tl.store(GP+p,gn,p<N)
    tl.store(GH+p,gd,p<N)
    tl.store(GL+p,-gn-gd,p<N)


@triton.jit
def _diag_grad_stage(MAX, MIN, GH, GL, GP, NEXTGH, NEXTGL,
                     N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,SHIFT:tl.constexpr,
                     ADD_PREFIX:tl.constexpr,B:tl.constexpr):
    p = tl.program_id(0)*B+tl.arange(0,B)
    r,c = (p//W)%H,p%W
    before = (p<N)&(r-SHIFT>=0)&(r-SHIFT<H)&(c-SHIFT>=0)&(c-SHIFT<W)
    after = (p<N)&(r+SHIFT>=0)&(r+SHIFT<H)&(c+SHIFT>=0)&(c+SHIFT<W)
    q,s = p-SHIFT*(W+1),p+SHIFT*(W+1)
    a = tl.load(MAX+p,p<N,other=0.)
    b = tl.load(MAX+q,before,other=-1000.)
    u = tl.load(MIN+p,p<N,other=0.)
    v = tl.load(MIN+q,before,other=1000.)
    # Elementwise torch.maximum/minimum split equal-value gradients in half,
    # including ties with constant padding (whose half is discarded).
    gh = tl.load(GH+p,p<N,other=0.)*tl.where(a>b,1.,tl.where(a==b,.5,0.))
    gl = tl.load(GL+p,p<N,other=0.)*tl.where(u<v,1.,tl.where(u==v,.5,0.))
    target_max = tl.load(MAX+s,after,other=0.)
    target_min = tl.load(MIN+s,after,other=0.)
    gh += tl.load(GH+s,after,other=0.)*tl.where(a>target_max,1.,tl.where(a==target_max,.5,0.))
    gl += tl.load(GL+s,after,other=0.)*tl.where(u<target_min,1.,tl.where(u==target_min,.5,0.))
    if ADD_PREFIX:
        gh += tl.load(GP+p,p<N,other=0.)
    tl.store(NEXTGH+p,gh,p<N)
    tl.store(NEXTGL+p,gl,p<N)


def _check(x, mask):
    if not x.is_cuda or mask.device!=x.device:
        raise ValueError('hard Triton cummax requires x and masks on the same CUDA device')
    if x.dtype!=torch.float32 or mask.dtype!=torch.float32:
        raise TypeError('hard Triton cummax supports float32 only')
    if mask.requires_grad:
        raise ValueError('mask gradients are unsupported; task masks must be fixed')
    if not x.numel():
        raise ValueError('hard Triton cummax requires nonempty tensors')


class _Axis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        _check(x,mask)
        x,mask=x.contiguous(),mask.expand_as(x).contiguous()
        length=x.shape[-1]
        if length>128:
            raise ValueError('V1 cardinal scan supports axis lengths up to 128')
        y=torch.empty_like(x); indices=torch.empty_like(x,dtype=torch.int32)
        hi=torch.empty(x.numel()//length,device=x.device,dtype=torch.int32);lo=torch.empty_like(hi)
        with torch.cuda.device(x.device):
            _axis_forward[(hi.numel(),)](x,mask,y,indices,hi,lo,L=length,B=triton.next_power_of_2(length),enable_fp_fusion=False)
        ctx.save_for_backward(x,mask,indices,hi,lo)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x,mask,index,hi,lo=ctx.saved_tensors
        dx=torch.empty_like(x);length=x.shape[-1]
        with torch.cuda.device(x.device):
            _axis_backward[(x.numel(),)](x,mask,index,hi,lo,grad.contiguous(),dx,L=length,
                                       B=triton.next_power_of_2(length),enable_fp_fusion=False)
        return dx,None


class _Diagonal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        _check(x,mask)
        x,mask=x.contiguous(),mask.expand_as(x).contiguous()
        h,w=x.shape[-2:];n=x.numel();rounds=(min(h,w)-1).bit_length()
        shifts=[sign*2**i for sign in (1,-1) for i in range(rounds)]
        high=torch.empty((len(shifts)+1,*x.shape),device=x.device,dtype=x.dtype)
        low=torch.empty_like(high)
        grid=(triton.cdiv(n,256),)
        with torch.cuda.device(x.device):
            _diag_init[grid](x,mask,high[0],low[0],N=n,B=256,enable_fp_fusion=False)
            for i,shift in enumerate(shifts):
                _diag_stage[grid](high[i],low[i],high[i+1],low[i+1],N=n,H=h,W=w,SHIFT=shift,B=256,enable_fp_fusion=False)
            y=torch.empty_like(x)
            _diag_output[grid](high[rounds],high[-1],low[-1],mask,y,N=n,B=256,enable_fp_fusion=False)
        ctx.save_for_backward(high,low,mask)
        ctx.shifts,ctx.rounds=shifts,rounds
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        high,low,mask=ctx.saved_tensors
        h,w=mask.shape[-2:];n=mask.numel();grid=(triton.cdiv(n,256),)
        gp,gh,gl=[torch.empty_like(mask) for _ in range(3)]
        with torch.cuda.device(mask.device):
            _diag_grad_output[grid](high[ctx.rounds],high[-1],low[-1],mask,grad.contiguous(),gp,gh,gl,N=n,B=256,enable_fp_fusion=False)
            scratch_h,scratch_l=torch.empty_like(gh),torch.empty_like(gl)
            for i in reversed(range(len(ctx.shifts))):
                _diag_grad_stage[grid](high[i],low[i],gh,gl,gp,scratch_h,scratch_l,
                    N=n,H=h,W=w,SHIFT=ctx.shifts[i],ADD_PREFIX=(i==ctx.rounds),B=256,enable_fp_fusion=False)
                gh,scratch_h=scratch_h,gh
                gl,scratch_l=scratch_l,gl
            dx=gh+gl
            if not ctx.shifts:
                dx=dx+gp
        return dx,None


def hard_cummax(x, dim, masks):
    """Cardinal primitive with the scan axis moved to the last dimension."""
    axis=dim%x.ndim
    mask=masks.expand_as(x)
    return _Axis.apply(x.movedim(axis,-1),mask.movedim(axis,-1)).movedim(-1,axis)


def hard_diagonal_cummax(x, dim1, dim2, masks):
    axes=(dim1%x.ndim,dim2%x.ndim)
    if axes[0]==axes[1]:
        raise ValueError('diagonal spatial axes must differ')
    mask=masks.expand_as(x)
    y=_Diagonal.apply(x.movedim(axes,(-2,-1)),mask.movedim(axes,(-2,-1)))
    return y.movedim((-2,-1),axes)


class TritonHardCummax(CummaxPrimitives):
    def __init__(self, multify):
        directional=make_directional_layer(hard_cummax,hard_diagonal_cummax)
        def apply_one(dims,x,weights,masks,**kwargs):
            return apply_residual(x,weights,lambda z:directional(dims,z,masks),**kwargs)
        self._apply=multify(only_do_for_certain_shapes((1,1,1,1,1),(1,0,1,1,1))(apply_one))
