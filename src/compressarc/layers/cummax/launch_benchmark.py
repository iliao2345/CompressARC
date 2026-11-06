"""Experimental launch layouts for benchmarking only; training defaults unchanged."""
import torch
import triton
import triton.language as tl
from . import triton_hard as base


@triton.jit
def _axis_backward_grouped(X, M, INDEX, HI, LO, DY, DX,
                           L: tl.constexpr, B: tl.constexpr, P: tl.constexpr):
    row = tl.program_id(0)
    position = tl.program_id(1) * P + tl.arange(0, P)
    j = tl.arange(0, B)
    valid = j < L
    idx = tl.load(INDEX + row*L+j, valid, other=0)
    mask = tl.load(M + row*L+j, valid, other=0.)
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
    contribution = tl.where(idx[None,:] == position[:,None], gn[None,:], 0.)
    contribution += tl.where(position[:,None] == hi, gd[None,:], 0.)
    contribution += tl.where(position[:,None] == lo, (-gn-gd)[None,:], 0.)
    dx = tl.sum(tl.where(valid[None,:], contribution, 0.), 1)
    tl.store(DX+row*L+position, dx, position < L)


def axis_plan(x, mask, dy, positions=1, warps=4):
    """Preallocate an axis forward/backward pipeline with configurable work tiling."""
    base._check(x, mask)
    if dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError('Upstream gradient must match input device/dtype')
    if x.ndim != 2 or not all(t.is_contiguous() for t in (x, mask, dy)):
        raise ValueError('Axis benchmark expects contiguous [rows,length] inputs')
    if x.shape != mask.shape or dy.shape != x.shape:
        raise ValueError('Inputs, fixed masks and upstream gradients must have the same shape')
    if positions not in (1,2,4,8,16,32) or warps not in (1,2,4,8):
        raise ValueError('Unsupported positions/warps')
    rows, length = x.shape
    if not 1 <= length <= 128:
        raise ValueError('Axis lengths must be in [1,128]')
    y, dx = torch.empty_like(x), torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int32)
    hi = torch.empty(rows, device=x.device, dtype=torch.int32)
    lo = torch.empty_like(hi)
    options = dict(L=length, B=triton.next_power_of_2(length),
                   num_warps=warps, enable_fp_fusion=False)
    def forward():
        base._axis_forward[(rows,)](x, mask, y, indices, hi, lo, **options)
    def backward():
        if positions == 1:
            base._axis_backward[(x.numel(),)](x, mask, indices, hi, lo, dy, dx, **options)
        else:
            _axis_backward_grouped[(rows, triton.cdiv(length,positions))](
                x, mask, indices, hi, lo, dy, dx, P=positions, **options)
    return dict(forward=forward, backward=backward, output=y, gradient=dx,
                forward_programs=rows, backward_programs=rows*triton.cdiv(length,positions),
                forward_launches=1, backward_launches=1)


def diagonal_plan(x, mask, dy, block=256, warps=4):
    """Same original diagonal tree and tie derivatives, varying block/warp layout."""
    base._check(x, mask)
    if dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError('Upstream gradient must match input device/dtype')
    if x.ndim != 3 or not all(t.is_contiguous() for t in (x,mask,dy)):
        raise ValueError('Diagonal benchmark expects contiguous [batch,height,width]')
    if x.shape != mask.shape or dy.shape != x.shape:
        raise ValueError('Inputs, masks and upstream gradients must have identical shapes')
    if block not in (64,128,256,512,1024) or warps not in (1,2,4,8):
        raise ValueError('Unsupported block/warps')
    h,w = x.shape[-2:]; n = x.numel(); rounds = (min(h,w)-1).bit_length()
    shifts = [sign*2**i for sign in (1,-1) for i in range(rounds)]
    high = torch.empty((len(shifts)+1,*x.shape), device=x.device, dtype=x.dtype)
    low = torch.empty_like(high)
    y, dx, gp, initial_h, initial_l, scratch_h, scratch_l = [torch.empty_like(x) for _ in range(7)]
    grid = (triton.cdiv(n,block),)
    options = dict(N=n, B=block, num_warps=warps, enable_fp_fusion=False)
    def forward():
        base._diag_init[grid](x,mask,high[0],low[0],**options)
        for i,shift in enumerate(shifts):
            base._diag_stage[grid](high[i],low[i],high[i+1],low[i+1],H=h,W=w,SHIFT=shift,**options)
        base._diag_output[grid](high[rounds],high[-1],low[-1],mask,y,**options)
    def backward():
        gh, gl, sh, sl = initial_h, initial_l, scratch_h, scratch_l
        base._diag_grad_output[grid](high[rounds],high[-1],low[-1],mask,dy,gp,gh,gl,**options)
        for i in reversed(range(len(shifts))):
            base._diag_grad_stage[grid](high[i],low[i],gh,gl,gp,sh,sl,
                                      H=h,W=w,SHIFT=shifts[i],ADD_PREFIX=(i==rounds),**options)
            gh, sh, gl, sl = sh, gh, sl, gl
        torch.add(gh, gl, out=dx)
        if not shifts:
            dx.add_(gp)
    return dict(forward=forward, backward=backward, output=y, gradient=dx,
                forward_programs=grid[0], backward_programs=grid[0],
                forward_launches=len(shifts)+2, backward_launches=len(shifts)+2+(not shifts))
