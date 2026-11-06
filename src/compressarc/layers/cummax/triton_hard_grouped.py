"""Opt-in eight-position cardinal backward; original forward/diagonal unchanged."""
import torch
import triton
from torch.autograd.function import once_differentiable
from . import triton_hard as base
from .launch_benchmark import _axis_backward_grouped
from ..helper import apply_residual, make_directional_layer, only_do_for_certain_shapes


class _AxisGrouped(base._Axis):
    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x,mask,index,hi,lo = ctx.saved_tensors
        dx = torch.empty_like(x)
        length = x.shape[-1]
        rows = x.numel()//length
        with torch.cuda.device(x.device):
            _axis_backward_grouped[(rows,triton.cdiv(length,8))](
                x,mask,index,hi,lo,grad.contiguous(),dx,
                L=length,B=triton.next_power_of_2(length),P=8,
                num_warps=4,enable_fp_fusion=False)
        return dx,None


def hard_cummax_grouped(x, dim, masks):
    axis = dim % x.ndim
    mask = masks.expand_as(x)
    return _AxisGrouped.apply(x.movedim(axis,-1),mask.movedim(axis,-1)).movedim(-1,axis)


class TritonHardCummaxGrouped(base.TritonHardCummax):
    def __init__(self, multify):
        directional = make_directional_layer(hard_cummax_grouped,base.hard_diagonal_cummax)
        def apply_one(dims,x,weights,masks,**kwargs):
            return apply_residual(x,weights,lambda z:directional(dims,z,masks),**kwargs)
        self._apply = multify(only_do_for_certain_shapes((1,1,1,1,1),(1,0,1,1,1))(apply_one))
