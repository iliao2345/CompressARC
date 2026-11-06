"""Per-forward sharing DAG reuse, with optional shape-compatible batching.

The plan caches only layout metadata. Values/masks/denominators and autograd
graphs are fresh each call. Destination addition order and the exact masked
reduction sequence (including each +1e-4) match the original primitive.
"""
from collections import defaultdict
from ..helper import affine,normalize
from .primitives import MultitensorShare
from ...model import multitensor_systems
from ...train.layer_profiling import profile_share_stage
import torch


def make_plan(layouts,direction,known_size):
    layouts=tuple(tuple(d) for d in layouts)
    nodes=[(None,('source',d,None)) for d in layouts]
    levels=[0]*len(nodes)
    lookup={}
    contributions=[]
    requested=0
    for target in layouts:
        inputs=[]
        for source_id,source in enumerate(layouts):
            eligible=all(a<=b for a,b in zip(source,target)) if direction==1 else all(a>=b for a,b in zip(source,target))
            if not eligible:continue
            parent=source_id
            axes=range(5) if direction==1 else reversed(range(5))
            for dim in axes:
                if source[dim]==target[dim]:continue
                requested+=1
                if direction==1:
                    op=('unsqueeze',sum(target[:dim]),None)
                else:
                    axis=sum(source[:dim])
                    if source[0] and known_size and dim in (3,4):
                        drop=(target[4]==0) if dim==3 else (source[3]==0)
                        op=('masked',axis,(sum(source[1:3]),dim,drop))
                    else:op=('mean',axis,None)
                key=(parent,op)
                if key not in lookup:
                    lookup[key]=len(nodes)
                    nodes.append(key)
                    levels.append(levels[parent]+1)
                parent=lookup[key]
            inputs.append(parent)
        contributions.append(tuple(inputs))
    return dict(layouts=layouts,nodes=nodes,levels=levels,contributions=contributions,
                requested_steps=requested)


class ReusedMultitensorShare(MultitensorShare):
    def __init__(self,batch=False,max_batch_elements=262144,fused_normalization=False):
        if max_batch_elements<1:raise ValueError('share_batch_max_elements must be positive')
        self.fused_normalization=fused_normalization
        self.batch=batch
        self.max_batch_elements=int(max_batch_elements)
        self._plans={}
        self.last_stats={}

    def communicate(self,projected,direction):
        system=projected.multitensor_system
        task=system.task
        known_size=bool(task.in_out_same_size or task.all_out_same_size) if direction==-1 else False
        layouts=tuple(tuple(d) for d in system)
        key=(layouts,direction,known_size)
        if key not in self._plans:self._plans[key]=make_plan(layouts,direction,known_size)
        plan=self._plans[key]
        values=[projected[d] for d in layouts]+[None]*(len(plan['nodes'])-len(layouts))
        masks={};denominators={};union=None
        stats=dict(requested_steps=plan['requested_steps'],unique_steps=len(plan['nodes'])-len(layouts),
                   reduction_calls=0,batch_groups=0,batched_nodes=0)

        def mask_and_denominator(op):
            nonlocal union
            _,axis,spec=op
            if spec not in masks:
                if union is None:
                    union=1-(1-task.masks[...,0])*(1-task.masks[...,1])
                expanded=union
                count,dim,drop=spec
                for _ in range(count):expanded=expanded[:,None,...]
                if drop:expanded=expanded[...,0] if dim==3 else expanded[...,0,:]
                masks[spec]=expanded[...,None]
            denominator_key=(axis,spec)
            if denominator_key not in denominators:
                denominators[denominator_key]=torch.sum(masks[spec],dim=axis)+1e-4
            return masks[spec],denominators[denominator_key]

        def reduce_one(value,op):
            stats['reduction_calls']+=1
            kind,axis,_=op
            if kind=='mean':return torch.mean(value,dim=axis)
            mask,denominator=mask_and_denominator(op)
            return torch.sum(value*mask,dim=axis)/denominator

        def single(index):
            parent,op=plan['nodes'][index]
            if op[0]=='unsqueeze':values[index]=torch.unsqueeze(values[parent],op[1])
            else:values[index]=reduce_one(values[parent],op)

        if not self.batch:
            for index in range(len(layouts),len(values)):single(index)
        else:
            # Level ordering batches only independent nodes with computed parents.
            for level in range(1,max(plan['levels'],default=0)+1):
                groups=defaultdict(list)
                for index in range(len(layouts),len(values)):
                    if plan['levels'][index]!=level:continue
                    parent,op=plan['nodes'][index]
                    if op[0]=='unsqueeze':single(index);continue
                    value=values[parent]
                    groups[(op,tuple(value.shape),value.dtype,value.device)].append(index)
                for (op,shape,dtype,device),indices in groups.items():
                    first=values[plan['nodes'][indices[0]][0]]
                    capacity=max(1,self.max_batch_elements//first.numel()) if first.numel() else 1
                    for start in range(0,len(indices),capacity):
                        chunk=indices[start:start+capacity]
                        if len(chunk)==1:
                            single(chunk[0]);continue
                        stacked=torch.stack([values[plan['nodes'][i][0]] for i in chunk])
                        if op[0]=='mean':result=stacked.mean(dim=op[1]+1)
                        else:
                            mask,denominator=mask_and_denominator(op)
                            result=(stacked*mask.unsqueeze(0)).sum(dim=op[1]+1)/denominator.unsqueeze(0)
                        for index,value in zip(chunk,result.unbind(0)):values[index]=value
                        stats['reduction_calls']+=1
                        stats['batch_groups']+=1
                        stats['batched_nodes']+=len(chunk)
        result=system.make_multitensor()
        for layout,inputs in zip(layouts,plan['contributions']):
            result[layout]=sum(values[i] for i in inputs)
        stats['mask_layouts']=len(masks)
        stats['mask_denominators']=len(denominators)
        self.last_stats['share_up' if direction==1 else 'share_down']=stats
        return result

    def share_direction(self,residual,share_weights,direction):
        if direction not in (1,-1):raise ValueError('direction must be 1 (up) or -1 (down)')
        component='share_up' if direction==1 else 'share_down'
        with profile_share_stage(component,'weight_selection'):
            down=multitensor_systems.multify(lambda dims,w:w[0])(share_weights)
            up=multitensor_systems.multify(lambda dims,w:w[1])(share_weights)
        with profile_share_stage(component,'source_projection'):
            projected=affine(residual,down,use_bias=False)
        with profile_share_stage(component,'communication'):
            communicated=self.communicate(projected,direction)
        with profile_share_stage(component,'normalization'):
            if self.fused_normalization:
                def normalize_one(dims,value):
                    # Optional Triton import is never required for CPU/FP64 paths.
                    if value.is_cuda and value.dtype==torch.float32:
                        from .normalization_triton import normalize_tensor
                        return normalize_tensor(value)
                    return normalize(value)
                communicated=multitensor_systems.multify(normalize_one)(communicated)
            else:
                communicated=normalize(communicated)
        with profile_share_stage(component,'destination_projection'):
            communicated=affine(communicated,up,use_bias=False)
        with profile_share_stage(component,'residual'):
            return multitensor_systems.multify(lambda dims,x,y:x+y)(residual,communicated)
