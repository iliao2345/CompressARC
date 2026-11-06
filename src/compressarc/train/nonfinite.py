"""Numerical checks with an optional pointer-table CUDA fast path."""
from contextlib import nullcontext
import torch


SUPPORTED = {torch.float16,torch.bfloat16,torch.float32,torch.float64}


def tile_metadata(tensors, block=1024):
    """A row is (tensor address, element offset, valid element count).

    Do not deduplicate aliases: the original concatenation counts each occurrence.
    data_ptr includes storage_offset. No detach/reshape views are needed.
    """
    rows = []
    for tensor in tensors:
        pointer,size = tensor.data_ptr(),tensor.numel()
        rows.extend((pointer,start,min(block,size-start)) for start in range(0,size,block))
    return rows


def prepare_direct(tensors):
    """Inspect only ordinary Tensor/Parameter metadata with dispatch disabled.

    Materialize iterators first so caller code never executes under the guard.
    Subclasses stay on the original path, with their dispatch hooks intact.
    No tensor allocation, value arithmetic, or GPU work occurs under this guard.
    """
    sources = list(tensors)
    fallback,fast,signatures,subclasses = {},{},{},[]
    guard = getattr(torch._C, 'DisableTorchFunction', nullcontext)
    with guard():
        for value in sources:
            if value is None:
                continue
            if type(value) not in (torch.Tensor,torch.nn.Parameter):
                subclasses.append(value)
                continue
            dtype,device = value.dtype,value.device
            if not (dtype.is_floating_point or dtype.is_complex):
                continue
            if device.type == 'cuda' and dtype in SUPPORTED and value.is_contiguous():
                size = value.numel()
                if size:
                    key = (device,dtype)
                    fast.setdefault(key,[]).append(value)
                    signatures.setdefault(key,[]).append((id(value),value.data_ptr(),size))
            else:
                fallback.setdefault(device,[]).append(value)
    if subclasses:
        # A subclass can override cat/classification for an entire mixed list.
        # Keep the whole observation on the original path in its original order.
        reference = {}
        for value in sources:
            if value is not None and (value.is_floating_point() or value.is_complex()):
                reference.setdefault(value.device,[]).append(value.detach().reshape(-1))
        return reference,{},{}
    # Preserve normal dispatch/autograd behavior for fallback view construction.
    reference = {device:[v.detach().reshape(-1) for v in values] for device,values in fallback.items()}
    return reference,fast,signatures


def tiles_from_signature(signature, block=1024):
    rows = []
    for _,pointer,size in signature:
        rows.extend((pointer,start,min(block,size-start)) for start in range(0,size,block))
    return rows


class NonfiniteCounter:
    def __init__(self, implementation='reference'):
        if implementation not in ('reference','triton'):
            raise ValueError('training.diagnostics_implementation must be reference or triton')
        self.implementation = implementation
        self.cache = {}

    def count(self, tensors, category, scope=None):
        scope = scope or (lambda name:nullcontext())
        reference,fast,signatures = {},{},{}
        with scope('prepare'):
            if self.implementation == 'reference':
                for value in tensors:
                    if value is not None and (value.is_floating_point() or value.is_complex()):
                        reference.setdefault(value.device,[]).append(value.detach().reshape(-1))
            else:
                reference,fast,signatures = prepare_direct(tensors)
        nan_count = inf_count = 0
        for values in reference.values():
            with scope('concatenate'):
                flattened = torch.cat(values)
            with scope('reduce'):
                counts = torch.stack((torch.isnan(flattened).sum(),torch.isinf(flattened).sum()))
            with scope('transfer'):
                counts = counts.cpu().tolist()
            nan_count += counts[0]
            inf_count += counts[1]
        if fast:
            from .nonfinite_triton import count_tiles
        for (device,dtype),values in fast.items():
            with scope('metadata'):
                # Addresses change when zero_grad replaces grad buffers or a
                # parameter changes storage. Rebuild instead of reading stale data.
                signature = tuple(signatures[(device,dtype)])
                key = (category,device,dtype)
                cached = self.cache.get(key)
                if cached is None or cached[0] != signature:
                    metadata = torch.tensor(tiles_from_signature(signature),dtype=torch.int64,device=device)
                    self.cache[key] = (signature,metadata)
                else:
                    metadata = cached[1]
            with scope('reduce'):
                counts = count_tiles(metadata,dtype)
            with scope('transfer'):
                counts = counts.cpu().tolist()
            nan_count += counts[0]
            inf_count += counts[1]
        return nan_count,inf_count
