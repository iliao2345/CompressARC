"""Direct, read-only NaN/Inf counts for lists of contiguous CUDA tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scan(META, PARTIAL, POINTER_TYPE:tl.constexpr, PROMOTE:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    address = tl.load(META+tile*3)
    offset = tl.load(META+tile*3+1)
    length = tl.load(META+tile*3+2)
    pointer = address.to(POINTER_TYPE)
    index = tl.arange(0, BLOCK)
    values = tl.load(pointer+offset+index, index<length, other=0.)
    # Older Triton/NVIDIA lowering can leave bf16 comparisons as i16 fcmp.
    # FP16/BF16 -> FP32 preserves all finite values and NaN/Inf classes.
    # Do not cast FP64: large finite values would become false infinities.
    if PROMOTE:
        values = values.to(tl.float32)
    # Count elements, not just a boolean flag. +/-Inf both count as infinity.
    nan_count = tl.sum(((values != values)&(index<length)).to(tl.int64),0)
    inf_count = tl.sum(((tl.abs(values)==float('inf'))&(index<length)).to(tl.int64),0)
    tl.store(PARTIAL+tile*2,nan_count)
    tl.store(PARTIAL+tile*2+1,inf_count)


DTYPES = {torch.float16:tl.float16,torch.bfloat16:tl.bfloat16,
          torch.float32:tl.float32,torch.float64:tl.float64}


def count_tiles(metadata, dtype):
    """Metadata owns no tensor storage; caller keeps source tensors alive."""
    partial = torch.empty((metadata.shape[0],2),device=metadata.device,dtype=torch.int64)
    with torch.cuda.device(metadata.device):
        # Construct on the host: inside a JIT function DTYPE is a constexpr
        # wrapper, which older Triton pointer_type constructors reject.
        pointer_type = tl.pointer_type(DTYPES[dtype])
        _scan[(metadata.shape[0],)](metadata,partial,POINTER_TYPE=pointer_type,
            PROMOTE=dtype in (torch.float16,torch.bfloat16),BLOCK=1024)
        return partial.sum(dim=0)
