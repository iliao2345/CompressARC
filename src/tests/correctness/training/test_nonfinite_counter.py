from contextlib import contextmanager
from types import SimpleNamespace
import sys
import pytest
import torch

from compressarc.train.nonfinite import NonfiniteCounter,tile_metadata,prepare_direct,tiles_from_signature
from compressarc.train.diagnostics import TrainingDiagnostics


def original_count(values):
    groups={}
    for v in values:
        if v is not None and (v.is_floating_point() or v.is_complex()):
            groups.setdefault(v.device,[]).append(v.detach().reshape(-1))
    totals=[0,0]
    for group in groups.values():
        flat=torch.cat(group)
        counts=torch.stack((torch.isnan(flat).sum(),torch.isinf(flat).sum())).cpu().tolist()
        totals=[a+b for a,b in zip(totals,counts)]
    return tuple(totals)


@pytest.mark.parametrize('mode',['reference','triton'])
def test_cpu_fallback_preserves_complex_alias_empty_and_noncontiguous_counts(mode,monkeypatch):
    monkeypatch.setitem(sys.modules,'compressarc.train.nonfinite_triton',None)
    p=torch.tensor([float('nan'),float('inf'),-float('inf'),2.],requires_grad=True)
    matrix=torch.tensor([[1.,float('nan')],[float('inf'),4.]]).T
    complex_values=torch.tensor([complex(float('nan'),float('inf')),complex(1,float('-inf'))])
    values=[p,p,None,torch.empty(0),torch.tensor([3,4]),matrix,complex_values]
    expected=original_count(values)
    observed=NonfiniteCounter(mode).count(iter(values),'parameters')
    assert observed==expected
    assert p.grad is None


def test_tile_metadata_preserves_aliases_offsets_and_partial_tiles():
    x=torch.ones(1030)
    view=x[2:1027]
    rows=tile_metadata([view,view,torch.empty(0)])
    assert rows==[(view.data_ptr(),0,1024),(view.data_ptr(),1024,1)]*2


def test_reference_timing_sections_are_separate_and_in_order():
    events=[]
    @contextmanager
    def scope(name):
        events.append('enter:'+name);yield;events.append('exit:'+name)
    assert NonfiniteCounter().count([torch.ones(3)],'parameters',scope)==(0,0)
    assert events==[s+name for name in ['prepare','concatenate','reduce','transfer'] for s in ['enter:','exit:']]


@pytest.mark.parametrize('mode',['reference','triton'])
def test_diagnostics_event_counts_and_step_summary_unchanged(mode):
    records=[]
    run=SimpleNamespace(log=records.append,summary={})
    with TrainingDiagnostics(run,SimpleNamespace(),None,implementation=mode) as diag:
        diag.step=2
        diag.observe('parameters',[torch.tensor([float('nan'),float('inf'),float('-inf')])])
        diag.observe('gradients',[torch.tensor([float('nan')])])
        diag.step=3;diag.observe('parameters',[torch.ones(2)])
    assert records==[{'train_step':2,'numerics/phase':'parameters','numerics/nan_count':1,'numerics/inf_count':2},
                     {'train_step':2,'numerics/phase':'gradients','numerics/nan_count':1,'numerics/inf_count':0}]
    assert run.summary['numerics/nonfinite_event_count']==2
    assert run.summary['numerics/nonfinite_step_count']==1
    assert run.summary['numerics/first_nonfinite_step']==3


def test_unknown_mode_rejected():
    with pytest.raises(ValueError,match='reference or triton'):NonfiniteCounter('typo')


@pytest.fixture
def cuda_counter():
    if not torch.cuda.is_available():pytest.skip('CUDA required for direct Triton nonfinite scans')
    pytest.importorskip('triton')
    return NonfiniteCounter('triton')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_cuda_exact_counts_readonly_and_mixed_fallbacks(cuda_counter,dtype):
    x=torch.ones(4099,device='cuda',dtype=dtype,requires_grad=True)
    with torch.no_grad():
        x[0]=float('nan');x[1023]=float('inf');x[1024]=float('-inf');x[-1]=float('nan')
        limit=torch.finfo(dtype)
        x[4:9]=torch.tensor([limit.max,-limit.max,limit.tiny,-limit.tiny,0.],device='cuda',dtype=dtype)
    x.grad=torch.ones_like(x)
    grad=x.grad.clone()
    view=torch.tensor([[1.,float('nan')],[float('inf'),4.]],device='cuda').T
    complex_values=torch.tensor([complex(float('nan'),float('inf'))],device='cuda')
    values=[x,x,None,view,complex_values,torch.empty(0,device='cuda'),torch.ones(3,dtype=torch.long,device='cuda')]
    before=x.detach().clone()
    assert cuda_counter.count(values,'parameters')==original_count(values)
    torch.testing.assert_close(x,before,rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(x.grad,grad,rtol=0,atol=0)


def test_cuda_cache_reads_current_values_and_invalidates_on_storage_change(cuda_counter):
    x=torch.ones(1025,device='cuda')
    assert cuda_counter.count([x],'parameters')==(0,0)
    key=next(iter(cuda_counter.cache));old_metadata=cuda_counter.cache[key][1]
    x[0]=float('nan')
    assert cuda_counter.count([x],'parameters')==(1,0)
    assert cuda_counter.cache[key][1] is old_metadata
    x.set_(torch.full((5,),float('inf'),device='cuda'))
    assert cuda_counter.count([x],'parameters')==(0,5)
    assert cuda_counter.cache[key][1] is not old_metadata


def test_cuda_many_tiny_tensors_and_replaced_gradients(cuda_counter):
    values=[torch.ones(4,device='cuda') for _ in range(1500)]
    values[-1][0]=float('inf');values[400][1]=float('nan')
    assert cuda_counter.count(values,'parameters')==(1,1)
    p=torch.ones(8,device='cuda',requires_grad=True)
    for bad in [float('nan'),float('inf'),-float('inf')]:
        p.grad=torch.full_like(p,bad)
        assert cuda_counter.count([p.grad,None],'gradients')==original_count([p.grad])
        p.grad=None
        assert cuda_counter.count([p.grad],'gradients')==(0,0)


@pytest.mark.parametrize('mode',['reference','triton'])
def test_diagnostic_detail_timings_integrate_without_changing_counts(mode):
    from compressarc.train.step_timing import StepTimer
    records=[];run=SimpleNamespace(log=records.append,summary={})
    with StepTimer('cpu','phases') as timer, TrainingDiagnostics(run,SimpleNamespace(),None,implementation=mode) as diag:
        timer.start(0);timer.mark('optimizer')
        with timer.optimizer_scope('parameter_checks'):
            diag.observe('parameters',[torch.tensor([float('nan'),float('inf')])],
                         scope=lambda name:timer.scope('numerics_parameters_'+name))
        metrics=timer.finish()
    fields=[f'timing/host_numerics_parameters_{stage}_seconds' for stage in ['prepare','concatenate','reduce','transfer']]
    assert all(metrics[k]>=0 for k in fields)
    assert metrics['timing/host_optimizer_parameter_checks_seconds']>=sum(metrics[k] for k in fields)
    assert records[0]['numerics/nan_count']==1
    assert records[0]['numerics/inf_count']==1


def test_metadata_guard_keeps_iterator_default_device_and_restores_dispatch():
    made=[]
    def source():
        value=torch.ones(2)
        made.append(value.device.type)
        yield value
    with torch.device('meta'):
        reference,fast,signatures=prepare_direct(source())
        assert made==['meta']
        assert not fast and not signatures
        assert next(iter(reference)).type=='meta'
        assert torch.ones(1).device.type=='meta'
    assert torch.ones(1).device.type=='cpu'


def test_metadata_guard_restored_when_input_iterator_raises():
    def broken():
        yield torch.ones(1)
        raise RuntimeError('broken iterator')
    with torch.device('meta'):
        with pytest.raises(RuntimeError,match='broken iterator'):
            prepare_direct(broken())
        assert torch.ones(1).device.type=='meta'


def test_subclasses_keep_their_dispatch_hooks_on_fallback():
    called=[]
    class TrackedTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls,func,types,args=(),kwargs=None):
            called.append(func.__name__)
            return super().__torch_function__(func,types,args,kwargs or {})
    value=torch.tensor([float('nan'),float('inf'),2.]).as_subclass(TrackedTensor)
    called.clear()
    assert NonfiniteCounter('triton').count([value],'parameters')==(1,1)
    assert 'is_floating_point' in called
    assert 'detach' in called


def test_cached_signature_tiles_equal_original_metadata_without_tensor_rereads():
    values=[torch.ones(1030)[2:1027],torch.ones(3),torch.empty(0)]
    values.append(values[1])
    signature=[(id(v),v.data_ptr(),v.numel()) for v in values]
    assert tiles_from_signature(signature)==tile_metadata(values)
