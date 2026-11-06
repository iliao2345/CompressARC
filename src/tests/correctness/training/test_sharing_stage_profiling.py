from contextlib import contextmanager
from types import SimpleNamespace
import json
import pytest
import torch

from compressarc.layers.share.primitives import MultitensorShare
from compressarc.model.initializers import Initializer
from compressarc.model.multitensor_systems import MultiTensorSystem
from compressarc.train.layer_profiling import sharing_stage_timing,profile_share_stage,profile_layer
from compressarc.train.step_timing import StepTimer
from compressarc.train.operator_profiling import OperatorProfiler
from compressarc.train.profile_analysis import analyze_trace

STAGES=['weight_selection','source_projection','communication','normalization','destination_projection','residual']


def make_case(known_size=True,policy='strict'):
    torch.manual_seed(48)
    masks=torch.ones(2,3,5,2)
    masks[0,1,2,:]=0
    task=SimpleNamespace(in_out_same_size=known_size,all_out_same_size=False,masks=masks)
    system=MultiTensorSystem(2,2,3,5,task=task,constraint_policy=policy)
    initializer=Initializer(system,lambda dims:4 if dims[2] else 2)
    weights=initializer.initialize_multiresidual(3,3)
    initializer.symmetrize_xy(weights)
    inputs=system.make_multitensor()
    for dims in system:
        inputs[dims]=torch.randn(system.shape(dims,initializer.channel_dim_fn(dims)),requires_grad=True)
    return system,inputs,weights,initializer.weights_list


@pytest.mark.parametrize('direction',['share_up','share_down'])
@pytest.mark.parametrize('known_size',[False,True])
@pytest.mark.parametrize('policy',['strict','relaxed'])
def test_stage_timing_preserves_outputs_gradients_and_operation_order(direction,known_size,policy):
    system,inputs,weights,parameters=make_case(known_size,policy)
    call=getattr(MultitensorShare(),direction)
    expected=call(inputs,weights)
    names=[]
    @contextmanager
    def collect(name):
        names.append(name)
        with timer.scope(name):yield
    with StepTimer('cpu','phases') as timer:
        timer.start(0);timer.mark('forward')
        with sharing_stage_timing(collect):actual=call(inputs,weights)
        metrics=timer.finish()
    assert names==['sharing_'+direction+'_'+stage for stage in STAGES]
    leaf_inputs=[inputs[dims] for dims in system]
    got_loss,wanted_loss=0,0
    for dims in system:
        torch.testing.assert_close(actual[dims],expected[dims],rtol=0,atol=0)
        upstream=torch.randn_like(actual[dims])
        got_loss=got_loss+(actual[dims]*upstream).sum()
        wanted_loss=wanted_loss+(expected[dims]*upstream).sum()
    got=torch.autograd.grad(got_loss,[*leaf_inputs,*parameters],allow_unused=True)
    wanted=torch.autograd.grad(wanted_loss,[*leaf_inputs,*parameters],allow_unused=True)
    for a,b in zip(got,wanted):
        if b is None:assert a is None
        else:torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all('timing/host_sharing_'+direction+'_'+stage+'_seconds' in metrics for stage in STAGES)


def test_stage_timer_context_restores_after_exception():
    calls=[]
    @contextmanager
    def scope(name):calls.append(name);yield
    with pytest.raises(RuntimeError,match='stop'):
        with sharing_stage_timing(scope):
            with profile_share_stage('share_up','communication'):pass
            raise RuntimeError('stop')
    with profile_share_stage('share_up','communication'):pass
    assert calls==['sharing_share_up_communication']


def test_cross_thread_backward_stage_mapping_preserves_layer_totals(tmp_path):
    def e(cat,name,ts,dur,tid=1,**args):
        return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,pid=10,tid=tid,args=args)
    events=[e('user_annotation','ttt/forward',0,100),
            e('user_annotation','layer/depth_0/share_down',1,90),
            e('user_annotation','stage/share_down/source_projection',2,80),
            e('cpu_op','aten::mm',3,20,**{'External id':1,'Sequence number':5}),
            e('cuda_runtime','cudaLaunchKernel',4,1,correlation=1,**{'External id':1}),
            e('kernel','gemm',6,1,correlation=1,**{'External id':1}),
            e('user_annotation','ttt/backward',100,100),
            e('cpu_op','autograd::engine::evaluate_function: MmBackward0',110,50,tid=2,**{'Sequence number':5}),
            e('cpu_op','aten::mm',115,20,tid=2,**{'External id':2}),
            e('cuda_runtime','cudaLaunchKernel',116,1,tid=2,correlation=2,**{'External id':2}),
            e('kernel','backward_gemm',118,2,correlation=2,**{'External id':2})]
    path=tmp_path/'trace.json';path.write_text(json.dumps({'traceEvents':events}))
    result=analyze_trace(path,tmp_path/'report')
    assert result['kernel_launches']==2
    assert result['sharing_stage_attributed_launches']==2
    assert result['sharing_layer_launches']==2
    stages={(r['phase'],r['stage']):r for r in result['sharing_stages']}
    assert stages['forward','source_projection']['kernel_launches']==1
    assert stages['backward','source_projection']['kernel_launches']==1
    assert stages['backward','source_projection']['gpu_kernel_us']==2


def test_real_cpu_sharing_stage_trace_and_backward(tmp_path):
    system,inputs,weights,parameters=make_case()
    with OperatorProfiler('cpu',tmp_path/'share',enabled=True,start_step=0,steps=1) as profiler:
        profiler.before_step(0);profiler.mark('forward')
        with profile_layer('depth_0/share_down'):
            result=MultitensorShare().share_down(inputs,weights)
        profiler.mark('backward')
        sum(result[dims].square().sum() for dims in system).backward()
        profiler.after_step(0)
    report=json.loads((tmp_path/'share.layers.json').read_text())
    forward={r['stage'] for r in report['sharing_stages'] if r['phase']=='forward'}
    assert forward==set(STAGES)
    backward={r['stage'] for r in report['sharing_stages'] if r['phase']=='backward'}
    assert {'source_projection','communication','normalization','destination_projection','residual'}<=backward
