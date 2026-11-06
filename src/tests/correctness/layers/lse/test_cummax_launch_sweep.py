"""Correctness-gated scan launch tuning; CUDA execution required for kernels."""
from types import SimpleNamespace
import pytest
import torch
import benchmark_cummax_launch as sweep


def test_case_grid_changes_work_layout_and_sizes_without_cuda():
    args=SimpleNamespace(kind='both',rows=[3],lengths=[6,30],positions=[1,4],warps=[4,8],
                         batches=[2],spatial_sizes=['3x5'],blocks=[128,256])
    cases=list(sweep.cases(args))
    assert len(cases)==12
    assert len([c for c in cases if c['kind']=='axis'])==8
    assert any(c['shape']==[2,3,5] for c in cases)


def test_winners_exclude_incorrect_results_and_compare_same_shape_baseline():
    rows=[dict(kind='axis',shape=[3,6],positions=p,warps=4,status=status,
               backward=dict(wall_median_us=us)) for p,status,us in [(1,'valid',10),(2,'valid',5),(4,'invalid',1)]]
    best=sweep.winners(rows)
    assert best[0]['best_config']['positions']==2
    assert best[0]['backward_speedup']==2


def test_dry_run_does_not_require_triton_or_gpu(capsys):
    sweep.main(['--dry-run','--kind','axis','--rows','3','--lengths','6',
                '--positions','1','4','--warps','4'])
    assert '"configurations": 2' in capsys.readouterr().out


@pytest.mark.parametrize('length',[1,3,6,30,64,128])
@pytest.mark.parametrize('positions',[1,2,4,8])
@pytest.mark.parametrize('warps',[4,8])
@pytest.mark.parametrize('pattern',['holes','ties','padding_empty'])
def test_axis_outputs_and_gradients(length,positions,warps,pattern):
    if not torch.cuda.is_available():pytest.skip('CUDA/Triton required')
    pytest.importorskip('triton')
    case=dict(kind='axis',shape=[3,length],positions=positions,warps=warps)
    data=sweep.make_data(case['shape'],pattern,'cuda')
    plan=sweep.make_plan(case,data)
    sweep.validate(plan,sweep.reference(case,data))
    assert plan['backward_programs']==3*((length+positions-1)//positions)
    assert plan['backward_launches']==1


@pytest.mark.parametrize('shape',[(2,1,1),(2,3,5),(2,16,30)])
@pytest.mark.parametrize('block',[128,256,512])
@pytest.mark.parametrize('warps',[4,8])
@pytest.mark.parametrize('pattern',['holes','ties','padding_empty'])
def test_diagonal_outputs_and_gradients(shape,block,warps,pattern):
    if not torch.cuda.is_available():pytest.skip('CUDA/Triton required')
    pytest.importorskip('triton')
    case=dict(kind='diagonal',shape=list(shape),block=block,warps=warps)
    data=sweep.make_data(shape,pattern,'cuda')
    plan=sweep.make_plan(case,data)
    wanted=sweep.reference(case,data)
    sweep.validate(plan,wanted)
    # Scratch swaps must reset correctly between repeated backwards.
    plan['backward']()
    torch.testing.assert_close(plan['gradient'],wanted[1],rtol=2e-3,atol=2e-3)
