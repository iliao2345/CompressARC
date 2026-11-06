"""Default-device dispatch may be removed without changing training semantics."""
import pytest
import torch
from config import load_config
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step
from compressarc.train.optimizer import create_optimizer
from compressarc.train.device_dispatch import runtime_device_dispatch


def test_restore_on_exception_and_keep_tensor_subclass_dispatch():
    previous = torch.get_default_device()
    calls = []
    class Tracked(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            calls.append(func.__name__)
            return super().__torch_function__(func, types, args, kwargs or {})
    try:
        torch.set_default_device('meta')
        with pytest.raises(RuntimeError, match='intentional'):
            with runtime_device_dispatch('explicit'):
                assert torch.empty(1).device.type == 'cpu'
                value = torch.ones(1).as_subclass(Tracked)
                value + 1
                assert 'add' in calls
                raise RuntimeError('intentional')
        assert torch.get_default_device().type == 'meta'
        with runtime_device_dispatch('global'):
            assert torch.empty(1).device.type == 'meta'
        with pytest.raises(ValueError):
            with runtime_device_dispatch('invalid'):
                pass
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('cummax', ['primitives', 'triton_hard'])
def test_real_training_weights_gradients_outputs_and_rng(device, cummax):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA validation requires M3')
    if cummax == 'triton_hard':
        if device == 'cpu':
            pytest.skip('Triton hard cummax requires CUDA')
        pytest.importorskip('triton')
    config = load_config(overrides=[
        'model.n_layers=1', 'model.channel_dim_shared=2', 'model.channel_dim_grid=2',
        'model.share_up_dim=2', 'model.share_down_dim=2', 'model.decoding_dim=2',
        'model.softmax_dim=2', 'model.cummax_dim=2', 'model.shift_dim=2',
        'model.nonlinear_dim=2', 'model.direction_share_dim=2',
        'model.shift_implementation=d4_tied_conv',
        'model.direction_share_implementation=vectorized',
        'model.softmax_implementation=fused', 'model.share_implementation=reuse',
        f'model.cummax_implementation={cummax}',
        'training.optimizer_implementation=' + ('fused' if device == 'cuda' else 'default'),
    ])
    previous = torch.get_default_device()
    results = []
    try:
        torch.set_default_device(device)
        for mode in ('global', 'explicit'):
            torch.manual_seed(13)
            task = Task('dispatch', {'train': [{'input': [[0,1],[1,0]], 'output': [[1,0],[0,1]]}],
                                     'test': [{'input': [[1,0],[0,1]]}]}, [[[0,1],[1,0]]])
            model = ARCCompressor(task, config.model)
            optimizer = create_optimizer(model.weights_list, config.training)
            states = []
            with runtime_device_dispatch(mode):
                for step in range(3):
                    torch.manual_seed(91 + step)
                    metrics = take_step(task, model, optimizer, step, config.training, return_outputs=True)
                    states.append((metrics['loss'], [x.detach().cpu().clone() for x in metrics['outputs']],
                                   [p.detach().cpu().clone() for p in model.weights_list],
                                   [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.weights_list],
                                   torch.get_rng_state().clone(),
                                   torch.cuda.get_rng_state().clone() if device == 'cuda' else None))
            assert torch.get_default_device().type == device
            results.append(states)
    finally:
        torch.set_default_device(previous)
    for left, right in zip(*results):
        assert left[0] == pytest.approx(right[0], rel=1e-6)
        for xs, ys in zip(left[1:4], right[1:4]):
            for x, y in zip(xs, ys):
                if x is None or y is None:
                    assert x is y
                else:
                    torch.testing.assert_close(x, y, rtol=1e-5 if device == 'cuda' else 0,
                                                atol=1e-6 if device == 'cuda' else 0)
        assert torch.equal(left[4], right[4])
        if device == 'cuda':
            assert torch.equal(left[5], right[5])
