"""Focused backward timings preserve aliases and use observed loss gradients."""
from types import SimpleNamespace
import json
import torch
import pytest
import profile_backward as bench
from config import load_config
from preprocessing import Task


def test_detached_boundary_preserves_aliases_and_noncontiguous_strides():
    x = torch.randn(4, 8, requires_grad=True)[:, ::2]
    a, b = bench.detach_inputs([x, x])
    assert a is b and a.is_leaf and a.requires_grad
    assert a.stride() == x.stride() and a.data_ptr() == x.data_ptr()
    torch.testing.assert_close(a, x)


def test_captured_upstream_replays_exact_parameter_and_input_gradients():
    weight = torch.randn(3, 3, requires_grad=True)
    x = torch.randn(2, 3, requires_grad=True)
    # The same weight participates twice; aliasing must survive replay.
    def layer(x, weights):
        return (x @ weights[0]) + (x @ weights[1])
    model = SimpleNamespace(share=SimpleNamespace(share_up=layer))
    with bench.capture_components(model, ['share_up']) as records:
        y = model.share.share_up(x, [weight, weight])
        y.square().sum().backward()
    expected_w, expected_x = weight.grad.clone(), x.grad.clone()
    record = records[0]
    assert record['args'][1][0] is record['args'][1][1] is weight
    before = weight.detach().clone()
    result = bench.benchmark_component(record, [weight], torch.device('cpu'), warmup=1, steps=2)
    torch.testing.assert_close(weight.grad, expected_w, rtol=0, atol=0)
    torch.testing.assert_close(record['args'][0].grad, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
    assert result['samples'] == 2 and result['cuda_stream_mean_seconds'] is None
    assert model.share.share_up is layer


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_real_task_all_components_and_reports_without_wandb(monkeypatch, tmp_path, device):
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            pytest.skip('CUDA timing and Triton replay require M3')
        pytest.importorskip('triton')
    config = load_config(overrides=[
        'model.n_layers=1', 'model.channel_dim_shared=2', 'model.channel_dim_grid=2',
        'model.share_up_dim=2', 'model.share_down_dim=2', 'model.decoding_dim=2',
        'model.softmax_dim=2', 'model.cummax_dim=2', 'model.shift_dim=2',
        'model.nonlinear_dim=2', 'model.direction_share_dim=2',
        'model.shift_implementation=d4_tied_conv', 'model.share_implementation=reuse',
        'model.direction_share_implementation=vectorized', 'model.softmax_implementation=fused',
        'training.runtime_device_dispatch=explicit', 'logging.wandb=false'] + (
            ['model.cummax_implementation=triton_hard', 'training.optimizer_implementation=fused']
            if device.startswith('cuda') else []))
    def tasks(*args, **kwargs):
        return [Task('tiny', {'train': [{'input': [[0,1],[1,0]], 'output': [[1,0],[0,1]]}],
                              'test': [{'input': [[1,0],[0,1]]}]}, [[[0,1],[1,0]]])]
    monkeypatch.setattr(bench, 'load_config', lambda *a: config)
    monkeypatch.setattr(bench, 'preprocess_tasks', tasks)
    previous = torch.get_default_device()
    report = bench.main(['--config', 'unused.yaml', '--task', 'tiny', '--device', device,
                         '--prepare-steps', '1', '--warmup-steps', '0', '--measured-steps', '1',
                         '--output-dir', str(tmp_path)])
    assert torch.get_default_device() == previous
    assert len(report['components']) == 11
    assert report['observed_boundary_count'] == report['expected_boundary_count'] == 11
    assert report['connected_model_backward']['samples'] == 1
    assert {r['name'].split('/')[-1] for r in report['components']} == set(bench.COMPONENTS)
    assert report['full_backward']['samples'] == 1
    assert all(r['active_outputs'] > 0 for r in report['components'])
    assert json.loads((tmp_path/'backward_latency.json').read_text())['task'] == 'tiny'
    assert (tmp_path/'backward_latency.txt').is_file()
    assert not list(tmp_path.glob('*.pt'))


def test_capture_patches_restore_after_exception():
    def fn(x):
        return x * 2
    model = SimpleNamespace(share=SimpleNamespace(share_up=fn))
    with pytest.raises(RuntimeError, match='stop'):
        with bench.capture_components(model, ['share_up']):
            model.share.share_up(torch.ones(2, requires_grad=True))
            raise RuntimeError('stop')
    assert model.share.share_up is fn


def test_connected_model_and_isolated_loss_reproduce_full_step_gradients():
    from compressarc.model.model import ARCCompressor
    from compressarc.train.optimizer import create_optimizer
    from unittest.mock import patch
    config = load_config(overrides=[
        'model.n_layers=1', 'model.channel_dim_shared=2', 'model.channel_dim_grid=2',
        'model.share_up_dim=2', 'model.share_down_dim=2', 'model.decoding_dim=2',
        'model.softmax_dim=2', 'model.cummax_dim=2', 'model.shift_dim=2',
        'model.nonlinear_dim=2', 'model.direction_share_dim=2',
        'model.direction_share_implementation=vectorized', 'model.share_implementation=reuse'])
    task = Task('gradients', {'train': [{'input': [[0,1],[1,0]], 'output': [[1,0],[0,1]]}],
                              'test': [{'input': [[1,0],[0,1]]}]}, [[[0,1],[1,0]]])
    model = ARCCompressor(task, config.model)
    optimizer = create_optimizer(model.weights_list, config.training)
    torch.manual_seed(782)
    with patch.object(optimizer, 'step', lambda: None), bench.capture_components(
            model, ['latent_decoder', 'output_heads', 'loss'], capture_connected=True) as records:
        bench.take_step(task, model, optimizer, 10, config.training)
    expected = [None if p.grad is None else p.grad.clone() for p in model.weights_list]
    connected = next(r for r in records if r['name'] == 'connected_model')
    loss = next(r for r in records if r['name'] == 'loss')
    torch.manual_seed(782)
    bench.benchmark_component(connected, model.weights_list, torch.device('cpu'), warmup=0, steps=1)
    for p, wanted in zip(model.weights_list, expected):
        if wanted is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, wanted, rtol=3e-5, atol=3e-5)
    bench.benchmark_component(loss, model.weights_list, torch.device('cpu'), warmup=0, steps=1)
    boundary = [x for x in bench.unique_tensors(loss['boundary_inputs']) if x.requires_grad]
    assert len(boundary) == len(connected['upstream'])
    for tensor, wanted in zip(boundary, connected['upstream']):
        if wanted is None:
            assert tensor.grad is None
        else:
            torch.testing.assert_close(tensor.grad, wanted, rtol=0, atol=0)
