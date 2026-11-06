"""Operation-stage labels distinguish matrix assembly from arithmetic backward."""
import json
import torch
from compressarc.layers.direction_share.vectorized import DirectionShareVectorized
from compressarc.layers.shift.D4_morphological import D4DirectionalConv
from compressarc.train.layer_profiling import profile_layer
from compressarc.train.operator_profiling import OperatorProfiler


def test_real_packing_projection_and_convolution_backward_labels_preserve_math(tmp_path):
    def run(enabled):
        torch.manual_seed(42)
        x = torch.randn(8, 3, requires_grad=True)
        # Repeated pairs exercise gradient accumulation into tied parameters.
        weight = torch.randn(3, 3, requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        weights = [[[weight, bias] for _ in range(8)] for _ in range(8)]
        share = DirectionShareVectorized(lambda fn: lambda *a, **kw: fn((0,0,1,0,0), *a, **kw))
        conv = D4DirectionalConv()
        image = torch.randn(2, 2, 4, 3, 5, requires_grad=True)
        with OperatorProfiler('cpu', tmp_path/('on' if enabled else 'off'), enabled=enabled,
                              start_step=0, steps=1) as profiler:
            profiler.before_step(0); profiler.mark('forward')
            with profile_layer('depth_0/direction_share'):
                shared = share(x, weights)
            with profile_layer('depth_0/shift'):
                shifted = conv(image)
            profiler.mark('loss')
            loss = shared.square().sum() + shifted.square().sum()
            profiler.mark('backward')
            gradients = torch.autograd.grad(loss, [x, weight, image, *conv.parameters()])
            profiler.after_step(0)
        return shared.detach(), shifted.detach(), gradients
    actual, expected = run(True), run(False)
    for x, y in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(actual[2], expected[2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    result = json.loads((tmp_path/'on.layers.json').read_text())
    stages = {(r['phase'],r['component'],r['stage']) for r in result['operation_stages']}
    for component, names in [('direction_share', ['normalization','weight_packing','input_layout','matmul','residual']),
                              ('shift', ['d4_projection','kernel_expansion','convolution'])]:
        for name in names:
            assert ('forward',component,name) in stages
            assert ('backward',component,name) in stages
    assert result['sharing_stages'] == []
    assert result['sharing_stage_attributed_launches'] == 0
    nodes = result['backward_nodes']
    assert any(r['stage']=='direction_share/weight_packing' and 'StackBackward' in r['node'] for r in nodes)
    assert any(r['stage']=='direction_share/matmul' and 'MmBackward' in r['node'] for r in nodes)
    assert any(r['stage']=='shift/convolution' and 'ConvolutionBackward' in r['node'] for r in nodes)
    assert all(r['cpu_operator_self_us'] >= 0 for r in nodes)
