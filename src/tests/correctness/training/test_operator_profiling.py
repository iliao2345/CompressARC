import json
import torch
import pytest
from compressarc.train.operator_profiling import OperatorProfiler


def test_disabled_does_not_create_files(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Profiler started when disabled")
    monkeypatch.setattr(torch.profiler, "profile", forbidden)
    prefix = tmp_path / "disabled"
    with OperatorProfiler("cpu", prefix) as profiler:
        profiler.before_step(10)
        profiler.mark("forward")
        with profiler.scope("numerics/outputs"):
            pass
        profiler.after_step(10)
    assert not list(tmp_path.iterdir())


def run_steps(prefix, enabled):
    torch.manual_seed(5)
    weight = torch.randn(4, 4, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=.01)
    with OperatorProfiler("cpu", prefix, enabled=enabled, start_step=1, steps=2) as profiler:
        for step in range(4):
            profiler.before_step(step)
            optimizer.zero_grad()
            profiler.mark("forward")
            output = weight @ weight
            with profiler.scope("numerics/outputs"):
                torch.isfinite(output).all()
            profiler.mark("backward")
            output.square().sum().backward()
            profiler.mark("optimizer")
            optimizer.step()
            profiler.after_step(step)
    return weight.detach()


def test_capture_window_attribution_and_training_parity(tmp_path):
    prefix = tmp_path / "profile"
    actual = run_steps(prefix, True)
    expected = run_steps(tmp_path / "off", False)
    assert torch.equal(actual, expected)
    manifest = json.loads(prefix.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["captured_steps_zero_based"] == [1, 2]
    phases = json.loads(prefix.with_suffix(".by_phase.json").read_text())
    for name in ("forward", "backward", "optimizer"):
        assert phases[name]
    assert any(row["scope"] == "numerics/outputs" for row in phases["forward"])
    trace = json.loads(prefix.with_suffix(".trace.json").read_text())
    assert trace["traceEvents"]
    assert prefix.with_suffix(".cpu.txt").stat().st_size
    assert prefix.with_suffix(".operators.csv").stat().st_size


def test_partial_capture_on_exception(tmp_path):
    prefix = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="intentional"):
        with OperatorProfiler("cpu", prefix, enabled=True, start_step=0, steps=3) as profiler:
            profiler.before_step(0)
            profiler.mark("forward")
            torch.ones(2).sum()
            profiler.after_step(0)
            raise RuntimeError("intentional")
    assert json.loads(prefix.with_suffix(".manifest.json").read_text())["status"] == "partial"


def test_cuda_kernel_export_with_real_interval_api(tmp_path, monkeypatch):
    # Exercise the CUDA event export branch without requiring a local GPU.
    from types import SimpleNamespace
    from torch.autograd.profiler_util import Interval

    class FakeProfiler:
        def __exit__(self, *args):
            pass

        def export_chrome_trace(self, path):
            from pathlib import Path
            Path(path).write_text('{"traceEvents": []}')

        def key_averages(self, **kwargs):
            return SimpleNamespace(table=lambda **kw: "operator table")

        def events(self):
            return [SimpleNamespace(device_type=torch.autograd.DeviceType.CUDA,
                                    name="test_kernel", time_range=Interval(10., 35.)),
                    SimpleNamespace(device_type=torch.autograd.DeviceType.CUDA,
                                    name="test_kernel", time_range=Interval(40., 55.))]

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock GPU")
    prefix = tmp_path / "cuda"
    profiler = OperatorProfiler("cuda:0", prefix, enabled=True, steps=1)
    profiler.profiler = FakeProfiler()
    profiler.active = True
    profiler.captured = [10]
    profiler._stop((None, None, None))
    assert json.loads(prefix.with_suffix(".kernels.json").read_text()) == [
        {"kernel": "test_kernel", "calls": 2, "duration_us": 40.}]
    assert prefix.with_suffix(".cuda.txt").is_file()
    assert json.loads(prefix.with_suffix(".manifest.json").read_text())["status"] == "complete"
