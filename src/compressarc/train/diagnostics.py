"""Read-only kernel norm and numerical diagnostics."""
from contextlib import nullcontext
import torch
from .nonfinite import NonfiniteCounter


class TrainingDiagnostics:
    def __init__(self, run, model, task, implementation="reference"):
        self.run,self.model,self.task = run,model,task
        self.counter = NonfiniteCounter(implementation)
        self.step = -1
        self.kernels,self.initial = {},{}
        self.nonfinite_steps = set()
        self.event_count = 0
        self.nan_seen = self.inf_seen = False
        self.gradient_metrics = {}
        if run is not None:
            for kind in ("shift","cummax","direction_share"):
                for depth,core in enumerate(getattr(getattr(model,kind,None),"models",())):
                    for name,parameter in core.named_parameters():
                        key = f"{kind}/depth_{depth}/{name.replace('.', '/')}"
                        self.kernels[key] = parameter
                        self.initial[key] = parameter.detach().clone()

    def __enter__(self):
        if self.run is not None:
            self.run.summary["kernels/parameter_tensor_count"] = len(self.kernels)
            self.run.summary["numerics/implementation"] = self.counter.implementation
        return self

    def __exit__(self, error_type, error, traceback):
        if self.run is not None:
            self.run.summary.update({
                "numerics/nonfinite_detected":bool(self.event_count),
                "numerics/nan_detected":self.nan_seen,
                "numerics/inf_detected":self.inf_seen,
                "numerics/nonfinite_event_count":self.event_count,
                "numerics/nonfinite_step_count":len(self.nonfinite_steps)})
            if self.nonfinite_steps:
                self.run.summary["numerics/first_nonfinite_step"] = min(self.nonfinite_steps)+1
        return False

    def observe(self, category, tensors, scope=None):
        if self.run is None:
            return
        scope = scope or (lambda name:nullcontext())
        nan_count,inf_count = self.counter.count(tensors,category,scope)
        if nan_count or inf_count:
            self.nonfinite_steps.add(self.step)
            self.event_count += 1
            self.nan_seen |= bool(nan_count)
            self.inf_seen |= bool(inf_count)
            self.run.log({"train_step":self.step,"numerics/phase":category,
                          "numerics/nan_count":nan_count,"numerics/inf_count":inf_count})
        if category == "gradients" and (self.step+1)%50 == 0:
            self.gradient_metrics = {
                f"kernels/{name}/gradient_norm": (float(torch.linalg.vector_norm(p.grad.detach().double()))
                                                  if p.grad is not None else None)
                for name,p in self.kernels.items()}

    def after_step(self):
        if self.run is None or (self.step+1)%50 != 0:
            return
        payload = {"train_step":self.step,**self.gradient_metrics}
        for name,parameter in self.kernels.items():
            prefix = f"kernels/{name}"
            value = parameter.detach().double()
            payload[prefix+"/parameter_norm"] = float(torch.linalg.vector_norm(value))
            payload[prefix+"/distance_from_initialization"] = float(
                torch.linalg.vector_norm(value-self.initial[name].double()))
        if self.kernels:
            self.run.log(payload)
