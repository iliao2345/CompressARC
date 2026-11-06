"""Layer labels only while the short operator capture is active."""
from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
import torch

active_layer_profile = ContextVar('active_layer_profile', default=False)


def profile_layer(name):
    if active_layer_profile.get():
        return torch.profiler.record_function('layer/' + name)
    return nullcontext()


_share_stage_timer = ContextVar('share_stage_timer', default=None)


@contextmanager
def sharing_stage_timing(timer=None):
    """Install a clean stage timer for this training call; restore on exceptions."""
    token = _share_stage_timer.set(timer)
    try:
        yield
    finally:
        _share_stage_timer.reset(token)


@contextmanager
def profile_share_stage(component, stage):
    timer = _share_stage_timer.get()
    timing_scope = timer('sharing_'+component+'_'+stage) if timer is not None else nullcontext()
    trace_scope = torch.profiler.record_function('stage/'+component+'/'+stage) if active_layer_profile.get() else nullcontext()
    with timing_scope, trace_scope:
        yield


def profile_operation_stage(component, stage):
    """Label forward operations for sequence-linked backward attribution only."""
    if active_layer_profile.get():
        return torch.profiler.record_function('stage/'+component+'/'+stage)
    return nullcontext()
