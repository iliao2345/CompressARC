"""Grouped hard scan stays a primitive-compatible opt-in backend."""
def test_grouped_factory_uses_primitive_call_contract(monkeypatch):
    import compressarc.model.layer_factory as factory
    from compressarc.layers.cummax.primitives import CummaxPrimitives
    from compressarc.model.model import ARCCompressor
    from preprocessing import Task
    from config import load_config
    calls=[]
    def fake(**kwargs):
        calls.append(kwargs)
        return CummaxPrimitives(**kwargs)
    monkeypatch.setattr(factory,'create_triton_hard_grouped_cummax',fake)
    cfg=load_config(overrides=['model.n_layers=1','model.cummax_implementation=triton_hard_grouped'])
    task=Task('factory',{'train':[{'input':[[0,1]],'output':[[1,0]]}],
                         'test':[{'input':[[1,0]]}]},None)
    model=ARCCompressor(task,cfg.model)
    assert len(calls)==1 and set(calls[0])=={'multify'}
    assert model._layer_call_options['cummax'] is False
