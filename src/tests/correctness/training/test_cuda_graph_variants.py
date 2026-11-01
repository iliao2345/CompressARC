import pytest
from cuda_graph_preflight import preflight_model_options

@pytest.mark.parametrize('variant,sharing,cummax',[
    ('conv','vectorized','triton_hard_grouped'),
    ('combined','d4','triton_hard_grouped'),
    ('combined_lse','d4','triton')])
def test_cuda_gate_uses_the_intended_architecture(variant,sharing,cummax):
    options=preflight_model_options(variant)
    assert options['shift_implementation']=='d4_tied_conv'
    assert options['direction_share_implementation']==sharing
    assert options['cummax_implementation']==cummax
    if variant!='conv':assert options['direction_share_packed']
    if variant=='combined_lse':assert options['lse_tau']==0.1

def test_unknown_variant_is_not_silently_substituted():
    with pytest.raises(ValueError):preflight_model_options('unknown')
