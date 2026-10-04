"""Generic discovery has no device policy and rejects ambiguous profiles."""
import pytest
from tensor.compiler.search import ScheduleSearch,ScheduleProfile


def test_retired_webgpu_search_has_no_compatibility_alias():
    import importlib.util
    assert importlib.util.find_spec('tensor.compiler.webgpu_search') is None


def test_finite_exploration_exhausts_and_filters_before_measurement():
    search=ScheduleSearch([{'family':'cpu','tile':1}],spaces={'cpu':{'tile':(1,2,3)}},
                          legal=lambda config:config['tile']!=2)
    assert search.next()=={'family':'cpu','tile':1}
    search.record({'family':'cpu','tile':1},.01)
    assert search.next()=={'family':'cpu','tile':3}
    with pytest.raises(StopIteration):search.next()


def test_cuda_and_webgpu_families_share_discovery_without_defaults():
    spaces={'cuda':{'threads':(64,128)},'webgpu':{'threads':(128,256)}}
    search=ScheduleSearch([],spaces=spaces)
    configs=[search.next() for _ in range(4)]
    assert {c['family'] for c in configs}==set(spaces)
    with pytest.raises(StopIteration):search.next()


def profile(entries):
    return ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_86',entries=entries))


def test_profile_uses_specific_semantic_selector_and_binds_target():
    value=profile([dict(operation='linear',parameters={'r':1},schedule={'threads':128}),
                   dict(operation='linear',parameters={'r':1,'k':2048},schedule={'threads':256})])
    assert value.select('linear',{'r':1,'k':2048,'o':512},provider='cuda',target='sm_86')=={'threads':256}
    assert value.select('unknown',{},provider='cuda',target='sm_86')=={}
    with pytest.raises(ValueError,match='target mismatch'):
        value.select('linear',{'r':1},provider='cuda',target='sm_90')
    assert value.sha256==ScheduleProfile(value.data).sha256


def test_duplicate_and_ambiguous_profile_selectors_rejected():
    entry=dict(operation='linear',parameters={'r':1},schedule={'threads':128})
    with pytest.raises(ValueError,match='duplicate'):profile([entry,entry])
    value=profile([entry,dict(operation='linear',parameters={'k':256},schedule={'threads':64})])
    with pytest.raises(ValueError,match='ambiguous'):
        value.select('linear',{'r':1,'k':256},provider='cuda',target='sm_86')
