"""Search grammar contracts independent of GPU and producer dependencies."""
import pytest
from tensor.compiler.webgpu_search import ScheduleSearch
from tensor.compiler.webgpu_lowering import streamed_gemv_schedule


def test_custom_search_space_does_not_enter_prefill_families():
    seed={'family':'decode','lanes':32,'threads':128}
    search=ScheduleSearch([seed],spaces={'decode':{'lanes':(16,32,64),'threads':(64,128)}})
    configs=[]
    for _ in range(6):
        config=search.next();search.record(config,.01);configs.append(config)
    assert len({tuple(sorted(c.items())) for c in configs})==6
    assert {c['family'] for c in configs}=={'decode'}


def test_gemv_extended_chains_and_non_power_of_two_k_coverage():
    # 2560-wide reductions admit factors absent from the original power-of-two
    # space. Explicit unroll slots still cover every K element exactly once.
    for unroll,accumulators in ((5,1),(10,2),(20,4)):
        text=streamed_gemv_schedule(2560,7,unroll=unroll,accumulators=accumulators)
        assert f'for tile in T.serial({2560//(32*4*unroll)})' in text
    text=streamed_gemv_schedule(1024,7,unroll=8,accumulators=8)
    assert 'acc0_7 = T.alloc_var' in text


@pytest.mark.parametrize('change',[{'lanes':24},{'threads':32},{'unroll':3},
                                  {'unroll':1,'accumulators':4},{'micro_rows':9},
                                  {'k_layout':'unknown'},{'dot_width':3}])
def test_gemv_rejects_incomplete_k_coverage_and_invalid_distributions(change):
    with pytest.raises(ValueError):streamed_gemv_schedule(128,7,**change)
