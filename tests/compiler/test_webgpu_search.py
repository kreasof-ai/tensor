"""Discovery rejects invalid scores and keeps exploring after convergence."""
import math
import pytest
from tensor.compiler.search import ScheduleSearch,neighbors,key
from tensor.compiler.webgpu_schedules import SPACES,coupled_moves
from tensor.compiler.webgpu_lowering import partitioned_matmul_schedule

SEED=dict(family='partitioned',tile_m=8,tile_n=32,threads=128,partitions=8,unroll=4,dot_width=1)

def test_invalid_timings_never_enter_beam():
    search=ScheduleSearch([SEED],spaces=SPACES,coupled=coupled_moves)
    for score in (math.inf,math.nan,-1,0):search.record(SEED,score)
    assert search.results==[]
    search.record(SEED,.001)
    assert search.results==[(.001,SEED)]

def test_restarts_continue_without_a_valid_incumbent_and_deduplicate():
    search=ScheduleSearch([SEED,SEED],spaces=SPACES,coupled=coupled_moves)
    configs=[search.next() for _ in range(100)]
    assert configs[0]==SEED
    assert len({key(c) for c in configs})==100

def test_coupled_move_can_preserve_output_owners():
    assert {**SEED,'threads':256,'partitions':16} in list(neighbors(SEED,SPACES,coupled=coupled_moves))


def test_outer_product_coupled_moves_reach_legal_microtiles():
    seed=dict(family='outer',tile_m=64,tile_n=128,micro_m=4,micro_n=8,threads=256)
    space=dict(tile_m=(32,64,128),tile_n=(64,128),micro_m=(2,4,8),
               micro_n=(4,8),threads=(64,128,256,512))
    moves=list(neighbors(seed,{'outer':space},coupled=coupled_moves))
    assert {**seed,'micro_m':8,'threads':128} in moves
    assert {**seed,'tile_m':128,'threads':512} in moves
    assert {**seed,'tile_n':64,'threads':128} in moves

def test_slower_new_family_still_gets_neighbor_exploration():
    fast={**SEED,'family':'partitioned'}
    slow={**SEED,'family':'partitioned_rows','tile_n':5,'dot_width':2,'unroll':8}
    search=ScheduleSearch([],width=1,spaces=SPACES,coupled=coupled_moves);search.seen={key(fast),key(slow)}
    search.record(fast,.001);search.record(slow,.002)
    configs=[search.next() for _ in range(35)]
    assert any(c['family']=='partitioned_rows' for c in configs)

@pytest.mark.parametrize('change',[{'partitions':3},{'unroll':3},{'dot_width':4,'unroll':1},
                                   {'tile_m':32,'tile_n':128,'partitions':32}, {'threads':0}])
def test_illegal_schedules_rejected_before_compilation(change):
    kwargs={k:v for k,v in {**SEED,**change}.items() if k!='family'}
    with pytest.raises(ValueError):partitioned_matmul_schedule(32,1024,2560,'x[{row}*1024+{k}]','w[{column}*1024+{k}]',**kwargs)
