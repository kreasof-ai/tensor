"""GPU-independent candidate discovery for portable SIMT matmul schedules.

The caller compiles, checks and measures candidates on its target adapter. Beam
selection accepts only finite timings supplied after correctness validation.
"""
from __future__ import annotations
import itertools
import math

SPACES = {
    'partitioned': {'tile_m': (4,8,16,32), 'tile_n': (4,8,16,32,64,128),
                    'threads': (64,128,256), 'partitions': (2,4,8,16,32),
                    'unroll': (1,2,4,8,16,32), 'dot_width': (1,4)},
    'staged': {'tile_m': (8,16,32), 'tile_n': (16,32,64,128),
               'tile_k': (16,32,64,128), 'threads': (64,128,256),
               'dot_width': (1,4), 'unroll': (False,True), 'lhs_pad': (0,1),
               'lhs_transpose': (False,True)},
}
SPACES['partitioned']['k_layout']=('blocked','striped')
SPACES['partitioned_rows']={**SPACES['partitioned'],'tile_n':(4,5,8,10,16,20,32,40,64), 'dot_width':(1,2,4)}

def key(config):
    return tuple(sorted(config.items()))

def neighbors(config,spaces=None):
    """Adjacent moves and coupled changes that preserve output ownership."""
    space=(SPACES if spaces is None else spaces)[config['family']]
    for axis,values in space.items():
        index=values.index(config.get(axis,values[0]))
        for offset in (-1,1):
            if 0<=index+offset<len(values):
                yield {**config,axis:values[index+offset]}
    if config['family'].startswith('partitioned'):
        for thread,partition in itertools.product(space['threads'],space['partitions']):
            if thread//partition==config['threads']//config['partitions']:
                yield {**config,'threads':thread,'partitions':partition}
    if all(axis in space for axis in ('tile_m','tile_n','micro_m','micro_n','threads')):
        # A tile or register-microtile move must also change the thread count.
        # Single-coordinate moves alone cannot cross this legality constraint.
        for axis in ('tile_m','tile_n','micro_m','micro_n'):
            values=space[axis];index=values.index(config[axis])
            for offset in (-1,1):
                if not 0<=index+offset<len(values):continue
                candidate={**config,axis:values[index+offset]}
                tm,tn,mm,mn=(candidate[a] for a in ('tile_m','tile_n','micro_m','micro_n'))
                if tm%mm or tn%mn:continue
                threads=(tm//mm)*(tn//mn)
                if threads in space['threads']:yield {**candidate,'threads':threads}

class ScheduleSearch:
    """Widenable beam with deterministic restarts across the legal space.

    Exploration keeps running after a local optimum. The benchmark owns the
    wall-clock deadline; there is no hidden device work or tolerance policy.
    """
    def __init__(self,seeds,width=8,spaces=None):
        if type(width) is not int or width<=0:raise ValueError('beam width must be a positive integer')
        self.width=width;self.pending=list(seeds);self.seen=set();self.results=[]
        self.spaces=SPACES if spaces is None else spaces
        self.restarts=self._restarts()

    def _restarts(self):
        for family,space in self.spaces.items():
            # Permute dimensions so successive restarts cover different tiles.
            axes=list(space)
            for values in itertools.product(*(space[a] for a in axes)):
                yield {'family':family,**dict(zip(axes,values))}

    def next(self):
        while True:
            if not self.pending:
                ranked=sorted(self.results,key=lambda r:r[0]);selected=ranked[:self.width]
                # Keep new schedule families alive even when their first seed
                # loses to an established family. Otherwise widening the action
                # space cannot discover a better neighbor of that slower seed.
                for family in self.spaces:
                    selected += [r for r in ranked if r[1]['family']==family][:2]
                self.pending=[n for _,c in selected
                              for n in neighbors(c,self.spaces) if key(n) not in self.seen]
                if not self.pending:
                    self.pending=[next(self.restarts)]
            config=self.pending.pop(0);identity=key(config)
            if identity not in self.seen:
                self.seen.add(identity);return config

    def record(self,config,seconds):
        if math.isfinite(seconds) and seconds>0:self.results.append((seconds,dict(config)))
