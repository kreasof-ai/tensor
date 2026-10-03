"""Replay exact cached FFN ASTs; use ordinary heuristics for every other kernel."""
import hashlib,pickle,sqlite3
from dataclasses import replace
from pathlib import Path
from tinygrad.codegen.opt import search
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.uop.ops import Ops
from tinygrad.engine import jit


class ProjectionReplay:
    def __init__(self,root):
        self.entries={};self.records={};self.misses=0;self.caches={}
        for name in ('ffn_gate','ffn_down'):
            path=Path(root)/f'{name}-beam8-up128-local1024/cache.db'
            self.caches[name]={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True) as db:
                for ast,device,val in db.execute('select ast,device,val from beam_search_24 where amt=8 and allow_test_size=0'):
                    self.entries[(ast,device)]=(name,pickle.loads(val))

    def select(self,s,rawbufs,var_vals,amt,allow_test_size=True):
        key=s.ast.replace(arg=replace(s.ast.arg,beam=8)).key
        found=self.entries.get((key,s.ren.target.device))
        if found is not None:
            name,opts=found;candidate=s.copy()
            for opt in opts:candidate.apply_opt(opt)
            record=self.records.setdefault(key.hex(),{'weight_shape':name,'uses':0,'opts':[str(opt) for opt in opts]})
            record['uses']+=1
            return candidate
        self.misses+=1
        # Match tinygrad's normal BEAM=0 fallback, including its stage guard.
        return s if any(u.op is Ops.STAGE for u in s.ast.backward_slice) else hand_coded_optimizations(s)

    def __enter__(self):
        self.original=search.beam_search;search.beam_search=self.select
        self.original_getenv=jit.getenv
        jit.getenv=lambda key,default=0:2 if key=='JITBEAM' else self.original_getenv(key,default)
        return self

    def __exit__(self,*exc):
        search.beam_search=self.original
        jit.getenv=self.original_getenv

    def report(self):
        return {'search':'disabled; exact AST cache replay only','caches':self.caches,
                'matched':self.records,'heuristic_fallbacks':self.misses,
                'ffn_boundaries':'materialized residual before norm, FP32 projection inputs and outputs; complete state writes before residual reuse'}
