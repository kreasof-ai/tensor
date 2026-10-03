"""Benchmark-local deadline and oracle hooks for the pinned tinygrad BEAM loop.

Does not edit upstream. Search buffers contain the actual F16 projection;
every timed candidate must overwrite a NaN sentinel and pass a float64 oracle.
"""
import hashlib,json,math,pickle,sqlite3,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import dtypes
from tinygrad.helpers import Context,diskcache_put
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import search


class SearchDeadline(KeyboardInterrupt):pass


class BudgetSearch:
    def __init__(self,seconds,out,x,weight,expected,tolerance,*,fallback_cache=None):
        self.seconds=seconds;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.x=np.ascontiguousarray(x);self.weight=np.ascontiguousarray(weight)
        self.expected=expected;self.tolerance=tolerance;self.fallback_cache=fallback_cache
        self.stats={'budget_seconds':seconds,'compiled_attempts':0,'compile_failures':0,'passed':0,'rejected':0,'incumbents':[]}
        self.started=None;self.best=None;self.programs={};self.key=None

    def save(self):
        self.stats['elapsed_seconds']=time.perf_counter()-self.started
        if self.best is not None:
            schedule,seconds,program=self.best
            self.stats['best_gpu_seconds']=seconds
            self.stats['best_opts']=[str(v) for v in schedule.applied_opts]
            self.stats['best_source_sha256']=hashlib.sha256(program.src[2].arg.encode()).hexdigest()
            (self.out/'best-source.txt').write_text(program.src[2].arg)
            diskcache_put('beam_search',self.key,schedule.applied_opts)
        (self.out/'search-progress.json').write_text(json.dumps(self.stats,indent=2)+'\n')

    def deadline(self):
        if time.perf_counter()-self.started>=self.seconds:raise SearchDeadline()

    def compile(self,item):
        self.deadline();self.stats['compiled_attempts']+=1
        result=self.original_compile(item)
        if result[1] is not None:
            program=result[1][0];self.programs[program.src[3].arg]=item[1].copy()
        else:self.stats['compile_failures']+=1
        if self.stats['compiled_attempts']%25==0:self.save()
        return result

    def timed(self,program,var_vals,rawbufs,*args,**kwargs):
        self.deadline()
        sentinel=np.full(self.expected.size,np.nan,np.float32)
        rawbufs[0].allocator._copyin(rawbufs[0]._buf,memoryview(sentinel).cast('B'))
        times=self.original_time(program,var_vals,rawbufs,*args,**kwargs)
        actual=rawbufs[0].numpy().reshape(self.expected.shape)
        valid=np.all(np.isfinite(actual)) and np.all(np.abs(actual-self.expected)<=self.tolerance)
        self.stats['passed' if valid else 'rejected']+=1
        if not valid:
            self.save();return [math.inf]*len(times)
        score=min(times)
        if math.isfinite(score) and (self.best is None or score<self.best[1]):
            self.best=(self.programs[program.src[3].arg].copy(),score,program)
            self.stats['incumbents'].append({'elapsed_seconds':time.perf_counter()-self.started,'compiled_attempts':self.stats['compiled_attempts'],
                                             'gpu_seconds':score,'opts':[str(v) for v in self.best[0].applied_opts]})
            self.save()
            print('search incumbent',self.stats['compiled_attempts'],round(score*1e6,3),'us',flush=True)
        return times

    def beam(self,s,rawbufs,var_vals,amt,allow_test_size=True):
        if self.started is not None:raise RuntimeError('budget hook expects one projection kernel')
        if allow_test_size:raise ValueError('oracle requires full-shape search timing')
        self.started=time.perf_counter()
        self.key={'ast':s.ast.key,'amt':amt,'allow_test_size':allow_test_size,'device':s.ren.target.device,'suffix':s.ren.suffix}
        self.stats.update(ast_sha256=s.ast.key.hex(),beam_width=amt)
        expected_buffers=[(self.expected.size,dtypes.float),(self.x.size,dtypes.float),(self.weight.size,dtypes.half)]
        if [(b.size,b.dtype) for b in rawbufs]!=expected_buffers:raise ValueError('unsupported projection parameter order/dtypes')
        for buf in rawbufs:buf.ensure_allocated()
        for buf,array in zip(rawbufs[1:],(self.x,self.weight)):
            buf.allocator._copyin(buf._buf,memoryview(array).cast('B'))
        fallback=s.copy()
        if self.fallback_cache:
            with sqlite3.connect('file:'+str(Path(self.fallback_cache).resolve())+'?mode=ro',uri=True) as db:
                ast_key=s.ast.replace(arg=replace(s.ast.arg,beam=2)).key if hasattr(s.ast.arg,'beam') else s.ast.key
                row=db.execute('select val from beam_search_24 where ast=? and amt=2 and allow_test_size=0 and device=?',
                               (ast_key,s.ren.target.device)).fetchone()
            if row is None:raise ValueError(('missing validated fallback schedule',s.ast.arg,s.ast.key.hex(),ast_key.hex()))
            for opt in pickle.loads(row[0]):fallback.apply_opt(opt)
        else:
            from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
            fallback=hand_coded_optimizations(fallback)
        program=to_program(fallback.get_optimized_ast(name_override='fallback'),s.ren)
        self.programs[program.src[3].arg]=fallback
        self.timed(program,var_vals,rawbufs,allow_test_size=False)
        if self.best is None:raise AssertionError('fallback failed the projection oracle')
        try:
            # Checkpoint cache entries must not short-circuit the running search.
            with Context(IGNORE_BEAM_CACHE=1):self.original_beam(s,rawbufs,var_vals,amt,False)
            self.stats['stop_reason']='converged'
        except SearchDeadline:self.stats['stop_reason']='time budget'
        self.save()
        print('search finished',self.stats,flush=True)
        return self.best[0].copy()

    def __enter__(self):
        self.original_compile,self.original_time,self.original_beam=search._try_compile,search._time_program,search.beam_search
        search._try_compile,search._time_program,search.beam_search=self.compile,self.timed,self.beam
        return self

    def __exit__(self,*exc):
        search._try_compile,search._time_program,search.beam_search=self.original_compile,self.original_time,self.original_beam
