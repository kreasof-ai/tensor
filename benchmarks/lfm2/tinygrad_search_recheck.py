"""Remeasure cached schedules together, without search, on held-out inputs."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,pickle,sqlite3,statistics,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor,TinyJit,Device,UOp
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import search
from tinygrad.engine.realize import time_call
from tinygrad.uop.ops import Ops
from tensor_llm import GGUF
from benchmarks.lfm2.tinygrad_reference import rounded_half
from benchmarks.lfm2.tinygrad_compare import tiny_info


def check(actual,inputs,weight):
    left=inputs.astype(np.float16).astype(np.float64);right=weight.astype(np.float64)
    expected=left@right.T;bound=np.abs(left)@np.abs(right).T*3e-6+1e-10
    if not np.isfinite(actual).all() or not np.all(np.abs(actual-expected)<=bound):raise AssertionError('recheck oracle failed')
    return {'maximum_absolute_error':float(np.max(np.abs(actual-expected))),
            'maximum_error_over_bound':float(np.max(np.abs(actual-expected)/bound))}


def run(model,root,fallback_cache,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);root=Path(root)
    summary=json.loads((root/'search-summary.json').read_text());gguf=GGUF(model);records=[]
    if not Device.DEFAULT.startswith('CL'):raise ValueError('recheck requires OpenCL')
    for suffix in ('ffn_gate','ffn_down'):
        weight_array=np.array(gguf.array('blk.0.'+suffix+'.weight',dtype=np.float16),copy=True);o,k=weight_array.shape
        inputs=(np.random.default_rng(29).normal(size=(32,k))*.01).astype(np.float32)
        input=Tensor(inputs).realize();weight=Tensor(weight_array).realize()
        specs=[{'name':'previous-beam2','cache':str(fallback_cache),'width':2}]
        specs += [{'name':stage['name'],'cache':str(root/stage['name']/'cache.db'),'width':stage['beam']}
                  for stage in summary['stages'] if stage['name'].startswith(suffix) and 'checkpoint' in stage]
        runners={};timers={};row={'weight':suffix,'shape':[32,k,o],'variants':{}}
        for spec in specs:
            def select(s,rawbufs,var_vals,amt,allow_test_size=True):
                key=s.ast.replace(arg=replace(s.ast.arg,beam=spec['width'])).key
                with sqlite3.connect('file:'+str(Path(spec['cache']).resolve())+'?mode=ro',uri=True) as db:
                    result=db.execute('select val from beam_search_24 where ast=? and amt=? and allow_test_size=0 and device=?',
                                      (key,spec['width'],s.ren.target.device)).fetchone()
                if result is None:raise ValueError('missing schedule for '+spec['name'])
                candidate=s.copy()
                for opt in pickle.loads(result[0]):candidate.apply_opt(opt)
                return candidate
            def operation(value):return (rounded_half(value)@rounded_half(weight).T).realize()
            # Compiled AST cache would otherwise bypass the per-variant selector.
            to_program_cache.clear();jit=TinyJit(operation);jit(input)
            original=search.beam_search;search.beam_search=select
            try:jit(input)
            finally:search.beam_search=original
            actual=jit(input).numpy();validation=check(actual,inputs,weight_array)
            programs=[u for u in jit.captured._linear.toposort() if u.op is Ops.PROGRAM]
            if len(programs)!=1:raise ValueError('expected one projection program')
            program=programs[0];source=next(u.arg for u in program.src if u.op is Ops.SOURCE)
            filename=f'{suffix}-{spec["name"]}.txt';(out/filename).write_text(source)
            output=Tensor.zeros(32,o).realize()
            call=program.call(*(UOp.from_buffer(t.uop.buffer) for t in (output,input,weight)))
            timer=time_call(call);next(timer)
            direct_validation=check(output.numpy(),inputs,weight_array)
            timers[spec['name']]=timer;runners[spec['name']]=jit
            row['variants'][spec['name']]={'source':filename,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                         'applied_opts':str(program.src[0].arg.applied_opts),'validation':[{'seed':29,'scale':.01,**validation}],
                                         'gpu_direct_validation':direct_validation,
                                         'gpu_samples_seconds':[],'completed_samples_seconds':[]}
        for jit in runners.values():
            deadline=time.perf_counter()+1
            while time.perf_counter()<deadline:
                for _ in range(10):jit(input)
                Device[Device.DEFAULT].synchronize()
        names=list(runners)
        for repeat in range(8):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                start=time.perf_counter()
                for _ in range(20):runners[name](input)
                Device[Device.DEFAULT].synchronize();completed=(time.perf_counter()-start)/20
                gpu=[next(timers[name]) for _ in range(20)]
                if repeat:
                    row['variants'][name]['completed_samples_seconds'].append(completed)
                    row['variants'][name]['gpu_samples_seconds'].append(gpu)
        for name,jit in runners.items():
            result=row['variants'][name]
            result['median_completed_seconds']=statistics.median(result['completed_samples_seconds'])
            result['median_gpu_seconds']=statistics.median(statistics.median(batch) for batch in result['gpu_samples_seconds'])
            for seed,scale in ((101,.01),(202,1.0),(303,.00001)):
                values=(np.random.default_rng(seed).normal(size=(32,k))*scale).astype(np.float32)
                actual=jit(Tensor(values).realize()).numpy()
                result['validation'].append({'seed':seed,'scale':scale,**check(actual,values,weight_array)})
        records.append(row);print('rechecked',suffix,{name:(v['median_gpu_seconds'],v['median_completed_seconds']) for name,v in row['variants'].items()},flush=True)
    report={'status':'passed','model_sha256':hashlib.file_digest(Path(model).open('rb'),'sha256').hexdigest(),'tinygrad':tiny_info(),
            'records':records,'protocol':{'search':'disabled; replay cached opts','input_seeds':[29,101,202,303],
                        'oracle':'float64 dot of nearest-even FP16 operands; abs-product * 3e-6 + 1e-10',
                        'timing':'same OpenCL device/weights/input; rotate variants; one discarded then seven batches of 20',
                        'gpu':'OpenCL event timing, median of batch medians','completed':'TinyJit calls plus synchronization, host copy excluded',
                        'cache':'hot weights; warm each variant for one second'}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','root','fallback-cache','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.root,a.fallback_cache,a.out)
