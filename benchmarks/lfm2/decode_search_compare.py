"""Paired full-model baseline/searched/native throughput with independent fixtures."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import tensor
from tensor_llm import LFM2
from benchmarks.lfm2.webgpu_run import cached_reference,metrics
from benchmarks.lfm2.vulkan_reference import Reference as NativeReference,COMMIT


def run(model,baseline,searched,reference,fixtures,out,repeats=7,*,runtime_bundle=None,baseline_wrapper=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[{key:row[key] for key in ('name','tokens','reset')} for row in json.loads((Path(fixtures)/'report.json').read_text())['validation']]
    expected,oracle=cached_reference(fixtures,model,cases);validation=[];benchmarks=[]
    report={'status':'validating','model_sha256':oracle['model_sha256'],'independent_reference':oracle,
            'native_commit':COMMIT,'validation':validation,'benchmarks':benchmarks,
            'protocol':{'context':512,'prefill_chunk':32,'warmups':3,'repeats':repeats,'decode_tokens':64,
                        'timing':'completed forward, host FP32 logits; loading, AOT compile, initial calls, reset and sampling excluded; rotate all three runners; sequential GPU'}}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with ExitStack() as stack:
        native=NativeReference(model,reference);stack.callback(native.close)
        device=stack.enter_context(tensor.Device(provider='webgpu'))
        before=stack.enter_context(LFM2(model,baseline,device,context=512));after=stack.enter_context(LFM2(model,searched,device,context=512))
        if baseline_wrapper:baseline_wrapper(before)
        runners={'tensor_before':before,'tensor_searched':after,'llama_cpp':native};names=list(runners)
        report.update(adapter=device.info,bundles={'before':before.manifest,'searched':after.manifest})
        if runtime_bundle is not None:
            middle=stack.enter_context(LFM2(model,runtime_bundle,device,context=512))
            runners={'tensor_before':before,'tensor_runtime':middle,'tensor_searched':after,'llama_cpp':native};names=list(runners)
            report['bundles']['runtime']=middle.manifest
        report['protocol']['timing']=report['protocol']['timing'].replace('all three runners','all runners')
        first={}
        for i,case in enumerate(cases):
            row={**case,'seconds':{}}
            for name,runner in runners.items():
                if case['reset']:runner.reset()
                start=time.perf_counter();actual=runner.forward(case['tokens']);row['seconds'][name]=time.perf_counter()-start
                row[name]=metrics(actual,expected[i]);np.save(out/f'{i}-{name}.npy',actual)
                check=row[name]
                if not np.isfinite(actual).all() or (name!='llama_cpp' and (check['relative_rms']>=.01 or check['cosine']<=.9999 or check['argmax'][0]!=check['argmax'][1])):
                    report.update(status='failed',failure={'fixture':case['name'],'runner':name,'metrics':check});save();raise AssertionError(report['failure'])
                if i==0:first[name]=actual.copy()
                if case['name']=='reset_chat' and not np.array_equal(first[name],actual):raise AssertionError('reset changed logits: '+name)
            validation.append(row);save();print('validation',case['name'],row['tensor_searched'],flush=True)
        prompt=cases[0]['tokens'];forced=np.resize(prompt,64).tolist();report['status']='measuring';save()
        for length in (32,128,384):
            prefix=np.resize(prompt,length).tolist();samples={name:[] for name in names}
            for repeat in range(repeats+3):
                for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                    runner=runners[name];runner.reset();start=time.perf_counter();runner.forward(prefix);prefill=time.perf_counter()-start
                    start=time.perf_counter()
                    for token in forced:runner.forward([token])
                    decode=time.perf_counter()-start
                    if repeat>=3:samples[name].append({'prefill_seconds':prefill,'decode_seconds':decode})
            row={'prompt_tokens':length,'decode_tokens':64,'samples':samples}
            for name,values in samples.items():row[name]={'prefill_tokens_per_second':length/statistics.median(v['prefill_seconds'] for v in values),
                                                         'decode_tokens_per_second':64/statistics.median(v['decode_seconds'] for v in values)}
            benchmarks.append(row);save();print('benchmark',length,{n:row[n] for n in names},flush=True)
        # Generated text and the host/GPU sampling paths must remain consistent.
        engines=[(name,runner) for name,runner in runners.items() if name.startswith('tensor_')]
        generation={};sample_times={name:[] for name,engine in engines}
        for name,engine in engines:
            first=engine.generate('What is 2 + 2?',max_tokens=96)
            host=engine.generate('What is 2 + 2?',max_tokens=96,gpu_greedy=False)
            if first!=host:raise AssertionError('host/GPU greedy mismatch: '+name)
            generation[name]=first
        if any(result!=generation['tensor_before'] for result in generation.values()):raise AssertionError('search changed greedy generation')
        for repeat in range(repeats+1):
            order=engines[repeat%len(engines):]+engines[:repeat%len(engines)]
            for name,engine in order:
                start=time.perf_counter();result=engine.generate('What is 2 + 2?',max_tokens=96);elapsed=time.perf_counter()-start
                if result!=generation[name]:raise AssertionError('nondeterministic generation')
                if repeat:sample_times[name].append(elapsed)
        report.update(status='passed',generation=generation['tensor_searched'],generation_samples_seconds=sample_times,
                      generation_median_seconds={name:statistics.median(samples) for name,samples in sample_times.items()})
        save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','baseline','searched','reference','fixtures','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=7);a=p.parse_args()
    run(a.model,a.baseline,a.searched,a.reference,a.fixtures,a.out,a.repeats)
