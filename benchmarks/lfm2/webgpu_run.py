"""Validate and measure the 230M WebGPU plan against independent public APIs."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import tensor
from tensor_llm import LFM2
from benchmarks.lfm2.numpy_reference import Reference as NumpyReference
from benchmarks.lfm2.vulkan_reference import Reference as NativeReference,COMMIT,RELEASE_SHA256


def metrics(actual,expected):
    x,y=actual.astype(np.float64),expected.astype(np.float64)
    return {'relative_rms':float(np.linalg.norm(x-y)/np.linalg.norm(y)),
            'cosine':float(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)),
            'argmax':[int(np.argmax(x)),int(np.argmax(y))]}


def run(model,bundle,reference,out,*,repeats=5,decode=64):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    native=NativeReference(model,reference,context=512)
    try:
        numpy=NumpyReference(model,context=512)
        with tensor.Device(provider='webgpu') as device,LFM2(model,bundle,device,context=512) as engine:
            if device.info['adapter']['backend_type']!='Vulkan':raise RuntimeError('this report requires the Vulkan adapter')
            prompt=engine.tokenizer.chat('What is 2 + 2?')
            cases=[{'name':'chat','tokens':prompt,'reset':True}]
            # Native greedy continuation is a shared forced-token fixture.
            native.reset();logits=native.forward(prompt)
            for i in range(3):
                token=int(np.argmax(logits));cases.append({'name':f'chat_decode_{i}','tokens':[token],'reset':False})
                logits=native.forward([token])
            for length in (31,32,33,127,128,129,511):
                cases.extend([{'name':f'prefix_{length}','tokens':np.resize(prompt,length).tolist(),'reset':True},
                              {'name':f'cached_{length}','tokens':[3097],'reset':False}])
            cases.append({'name':'reset_chat','tokens':prompt,'reset':True})
            validation=[];first=None
            for i,case in enumerate(cases):
                if case['reset']:
                    engine.reset();numpy.reset();native.reset()
                actual=engine.forward(case['tokens']);expected=numpy.forward(case['tokens']);baseline=native.forward(case['tokens'])
                independent=metrics(actual,expected);comparison=metrics(actual,baseline)
                if not np.all(np.isfinite(actual)) or independent['relative_rms']>=.01 or independent['cosine']<=.9999 or independent['argmax'][0]!=independent['argmax'][1]:
                    raise AssertionError((case['name'],independent))
                if first is None:first=actual.copy()
                if case['name']=='reset_chat' and not np.array_equal(first,actual):raise AssertionError('reset must be bitwise identical')
                np.save(out/f'{i}-tensor-logits.npy',actual);np.save(out/f'{i}-numpy-logits.npy',expected);np.save(out/f'{i}-native-logits.npy',baseline)
                validation.append({**case,'numpy':independent,'llama_cpp':comparison})
                print(case['name'],independent,flush=True)
            del numpy
            for tokens,message in (([], 'nonempty'),([engine.config.vocab],'vocabulary'),([1]*513,'capacity')):
                try:engine.forward(tokens)
                except ValueError as error:
                    if message not in str(error):raise
                else:raise AssertionError('invalid input accepted')
            benchmarks=[]
            forced=np.resize(prompt,decode).tolist()
            # Rotate engine order; each call includes completion and host logits.
            for length in (32,128,384):
                prefix=np.resize(prompt,length).tolist();times={'tensor':[],'llama_cpp':[]}
                for repeat in range(repeats+1):
                    order=(('tensor',engine),('llama_cpp',native)) if repeat%2==0 else (('llama_cpp',native),('tensor',engine))
                    for name,runner in order:
                        runner.reset();start=time.perf_counter();runner.forward(prefix);prefill=time.perf_counter()-start
                        start=time.perf_counter()
                        for token in forced:runner.forward([token])
                        elapsed=time.perf_counter()-start
                        if repeat:times[name].append({'prefill_seconds':prefill,'decode_seconds':elapsed})
                row={'prompt_tokens':length,'decode_tokens':decode,'samples':times}
                for name,samples in times.items():
                    row[name]={'prefill_tokens_per_second':length/statistics.median(s['prefill_seconds'] for s in samples),
                               'decode_tokens_per_second':decode/statistics.median(s['decode_seconds'] for s in samples)}
                benchmarks.append(row);print('benchmark',length,row['tensor'],row['llama_cpp'],flush=True)
            generation=engine.generate('What is 2 + 2?',max_tokens=96)
            generation_samples={'gpu_greedy':[],'host_greedy':[]}
            for repeat in range(repeats+1):
                modes=(True,False) if repeat%2==0 else (False,True)
                for gpu in modes:
                    start=time.perf_counter();result=engine.generate('What is 2 + 2?',max_tokens=96,gpu_greedy=gpu)
                    elapsed=time.perf_counter()-start
                    if result!=generation:raise AssertionError('GPU/host greedy generation mismatch')
                    if repeat:generation_samples['gpu_greedy' if gpu else 'host_greedy'].append(elapsed)
            report={'status':'passed','model':str(Path(model).resolve()),'model_sha256':hashlib.file_digest(Path(model).open('rb'),'sha256').hexdigest(),
                    'encodings':dict(Counter(t.encoding for t in engine.gguf.tensors.values())),
                    'platform':platform.platform(),'adapter':device.info,'allocated_bytes':engine.allocated_bytes,
                    'llama_cpp':{'commit':COMMIT,'release_archive_sha256':RELEASE_SHA256,'cache':'F16','flash_attention':True,'gpu_layers':-1,'threads':6},
                    'protocol':{'context':512,'prefill_chunk':32,'warmups':1,'repeats':repeats,'last_token_logits':'host FP32','sampling':'excluded','loading':'excluded'},
                    'validation':validation,'benchmarks':benchmarks,'generation':generation,
                    'dispatches':{r:len(plan) for r,plan in engine.plans.items()},
                    'generation_timing':{'protocol':'full generate call, including prompt tokenization/prefill, reset, greedy sampling and completion; model loading excluded',
                                         'generated_tokens':len(generation['generated_tokens']),'samples_seconds':generation_samples,
                                         'median_seconds':{name:statistics.median(samples) for name,samples in generation_samples.items()}},
                    'implementation':engine.manifest['implementation']}
            (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (out/'native.log').write_text(''.join(native.logs),encoding='utf-8')
        return report
    finally:native.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--decode',type=int,default=64)
    a=p.parse_args()
    if a.repeats<1 or not 1<=a.decode<=128:p.error('requires positive repeats and 1..128 decode tokens')
    run(a.model,a.bundle,a.reference,a.out,repeats=a.repeats,decode=a.decode)
