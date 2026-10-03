"""Full-model prefill variants and native chunk-size controls on one GPU."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor_llm import LFM2
from benchmarks.lfm2.webgpu_run import cached_reference,metrics
from benchmarks.lfm2.vulkan_reference import Reference,COMMIT


def run(model,root,reference,fixtures,out,repeats=7,bundles=None,max_buffer_size=None,allow_decode_changes=False,native_chunks=(32,64,128)):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[{key:row[key] for key in ('name','tokens','reset')} for row in json.loads((Path(fixtures)/'report.json').read_text())['validation']]
    expected,oracle=cached_reference(fixtures,model,cases)
    names=bundles or ['baseline32','chunked64','chunked128','unrolled32','unrolled64','unrolled128','outer32','outer64','outer128']
    report=dict(status='validating',model_sha256=oracle['model_sha256'],independent_reference=oracle,
        native_commit=COMMIT,validation=[],benchmarks=[],sources={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),text=Path(p).read_text()) for p in
        (__file__,'benchmarks/lfm2/vulkan_reference.py','src/tensor/compiler/webgpu_lowering.py','src/tensor/compiler/webgpu.py',
         'packages/tensor-llm/src/tensor_llm/model.py','packages/tensor-llm/src/tensor_llm/webgpu_kernels.py','benchmarks/lfm2/producer.py')},
        protocol=dict(context=512,warmups=3,repeats=repeats,decode_tokens=64,
            max_buffer_size=max_buffer_size,
            allow_decode_changes=allow_decode_changes,
            native_chunks=list(native_chunks),
            timing='Completed forward returning host F32 logits. Loading, AOT compilation, reset and sampling excluded. Rotate all runners on the GPU sequentially; native batch/ubatch match its chunk. Native ordinary Vulkan arithmetic. Tensor quant_searched uses nearest-even F16 prefill operands/F32 accumulation. Experimental prefill_q16 rounds activations to F16, approximates selected projections with two Q8 components, exact decoded F32 weight values and F32 accumulation; quantization dispatches are timed. Experimental prefill_mixed replaces selected integer projections with packed F16 pairs, short F16 FMA chains and F32 totals. Optional signed-byte weight caches preserve Q4_0 values at doubled matrix storage. The bundle parameters record each projection contract.'))
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with ExitStack() as stack:
        natives={f'llama{chunk}':Reference(model,reference,prefill_chunk=chunk) for chunk in native_chunks}
        for native in natives.values():stack.callback(native.close)
        device=stack.enter_context(tensor.Device(provider='webgpu',max_buffer_size=max_buffer_size))
        engines={name:stack.enter_context(LFM2(model,root/name,device,context=512)) for name in names}
        runners={**engines,**natives};names=list(runners)
        report.update(adapter=device.info,bundles={n:e.manifest for n,e in engines.items()},
            allocated_bytes={n:e.allocated_bytes for n,e in engines.items()},
            state_only_dispatch_counts={n:{str(r):len(p.nodes) for r,p in e.prepared_states.items()} for n,e in engines.items()},
            dispatch_counts={n:{str(r):len(p.nodes) for r,p in e.prepared.items()} for n,e in engines.items()},
            runtime_specialization={n:dict(last_token_ffn=e.prefill_last,tail_rows=e.prefill_tail,
                description='When enabled: store all final-attention K/V, retain the last eight queries and residual rows, run the two causal convolution layers on that suffix, and run only the final FFN row. Intermediate prefill chunks stop after updating persistent state; the final chunk always computes logits.') for n,e in engines.items()})
        for name,native in natives.items():(out/f'{name}.log').write_text(''.join(native.logs),encoding='utf-8')
        # Projection decode code must be identical across all prefill experiments.
        baseline_name='baseline32' if 'baseline32' in engines else next(iter(engines))
        report['baseline_bundle']=baseline_name
        baseline=engines[baseline_name];projection_hashes={}
        for n,e in engines.items():
            projection_hashes[n]={}
            for key,record in e.manifest['kernels'].items():
                if record['parameters'].get('r')==1 and record['kind'] in ('linear','linear_add','ffn'):
                    projection_hashes[n][key]=hashlib.sha256(read_artifact(e.directory/record['artifact'])[1]['kernel.wgsl']).hexdigest()
            if not allow_decode_changes and projection_hashes[n]!=projection_hashes[baseline_name]:raise AssertionError('prefill changed decode projection code')
        report['decode_projection_wgsl']=projection_hashes
        first={}
        for i,case in enumerate(cases):
            row={**case,'seconds':{}}
            for name,runner in runners.items():
                if case['reset']:runner.reset()
                start=time.perf_counter();actual=runner.forward(case['tokens']);row['seconds'][name]=time.perf_counter()-start
                quality=metrics(actual,expected[i]);row[name]=quality;np.save(out/f'{i}-{name}.npy',actual)
                if not np.isfinite(actual).all() or (name in engines and (quality['relative_rms']>=.01 or quality['cosine']<=.9999 or quality['argmax'][0]!=quality['argmax'][1])):
                    report.update(status='failed',failure=dict(fixture=case['name'],runner=name,metrics=quality));save();raise AssertionError(report['failure'])
                if i==0:first[name]=actual.copy()
                if case['name']=='reset_chat' and not np.array_equal(first[name],actual):raise AssertionError('reset changed logits: '+name)
            report['validation'].append(row);save();print('validated',case['name'],flush=True)
        prompt=cases[0]['tokens'];forced=np.resize(prompt,64).tolist();report['status']='measuring';save()
        for length in (32,128,384):
            prefix=np.resize(prompt,length).tolist();samples={name:[] for name in names}
            for iteration in range(repeats+3):
                for name in names[iteration%len(names):]+names[:iteration%len(names)]:
                    runner=runners[name];runner.reset();start=time.perf_counter();runner.forward(prefix);prefill=time.perf_counter()-start
                    start=time.perf_counter()
                    for token in forced:runner.forward([token])
                    decode=time.perf_counter()-start
                    if iteration>=3:samples[name].append(dict(prefill_seconds=prefill,decode_seconds=decode))
            row=dict(prompt_tokens=length,decode_tokens=64,samples=samples)
            for name,values in samples.items():
                row[name]=dict(prefill_tokens_per_second=length/statistics.median(v['prefill_seconds'] for v in values),
                    decode_tokens_per_second=64/statistics.median(v['decode_seconds'] for v in values))
            report['benchmarks'].append(row);save()
            print('benchmark',length,{n:round(row[n]['prefill_tokens_per_second'],1) for n in names},flush=True)
        generation={}
        for name,engine in engines.items():
            gpu=engine.generate('What is 2 + 2?',max_tokens=96)
            host=engine.generate('What is 2 + 2?',max_tokens=96,gpu_greedy=False)
            if gpu!=host:raise AssertionError('host/GPU greedy mismatch: '+name)
            generation[name]=gpu
        if any(v!=generation[baseline_name] for v in generation.values()):raise AssertionError('prefill changed generated token IDs')
        report['generation']=generation
        report['numerical_summary']={name:dict(max_relative_rms=max(v[name]['relative_rms'] for v in report['validation']),
            min_cosine=min(v[name]['cosine'] for v in report['validation']),matching_argmax=sum(v[name]['argmax'][0]==v[name]['argmax'][1] for v in report['validation'])) for name in names}
        report['status']='passed';save()
        for name,native in natives.items():(out/f'{name}.log').write_text(''.join(native.logs),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','root','reference','fixtures','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--bundles',nargs='+')
    p.add_argument('--max-buffer-size',type=int)
    p.add_argument('--allow-decode-changes',action='store_true',help='compare profiles that also change decode projections; numerical gates remain active')
    p.add_argument('--native-chunks',type=int,nargs='+',choices=(32,64,128),default=[32,64,128])
    a=p.parse_args();run(a.model,a.root,a.reference,a.fixtures,a.out,a.repeats,a.bundles,a.max_buffer_size,a.allow_decode_changes,tuple(a.native_chunks))
