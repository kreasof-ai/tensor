"""Matched single-sequence LFM2 API latency against llama.cpp CUDA."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,time
from importlib.metadata import version
import numpy as np
import tensor
from tensor_llm import GGUF
from tensor_llm.lfm2.model import LFM2
from tensor_llm.common.tokenizer import Tokenizer


def benchmark(model,bundle,reference_executable,out,*,depths=(128,512,2048,8192),generated=256,repeats=5,engine_cls=None):
    if generated < 1 or repeats < 1 or not depths or any(depth < 1 for depth in depths):
        raise ValueError('requires positive depths, decode tokens and repeats')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'benchmark.json').unlink(missing_ok=True);gguf=GGUF(model);tokenizer=Tokenizer(gguf.metadata)
    pattern=tokenizer.encode('Tensor evaluates a fixed sequence to compare cached language model inference. The same tokens are used by both engines. ',add_bos=False)
    cases=[]
    for depth in depths:
        prompt=[tokenizer.bos]+[pattern[i%len(pattern)] for i in range(depth-1)]
        decode=[pattern[i%len(pattern)] for i in range(generated)]
        cases.append({'name':f'pp{depth}-tg{generated}','prompt':prompt,'decode':decode,'repeats':repeats})
    spec={'model':str(Path(model).resolve()),'out':str((out/'llama').resolve()),'context':max(depths)+generated,'benchmarks':cases}
    specfile=out/'reference-spec.json';specfile.write_text(json.dumps(spec,indent=2)+'\n')
    with (out/'llama.log').open('w') as log:subprocess.run([str(reference_executable.resolve()),str(specfile)],check=True,stderr=log,stdout=log)
    reference=json.loads((out/'llama/reference.json').read_text());results=[]
    if len(reference['benchmarks']) != len(cases):raise ValueError('incomplete llama.cpp benchmark')
    with tensor.Device() as device,(engine_cls or LFM2)(model,bundle,device,context=max(depths)+generated) as network:
        for item,baseline in zip(cases,reference['benchmarks']):
            samples=[]
            for repeat in range(-1,repeats):
                network.reset();start=time.perf_counter();network.forward(item['prompt']);prefill=time.perf_counter()-start
                latencies=[];start=time.perf_counter()
                for token in item['decode']:
                    begin=time.perf_counter();network.forward([token]);latencies.append(time.perf_counter()-begin)
                seconds=time.perf_counter()-start
                if repeat>=0:samples.append({'prefill_seconds':prefill,'decode_seconds':seconds,'decode_latencies':latencies})
            tensor_prefill=float(np.median([s['prefill_seconds'] for s in samples]));llama_prefill=float(np.median([s['prefill_seconds'] for s in baseline['samples']]))
            tensor_decode=float(np.median([s['decode_seconds'] for s in samples]));llama_decode=float(np.median([s['decode_seconds'] for s in baseline['samples']]))
            row={'name':item['name'],'prompt_tokens':len(item['prompt']),'decode_tokens':generated,'samples':samples,'llama_samples':baseline['samples'],
                 'tensor_prefill_tokens_per_second':len(item['prompt'])/tensor_prefill,'llama_prefill_tokens_per_second':len(item['prompt'])/llama_prefill,
                 'tensor_decode_tokens_per_second':generated/tensor_decode,'llama_decode_tokens_per_second':generated/llama_decode,
                 'tensor_decode_milliseconds':tensor_decode/generated*1000,'llama_decode_milliseconds':llama_decode/generated*1000,
                 'prefill_speedup':llama_prefill/tensor_prefill,'decode_speedup':llama_decode/tensor_decode}
            results.append(row);print({k:v for k,v in row.items() if 'samples' not in k},flush=True)
        report={'schema':'tensor.lfm2-benchmark.v1','status':'passed','adapter':device.info,'platform':platform.platform(),
                'model_sha256':hashlib.file_digest(Path(model).open('rb'),'sha256').hexdigest(),'model_file':Path(model).name,
                'bundle_sha256':hashlib.file_digest((Path(bundle)/'inference.json').open('rb'),'sha256').hexdigest(),
                'llama_cpp':{'commit':'f7b384c1e5c5b2c5b321a4a7cefea04b15b54cb7',
                    'helper_sha256':hashlib.file_digest(reference_executable.open('rb'),'sha256').hexdigest(),
                    'helper_source_sha256':hashlib.sha256(Path(__file__).with_name('reference.cpp').read_bytes()).hexdigest(),
                    'configuration':{k:v for k,v in reference.items() if k not in ('benchmarks','validation','tokenization')}},
                'protocol':'one sequence; prefill chunks <=128; one-token decode; last-token logits available on host; excludes loading/tokenization/sampling',
                'cache_type':'f16','cuda_graphs':True,'owned_device_bytes':network.allocated_bytes,'launches_per_decode':len(network.plans[1]),
                'versions':{name:version(name) for name in ('numpy','tensor-workspace','tensor-llm','regex')},'cases':results}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--depths',type=int,nargs='+',default=[128,512,2048,8192]);p.add_argument('--generated',type=int,default=256)
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    benchmark(a.model,a.bundle,a.reference,a.out,depths=a.depths,generated=a.generated,repeats=a.repeats)

if __name__=='__main__':main()
