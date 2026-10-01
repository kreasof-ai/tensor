"""Time the same natural chat on frozen or current compiler-free wheels."""
import argparse,json,statistics,time
from pathlib import Path
import tensor
from tensor_llm import LFM2


def run(model,bundle,out,repeats):
    with tensor.Device(provider='webgpu') as device,LFM2(model,bundle,device,context=512) as engine:
        samples=[];expected=None
        for repeat in range(repeats+1):
            start=time.perf_counter();result=engine.generate('What is 2 + 2?',max_tokens=96)
            device.synchronize();elapsed=time.perf_counter()-start
            if expected is None:expected=result
            if result!=expected:raise AssertionError('greedy replay differs')
            if repeat:samples.append(elapsed)
        report={'protocol':'default full generate call plus completion; reset/tokenization/prefill/sampling included; load excluded',
                'adapter':device.info,'implementation':engine.manifest['implementation'],'generation':expected,
                'samples_seconds':samples,'median_seconds':statistics.median(samples)}
        Path(out).write_text(json.dumps(report,indent=2)+'\n');print(report['median_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args();run(a.model,a.bundle,a.out,a.repeats)
