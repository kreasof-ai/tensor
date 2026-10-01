"""Compare native/Python encoding with identical shaders, buffers and inputs."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import LFM2


def run(model,bundle,out):
    samples={'native':[],'python':[]}
    with tensor.Device(provider='webgpu') as device,LFM2(model,bundle,device,context=512) as engine:
        encoders={r:p._encode for r,p in engine.prepared.items()}
        assert all(encoders.values()),'requires the optional native extension'
        prompt=engine.tokenizer.chat('What is 2 + 2?');prefix=np.resize(prompt,128).tolist();forced=np.resize(prompt,64).tolist()
        expected=None
        for repeat in range(6):
            for mode in (('native','python') if repeat%2==0 else ('python','native')):
                for r,plan in engine.prepared.items():plan._encode=encoders[r] if mode=='native' else None
                engine.reset();start=time.perf_counter();engine.forward(prefix);prefill=time.perf_counter()-start
                start=time.perf_counter()
                for token in forced:actual=engine.forward([token])
                decode=time.perf_counter()-start
                if expected is None:expected=actual.copy()
                np.testing.assert_array_equal(actual,expected)
                if repeat:samples[mode].append({'prefill_seconds':prefill,'decode_seconds':decode})
        result={'status':'passed','model':str(model),'adapter':device.info,'samples':samples,'implementation':engine.manifest['implementation'],
                'protocol':'identical shaders/state/inputs, alternate encoders, five repetitions after warmup, prefix 128 plus 64 forced decode, completed host FP32 logits, bitwise final-logit equality',
                'tokens_per_second':{mode:{'prefill':128/statistics.median(s['prefill_seconds'] for s in rows),
                    'decode':64/statistics.median(s['decode_seconds'] for s in rows)} for mode,rows in samples.items()}}
        Path(out).write_text(json.dumps(result,indent=2)+'\n');print(result['tokens_per_second'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.bundle,a.out)
