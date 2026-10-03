"""Host API breakdown; instrumented timings are diagnostic, not throughput."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,cProfile,hashlib,io,json,pstats,time
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import LFM2


def run(model,bundle,out):
    with tensor.Device(provider='webgpu') as device,LFM2(model,bundle,device,context=512) as engine:
        prompt=engine.tokenizer.chat('What is 2 + 2?')
        engine.forward(np.resize(prompt,128))
        for token in np.resize(prompt,32):engine.forward([int(token)])
        profile=cProfile.Profile();profile.enable()
        start=time.perf_counter()
        for token in np.resize(prompt,64):engine.forward([int(token)])
        elapsed=time.perf_counter()-start;profile.disable()
        output=io.StringIO();pstats.Stats(profile,stream=output).strip_dirs().sort_stats('cumulative').print_stats(55)
        stats=[]
        for (file,line,name),(primitive,calls,self_time,total,callers) in pstats.Stats(profile).stats.items():
            stats.append(dict(file=file,line=line,name=name,calls=calls,self_seconds=self_time,cumulative_seconds=total))
        report=dict(adapter=device.info,bundle=engine.manifest,decode_tokens=64,instrumented_seconds=elapsed,
                    stats=sorted(stats,key=lambda row:-row['cumulative_seconds']),text=output.getvalue())
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
        print(output.getvalue(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.bundle,a.out)
