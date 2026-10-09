"""Experimental Q8 activation dots; retain the existing full-logit gate."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.runtime.abi import BoundCall
from tensor_llm import LFM2
from tensor_llm.lfm2.kernels.webgpu import source
from benchmarks.lfm2.webgpu_run import metrics


def run(model,bundle,evidence,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    kernels={}
    for kind,p in (('quantize_q8',{'k':2560}),('linear_q8',{'k':2560,'o':1024})):
        path=out/(kind+'.py');artifact=path.with_suffix('.tbin');text=source(kind,p)
        if path.exists() and path.read_text()!=text:artifact.unlink(missing_ok=True)
        path.write_text(text)
        if not artifact.exists():tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
        kernels[kind]=artifact
    saved=json.loads((evidence/'report.json').read_text());cases=[]
    with tensor.Device(provider='webgpu') as device,LFM2(model,bundle,device,context=512) as engine:
        q=device.zeros(640,dtype='uint32');scales=device.zeros(80);sums=device.zeros(80,dtype='int32')
        extras=[device.load(kernels[kind]) for kind in ('quantize_q8','linear_q8')]
        replaced=[];plan=[]
        weight_names={id(w):name for name,w in engine.weights.items()}
        for kernel,call in engine.plans[1]:
            name=next((weight_names[id(w)] for w in call.storage if id(w) in weight_names),None)
            if name and name.endswith('ffn_down.weight'):
                arguments=dict(zip((a['name'] for a in kernel.manifest['abi']),call.storage))
                x,w,y=(arguments[key] for key in ('x','w','out'))
                for executable,args in ((extras[0],(x,q,scales,sums)),(extras[1],(q,scales,sums,w,y))):
                    values,symbols,launch=executable._bind(args,{},include_outputs=True)
                    plan.append((executable,BoundCall(device,executable.manifest,values,symbols,launch,validated=True)))
                replaced.append(name)
            else:plan.append((kernel,call))
        assert len(replaced)==14
        engine.prepared[1].close();engine.prepared[1]=device.prepare_plan(plan)
        for i,case in enumerate(saved['validation']):
            if case['reset']:engine.reset()
            actual=engine.forward(case['tokens']);expected=np.load(evidence/f'{i}-numpy-logits.npy');m=metrics(actual,expected)
            passed=np.isfinite(actual).all() and m['relative_rms']<.01 and m['cosine']>.9999 and m['argmax'][0]==m['argmax'][1]
            row={'name':case['name'],'metrics':m,'passed':bool(passed)};cases.append(row);print(row,flush=True)
        report={'status':'passed' if all(c['passed'] for c in cases) else 'rejected','replaced':replaced,'cases':cases,
                'note':'benchmark-only FFN-down activation quantization; not selected by the installed LFM2 producer'}
        (out/'report.json').write_text(json.dumps(report,indent=2))
        engine.prepared[1].close()
        for executable in extras:executable.release()
        for buffer in (q,scales,sums):buffer.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','evidence','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.model,a.bundle,a.evidence,a.out)
