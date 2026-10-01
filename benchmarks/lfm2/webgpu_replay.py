"""Quick optimization gate against the retained independent NumPy logits."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import LFM2
from benchmarks.lfm2.webgpu_run import metrics


def run(model,bundle,evidence,out=None):
    report=json.loads((evidence/'report.json').read_text())
    validation=[]
    with tensor.Device(provider='webgpu') as device,LFM2(model,bundle,device,context=512) as engine:
        first=None
        for i,case in enumerate(report['validation']):
            if case['reset']:engine.reset()
            actual=engine.forward(case['tokens']);expected=np.load(evidence/f'{i}-numpy-logits.npy');result=metrics(actual,expected)
            if not np.isfinite(actual).all() or result['relative_rms']>=.01 or result['cosine']<=.9999 or result['argmax'][0]!=result['argmax'][1]:raise AssertionError((case['name'],result))
            if first is None:first=actual.copy()
            if case['name']=='reset_chat':np.testing.assert_array_equal(actual,first)
            validation.append({'name':case['name'],'numpy':result})
            print(case['name'],result,flush=True)
        result={'status':'passed','model':str(model),'bundle':str(bundle),'implementation':engine.manifest['implementation'],'validation':validation}
        if out is not None:
            out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(result,indent=2)+'\n')
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','evidence'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args();run(a.model,a.bundle,a.evidence,a.out)
