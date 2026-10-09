"""Independent finalist replay with all model weights and held-out activations."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor.artifacts.format import read_artifact
from tensor_llm import GGUF
from tensor_llm.lfm2.model import DECODE_GEMV,DECODE_GEMV_COMMON
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,bind,check
from benchmarks.lfm2.decode_kernel_search import TrafficTimer


def run(model,root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    search=json.loads((root/'report.json').read_text())
    if search['status']!='finished':raise ValueError('search must finish first')
    gguf=GGUF(model);infos=[t for t in gguf.tensors.values() if len(t.shape)==2 and not t.name.endswith('conv.weight')]
    weights={t.name:np.array(gguf.array(t.name,dtype=np.float16),copy=True) for t in infos}
    device=Device();device._adapter=TimestampAdapter(device._adapter);records=[]
    with device:
        buffers={n:device.from_numpy(w.ravel()) for n,w in weights.items()}
        ins={k:device.from_numpy(np.random.default_rng(29).normal(size=k).astype(np.float32)) for _,k in {t.shape for t in infos}}
        # Independent outputs avoid introducing write-after-write hazards
        # between otherwise unrelated projections in this replay.
        outs={t.name:device.full(t.shape[0],np.nan) for t in infos}
        kernels={shape:device.load(root/f'baseline-{shape[0]}x{shape[1]}.tbin') for shape in {t.shape for t in infos}}
        calls=[]
        for t in infos:
            single=bind(device,kernels[t.shape],(ins[t.shape[1]],buffers[t.name],outs[t.name]));calls.append(single.calls[0]);single.close()
        for row in search['records']:
            indices=[i for i,t in enumerate(infos) if t.name in row['matrices']];o,k=row['shape'];seen=set();selected=[]
            passed=sorted((c for c in row['candidates'] if c['status']=='passed'),key=lambda c:c['median_gpu_seconds'])
            for candidate in passed:
                _,files=read_artifact(root/candidate['artifact']);digest=hashlib.sha256(files['kernel.wgsl']).hexdigest()
                if digest in seen:continue
                seen.add(digest);selected.append(candidate)
                if len(selected)==4:break
            for family in {c['config']['family'] for c in passed}:
                candidate=next(c for c in passed if c['config']['family']==family)
                if candidate not in selected:selected.append(candidate)
            if search.get('extended'):
                current={**DECODE_GEMV_COMMON,**DECODE_GEMV[(k,o)]}
                candidate=next(c for c in passed if c['config']==current)
                if candidate not in selected:selected.append(candidate)
            runners={};result={'group':row['group'],'shape':row['shape'],'variants':{}}
            for candidate in [None,*selected]:
                name='baseline' if candidate is None else str(candidate['index']);paired=candidate is not None and candidate['config']['family']=='paired'
                kernel=None if candidate is None else device.load(root/candidate['artifact']);updated=[];targets=[];checks=[]
                for i,(t,call) in enumerate(zip(infos,calls)):
                    if i not in indices:updated.append(call);continue
                    if paired and '.ffn_up.' in t.name:continue
                    actual_kernel=kernel if kernel is not None else kernels[t.shape]
                    args=(ins[k],buffers[t.name],outs[t.name])
                    if paired:args=(ins[k],buffers[t.name],buffers[t.name.replace('_gate.','_up.')],outs[t.name])
                    single=bind(device,actual_kernel,args)
                    for seed,scale in ((101,.01),(202,1),(303,1e-5)):
                        x=(np.random.default_rng(seed).normal(size=k)*scale).astype(np.float32);w=weights[t.name].astype(np.float64)
                        expected=w@x.astype(np.float64);bound=np.abs(w)@np.abs(x.astype(np.float64))*3e-6+1e-10
                        if paired:
                            w2=weights[t.name.replace('_gate.','_up.')].astype(np.float64);u=w2@x.astype(np.float64)
                            ub=np.abs(w2)@np.abs(x.astype(np.float64))*3e-6+1e-10;silu=expected/(1+np.exp(-expected))
                            bound=np.abs(u)*1.1*bound+np.abs(silu)*ub+1.1*bound*ub+np.abs(silu*u)*2e-6+1e-10;expected=silu*u
                        device._gpu.queue.write_buffer(ins[k]._storage,0,x.tobytes())
                        device._gpu.queue.write_buffer(outs[t.name]._storage,0,np.full(o,np.nan,np.float32).tobytes())
                        single.launch();checks.append({'weight':t.name,'seed':seed,'scale':scale,**check(outs[t.name].to_numpy(),expected,bound)})
                    targets.append(len(updated));updated.append(single.calls[0]);single.close()
                plan=device.prepare_plan(updated);timer=TrafficTimer(device,plan,targets);runners[name]=(plan,timer,kernel)
                result['variants'][name]={'config':candidate['config'] if candidate else None,'artifact':candidate['artifact'] if candidate else None,
                                         'validation':checks,'samples':[]}
            for plan,_,_ in runners.values():
                warm=time.perf_counter()+1
                while time.perf_counter()<warm:plan.launch();device.synchronize()
            names=list(runners)
            for repeat in range(8):
                for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                    sample=runners[name][1].sample()
                    if repeat:result['variants'][name]['samples'].append(sample)
            for name,(plan,timer,kernel) in runners.items():
                value=result['variants'][name];value['median_gpu_seconds']=statistics.median(s['target_seconds'] for s in value['samples'])
                timer.close();plan.close()
                if kernel:kernel._dispose()
            winner=min(result['variants'],key=lambda name:result['variants'][name]['median_gpu_seconds']);result['winner']=winner
            records.append(result);print(row['group'],winner,{n:round(v['median_gpu_seconds']*1e6,2) for n,v in result['variants'].items()},flush=True)
            (out/'report.json').write_text(json.dumps({'status':'checking','records':records},indent=2)+'\n')
        for kernel in kernels.values():kernel._dispose()
        for b in [*buffers.values(),*ins.values(),*outs.values()]:b.release()
        report={'status':'passed','adapter':device.info,'records':records,
                'model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                'protocol':'saved kernel replay; all F16 matrix traffic; three held-out inputs on every affected layer, NaN sentinels; 1-second warmup per finalist; rotate 1 discarded and 7 GPU samples'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','root','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.root,a.out)
