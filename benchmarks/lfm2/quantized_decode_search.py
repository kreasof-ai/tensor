"""Exact packed GEMV search over complete model weight traffic and fresh replay."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor_llm import GGUF
from tensor_llm.common.gguf import dequantize
from tensor_llm.lfm2.kernels.webgpu import source
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,bind,check
from benchmarks.lfm2.decode_fusion_search import PlanTimer


def candidates(kind,k,o,q):
    p=dict(r=1,k=k,o=o,type=q,sg=True)
    yield 'control',p
    if q==2:
        lanes=32 if min(k,o)>=2048 else 16
        for u in (2,4,8,16):
            for chains in (1,2,4,8):
                if chains>u or k%(lanes*8*u):continue
                yield f'u{u}-a{chains}',dict(p,gemv_unroll=u,gemv_chains=chains)
        for l,t in ((16,64),(16,128),(16,256),(32,64),(32,256),(32,512),(64,128),(64,256)):
            for u in (1,4):
                if k%(l*8*u):continue
                yield f'l{l}-t{t}-u{u}',dict(p,gemv_lanes=l,gemv_threads=t,gemv_dot=True,gemv_unroll=u)
    else:
        yield 'selected-control',dict(p,gemv_threads=64,gemv_unroll=8)
        for t in (64,128,256,512):
            for u,chains in ((1,1),(2,1),(2,2),(4,1),(4,2),(4,4),(8,1),(8,4)):
                if k%(256*u):continue
                yield f't{t}-u{u}-a{chains}',dict(p,gemv_threads=t,gemv_unroll=u,gemv_chains=chains)
        for t in (64,128,256):
            for u,chains in ((1,1),(2,1),(2,2),(4,1),(4,4),(8,1)):
                if k%(256*u):continue
                yield f'dot-t{t}-u{u}-a{chains}',dict(p,gemv_q6_dot=True,gemv_threads=t,gemv_unroll=u,gemv_chains=chains)


def reference(gguf,names,x):
    values=[];bounds=[]
    for name in names:
        info=gguf.tensors[name]
        w=dequantize(gguf.packed(name),info.type).reshape(info.shape).astype(np.float64)
        values.append(w@x.astype(np.float64))
        bounds.append(np.sum(np.abs(w*x),axis=1)*3e-6+1e-10)
    if len(values)==1:return values[0],bounds[0]
    g,u=values;gb,ub=bounds
    silu=g/(1+np.exp(-g))
    return silu*u,1.1*gb*np.abs(u)+np.abs(silu)*ub+1.1*gb*ub+1e-7


def run(model,out,encoding=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model);groups={}
    for info in gguf.tensors.values():
        if len(info.shape)!=2 or info.type not in (2,14) or info.name.endswith(('conv.weight','ffn_up.weight')):continue
        if encoding is not None and info.type!=encoding:continue
        o,k=info.shape;kind='ffn' if info.name.endswith('ffn_gate.weight') else 'linear'
        pair=[info.name]
        if kind=='ffn':pair.append(info.name.replace('ffn_gate','ffn_up'))
        groups.setdefault((kind,k,o,info.type),[]).append(pair)
    report=dict(status='searching',model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),groups=[],
        sources={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),text=Path(p).read_text()) for p in
            (__file__,'packages/tensor-llm/src/tensor_llm/lfm2/kernels/webgpu.py')},
        protocol='Native packed weights and F32 activations/output, floating dots, no activation quantization. Every matrix in each group streamed with distinct output; 150ms continuous warmup and 7 two-plan GPU timestamp samples. Control and best three independently replayed in rotating order after three warmups; winners checked on every affected weight with three held-out scales. Timestamp period 10ns.')
    started=time.perf_counter()
    def save():
        report['wall_seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    device=Device(max_buffer_size=268435456);device._adapter=TimestampAdapter(device._adapter)
    with device:
        report['adapter']=device.info
        for kind,k,o,q in sorted(groups,key=lambda g:(g[0]!='ffn',-g[2])):
            names=groups[(kind,k,o,q)];row=dict(kind=kind,k=k,o=o,type=q,names=names,records=[]);report['groups'].append(row)
            uploaded={name:device.from_numpy(gguf.packed(name).view(np.uint32)) for pair in names for name in pair}
            inp=device.zeros(k);outputs=[device.full(o,np.nan) for _ in names]
            x=(np.random.default_rng(29).normal(size=k)*.01).astype(np.float32);device.write(inp,x)
            expected,bound=reference(gguf,names[0],x)
            def prepare(kernel):
                calls=[]
                for pair,output in zip(names,outputs):
                    single=bind(device,kernel,(inp,*[uploaded[n] for n in pair],output));calls+=single.calls;single.close()
                return device.prepare_plan(calls)
            try:
                seen=set()
                for label,p in candidates(kind,k,o,q):
                    key=json.dumps(p,sort_keys=True)
                    if key in seen:continue
                    seen.add(key);record=dict(label=label,parameters=p);kernel=plan=timer=None
                    try:
                        path=out/f'{kind}-{k}-{o}-{q}-{len(row["records"])}.py';path.write_text(source(kind,p));artifact=path.with_suffix('.tbin');artifact.unlink(missing_ok=True)
                        tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact);plan=prepare(kernel)
                        quality=check(plan.launch(readback=outputs[0]),expected,bound)
                        timer=PlanTimer(device,plan);deadline=time.perf_counter()+.15
                        while time.perf_counter()<deadline:timer.sample(2)
                        samples=[timer.sample(2)/(2*len(names)) for _ in range(7)]
                        record.update(status='passed',validation=quality,samples_seconds=samples,median_gpu_seconds=statistics.median(samples),
                            artifact=artifact.name,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                    except Exception as error:record.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-700:]}')
                    finally:
                        if timer:timer.close()
                        if plan:plan.close()
                        if kernel:kernel._dispose()
                    row['records'].append(record);save()
                valid=[v for v in row['records'] if v['status']=='passed'];ranked=sorted(valid,key=lambda v:v['median_gpu_seconds'])
                finalists=list({v['artifact']:v for v in [next(v for v in valid if v['label']=='control')]+ranked[:3]}.values());runners=[]
                for record in finalists:
                    kernel=device.load(out/record['artifact']);plan=prepare(kernel);timer=PlanTimer(device,plan)
                    runners.append((record,kernel,plan,timer,[]))
                try:
                    for iteration in range(10):
                        for record,kernel,plan,timer,samples in runners[iteration%len(runners):]+runners[:iteration%len(runners)]:
                            value=timer.sample(3)/(3*len(names))
                            if iteration>=3:samples.append(value)
                    for record,kernel,plan,timer,samples in runners:
                        record.update(replay_samples_seconds=samples,replay_median_seconds=statistics.median(samples))
                    selected=min(finalists,key=lambda v:v['replay_median_seconds']);plan=next(v[2] for v in runners if v[0] is selected);validation=[]
                    for seed,scale in ((811,.01),(827,1),(843,1e-5)):
                        x=(np.random.default_rng(seed).normal(size=k)*scale).astype(np.float32);device.write(inp,x)
                        for output in outputs:device.write(output,np.full(o,np.nan,np.float32))
                        plan.launch()
                        for pair,output in zip(names,outputs):
                            expected,bound=reference(gguf,pair,x)
                            validation.append(dict(seed=seed,scale=scale,weights=pair,**check(output.to_numpy(),expected,bound)))
                    row['best']={**selected,'heldout':validation};row['control']=next(v for v in valid if v['label']=='control')
                    print('best',kind,k,o,q,selected['label'],round(selected['replay_median_seconds']*1e6,2),'control',round(row['control']['replay_median_seconds']*1e6,2),flush=True)
                finally:
                    for _,kernel,plan,timer,_ in runners:timer.close();plan.close();kernel._dispose()
            finally:
                inp.release()
                for b in (*outputs,*uploaded.values()):b.release()
            save()
    report['status']='finished';save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--encoding',type=int,choices=(2,14))
    args=parser.parse_args();run(args.model,args.out,args.encoding)
