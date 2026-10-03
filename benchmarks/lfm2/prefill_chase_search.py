"""F16 prefill schedule replay with fused FFNs and complete group weight traffic."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor_llm import GGUF
from tensor_llm.model import webgpu_parameters
from tensor_llm.webgpu_kernels import source
from tensor.compiler.webgpu_lowering import outer_product_matmul_schedule
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,bind,oracle,check
from benchmarks.lfm2.decode_fusion_search import PlanTimer


def candidates(kind,r,k,o):
    p=dict(r=r,k=k,o=o,type=1)
    baseline=webgpu_parameters(kind,p,'prefill_chunked')
    yield 'control',baseline
    yield 'unrolled',webgpu_parameters(kind,p,'prefill_unrolled')
    if baseline.get('schedule')=='partitioned':
        for u in (8,16):
            yield 'partitioned-u'+str(u),{**baseline,'unroll':u,'explicit_unroll':True}
    for tm,tn,mm,mn in ((16,32,2,4),(32,32,2,2),(32,64,4,4),(32,64,2,4),
                         (32,128,4,4),(64,64,4,4),(64,128,4,8)):
        for tk,layout,fma in ((16,'km',True),(32,'mk',False)):
            config=dict(tile_m=tm,tile_n=tn,tile_k=tk,micro_m=mm,micro_n=mn,
                threads=tm*tn//(mm*mn),lhs_layout=layout,lhs_pad=0,rhs_pad=0,
                owner_axis='column',unroll=8,explicit_unroll=True,fma=fma)
            if kind=='ffn' and mm*mn>32:continue
            outer_product_matmul_schedule(r,k,o,dtype='float16',**config)
            yield f'outer-{tm}-{tn}-{tk}-{mm}-{mn}',{**p,'sg':True,'schedule':'outer','outer':config}


def reference(x,weights):
    g,gb=oracle(x,weights[0])
    if len(weights)==1:return g,gb
    u,ub=oracle(x,weights[1]);silu=g/(1+np.exp(-g));expected=silu*u
    bound=np.abs(u)*1.1*gb+np.abs(silu)*ub+1.1*gb*ub+np.abs(expected)*2e-6+1e-10
    return expected,bound


def run(model,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model)
    groups={}
    for info in gguf.tensors.values():
        if len(info.shape)!=2 or info.type!=1 or info.name.endswith('conv.weight') or info.name in ('token_embd.weight','output.weight'):continue
        if info.name.endswith('ffn_up.weight'):continue
        o,k=info.shape;kind='ffn' if info.name.endswith('ffn_gate.weight') else 'linear'
        names=[info.name]
        if kind=='ffn':names.append(info.name.replace('ffn_gate','ffn_up'))
        groups.setdefault((kind,k,o),[]).append(names)
    order=sorted(groups,key=lambda g:(g[0]!='ffn',g[1]!=2560,-g[2]))
    sources={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),text=Path(p).read_text()) for p in
        (__file__,'src/tensor/compiler/webgpu_lowering.py','packages/tensor-llm/src/tensor_llm/webgpu_kernels.py','packages/tensor-llm/src/tensor_llm/model.py')}
    report=dict(status='searching',model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),sources=sources,groups=[],
        protocol='Native F16 weights, F32 activation ABI rounded nearest-even to F16, F32 accumulation/output, fused gate/up/SwiGLU. All matrices in the shape group streamed per sample with distinct outputs. 100ms warmup, 7 timestamp batches of two complete group plans, normalized per matrix. Every candidate checked against float64, winners checked on all weights and three held-out input scales. Vulkan timestamp period 10ns.')
    start=time.perf_counter()
    def save():
        report['wall_seconds']=time.perf_counter()-start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    with device:
        report['adapter']=device.info
        weights={n:np.array(gguf.array(n,dtype=np.float16),copy=True) for names in groups.values() for pair in names for n in pair}
        uploaded={n:device.from_numpy(w.ravel()) for n,w in weights.items()}
        for r in (32,64,128):
            for kind,k,o in order:
                names=groups[(kind,k,o)]
                row=dict(kind=kind,rows=r,k=k,o=o,names=names,records=[]);report['groups'].append(row)
                x=(np.random.default_rng(29).normal(size=(r,k))*.01).astype(np.float32)
                expected,bound=reference(x,[weights[n] for n in names[0]])
                inp=device.from_numpy(x.ravel());outputs=[device.full(r*o,np.nan) for _ in names]
                seen=set()
                try:
                    for label,p in candidates(kind,r,k,o):
                        identity=json.dumps(p,sort_keys=True)
                        if identity in seen:continue
                        seen.add(identity);i=len(row['records']);record=dict(label=label,parameters=p)
                        kernel=plan=timer=None
                        try:
                            path=out/f'{kind}-{r}-{k}-{o}-{i}.py';path.write_text(source(kind,p));artifact=path.with_suffix('.tbin');artifact.unlink(missing_ok=True)
                            tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact)
                            calls=[]
                            for pair,output in zip(names,outputs):
                                single=bind(device,kernel,(inp,*[uploaded[n] for n in pair],output));calls+=single.calls;single.close()
                            plan=device.prepare_plan(calls)
                            device.write(outputs[0],np.full(r*o,np.nan,np.float32))
                            actual=plan.launch(readback=outputs[0]).reshape(r,o)
                            quality=check(actual,expected,bound)
                            timer=PlanTimer(device,plan);warm_end=time.perf_counter()+.1
                            while time.perf_counter()<warm_end:timer.sample(2)
                            samples=[timer.sample(2)/(2*len(names)) for _ in range(7)]
                            record.update(status='passed',validation=quality,median_gpu_seconds=statistics.median(samples),samples_seconds=samples,
                                artifact=artifact.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        except Exception as error:record.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-700:]}')
                        finally:
                            if timer:timer.close()
                            if plan:plan.close()
                            if kernel:kernel._dispose()
                        row['records'].append(record);save()
                    valid=[v for v in row['records'] if v['status']=='passed']
                    if not valid:raise AssertionError('no valid schedule')
                    ranked=sorted(valid,key=lambda v:v['median_gpu_seconds'])
                    # Replay the control and best three in rotating order.
                    finalists=[next(v for v in valid if v['label']=='control')]+ranked[:3]
                    finalists=list({v['artifact']:v for v in finalists}.values());runners=[]
                    for record in finalists:
                        kernel=device.load(out/record['artifact']);calls=[]
                        for pair,output in zip(names,outputs):
                            single=bind(device,kernel,(inp,*[uploaded[n] for n in pair],output));calls+=single.calls;single.close()
                        plan=device.prepare_plan(calls);timer=PlanTimer(device,plan)
                        runners.append((record,kernel,plan,timer,[]))
                    try:
                        for iteration in range(10):
                            for record,kernel,plan,timer,samples in runners[iteration%len(runners):]+runners[:iteration%len(runners)]:
                                value=timer.sample(3)/(3*len(names))
                                if iteration>=3:samples.append(value)
                        for record,kernel,plan,timer,samples in runners:
                            record['replay_samples_seconds']=samples;record['replay_median_seconds']=statistics.median(samples)
                        selected=min(finalists,key=lambda v:v['replay_median_seconds'])
                        runner=next(v for v in runners if v[0] is selected);plan=runner[2];validation=[]
                        for seed,scale in ((811,.01),(827,1),(843,1e-5)):
                            x=(np.random.default_rng(seed).normal(size=(r,k))*scale).astype(np.float32);device.write(inp,x.ravel())
                            for output in outputs:device.write(output,np.full(r*o,np.nan,np.float32))
                            plan.launch()
                            for pair,output in zip(names,outputs):
                                expected,bound=reference(x,[weights[n] for n in pair])
                                validation.append(dict(seed=seed,scale=scale,weights=pair,**check(output.to_numpy().reshape(r,o),expected,bound)))
                        row['best']={**selected,'heldout':validation}
                        row['control']=next(v for v in valid if v['label']=='control')
                        print('best',kind,r,k,o,selected['label'],round(selected['replay_median_seconds']*1e6,2),
                              'control',round(row['control']['replay_median_seconds']*1e6,2),flush=True)
                    finally:
                        for _,kernel,plan,timer,_ in runners:timer.close();plan.close();kernel._dispose()
                finally:
                    inp.release()
                    for output in outputs:output.release()
                save()
        for buffer in uploaded.values():buffer.release()
    report['status']='finished';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.out)
