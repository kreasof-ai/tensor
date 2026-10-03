"""Oracle-gated packed integer prefill search including two-component preparation."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor_llm import GGUF
from tensor_llm.gguf import prepack_q4_0
from tensor_llm.model import webgpu_parameters,projection_tile
from tensor_llm.webgpu_kernels import source
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,bind,check
from benchmarks.lfm2.decode_fusion_search import PlanTimer


def configs(kind,wide_k=False):
    if wide_k:
        for tm,tn,mm,mn in ((8,32,1,2),(16,32,2,2),(16,64,2,4),(32,32,2,2),(32,64,2,4),(32,64,4,4),(64,64,4,4),(64,128,4,8),
                           (8,128,2,4),(16,128,2,8),(16,64,4,4),(32,128,4,8)):
            for tk in (32,64,128,256):
                for axis in ('column','row'):
                    parts=2 if kind=='ffn' else 1
                    if (tm*80+parts*tn*36)*(tk//32)>32768:continue
                    yield dict(tile_m=tm,tile_n=tn,micro_m=mm,micro_n=mn,threads=tm*tn//(mm*mn),
                        owner_axis=axis,group_order='row',tile_k=tk)
        return
    for tm,tn,mm,mn in ((16,32,2,2),(32,32,2,2),(32,64,4,4),(32,64,2,4),
                       (64,64,4,4),(64,128,4,8),(32,128,4,4),(64,64,8,4)):
        for axis in ('column','row'):
            yield dict(tile_m=tm,tile_n=tn,micro_m=mm,micro_n=mn,
                threads=tm*tn//(mm*mn),owner_axis=axis,group_order='row')


def reference(gguf,names,x,half_weights=False):
    values=[];bounds=[]
    for name in names:
        w=gguf.array(name,dtype=np.float16 if half_weights else np.float32).astype(np.float64)
        values.append(x@w.T);bounds.append(np.abs(x)@np.abs(w).T*3e-6+1e-10)
    if len(values)==1:return values[0],bounds[0]
    g,u=values;gb,ub=bounds;silu=g/(1+np.exp(-g))
    return silu*u,np.abs(u)*1.1*gb+np.abs(silu)*ub+1.1*gb*ub+np.abs(silu*u)*2e-6+1e-10


def run(model,out,rows=(128,),quantize_subgroup=False,wide_k=False,shapes=None,split_ffn=False,fixed_residual=False,prepacked=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model);groups={}
    for info in gguf.tensors.values():
        if len(info.shape)!=2 or info.type!=2 or info.name.endswith(('conv.weight','ffn_up.weight')) or info.name in ('token_embd.weight','output.weight'):continue
        o,k=info.shape;kind='ffn' if info.name.endswith('ffn_gate.weight') else 'linear'
        names=[info.name]
        if kind=='ffn':names += [info.name.replace('ffn_gate','ffn_up')]
        groups.setdefault((kind,k,o),[]).append(names)
    if shapes:
        selected={(v.split(':')[0],int(v.split(':')[1]),int(v.split(':')[2])) for v in shapes}
        groups={key:value for key,value in groups.items() if key in selected}
        if not groups or set(groups)!=selected:raise ValueError('missing requested shape')
    report=dict(status='searching',model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),groups=[],quantize_subgroup=quantize_subgroup,wide_k=wide_k,split_ffn=split_ffn,fixed_residual=fixed_residual,prepacked=prepacked,
        sources={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),text=Path(p).read_text()) for p in
            (__file__,'src/tensor/compiler/webgpu_lowering.py','packages/tensor-llm/src/tensor_llm/webgpu_kernels.py','packages/tensor-llm/src/tensor_llm/gguf.py')},
        protocol='Native packed Q4_0 weights. Current control uses nearest-even F16 operands/F32 accumulation. Integer candidates use two Q8 activation planes after nearest-even half rounding and exact decoded F32 weights, F32 output accumulation. Both checked against their independent float64 arithmetic oracle; activation reconstruction separately bounded. All matrices streamed with distinct outputs, quantizer dispatched before EVERY integer projection, included in timing. 100ms warmup, 7 two-plan samples; control/best3 fresh rotating replay after 3 warmups; every selected weight checked at 3 held-out scales. Timestamp 10ns.')
    if prepacked:report['protocol']+=' Integer candidates use an aligned signed-byte derived cache (36 bytes per Q4_0 block), doubling matrix storage while preserving values; control retains native 18-byte blocks.'
    if fixed_residual:report['protocol']+=' Low-component scale is high scale /254, allowing an integer combination before conversion to F32.'
    started=time.perf_counter()
    def save():
        report['wall_seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    device=Device(max_buffer_size=268435456);device._adapter=TimestampAdapter(device._adapter)
    with device:
        report['adapter']=device.info
        for r in rows:
            for kind,k,o in sorted(groups,key=lambda g:(g[0]!='ffn',-g[2])):
                names=groups[(kind,k,o)];row=dict(kind=kind,rows=r,k=k,o=o,names=names,records=[]);report['groups'].append(row)
                uploaded={name:device.from_numpy(gguf.packed(name).view(np.uint32)) for pair in names for name in pair}
                cache={name:device.from_numpy(prepack_q4_0(gguf.packed(name))) for pair in names for name in pair} if prepacked else {}
                row['derived_cache_bytes']=sum(v.nbytes for v in cache.values())
                inp=device.zeros(r*k);packed=device.zeros(r*k//2,dtype='uint32');scales=device.zeros(r*k//16);sums=device.zeros(r*k//16,dtype='int32')
                outputs=[device.full(r*o,np.nan) for _ in names]
                qp=out/f'quant-{r}-{k}.py';qa=qp.with_suffix('.tbin');text=source('quantize_q16',dict(r=r,k=k,sg=quantize_subgroup,fixed_residual=fixed_residual))
                if not qp.exists() or qp.read_text()!=text:qa.unlink(missing_ok=True)
                qp.write_text(text)
                if not qa.exists():tensor.build(qp,qa,provider='webgpu',cache_dir=out/'cache')
                quant=device.load(qa)
                qargs=(inp,packed,scales,sums);qplan=bind(device,quant,qargs)
                temporaries=[];swiglu=None
                if split_ffn and kind=='ffn':
                    temporaries=[(device.zeros(r*o),device.zeros(r*o)) for _ in names]
                    sp=out/f'swiglu-{r}-{o}.py';sa=sp.with_suffix('.tbin');sp.write_text(source('swiglu',dict(r=r,c=o)))
                    if not sa.exists():tensor.build(sp,sa,provider='webgpu',cache_dir=out/'cache')
                    swiglu=device.load(sa)
                def prepare(kernel,integer,split=False):
                    calls=[]
                    storage=cache if integer and prepacked else uploaded
                    for index,(pair,output) in enumerate(zip(names,outputs)):
                        if split:
                            for name,temp in zip(pair,temporaries[index]):
                                calls += qplan.calls
                                single=bind(device,kernel,(packed,scales,sums,storage[name],temp));calls += single.calls;single.close()
                            single=bind(device,swiglu,(*temporaries[index],output));calls += single.calls;single.close()
                            continue
                        if integer:calls += qplan.calls
                        inputs=(packed,scales,sums) if integer else (inp,)
                        single=bind(device,kernel,(*inputs,*[storage[n] for n in pair],output));calls += single.calls;single.close()
                    return device.prepare_plan(calls)
                def activation(x,integer):
                    device.write(inp,x.ravel())
                    rounded=x.astype(np.float16).astype(np.float64)
                    if not integer:return rounded
                    qplan.launch()
                    q=packed.to_numpy().view(np.int8).reshape(2,r,k//32,32)
                    factors=scales.to_numpy().reshape(2,r,k//32)
                    np.testing.assert_array_equal(sums.to_numpy().reshape(2,r,k//32),q.astype(np.int32).sum(axis=-1))
                    actual=np.sum(q.astype(np.float64)*factors[:,:,:,None],axis=0).reshape(r,k)
                    maxima=np.max(np.abs(rounded.reshape(r,k//32,32)),axis=-1)
                    bound=np.repeat(maxima/127**2*.51+1e-12,32,axis=1)
                    if not np.all(np.abs(actual-rounded)<=bound):raise AssertionError('two-component activation bound failed')
                    return actual
                x=(np.random.default_rng(29).normal(size=(r,k))*.01).astype(np.float32)
                control=webgpu_parameters(kind,dict(r=r,k=k,o=o,type=2,**projection_tile(r,o)),'quant_searched')
                options=[('control',control)]+[(f'integer-{i}',dict(r=r,k=k,o=o,type=2,q4_prepacked=prepacked,integer=dict(c,fixed_residual=fixed_residual),split_ffn=split_ffn and kind=='ffn')) for i,c in enumerate(configs('linear' if split_ffn and kind=='ffn' else kind,wide_k))]
                if not (prepacked or fixed_residual or split_ffn):
                    current=webgpu_parameters(kind,dict(r=r,k=k,o=o,type=2),'prefill_q16')
                    if current.get('q16'):options.insert(1,('integer-current',current))
                expected={flag:reference(gguf,names[0],activation(x,flag),half_weights=not flag) for flag in (False,True)}
                qt=PlanTimer(device,qplan)
                for _ in range(3):qt.sample(2)
                row['quantizer_seconds']=statistics.median(qt.sample(2)/2 for _ in range(7));qt.close()
                try:
                    for label,p in options:
                        integer=label!='control';record=dict(label=label,parameters=p,integer=integer);kernel=plan=timer=None
                        try:
                            path=out/f'{kind}-{r}-{k}-{o}-{label}.py';path.write_text(source('linear_q16' if p.get('split_ffn') else kind+'_q16' if integer else kind,p));artifact=path.with_suffix('.tbin')
                            artifact.unlink(missing_ok=True)
                            tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact);plan=prepare(kernel,integer,p.get('split_ffn',False))
                            device.write(outputs[0],np.full(r*o,np.nan,np.float32))
                            quality=check(plan.launch(readback=outputs[0]).reshape(r,o),*expected[integer])
                            timer=PlanTimer(device,plan);deadline=time.perf_counter()+.1
                            while time.perf_counter()<deadline:timer.sample(2)
                            samples=[timer.sample(2)/(2*len(names)) for _ in range(7)]
                            record.update(status='passed',validation=quality,artifact=artifact.name,
                                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),samples_seconds=samples,median_gpu_seconds=statistics.median(samples))
                        except Exception as error:record.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-700:]}')
                        finally:
                            if timer:timer.close()
                            if plan:plan.close()
                            if kernel:kernel._dispose()
                        row['records'].append(record);save()
                    valid=[v for v in row['records'] if v['status']=='passed'];ranked=sorted(valid,key=lambda v:v['median_gpu_seconds'])
                    controls=[v for v in valid if v['label'] in ('control','integer-current')]
                    finalists=list({v['artifact']:v for v in controls+ranked[:3]}.values());runners=[]
                    for record in finalists:
                        kernel=device.load(out/record['artifact']);plan=prepare(kernel,record['integer'],record['parameters'].get('split_ffn',False));timer=PlanTimer(device,plan)
                        runners.append((record,kernel,plan,timer,[]))
                    try:
                        for iteration in range(10):
                            for record,kernel,plan,timer,samples in runners[iteration%len(runners):]+runners[:iteration%len(runners)]:
                                value=timer.sample(3)/(3*len(names))
                                if iteration>=3:samples.append(value)
                        for record,kernel,plan,timer,samples in runners:record.update(replay_samples_seconds=samples,replay_median_seconds=statistics.median(samples))
                        selected=min(finalists,key=lambda v:v['replay_median_seconds']);plan=next(v[2] for v in runners if v[0] is selected);heldout=[]
                        for seed,magnitude in ((811,.01),(827,1.),(843,1e-5)):
                            x=(np.random.default_rng(seed).normal(size=(r,k))*magnitude).astype(np.float32);effective=activation(x,selected['integer'])
                            for output in outputs:device.write(output,np.full(r*o,np.nan,np.float32))
                            plan.launch()
                            for pair,output in zip(names,outputs):heldout.append(dict(seed=seed,scale=magnitude,weights=pair,
                                **check(output.to_numpy().reshape(r,o),*reference(gguf,pair,effective,half_weights=not selected['integer']))))
                        row['best']={**selected,'heldout':heldout};row['control']=next(v for v in valid if v['label']=='control')
                        print('best',kind,r,k,o,selected['label'],round(selected['replay_median_seconds']*1e6,2),'control',round(row['control']['replay_median_seconds']*1e6,2),flush=True)
                    finally:
                        for _,kernel,plan,timer,_ in runners:timer.close();plan.close();kernel._dispose()
                finally:
                    qplan.close();quant._dispose()
                    if swiglu:swiglu._dispose()
                    for pair in temporaries:
                        for buffer in pair:buffer.release()
                    for buffer in (*uploaded.values(),*cache.values(),inp,packed,scales,sums,*outputs):buffer.release()
                save()
    report['status']='finished';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',type=int,nargs='+',choices=(32,64,128),default=[128])
    p.add_argument('--quantize-subgroup',action='store_true')
    p.add_argument('--wide-k',action='store_true')
    p.add_argument('--shapes',nargs='+')
    p.add_argument('--split-ffn',action='store_true')
    p.add_argument('--fixed-residual',action='store_true')
    p.add_argument('--prepacked',action='store_true')
    a=p.parse_args();run(a.model,a.out,tuple(a.rows),a.quantize_subgroup,a.wide_k,a.shapes,a.split_ffn,a.fixed_residual,a.prepacked)
