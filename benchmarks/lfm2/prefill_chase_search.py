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
from tensor_llm.model import webgpu_parameters,projection_tile
from tensor_llm.webgpu_kernels import source
from tensor.compiler.webgpu_lowering import outer_product_matmul_schedule
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,bind,oracle,check
from benchmarks.lfm2.decode_fusion_search import PlanTimer


def candidates(kind,r,k,o,encoding=1,wide=False,cached=False,dots=False,packed_pairs=False,half_accum=False,half_layouts=False):
    p=dict(r=r,k=k,o=o,type=encoding,**projection_tile(r,o))
    if half_accum:
        yield 'control',webgpu_parameters(kind,p,'quant_searched')
        if half_layouts:
            current=webgpu_parameters(kind,p,'prefill_mixed')
            if not current.get('outer',{}).get('half_accum'):raise ValueError('requires selected half shape')
            yield 'half-current',current
            for owner in ('column','row'):
                for group in ('column','row'):
                    for padding in (0,1):
                        config={**current['outer'],'owner_axis':owner,'group_order':group,
                                'lhs_pad':padding,'rhs_pad':padding}
                        yield f'half-layout-{owner}-{group}-{padding}',{**current,'outer':config}
            return
        shapes=((16,32,2,4),(32,64,4,4),(64,64,4,4),(64,128,4,8))
        if wide:shapes += ((128,64,8,4),(64,64,8,4),(32,64,4,2),(64,64,8,2),(32,128,4,8),(128,128,8,8))
        for tm,tn,mm,mn in shapes:
            for tk in (16,32):
                for unroll in (2,4,8):
                    config=dict(tile_m=tm,tile_n=tn,tile_k=tk,micro_m=mm,micro_n=mn,
                        threads=tm*tn//(mm*mn),lhs_layout='km',unroll=unroll,explicit_unroll=True,
                        dot_width=2,packed_pairs=True,half_accum=True)
                    yield f'half-{tm}-{tn}-{tk}-{mm}-{mn}-{unroll}',{**p,'sg':True,'schedule':'outer','outer':config}
        return
    if packed_pairs:
        yield 'control',webgpu_parameters(kind,p,'quant_searched')
        for tm,tn,mm,mn in ((32,32,2,2),(32,64,4,4),(64,64,4,4),(64,128,4,8),(32,128,4,4),(16,32,2,4)):
            for tk in (16,32,64):
                if tk*(tm+(2 if kind=='ffn' else 1)*tn)*2>32768:continue
                config=dict(tile_m=tm,tile_n=tn,tile_k=tk,micro_m=mm,micro_n=mn,
                    threads=tm*tn//(mm*mn),lhs_layout='km',lhs_pad=0,rhs_pad=0,
                    owner_axis='column',unroll=4,explicit_unroll=True,fma=False,dot_width=2,packed_pairs=True)
                yield f'packed-{tm}-{tn}-{tk}-{mm}-{mn}',{**p,'sg':True,'schedule':'outer','outer':config}
        return
    if dots:
        current=webgpu_parameters(kind,p,'quant_searched')
        yield 'control',current
        for tm,tn,mm,mn in ((64,64,4,4),(64,128,4,8),(32,64,4,4),(32,32,2,2),(64,32,4,4)):
            for tk in (32,64):
                for width in (2,4):
                    config=dict(tile_m=tm,tile_n=tn,tile_k=tk,micro_m=mm,micro_n=mn,
                        threads=tm*tn//(mm*mn),lhs_layout='km',lhs_pad=0,rhs_pad=0,
                        owner_axis='column',unroll=4,explicit_unroll=True,fma=False,dot_width=width)
                    if tk*(tm+(2 if kind=='ffn' else 1)*tn)*2>32768:continue
                    outer_product_matmul_schedule(r,k,o,dtype='float16',**config)
                    yield f'dot{width}-{tm}-{tn}-{tk}-{mm}-{mn}',{**p,'sg':True,'schedule':'outer','outer':config}
        return
    if cached:
        current=webgpu_parameters(kind,p,'quant_searched')
        yield 'control',current
        if current.get('schedule')=='outer':
            for dtype in ('float16','float32'):
                yield 'cached-current-'+dtype,{**current,'type':1,'outer_shared_dtype':dtype}
        for tm,tn,mm,mn in ((64,64,4,4),(64,128,4,8),(32,64,4,4),(32,32,2,2),(64,32,4,4)):
            for tk in (32,64,128):
                for dtype in ('float16','float32'):
                    config=dict(tile_m=tm,tile_n=tn,tile_k=tk,micro_m=mm,micro_n=mn,
                        threads=tm*tn//(mm*mn),lhs_layout='km',lhs_pad=0,rhs_pad=0,
                        owner_axis='column',unroll=8,explicit_unroll=True,fma=True)
                    shared=tk*(tm+(2 if kind=='ffn' else 1)*tn)*(2 if dtype=='float16' else 4)
                    if shared>32768:continue
                    outer_product_matmul_schedule(r,k,o,dtype=dtype,**config)
                    yield f'cached-{tm}-{tn}-{tk}-{mm}-{mn}-{dtype}',{**p,'type':1,'sg':True,
                        'schedule':'outer','outer_shared_dtype':dtype,'outer':config}
        return
    if wide:
        current=webgpu_parameters(kind,p,'quant_searched')
        yield 'control',current
        if current.get('schedule')=='outer':yield 'current-f32-shared',{**current,'outer_shared_dtype':'float32'}
        for tm,tn,mm,mn in ((64,64,4,4),(64,128,4,8),(64,128,8,4),(128,64,8,4),
                            (64,64,8,4),(64,128,8,8),(128,128,8,8),(32,128,4,8)):
            for tk in (16,32):
                for layout,ap,bp in (('km',0,0),('km',1,1),('mk',0,1)):
                    config=dict(tile_m=tm,tile_n=tn,tile_k=tk,micro_m=mm,micro_n=mn,
                        threads=tm*tn//(mm*mn),lhs_layout=layout,lhs_pad=ap,rhs_pad=bp,
                        owner_axis='column',unroll=8,explicit_unroll=True,fma=True)
                    shared=tk*(tm+ap+(2 if kind=='ffn' else 1)*(tn+bp))*4
                    if shared>32768:continue
                    outer_product_matmul_schedule(r,k,o,dtype='float32',**config)
                    yield f'wide-{tm}-{tn}-{tk}-{mm}-{mn}-{layout}-{ap}',{**p,'sg':True,
                        'schedule':'outer','outer_shared_dtype':'float32','outer':config}
        return
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


def partial_half_bound(x,weights,p):
    """Float64 bound for short F16 FMA chains, including gradual underflow.

    This changes the kernel arithmetic contract, not the model acceptance gate.
    """
    x=x.astype(np.float16).astype(np.float64);count=p['outer']['unroll']
    gamma=(count*2**-11)/(1-count*2**-11)+3e-6
    values=[x@w.astype(np.float64).T for w in weights]
    bounds=[np.abs(x)@np.abs(w.astype(np.float64)).T*gamma+x.shape[1]*2**-25+1e-10 for w in weights]
    if len(weights)==1:return bounds[0]
    g,u=values;gb,ub=bounds;silu=g/(1+np.exp(-g))
    return np.abs(u)*1.1*gb+np.abs(silu)*ub+1.1*gb*ub+np.abs(silu*u)*2e-6+1e-10


def run(model,out,encoding=1,rows=(32,64,128),replay=None,wide=False,shapes=None,cached=False,dots=False,packed_pairs=False,half_accum=False,half_layouts=False):
    if half_layouts and not half_accum:raise ValueError('half layout search requires half accumulation')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model)
    groups={}
    for info in gguf.tensors.values():
        if len(info.shape)!=2 or info.type!=encoding or info.name.endswith('conv.weight') or info.name in ('token_embd.weight','output.weight'):continue
        if info.name.endswith('ffn_up.weight'):continue
        o,k=info.shape;kind='ffn' if info.name.endswith('ffn_gate.weight') else 'linear'
        names=[info.name]
        if kind=='ffn':names.append(info.name.replace('ffn_gate','ffn_up'))
        groups.setdefault((kind,k,o),[]).append(names)
    if shapes is not None:
        selected={(v.split(':')[0],int(v.split(':')[1]),int(v.split(':')[2])) for v in shapes}
        groups={key:value for key,value in groups.items() if key in selected}
        if not groups or set(groups)!=selected:raise ValueError('requested shape missing from checkpoint')
    order=sorted(groups,key=lambda g:(g[0]!='ffn',g[1]!=2560,-g[2]))
    prior=json.loads(Path(replay).read_text()) if replay is not None else None
    sources={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),text=Path(p).read_text()) for p in
        (__file__,'src/tensor/compiler/webgpu_lowering.py','packages/tensor-llm/src/tensor_llm/webgpu_kernels.py','packages/tensor-llm/src/tensor_llm/model.py')}
    report=dict(status='searching',model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),sources=sources,groups=[],encoding=encoding,wide=wide,cached=cached,dots=dots,packed_pairs=packed_pairs,half_accum=half_accum,half_layouts=half_layouts,
        protocol='Native GGUF weights (F16 or packed), F32 activation ABI and decoded weights rounded nearest-even to F16 operands, F32 accumulation/output, fused gate/up/SwiGLU. All matrices in the shape group streamed per sample with distinct outputs. 100ms warmup, 7 timestamp batches of two complete group plans, normalized per matrix. Every candidate checked against float64, winners checked on all weights and three held-out input scales. Vulkan timestamp period 10ns.')
    if half_accum:report['protocol']+=' Half candidates explicitly use short F16 FMA chains (outer.unroll operations per even/odd K chain), then F32 totals. Float64 kernel bounds include half rounding/underflow. Full-model acceptance keeps the original independent NumPy logits gate.'
    if prior is not None:
        if prior['status']!='finished' or prior['model_sha256']!=report['model_sha256'] or prior.get('encoding',1)!=encoding:
            raise ValueError('replay requires completed discovery on this checkpoint and encoding')
        report['discovery_report_sha256']=hashlib.sha256(Path(replay).read_bytes()).hexdigest()
    start=time.perf_counter()
    def save():
        report['wall_seconds']=time.perf_counter()-start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    device=Device(max_buffer_size=268435456 if encoding!=1 else None);device._adapter=TimestampAdapter(device._adapter)
    with device:
        report['adapter']=device.info
        weights={n:np.array(gguf.array(n,dtype=np.float16),copy=True) for names in groups.values() for pair in names for n in pair}
        uploaded={n:device.from_numpy(w.ravel() if encoding==1 else gguf.packed(n).view(np.uint32)) for n,w in weights.items()}
        half_cache={n:device.from_numpy(w.ravel()) for n,w in weights.items()} if cached else {}
        report['cached_weight_bytes']=sum(v.nbytes for v in half_cache.values())
        for r in rows:
            for kind,k,o in order:
                names=groups[(kind,k,o)]
                row=dict(kind=kind,rows=r,k=k,o=o,names=names,records=[]);report['groups'].append(row)
                x=(np.random.default_rng(29).normal(size=(r,k))*.01).astype(np.float32)
                expected,bound=reference(x,[weights[n] for n in names[0]])
                inp=device.from_numpy(x.ravel());outputs=[device.full(r*o,np.nan) for _ in names]
                seen=set()
                try:
                    options=candidates(kind,r,k,o,encoding,wide,cached,dots,packed_pairs,half_accum,half_layouts)
                    if prior is not None:
                        group=next(v for v in prior['groups'] if (v['kind'],v['rows'],v['k'],v['o'])==(kind,r,k,o))
                        ranked=sorted((v for v in group['records'] if v['status']=='passed'),key=lambda v:v['median_gpu_seconds'])[:3]
                        options=[next(candidates(kind,r,k,o,encoding,wide,cached,dots,packed_pairs,half_accum))]+[(v['label'],v['parameters']) for v in ranked if v['label']!='control']
                        configured=webgpu_parameters(kind,dict(r=r,k=k,o=o,type=encoding,**projection_tile(r,o)),'prefill_mixed')
                        if half_accum and configured.get('outer',{}).get('half_accum'):options.append(('configured-short',configured))
                        if cached and half_accum:
                            options += [('cached-'+label,dict(p,type=1)) for label,p in list(options) if p.get('outer',{}).get('half_accum')]
                    for label,p in options:
                        identity=json.dumps(p,sort_keys=True)
                        if identity in seen:continue
                        seen.add(identity);i=len(row['records']);record=dict(label=label,parameters=p)
                        kernel=plan=timer=None
                        try:
                            path=out/f'{kind}-{r}-{k}-{o}-{i}.py';path.write_text(source(kind,p));artifact=path.with_suffix('.tbin');artifact.unlink(missing_ok=True)
                            tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact)
                            calls=[]
                            storage=half_cache if cached and p['type']==1 else uploaded
                            for pair,output in zip(names,outputs):
                                single=bind(device,kernel,(inp,*[storage[n] for n in pair],output));calls+=single.calls;single.close()
                            plan=device.prepare_plan(calls)
                            device.write(outputs[0],np.full(r*o,np.nan,np.float32))
                            actual=plan.launch(readback=outputs[0]).reshape(r,o)
                            candidate_bound=partial_half_bound(x,[weights[n] for n in names[0]],p) if p.get('outer',{}).get('half_accum') else bound
                            quality=check(actual,expected,candidate_bound)
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
                    if half_layouts:
                        finalists += [v for v in valid if v['label'] in ('half-current','configured-short')]
                    finalists=list({v['artifact']:v for v in finalists}.values());runners=[]
                    for record in finalists:
                        kernel=device.load(out/record['artifact']);calls=[]
                        storage=half_cache if cached and record['parameters']['type']==1 else uploaded
                        for pair,output in zip(names,outputs):
                            single=bind(device,kernel,(inp,*[storage[n] for n in pair],output));calls+=single.calls;single.close()
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
                                if selected['parameters'].get('outer',{}).get('half_accum'):bound=partial_half_bound(x,[weights[n] for n in pair],selected['parameters'])
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
        for buffer in half_cache.values():buffer.release()
    report['status']='finished';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--encoding',type=int,choices=(1,2,14),default=1)
    p.add_argument('--rows',type=int,nargs='+',choices=(32,64,128),default=[32,64,128])
    p.add_argument('--replay',type=Path,help='freshly replay the control and three discovery finalists')
    p.add_argument('--wide',action='store_true',help='larger register tiles and F32 shared storage against current quantized profile')
    p.add_argument('--shapes',nargs='+',help='restrict streamed groups, e.g. ffn:2048:10752 linear:10752:2048')
    p.add_argument('--cached',action='store_true',help='compare predecoded F16 weight caches against native packed control')
    p.add_argument('--dots',action='store_true',help='floating vec2/vec4 dot products with F32 accumulation')
    p.add_argument('--packed-pairs',action='store_true')
    p.add_argument('--half-accum',action='store_true')
    p.add_argument('--half-layouts',action='store_true',help='test ownership, shared padding and group order for selected short-half tiles')
    a=p.parse_args();run(a.model,a.out,a.encoding,tuple(a.rows),a.replay,a.wide,a.shapes,a.cached,a.dots,a.packed_pairs,a.half_accum,a.half_layouts)
