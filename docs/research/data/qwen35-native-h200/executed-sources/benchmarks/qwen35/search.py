"""Correctness-gated Tensor beam search on the resident native decoder's buffers.

This measures individual projection pipelines. It does not qualify model
quality, prefill, or the 600 tok/s complete finite replay target.
"""
import ctypes as ct
import hashlib,json,time
from pathlib import Path
import numpy as np
from tensor.compiler.entry import export_source
from tensor.compiler.build import build_artifact
from tensor.compiler.search import ScheduleSearch
from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from tensor.runtime.signature import resolve_shape


def reference(bindings,grouped,routed):
    import torch
    torch.set_num_threads(2)
    x=torch.from_numpy(bindings['x'].to_numpy()).view(torch.float8_e4m3fn).float()
    ascales=torch.from_numpy(bindings['activation_scales'].to_numpy())
    x=(x.reshape(*x.shape[:-1],x.shape[-1]//128,128)*ascales[...,None]).reshape(x.shape)
    bits=bindings['w'].to_numpy()
    scales=torch.from_numpy(bindings['scales'].to_numpy()).view(torch.bfloat16).float() if bindings['scales'].dtype=='uint16' else torch.from_numpy(bindings['scales'].to_numpy())
    # Buffer.to_numpy decodes BF16 into float32; raw FP8 weights remain uint8.
    scales=scales.float()
    if not grouped:
        w=torch.from_numpy(bits).view(torch.float8_e4m3fn).float()
        w*=scales.repeat_interleave(128,0).repeat_interleave(128,1)
        return (x@w.T).numpy()
    ids=bindings['experts'].to_numpy();routes=bindings['routes'].to_numpy()
    r=x.shape[0];top=routes.shape[0]//r;o=bits.shape[1]
    out=np.full((r,top,o),np.nan,'float32')
    for group,expert in enumerate(ids):
        if expert<0:continue
        w=torch.from_numpy(bits[expert].copy()).view(torch.float8_e4m3fn).float()
        w*=scales[expert].repeat_interleave(128,0).repeat_interleave(128,1)
        for row in range(r):
            rank=routes[group,row]
            if rank>=0:out[row,rank]=(x[row,rank]@w.T if routed else x[row]@w.T).numpy()
    return out


def search(model,out,*,candidates=12,width=4):
    model._check();d=model.device;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((model.bundle/'inference.json').read_text())
    results=[];winners={}
    def save():
        p=out/'search.json.tmp';p.write_text(json.dumps(dict(schema='tensor.qwen35-projection-search.v1',
            protocol='native resident weights; independent decoded FP8 CPU reference; captured repeated pipeline GPU timing',
            model_throughput_qualified=False,full_stress_target_reached=False,records=results,winners=winners),indent=2)+'\n');p.replace(out/'search.json')
    for key,row in manifest['kernels'].items():
        kind,p=row['kind'],row['parameters']
        if kind not in ('fp8_linear','fp8_experts'):continue
        existing=model.kernels[key]
        call=next(call for kernel,call,_ in reversed(model.plan) if kernel is existing)
        bindings={a['name']:b for a,b in zip(existing.manifest['abi'],call.storage)}
        if 'activation_scales' not in bindings:raise ValueError('search requires a prequantized incumbent')
        expected=reference(bindings,kind=='fp8_experts',p.get('routed_input',False))
        spaces={'mma':dict(columns=(32,64,128),threads=(128,256),partitions=(1,2,4,8,16),stages=(1,2))}
        legal=lambda c:p['k']%(128*c['partitions'])==0 and p['o']%c['columns']==0
        explorer=ScheduleSearch([dict(family='mma',columns=64,threads=128,partitions=min(8,p['k']//128),stages=1)],spaces=spaces,width=width,legal=legal)
        for index in range(candidates):
            try:config=explorer.next()
            except StopIteration:break
            candidate=dict(operation=key,kind=kind,parameters=p,schedule=config,status='building')
            results.append(candidate);save();owned=[];kernels=[];graph=None
            try:
                cp={**p,**{k:v for k,v in config.items() if k!='family'}}
                tag=hashlib.sha256(json.dumps([key,cp],sort_keys=True).encode()).hexdigest()[:24]
                entry=out/(tag+'.py');artifact=entry.with_suffix('.tbin')
                text=export_source('tensor_llm.qwen35.kernels.matmul','make_kernel',kind+'_mma_prequantized',cp,
                    dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering'))
                if not artifact.is_file() or not entry.is_file() or entry.read_text()!=text:
                    artifact.unlink(missing_ok=True);entry.write_text(text)
                    build_artifact(entry,artifact,target=d.info['arch'],compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
                kernel=d.load(artifact);kernels.append(kernel)
                desc=next(a for a in kernel.manifest['arguments'] if a['name']=='out')
                partial=d.empty(resolve_shape(desc['shape'],{}),desc['dtype']);owned.append(partial)
                def bind(k,bb):
                    args=tuple(bb[a['name']] for a in k.manifest['arguments'])
                    values,symbols,launch=k._bind(args,{},include_outputs=True)
                    return BoundCall(d,k.manifest,values,symbols,launch,validated=True)
                pipeline=[(kernel,bind(kernel,{**bindings,'out':partial}))]
                actual=partial
                merge_record=None
                if config['partitions']>1:
                    mp=dict(r=p['r'],o=p['o'],partitions=config['partitions'])
                    if kind=='fp8_experts':mp['top']=p['top']
                    me=out/(tag+'-merge.py');ma=me.with_suffix('.tbin')
                    mt=export_source('tensor_llm.qwen35.kernels.matmul','merge_kernel',mp,dependencies=('tensor.compiler.entry',))
                    if not ma.is_file() or not me.is_file() or me.read_text()!=mt:
                        ma.unlink(missing_ok=True);me.write_text(mt);build_artifact(me,ma,target=d.info['arch'],compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
                    merge=d.load(ma);kernels.append(merge)
                    actual=d.empty(expected.shape);owned.append(actual)
                    pipeline.append((merge,bind(merge,dict(partial=partial,out=actual))))
                    merge_record=dict(path=ma.name,sha256=hashlib.sha256(ma.read_bytes()).hexdigest())
                for k,c in pipeline:d._launch(k,c)
                got=actual.to_numpy();rms=float(np.linalg.norm(got-expected)/np.linalg.norm(expected))
                candidate['relative_rms']=rms
                if not np.isfinite(got).all() or rms>.001:raise AssertionError(f'projection reference mismatch: {rms}')
                repeats=32
                def submit():
                    for _ in range(repeats):
                        for k,c in pipeline:d._launch(k,c)
                graph=CudaGraph(d,submit,resources=(*bindings.values(),*owned,*kernels))
                for _ in range(3):graph.launch()
                d.synchronize();times=[]
                events=[]
                try:
                    for _ in range(2):
                        event=ct.c_void_p();d.driver.call('cuEventCreate',ct.byref(event),0);events.append(event)
                    for _ in range(5):
                        d.driver.call('cuEventRecord',events[0],d.stream);graph.launch()
                        d.driver.call('cuEventRecord',events[1],d.stream);d.driver.call('cuEventSynchronize',events[1])
                        elapsed=ct.c_float();d.driver.call('cuEventElapsedTime',ct.byref(elapsed),*events)
                        times.append(elapsed.value/1000/repeats)
                finally:
                    for event in events:d.driver.call('cuEventDestroy_v2',event)
                seconds=float(np.median(times));candidate.update(status='passed',seconds=seconds,samples=times,artifact=dict(path=artifact.name,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),merge=merge_record)
                explorer.record(config,seconds)
                if key not in winners or seconds<winners[key]['seconds']:winners[key]=candidate.copy()
                print('Tensor projection search',kind,p,config,round(seconds*1e6,2),'us','RMS',rms,flush=True)
            except Exception as error:
                candidate.update(status='rejected',error=f'{type(error).__name__}: {error}')
                print('Tensor projection rejected',kind,config,candidate['error'],flush=True)
            finally:
                if graph:graph.close()
                for b in owned:b.release()
                for k in kernels:k.release()
                save()
    return winners
