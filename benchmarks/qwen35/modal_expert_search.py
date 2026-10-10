"""Bounded Hopper expert schedule search with exact arithmetic checks."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume
app=modal.App('tensor-qwen35-h200-expert-search')


@app.function(image=image,cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_candidates():
    import hashlib,json,os
    from pathlib import Path
    from tensor.compiler.entry import export_source
    from benchmarks.qwen35.build import build_artifact
    os.chdir('/workspace');volume.reload()
    source=Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_experts.py')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    root=Path('/cache/expert-search')/digest[:16];root.mkdir(parents=True,exist_ok=True)
    schedules=[dict(block_m=m,columns=n,threads=t,stages=s) for m,n,t,s in
               [(64,64,256,1),(64,128,256,1)]]
    schedules=[dict(s,packed_gather=True,bf16_mma=bf16) for s in schedules for bf16 in (False,True)]
    result=dict(source_sha256=digest,root=str(root),cases=[])
    def compile(name,module,factory,p,target):
        entry=root/(name+'.py');entry.write_text(export_source(module,factory,p,
                                                   dependencies=('tensor.compiler.entry',)))
        artifact=entry.with_suffix('.tbin');artifact.unlink(missing_ok=True)
        build_artifact(entry,artifact,target=target)
        return str(artifact)
    for routed,k,o in [(False,2048,512),(True,512,2048)]:
        case=dict(routed=routed,k=k,o=o,candidates=[])
        p=dict(rows=4096,k=k,o=o,routed_input=routed,block_m=64,threads=256,compact=True)
        case['control']=compile(f'control-{k}','tensor_llm.qwen35.kernels.prefill','expert_kernel',p,'sm_90')
        for index,schedule in enumerate(schedules):
            p.update(schedule)
            artifact=compile(f'candidate-{k}-{index}','tensor_llm.qwen35.kernels.hopper_experts',
                             'expert_kernel',p,'sm_90a')
            case['candidates'].append(dict(schedule=schedule,path=artifact))
            print('Prepared expert candidate',k,schedule,flush=True)
        result['cases'].append(case)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def search(prepared,normal_inputs=False):
    import ctypes as ct
    import json,os,time
    from pathlib import Path
    import numpy as np
    import tensor
    from tensor.providers.cuda_graph import CudaGraph
    from tensor.runtime.abi import BoundCall
    os.chdir('/workspace');volume.reload()
    rng=np.random.default_rng(623);rows=4096;top=8
    result=dict(prepared=prepared,cases=[],normal_inputs=normal_inputs,
                timing='CUDA graph replay; isolated projection, not model throughput')
    with tensor.Device() as device:
        result['device']=device.info
        def measured(kernel,args):
            storage,symbols,launch=kernel._bind(tuple(args),{},include_outputs=True)
            call=BoundCall(device,kernel.manifest,storage,symbols,launch,validated=True)
            with CudaGraph(device,lambda:[device._launch(kernel,call) for _ in range(8)],resources=(kernel,*args)) as graph:
                graph.launch();device.synchronize();samples=[]
                a,b=ct.c_void_p(),ct.c_void_p()
                device.driver.call('cuEventCreate',ct.byref(a),0);device.driver.call('cuEventCreate',ct.byref(b),0)
                try:
                    for _ in range(3):
                        device.driver.call('cuEventRecord',a,device.stream);graph.launch()
                        device.driver.call('cuEventRecord',b,device.stream);device.synchronize()
                        value=ct.c_float();device.driver.call('cuEventElapsedTime',ct.byref(value),a,b)
                        samples.append(value.value/8)
                finally:
                    device.driver.call('cuEventDestroy_v2',a);device.driver.call('cuEventDestroy_v2',b)
                return samples
        for case in prepared['cases']:
            k,o,routed=case['k'],case['o'],case['routed']
            shape=(rows,top,k) if routed else (rows,k)
            def fp8(shape):
                if not normal_inputs:return rng.integers(0,127,shape,dtype='uint8')
                import torch
                torch.set_num_threads(4)
                return torch.from_numpy(rng.standard_normal(shape,dtype='float32')).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            inputs=[device.from_numpy(v,dtype=dtype) for v,dtype in [
                (fp8(shape),'uint8'),
                (rng.uniform(.001,.01,(*shape[:-1],k//128)).astype('float32'),'float32'),
                (fp8((256,o,k)),'uint8'),
                (rng.uniform(.001,.01,(256,o//128,k//128)).astype('float32'),'bfloat16')]]
            counts=device.empty((256,),'int32');routes=device.empty((256,rows),'int32')
            out=device.empty((rows,top,o),'float32');control=device.load(case['control'])
            resources=[*inputs,counts,routes,out,control]
            try:
                for hot in (False,True):
                    ids=np.tile(np.arange(top),(rows,1)) if hot else (
                        np.arange(rows)[:,None]*top+np.arange(top)[None,:])%256
                    route=np.full((256,rows),-1,'int32');count=np.zeros(256,'int32')
                    for expert in range(256):
                        row,rank=np.nonzero(ids==expert);selected=row*top+rank
                        rng.shuffle(selected);count[expert]=len(selected);route[expert,:len(selected)]=selected
                    device.driver.call('cuMemcpyHtoD_v2',counts.pointer,ct.c_void_p(count.ctypes.data),count.nbytes)
                    device.driver.call('cuMemcpyHtoD_v2',routes.pointer,ct.c_void_p(route.ctypes.data),route.nbytes)
                    def mapping(m):
                        e=np.full((rows*top+m-1)//m+255,-1,'int32');t=np.zeros_like(e);pos=0
                        for expert,n in enumerate(count):
                            for tile in range((int(n)+m-1)//m):e[pos]=expert;t[pos]=tile;pos+=1
                        return device.from_numpy(e),device.from_numpy(t)
                    maps=mapping(64)
                    control.launch(*inputs,counts,routes,out,*maps);expected=out.to_numpy()
                    baseline=measured(control,[*inputs,counts,routes,out,*maps])
                    for b in maps:b.release()
                    record=dict(k=k,o=o,routed=routed,routing='hot8' if hot else 'balanced256',
                                control_milliseconds=baseline,candidates=[])
                    for candidate in case['candidates']:
                        kernel=device.load(candidate['path']);maps=mapping(candidate['schedule']['block_m'])
                        try:
                            args=[*inputs,counts,routes,out,*maps];kernel.launch(*args);actual=out.to_numpy()
                            equal=bool(np.array_equal(actual,expected))
                            row=dict(candidate,bitwise_equal=equal,finite=bool(np.isfinite(actual).all()),
                                     maximum_absolute_error=float(np.max(np.abs(actual-expected))))
                            row['relative_rms']=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                            # Timing is diagnostic even when numerical equivalence
                            # fails. No candidate is selected from timing alone.
                            row['milliseconds']=measured(kernel,args) if row['finite'] else None
                            record['candidates'].append(row);print('Expert result',record['routing'],k,row,flush=True)
                        finally:
                            kernel.release()
                            for b in maps:b.release()
                    result['cases'].append(record)
                    del expected
            finally:
                for resource in resources:resource.release()
    root=Path(prepared['root']);(root/('search-normal.json' if normal_inputs else 'search.json')).write_text(json.dumps(result,indent=2)+'\n')
    volume.commit();return result


@app.local_entrypoint()
def main(out:str='build/qwen35-h200-expert-search',prepared_file:str='',normal_inputs:bool=False):
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    prepared=json.loads(Path(prepared_file).read_text()) if prepared_file else prepare_candidates.remote()
    (root/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
    result=search.remote(prepared,normal_inputs);(root/'search.json').write_text(json.dumps(result,indent=2)+'\n')
