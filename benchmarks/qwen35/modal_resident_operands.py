"""Measure a lossless FP16 operand cache before allocating one for the model."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume

app=modal.App('tensor-qwen35-h200-resident-operands')


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def measure():
    import hashlib,json,os
    from pathlib import Path
    import numpy as np
    import tensor,torch
    from tensor.compiler.entry import export_source
    from benchmarks.qwen35.build import build_artifact
    from benchmarks.qwen35.kernel_timing import time_calls
    os.chdir('/workspace');volume.reload();torch.set_num_threads(4)
    sources=[Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/'+name+'.py')
             for name in ('hopper_experts','resident_operands')]+[Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/resident-operands')/identity;root.mkdir(parents=True,exist_ok=True)
    def compile(name,module,factory,p):
        entry=root/(name+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,p,
                                      dependencies=('tensor.compiler.entry',)))
        artifact.unlink(missing_ok=True);build_artifact(entry,artifact,target='sm_90a')
        return artifact
    rng=np.random.default_rng(19163);result=dict(source_hashes=hashes,cases=[],
        scope='isolated projection; official FP8 values widened without applying scales')
    with tensor.Device() as dev:
        result['device']=dev.info
        for routed,k,o in ((False,2048,512),(True,512,2048)):
            encoded=rng.integers(0,254,(256,o,k),dtype='uint8');encoded+=encoded>=127
            weights=dev.from_numpy(encoded);wide=dev.empty(encoded.shape,'float16')
            widening=dev.load(compile(f'widen-{k}','resident_operands','widen',dict(shape=list(encoded.shape))))
            widening.launch(weights,wide)
            # Exhaust finite encodings and both signs; cache order must match K32 pairing.
            expected=torch.from_numpy(encoded).view(torch.float8_e4m3fn).half().numpy()
            j=np.arange(32);perm=(j%16)//2*4+j%2+j//16*2
            expected=expected.reshape(256,o,k//32,32)[...,perm].reshape(encoded.shape)
            np.testing.assert_array_equal(wide.to_numpy(),expected);del expected
            scales=dev.from_numpy(rng.uniform(.001,.01,(256,o//128,k//128)).astype('float32'),dtype='bfloat16')
            try:
                for rows in (512,4096):
                    shape=(rows,8,k) if routed else (rows,k)
                    x=rng.integers(0,254,shape,dtype='uint8');x+=x>=127
                    x=dev.from_numpy(x);a=dev.from_numpy(rng.uniform(.001,.01,(*shape[:-1],k//128)).astype('float32'))
                    counts=dev.empty((256,),'int32');routes=dev.empty((256,rows),'int32');out=dev.empty((rows,8,o))
                    kernels=[]
                    for resident,n,threads in ((False,128,256),(True,128,256),(True,64,128)):
                        p=dict(rows=rows,k=k,o=o,routed_input=routed,block_m=64,columns=n,threads=threads,
                               compact=True,packed_gather=True,bf16_mma=True,mma_reduction=32,mma_reorder=True,
                               resident_weights=resident)
                        kernel=dev.load(compile(f'expert-{k}-{rows}-{resident}-{n}','hopper_experts','expert_kernel',p))
                        kernels.append((p,kernel))
                    try:
                        for hot in (False,True):
                            ids=np.tile(np.arange(8),(rows,1)) if hot else (np.arange(rows)[:,None]*8+np.arange(8))%256
                            rr=np.full((256,rows),-1,'int32');cc=np.zeros(256,'int32');ee=[];tt=[]
                            for expert in range(256):
                                row,rank=np.nonzero(ids==expert);selected=row*8+rank;rng.shuffle(selected)
                                cc[expert]=len(selected);rr[expert,:len(selected)]=selected
                                for offset in range((len(selected)+63)//64):ee.append(expert);tt.append(offset)
                            size=(rows*8+63)//64+255
                            ee=np.pad(np.array(ee,'int32'),(0,size-len(ee)),constant_values=-1)
                            tt=np.pad(np.array(tt,'int32'),(0,size-len(tt)))
                            maps=[dev.from_numpy(ee),dev.from_numpy(tt)]
                            import ctypes as ct
                            for buffer,value in ((counts,cc),(routes,rr)):
                                dev.driver.call('cuMemcpyHtoD_v2',buffer.pointer,ct.c_void_p(value.ctypes.data),value.nbytes)
                            record=dict(rows=rows,k=k,o=o,routing='hot8' if hot else 'balanced256',candidates=[])
                            expected=None
                            try:
                                for p,kernel in kernels:
                                    args=(x,a,wide if p['resident_weights'] else weights,scales,counts,routes,out,*maps)
                                    kernel.launch(*args);actual=out.to_numpy()
                                    if expected is None:expected=actual.copy()
                                    equal=bool(np.array_equal(actual,expected))
                                    samples=time_calls(dev,[(kernel,args)])
                                    record['candidates'].append(dict(schedule=p,bitwise_equal=equal,
                                        finite=bool(np.isfinite(actual).all()),maximum_absolute_error=float(np.max(np.abs(actual-expected))),
                                        milliseconds=samples,cubin_sha256=kernel.manifest['files']['kernel.cubin']))
                                result['cases'].append(record);print('Resident operand result',record,flush=True)
                            finally:
                                for buffer in maps:buffer.release()
                    finally:
                        for p,kernel in kernels:kernel.release()
                        for buffer in (x,a,counts,routes,out):buffer.release()
            finally:
                for resource in (weights,wide,scales,widening):resource.release()
    (root/'measured.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result


@app.local_entrypoint()
def main(out:str='build/qwen35-h200-resident-operands.json'):
    import json
    from pathlib import Path
    result=measure.remote();Path(out).write_text(json.dumps(result,indent=2)+'\n')
