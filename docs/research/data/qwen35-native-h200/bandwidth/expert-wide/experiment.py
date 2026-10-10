import modal
from benchmarks.qwen35.modal_h200 import image,volume
app=modal.App('tensor-qwen35-h200-expert-wide')
@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,volumes={'/cache':volume},scaledown_window=2)
def measure():
 import ctypes as ct,hashlib,json,os
 from pathlib import Path
 import numpy as np,tensor
 from tensor.compiler.entry import export_source
 from benchmarks.qwen35.build import build_artifact
 from benchmarks.qwen35.kernel_timing import time_calls
 os.chdir('/workspace');volume.reload();rng=np.random.default_rng(718);
 root=Path('/cache/expert-wide-search')/hashlib.sha256(Path('/root/qwen35-expert-wide-micro.py').read_bytes()).hexdigest()[:16];root.mkdir(parents=True,exist_ok=True)
 result=dict(cases=[],scope='C8 expert projection wide tiles; not full-model throughput')
 with tensor.Device() as dev:
  result['device']=dev.info
  for rows,routed,k,o in ((512,False,2048,512),(512,True,512,2048),(4096,False,2048,512),(4096,True,512,2048)):
   parts=4;shape=(rows,8,k) if routed else (rows,k)
   def encoded(shape):
    bits=rng.integers(0,254,shape,dtype='uint8');bits+=bits>=127;return bits
   base=[dev.from_numpy(encoded(shape)),dev.from_numpy(rng.uniform(.001,.01,(*shape[:-1],k//128)).astype('float32')),dev.from_numpy(encoded((256,o,k))),dev.from_numpy(rng.uniform(.001,.01,(256,o//128,k//128)).astype('float32'),dtype='bfloat16')]
   counts=dev.empty((256,),'int32');routes=dev.empty((256,rows),'int32');out=dev.empty((rows,8,parts,o));kernels=[]
   try:
    for m,n,t in ((64,128,256),(64,256,256),(128,128,256),(128,256,256)):
     p=dict(rows=rows,k=k,o=o,routed_input=routed,block_m=m,columns=n,threads=t,partitions=parts,compact=True,packed_gather=True,bf16_mma=True,mma_reduction=32,mma_reorder=True)
     entry=root/f'projection-{k}-{m}-{n}-{t}.py';artifact=entry.with_suffix('.tbin')
     entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_experts','expert_kernel',p,dependencies=('tensor.compiler.entry',)));artifact.unlink(missing_ok=True)
     build_artifact(entry,artifact,target='sm_90a');kernels.append((p,dev.load(artifact),hashlib.sha256(artifact.read_bytes()).hexdigest()))
    for hot in (False,True):
     ids=np.tile(np.arange(8),(rows,1)) if hot else (np.arange(rows)[:,None]*8+np.arange(8))%256
     rr=np.full((256,rows),-1,'int32');cc=np.zeros(256,'int32')
     for expert in range(256):
      row,rank=np.nonzero(ids==expert);selected=row*8+rank;rng.shuffle(selected);cc[expert]=len(selected);rr[expert,:len(selected)]=selected
     for buffer,value in ((counts,cc),(routes,rr)):dev.driver.call('cuMemcpyHtoD_v2',buffer.pointer,ct.c_void_p(value.ctypes.data),value.nbytes)
     record=dict(rows=rows,k=k,o=o,routed=routed,routing='hot8' if hot else 'balanced256',candidates=[]);expected=None
     for p,kernel,sha in kernels:
      m=p['block_m'];size=(rows*8+m-1)//m+255;ee=np.full(size,-1,'int32');tt=np.zeros(size,'int32');pos=0
      for expert,count in enumerate(cc):
       for tile in range((int(count)+m-1)//m):ee[pos]=expert;tt[pos]=tile;pos+=1
      maps=[dev.from_numpy(ee),dev.from_numpy(tt)]
      try:
       args=[*base,counts,routes,out,*maps];kernel.launch(*args);actual=out.to_numpy()
       if expected is None:expected=actual.copy()
       record['candidates'].append(dict(schedule=p,bitwise_equal=bool(np.array_equal(actual,expected)),finite=bool(np.isfinite(actual).all()),maximum_absolute_error=float(np.max(np.abs(actual-expected))),milliseconds=time_calls(dev,[(kernel,args)]),artifact_sha256=sha))
      finally:
       for b in maps:b.release()
     result['cases'].append(record);print('Expert shape result',record,flush=True)
   finally:
    for b in (*base,counts,routes,out):b.release()
    for p,k,sha in kernels:k.release()
 result['source_sha256']=hashlib.sha256(Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_experts.py').read_bytes()).hexdigest()
 (root/'measured.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result
@app.local_entrypoint()
def main():
 import json
 from pathlib import Path
 Path('build/qwen35-h200-akbar-expert-wide-v1.json').write_text(json.dumps(measure.remote(),indent=2)+'\n')
