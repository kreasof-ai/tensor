import modal
from benchmarks.qwen35.modal_h200 import image,volume
app=modal.App('tensor-qwen35-h200-scan-geometry')
@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,volumes={'/cache':volume},scaledown_window=2)
def measure():
 import ctypes as ct,hashlib,json,os
 from pathlib import Path
 import numpy as np,tensor
 from tensor.compiler.entry import export_source
 from benchmarks.qwen35.build import build_artifact
 from benchmarks.qwen35.kernel_timing import time_calls
 os.chdir('/workspace');volume.reload();rng=np.random.default_rng(681)
 root=Path('/cache/scan-geometry')/hashlib.sha256(Path('/root/qwen35-scan-geometry-micro.py').read_bytes()).hexdigest()[:16];root.mkdir(parents=True,exist_ok=True)
 result=dict(cases=[],scope='scan value tile changes; isolated kernels, not client throughput')
 with tensor.Device() as dev:
  result['device']=dev.info
  for chunk in (128,2048):
   rows=8*chunk;lengths=np.array([chunk,chunk-1,chunk//2+5,0,1,3,chunk,3*chunk//4],'int32')
   q=rng.normal(0,1/(128**.5),(rows,32,128)).astype('float32');k=rng.normal(size=q.shape).astype('float32');k/=np.linalg.norm(k,axis=-1,keepdims=True)
   v=rng.normal(0,.1,q.shape).astype('float32');g=rng.uniform(-.2,-.01,(rows,32)).astype('float32');beta=rng.uniform(.01,.99,g.shape).astype('float32')
   initial=rng.normal(0,.01,(8,32,128,128)).astype('float32')
   base=[dev.from_numpy(x) for x in (q,k,v,g,beta,lengths)];state=dev.empty(initial.shape);out=dev.empty(q.shape);expected=None;record=dict(chunk=chunk,lengths=lengths.tolist(),candidates=[])
   try:
    for tile in (32,16,8,64):
     p=dict(slots=8,chunk=chunk,value_tile=tile);entry=root/f'scan-{chunk}-{tile}.py';artifact=entry.with_suffix('.tbin')
     entry.write_text(export_source('tensor_llm.qwen35.kernels.prefill','make_kernel','gdn_scan',p,dependencies=('tensor.compiler.entry',)));artifact.unlink(missing_ok=True);build_artifact(entry,artifact,target='sm_90')
     kernel=dev.load(artifact)
     try:
      dev.driver.call('cuMemcpyHtoD_v2',state.pointer,ct.c_void_p(initial.ctypes.data),initial.nbytes)
      args=[*base,state,out];kernel.launch(*args);actual=(out.to_numpy(),state.to_numpy())
      if expected is None:expected=actual
      row=dict(value_tile=tile,outputs_bitwise_equal=bool(np.array_equal(expected[0],actual[0])),states_bitwise_equal=bool(np.array_equal(expected[1],actual[1])),finite=bool(all(np.isfinite(x).all() for x in actual)),milliseconds=time_calls(dev,[(kernel,args)]),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest());record['candidates'].append(row);print('Scan geometry',chunk,row,flush=True)
     finally:kernel.release()
    result['cases'].append(record)
   finally:
    for b in (*base,state,out):b.release()
 result['source_sha256']=hashlib.sha256(Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/prefill.py').read_bytes()).hexdigest()
 (root/'measured.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result
@app.local_entrypoint()
def main():
 import json
 from pathlib import Path
 Path('build/qwen35-h200-akbar-scan-geometry-v1.json').write_text(json.dumps(measure.remote(),indent=2)+'\n')
