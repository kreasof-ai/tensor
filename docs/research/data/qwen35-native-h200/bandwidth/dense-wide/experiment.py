import modal
from benchmarks.qwen35.modal_h200 import image,volume
app=modal.App('tensor-qwen35-h200-dense-wide')
@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,volumes={'/cache':volume},scaledown_window=2)
def measure():
 import hashlib,json,os
 from pathlib import Path
 import numpy as np,tensor
 from tensor.compiler.entry import export_source
 from benchmarks.qwen35.build import build_artifact
 from benchmarks.qwen35.kernel_timing import time_calls
 os.chdir('/workspace');volume.reload();rng=np.random.default_rng(619)
 digest=hashlib.sha256(Path('/root/qwen35-dense-wide-micro.py').read_bytes()).hexdigest()
 root=Path('/cache/dense-wide-search')/digest[:16];root.mkdir(parents=True,exist_ok=True)
 result=dict(cases=[],scope='dense projections with finite signed FP8 operands; isolated kernels, not full-model throughput',source_sha256=hashlib.sha256(Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_dense.py').read_bytes()).hexdigest())
 def encoded(shape):
  bits=rng.integers(0,254,shape,dtype='uint8');bits+=bits>=127;return bits
 with tensor.Device() as dev:
  result['device']=dev.info
  for rows,k,o in ((4096,2048,8192),(4096,4096,2048),(4096,2048,512)):
   base=[dev.from_numpy(encoded((rows,k))),dev.from_numpy(encoded((o,k))),dev.from_numpy(rng.uniform(.001,.01,(o//128,k//128)).astype('float32'),dtype='bfloat16'),dev.from_numpy(rng.uniform(.001,.01,(rows,k//128)).astype('float32'))]
   out=dev.empty((rows,o));expected=None;record=dict(rows=rows,k=k,o=o,candidates=[])
   try:
    for m,n,t in ((64,128,256),(64,256,256),(128,128,256),(128,256,256)):
     p=dict(r=rows,k=k,o=o,block_m=m,columns=n,threads=t,packed_gather=True,stages=1,mma_reduction=32,mma_reorder=True)
     entry=root/f'dense-{k}-{o}-{m}-{n}-{t}.py';artifact=entry.with_suffix('.tbin')
     entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_dense','make_kernel',p,dependencies=('tensor.compiler.entry',)));artifact.unlink(missing_ok=True);build_artifact(entry,artifact,target='sm_90a')
     kernel=dev.load(artifact)
     try:
      args=[*base,out];kernel.launch(*args);actual=out.to_numpy()
      if expected is None:expected=actual.copy()
      row=dict(schedule=p,bitwise_equal=bool(np.array_equal(actual,expected)),finite=bool(np.isfinite(actual).all()),maximum_absolute_error=float(np.max(np.abs(actual-expected))),milliseconds=time_calls(dev,[(kernel,args)]),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest());record['candidates'].append(row);print('Dense wide',row,flush=True)
     finally:kernel.release()
    result['cases'].append(record)
   finally:
    for b in (*base,out):b.release()
 (root/'measured.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result
@app.local_entrypoint()
def main():
 import json
 from pathlib import Path
 Path('build/qwen35-h200-akbar-dense-wide-v1.json').write_text(json.dumps(measure.remote(),indent=2)+'\n')
