import modal
from benchmarks.qwen35.modal_h200 import image,volume
from benchmarks.qwen35.modal_bandwidth import app as micro_app,measure_micro
app=modal.App('tensor-qwen35-h200-batched-attention');app.include(micro_app)
@app.function(image=image,cpu=16,memory=32768,timeout=600,volumes={'/cache':volume},scaledown_window=2)
def prepare(prepared):
 import hashlib,json,os
 from pathlib import Path
 from tensor.compiler.entry import export_source
 from benchmarks.qwen35.build import build_artifact
 os.chdir('/workspace');volume.reload()
 digest=hashlib.sha256(Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_prefill_batched.py').read_bytes()).hexdigest()
 root=Path('/cache/batched-attention')/digest[:16];root.mkdir(parents=True,exist_ok=True)
 result=dict(prepared,root=str(root),batched_source_sha256=digest,candidates=[prepared['candidates'][0],prepared['candidates'][5]])
 for q,t in ((128,256),(64,256),(64,128)):
  p=dict(slots=8,chunk=512,capacity=48000,query_rows=q,threads=t,packed_loads=True,decoded_kv=True,batched_keys=True)
  entry=root/f'attention-q{q}-t{t}.py';artifact=entry.with_suffix('.tbin');entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_prefill_batched','attention',p,dependencies=('tensor.compiler.entry',)))
  artifact.unlink(missing_ok=True);build_artifact(entry,artifact,target='sm_90a')
  result['candidates'].append(dict(schedule=p,path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),compiled=True))
 (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result
@app.local_entrypoint()
def main():
 import json
 from pathlib import Path
 root=Path('build/qwen35-h200-akbar-batched-attention-v1');root.mkdir(exist_ok=True)
 base=json.loads(Path('build/qwen35-h200-akbar-decoded-micro-v1/prepared.json').read_text())
 ready=prepare.remote(base);(root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
 result=measure_micro.remote(ready);(root/'measured.json').write_text(json.dumps(result,indent=2)+'\n')
