import modal
from benchmarks.qwen35.modal_h200 import image,volume
app=modal.App('tensor-qwen35-h200-pool-trace')
@app.function(image=image,gpu='H200',cpu=16,memory=98304,timeout=600,volumes={'/cache':volume},scaledown_window=2)
def trace(prepared):
 import json,os
 from pathlib import Path
 import numpy as np,tensor
 from tensor_llm import Qwen35Batch,Qwen35Prefill
 from tensor_llm.qwen35.compact_prefill import install as compact
 from tensor_llm.qwen35.hopper import install as hopper
 from tensor_llm.qwen35.attention_workspace import install as workspace
 from benchmarks.qwen35.spec_run import make_verifier_pool
 from benchmarks.qwen35.whole_tune import Snapshot
 os.chdir('/workspace');volume.reload();base=prepared['base'];paths=base['paths']
 prompts=np.array([r['prompt_token_ids'] for r in json.loads(Path('workload.json').read_text())['requests']],'int32')
 with tensor.Device() as dev,Qwen35Batch(base['checkpoint'],paths['decoder'],dev) as model:
  prefill=Qwen35Prefill(model,prepared['prefill']);compact(prefill,prepared['prefill']);hopper(prefill,prepared['hopper']);workspace(prefill,prepared['attention_workspace'])
  for start in range(0,32000,prefill.chunk):
   count=min(prefill.chunk,32000-start);block=np.zeros((8,prefill.chunk),'int32');block[:,:count]=prompts[:,start:start+count];prefill.forward(block,np.full(8,count,'int32'))
  prefill.close();prefix=Snapshot(model);lengths=np.array([8,7,5,0,1,3,8,6],'int32');tokens=np.repeat(prefix.tokens[:,None],128,axis=1)
  def observe(bundle):
   prefix.restore(model);owner=make_verifier_pool(model,bundle);ctx=owner.contexts[8] if hasattr(owner,'contexts') else owner
   manifest=json.loads((Path(bundle)/('profiles/8/verify/prefill.json' if hasattr(owner,'contexts') else 'prefill.json')).read_text())
   names={id(k):manifest['kernels'].get(key.removeprefix('_compact_'),{}).get('kind',key) for key,k in ctx.kernels.items()}
   valid=(np.arange(ctx.chunk)[None,:]<lengths[:,None]).reshape(-1);records=[]
   kinds={'embedding','rms','small_bf16','split_linear_merge','gdn_conv','spec_gdn_conv','gdn_prepare','gdn_scan','spec_gdn_scan','gdn_norm','attention_q','split_merge','add_rms','moe_combine'}
   def submit():
    for kernel,bound in ctx.plan:
     dev._launch(kernel,bound);kind=names.get(id(kernel),'unknown')
     if kind not in kinds:continue
     values=dict(zip((a['name'] for a in kernel.manifest.get('abi',kernel.manifest['arguments'])),bound.storage));out=values.get('out')
     if out is not None and out.shape[0]==ctx.rows:
      records.append((kind.removeprefix('spec_'),out.to_numpy()[valid].copy()))
   ctx.graph.launch=submit
   try:owner.forward(tokens,lengths)
   finally:owner.close();prefix.restore(model)
   return records
  expected=observe(prepared['adaptive_control_bundle']);actual=observe(paths['verify']);result=dict(control_records=len(expected),candidate_records=len(actual),differences=[])
  for i,((ek,e),(ak,a)) in enumerate(zip(expected,actual)):
   if ek!=ak or e.shape!=a.shape:result['differences'].append(dict(index=i,control=ek,candidate=ak,control_shape=e.shape,candidate_shape=a.shape));break
   if not np.array_equal(e,a):
    result['differences'].append(dict(index=i,kind=ek,shape=e.shape,relative_rms=float(np.linalg.norm(a-e)/max(np.linalg.norm(e),1e-20)),maximum_absolute_error=float(np.max(np.abs(a-e)))))
  print('Trace summary',json.dumps(result),flush=True);return result
@app.local_entrypoint()
def main():
 import json
 from pathlib import Path
 r=trace.remote(json.loads(Path('build/qwen35-h200-akbar-pool128-v2/prepared.json').read_text()));Path('build/qwen35-h200-akbar-pool-trace-v1.json').write_text(json.dumps(r,indent=2)+'\n')
