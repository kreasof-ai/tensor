import json,time,urllib.request,subprocess,sys,os
from pathlib import Path
engine,profile,port=sys.argv[1:];port=int(port)
root=Path('/tmp/tensor-qwen08');base=f'http://127.0.0.1:{port}'
def get(path):
 with urllib.request.urlopen(base+path,timeout=3) as r:return r.read()
for _ in range(300):
 try:
  get('/health');break
 except Exception:time.sleep(2)
else:raise RuntimeError('server startup timeout')
print('healthy',engine,profile,flush=True)
py=f'build/{engine}-env/bin/python'
cmd=[py,'-m','benchmarks.qwen35.dense_scaling','run','--engine',engine,'--port',str(port),'--workloads',str(root/'workloads/train'),'--out',str(root/'results'/profile)]
if 'mtp' in profile:cmd+=['--speculative']
try:(root/f'{profile}-metrics-before.txt').write_bytes(get('/metrics'))
except Exception:pass
subprocess.run(cmd,check=True,env=os.environ|{'PYTHONPATH':'src:packages/tensor-llm/src'})
try:(root/f'{profile}-metrics-after.txt').write_bytes(get('/metrics'))
except Exception:pass
references=json.loads(Path('docs/research/data/qwen35-08b-local-scaling/sglang-quality.json').read_text())
quality=[]
for item in references:
 tokens=item['request']['prompt_token_ids']
 if engine=='vllm':
  path='/v1/completions';payload=dict(model='qwen08',prompt=tokens,max_tokens=32,temperature=0,ignore_eos=True,add_special_tokens=False,return_token_ids=True,stream=False)
 else:
  path='/generate';payload=dict(input_ids=tokens,stream=False,return_logprob=True,sampling_params=dict(max_new_tokens=32,temperature=0,ignore_eos=True))
 req=urllib.request.Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r:response=json.load(r)
 quality.append(dict(request=item['request'],response=response))
(root/f'{profile}-quality.json').write_text(json.dumps(quality,indent=2)+'\n')
print('complete',profile,flush=True)
