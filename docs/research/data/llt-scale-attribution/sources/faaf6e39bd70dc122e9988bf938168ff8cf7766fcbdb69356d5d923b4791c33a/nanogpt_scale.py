"""Sequential L40S inference/training profiles after Tensor kernel optimization."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tensor_torch.llt import Operators, AdamW
from model import Config
from tensor_model import BackendTransformer
from nanogpt_adapter import make as nano, UPSTREAM_SHA, UPSTREAM_URL
from prepared_inference import prepare as prepare_torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s-nanogpt'
TENSOR=Path(os.environ.get('TENSOR_CHECKOUT',ROOT.parent/'tensor'))


def git(path):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()


def save(name,data):
 OUT.mkdir(parents=True,exist_ok=True)
 files=list((ROOT/'experiments/l40s').glob('*.py'))+[ROOT/'experiments/l40s/third_party/nanogpt/model.py',ROOT/'experiments/l40s/third_party/nanogpt/LICENSE']
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 for p in files:
  dest=OUT/'sources'/sources[str(p.relative_to(ROOT))]/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.write_bytes(p.read_bytes())
 data['provenance']=dict(llt_commit=git(ROOT),tensor_commit=git(TENSOR),tensor_repository='https://github.com/kreasof-ai/tensor',
  nanogpt_repository=UPSTREAM_URL,nanogpt_commit=UPSTREAM_SHA,sources=sources,
  torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
  nvrtc=os.environ.get('TENSOR_NVRTC_HOME'),driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
 (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def measure(fn,samples=9,reset=None):
 for _ in range(3):
  if reset:reset()
  result=fn();del result
 torch.cuda.synchronize()
 gpu=[];wall=[];memory=[]
 for _ in range(samples):
  if reset:reset()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  t=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
  gpu.append(a.elapsed_time(b));wall.append((time.perf_counter()-t)*1000)
  memory.append(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_after_bytes=torch.cuda.memory_allocated()))
  del result
 return dict(gpu_samples_ms=gpu,wall_samples_ms=wall,gpu_median_ms=statistics.median(gpu),wall_median_ms=statistics.median(wall),
             memory_samples=memory,peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in memory))


def graph_measure(fn,samples=9):
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):result=fn();del result
 torch.cuda.current_stream().wait_stream(stream)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):result=fn()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 times=[]
 for _ in range(samples):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  a.record()
  for _ in range(10):g.replay()
  b.record();b.synchronize();times.append(a.elapsed_time(b)/10)
 out=dict(gpu_samples_ms=times,gpu_median_ms=statistics.median(times),
          allocated_graph_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
 del g,result;gc.collect();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
 return out


def fixture(a):
 torch.manual_seed(9505);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 ops=Operators(OUT/'artifacts',cache_inference_weights=a.phase=='inference') if a.backend=='tensor' else None
 if a.model=='nanogpt':
  m=nano(ops,n_embd=768,n_head=12,n_layer=12,vocab_size=50304,block_size=1024,dropout=0,bias=True)
 else:
  c=Config(kind=a.model,width=768,heads=12,layers=12,loops=a.loops,rank=64,vocab=50304,max_seq=1028,gelu='none')
  m=BackendTransformer(c,ops).cuda()
 x=torch.randint(50304,(a.batch,1024),device='cuda');y=torch.randint(50304,x.shape,device='cuda')
 return m,ops,x,y


def correctness(a):
 # Full geometry, all parameter gradients. Two models retained for comparison.
 m,ops,x,y=fixture(a)
 if a.model=='nanogpt':ref=nano(n_embd=768,n_head=12,n_layer=12,vocab_size=50304,block_size=1024,dropout=0,bias=True)
 else:ref=BackendTransformer(m.c).cuda()
 ref.load_state_dict(m.state_dict())
 x,y=x[:1],y[:1]
 with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
  if a.model=='nanogpt':out,l=m(x,y);expected,lr=ref(x,y)
  else:out=m(x);expected=ref(x);l=ops.cross_entropy(out.reshape(-1,50304),y.reshape(-1));lr=torch.nn.functional.cross_entropy(expected.float().reshape(-1,50304),y.reshape(-1))
 l.backward();lr.backward();errors={}
 for (name,p),(name2,q) in zip(m.named_parameters(),ref.named_parameters()):
  assert name==name2 and p.grad is not None and q.grad is not None,(name,name2)
  rel=((p.grad-q.grad).norm()/q.grad.norm().clamp_min(1e-12)).item();assert rel<.08,(name,rel);errors[name]=rel
 assert abs(l.item()-lr.item())<.15
 rel=((out.float()-expected.float()).norm()/expected.float().norm()).item();assert rel<.04,rel
 if a.model!='nanogpt':
  # Final-logit selection equals full output; cached trajectory equals full causal forward.
  m.eval()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   full=m(x);last=m(x,last_logits=True);torch.testing.assert_close(last,full[:,-1:],atol=.01,rtol=.03)
   _,state=m.prefill(x[:,:-4],last_logits=True)
   cached=[]
   for t in range(x.shape[1]-4,x.shape[1]):cached.append(m.decode_token(x[:,t:t+1],state))
   cached=torch.cat(cached,1);err=(cached.float()-full[:,-4:].float()).abs().max().item();assert err<.04,err
 else:err=None
 save('correctness-'+a.model+'-t'+str(a.loops),dict(status='passed',arguments=vars(a),loss=l.item(),reference_loss=lr.item(),
  output_relative_l2=rel,gradient_relative_l2=errors,cached_max_error=err,coverage=ops.report))
 print('correctness',a.model,a.loops,'passed',flush=True)


def run(a):
 m,ops,x,y=fixture(a)
 out=dict(status='running',arguments=vars(a),parameter_count=sum(p.numel() for p in m.parameters()),
          parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()))
 stem=a.phase+'-'+a.model+'-'+a.backend+'-b'+str(a.batch)+'-t'+str(a.loops)+'-'+a.policy
 if a.phase=='training':
  m.train();groups=[dict(params=[p for p in m.parameters() if p.ndim>=2],weight_decay=.1),dict(params=[p for p in m.parameters() if p.ndim<2],weight_decay=0)]
  opt=AdamW(groups,ops,lr=.0006,betas=(.9,.95),max_norm=1.) if ops else torch.optim.AdamW(groups,lr=.0006,betas=(.9,.95),foreach=False,fused=False)
  losses=[]
  def step():
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    l=m(x,y)[1] if a.model=='nanogpt' else m.loss(x,y,a.policy)
   l.backward()
   if not ops:torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   opt.step();losses.append(l.detach());return l.detach()
  out['training']=measure(step,a.samples);out['losses']=[v.item() for v in losses]
  assert all(torch.isfinite(v).item() for v in losses)
  out['tokens_per_second']=a.batch*1024/(out['training']['wall_median_ms']/1000)
 else:
  m.eval()
  if not ops:prepare_torch(m)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   prefill=(lambda:m(x)[0]) if a.model=='nanogpt' else (lambda:m(x,last_logits=True))
   out['causal_last_logits']=dict(eager=measure(prefill,a.samples),graph=graph_measure(prefill,a.samples))
   if a.model=='nanogpt':
    # Genuine upstream recomputes a cropped prefix each generation step; no KV cache.
    def decode():
     idx=x
     for _ in range(4):
      logits,_=m(idx[:,-1024:]);idx=torch.cat((idx,logits[:,-1].argmax(-1,keepdim=True)),1)
     return idx
    out['four_token_recompute']=dict(eager=measure(decode,a.samples),graph=graph_measure(decode,a.samples))
    out['cache_bytes']=0;out['protocol']='actual upstream forward; greedy full-prefix recomputation, cropped to context 1024'
   else:
    def startup():return m.prefill(x,capacity=1028,last_logits=True)
    out['serving_startup']=dict(eager=measure(startup,a.samples),graph=graph_measure(startup,a.samples))
    _,state=startup();next_tokens=torch.randint(50304,(a.batch,4),device='cuda')
    out['cache_bytes']=sum(c.nbytes for c in state['caches'])
    out['fold_bytes']=sum(w.numel()*w.element_size() for pair in state['folds'] for w in pair)
    def reset():m.rewind(state,1024)
    def decode():
     result=None
     for t in range(4):result=m.decode_token(next_tokens[:,t:t+1],state)
     return result
    eager=measure(decode,a.samples,reset)
    def graph_decode():reset();return decode()
    out['four_token_cached']=dict(eager=eager,graph=graph_measure(graph_decode,a.samples))
    out['protocol']='serving startup includes KV allocation/copy and fold rebuild; cached continuation uses fixed supplied tokens; graph includes rewind'
  out['precision']='FP32 masters, embeddings and residuals; BF16 projection/attention; explicit prepared parameter copies for both Tensor and Torch'
  if not ops:
   out['prepared_weight_bytes']=sum(entry[1].numel()*entry[1].element_size() for entry in m.prepared_weight_casts.values())
 if ops:
  assert not ops.report['fallbacks'];out['coverage']=ops.report
  if a.phase=='inference':
   assert ops.report['inference_weight_cast_hits']>0
   out['prepared_weight_bytes']=sum(entry[2].numel()*entry[2].element_size() for entry in ops.inference_weight_casts.values())
 out['status']='passed';save(stem,out)
 print(stem,'passed',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['training','inference','correctness'])
 p.add_argument('--model',choices=['llt','naive','nanogpt'],required=True);p.add_argument('--backend',choices=['tensor','torch'],default='tensor')
 p.add_argument('--batch',type=int,default=1);p.add_argument('--loops',type=int,default=1);p.add_argument('--policy',choices=['none','loop'],default='none');p.add_argument('--samples',type=int,default=9)
 a=p.parse_args()
 if a.model=='nanogpt' and (a.policy!='none' or a.loops!=1):raise ValueError('upstream nanoGPT has one pass and no loop checkpoint')
 if a.phase=='correctness':correctness(a)
 else:run(a)
