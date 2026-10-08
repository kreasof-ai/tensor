"""Whole-model optimization controls and actual nanoGPT adapter qualification."""
import gc
import os
import sys
import copy
from pathlib import Path
import torch
from .optimize import ROOT, BUILD, save, model_fixture, graph_measure
from tensor_torch.llt import Operators, AdamW
sys.path.insert(0,str(Path(os.environ.get('LLT_CHECKOUT',ROOT.parent/'loop-latent-transformer'))/'experiments/l40s'))
from nanogpt_adapter import make

def nano_check():
 torch.manual_seed(9504)
 ops=Operators(BUILD/'artifacts')
 c=dict(n_layer=2,n_head=2,n_embd=128,vocab_size=256,block_size=65,dropout=0,bias=True)
 m=make(ops,**c);ref=make(**c);ref.load_state_dict(m.state_dict())
 x=torch.randint(256,(2,33),device='cuda');y=x.roll(-1,1);y[0,0]=-1
 opt=AdamW(m.parameters(),ops,lr=.0006,weight_decay=.1)
 ropt=torch.optim.AdamW(ref.parameters(),lr=.0006,weight_decay=.1,foreach=False,fused=False)
 rows=[]
 for i in range(8):
  opt.zero_grad(set_to_none=True);ropt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits,l=m(x,y);rl,lr=ref(x,y)
  l.backward();lr.backward();errors={}
  for (name,p),(name2,q) in zip(m.named_parameters(),ref.named_parameters()):
   assert name==name2 and p.grad is not None and q.grad is not None
   rel=((p.grad-q.grad).norm()/q.grad.norm().clamp_min(1e-12)).item()
   # After independent updates accumulated parameter drift also contributes.
   assert rel < (.08 if i==0 else .2),(i,name,rel)
   errors[name]=rel
  assert abs(l.item()-lr.item())<.15
  torch.nn.utils.clip_grad_norm_(ref.parameters(),1)
  opt.step();ropt.step()
  rows.append(dict(step=i,loss=l.item(),reference_loss=lr.item(),gradient_relative_l2=errors))
 # Tied parameter identity and last-token inference protocol.
 assert m.lm_head.weight is m.transformer.wte.weight
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  out,_=m(x);expected,_=ref(x)
  assert out.shape==(2,1,256)
  rel=((out.float()-expected.float()).norm()/expected.float().norm()).item();assert rel<.1
 save('nanogpt-checks',dict(status='passed',config=c,steps=rows,last_logits_relative_l2=rel,
      tied_embedding=True,coverage=ops.report))
 print('nanoGPT qualification passed',flush=True)


def comparison():
 rows=[]
 for backend in ('torch','legacy','optimized'):
  ops=None if backend=='torch' else Operators(BUILD/'artifacts',gemm_profile=backend)
  m,tokens=model_fixture(ops)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for block in m.blocks:block.to(dtype=torch.bfloat16)
   m.down.to(dtype=torch.bfloat16);m.output.to(dtype=torch.bfloat16)
   _,state=m.prefill(tokens[:,:-1])
   def fn():
    m.rewind(state,1024)
    return m.decode_token(tokens[:,-1:],state)
   row=dict(backend=backend,decode=graph_measure(fn),config=vars(m.c),coverage=ops.report if ops else None)
   rows.append(row);save('model-comparison',dict(status='running',rows=rows))
   print(backend,row['decode']['median_ms'],flush=True)
  del fn,state,m,tokens,ops;gc.collect();torch.cuda.empty_cache()
 save('model-comparison',dict(status='passed',rows=rows,protocol='one token, prepared BF16 weights, 1024 prefix, CUDA graph; sequential GPU processes'))

if __name__=='__main__':
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 nano_check();gc.collect();torch.cuda.empty_cache();comparison()
