"""Numerical qualification for coalesced GEMMs and nanoGPT affine operations."""
import copy
import json
import torch
from torch.nn import functional as F
from tensor_torch.llt import Operators
from .optimize import ROOT, BUILD, save, model_fixture, graph_measure

def main():
 torch.set_num_threads(4);torch.manual_seed(9503)
 torch.backends.cuda.matmul.allow_tf32=False
 ops=Operators(BUILD/'artifacts'); rows=[]
 for dt in (torch.bfloat16,torch.float16):
  ops.compute_dtype=dt
  for m,k,n in ((1,79,37),(3,768,256),(17,69,33),(65,129,97)):
   for ta,tb in ((False,False),(False,True),(True,False),(True,True)):
    x=torch.randn((k,m) if ta else (m,k),device='cuda',dtype=dt)*.05
    w=torch.randn((n,k) if tb else (k,n),device='cuda',dtype=dt)*.05
    expected=(x.float().T if ta else x.float())@(w.float().T if tb else w.float())
    actual=ops.gemm(x,w,ta,tb)
    torch.testing.assert_close(actual.float(),expected,atol=.001,rtol=.035)
    rows.append(dict(kind='gemm',shape=[m,k,n],ta=ta,tb=tb,dtype=str(dt),max_error=(actual.float()-expected).abs().max().item()))
 ops.compute_dtype=torch.bfloat16
 for dt in (torch.float32,torch.bfloat16):
  for r,c in ((1,64),(37,79),(1025,768)):
   x=torch.randn(r,c,device='cuda',dtype=dt,requires_grad=True); xr=x.detach().clone().requires_grad_()
   w=torch.randn(c,device='cuda',requires_grad=True);wr=w.detach().clone().requires_grad_()
   b=torch.randn(c,device='cuda',requires_grad=True);br=b.detach().clone().requires_grad_()
   dy=torch.randn_like(x)
   y=ops.layer_norm(x,w,b);yr=F.layer_norm(xr.float(),(c,),wr,br).to(dt)
   torch.testing.assert_close(y,yr,atol=.035 if dt==torch.bfloat16 else 1e-5,rtol=.035 if dt==torch.bfloat16 else 1e-5)
   y.backward(dy);yr.backward(dy)
   errors=[]
   for a,z in ((x.grad,xr.grad),(w.grad,wr.grad),(b.grad,br.grad)):
    rel=(a.float()-z.float()).norm()/z.float().norm().clamp_min(1e-12)
    assert rel<.025,(r,c,dt,rel);errors.append(rel.item())
   rows.append(dict(kind='layer_norm',shape=[r,c],dtype=str(dt),gradient_relative_l2=errors))
 # Bias gradient reduction including tails and multiple reduction passes.
 x=torch.randn(37,79,device='cuda',requires_grad=True);w=torch.randn(53,79,device='cuda',requires_grad=True);b=torch.randn(53,device='cuda',requires_grad=True)
 xr=x.detach().clone().requires_grad_();wr=w.detach().clone().requires_grad_();br=b.detach().clone().requires_grad_()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  y=ops.linear(x,w,b);yr=F.linear(xr,wr,br)
 dy=torch.randn_like(y);y.backward(dy);yr.backward(dy)
 for a,z in ((x.grad,xr.grad),(w.grad,wr.grad),(b.grad,br.grad)):
  assert (a-z).norm()/z.norm()<.025
 rows.append(dict(kind='linear_bias',max_error=(y-yr).abs().max().item()))
 save('optimized-checks',dict(status='passed',checks=rows,coverage=ops.report))
 print('passed',len(rows),'checks',flush=True)
 # Whole-model output and every parameter gradient against matched Torch.
 for kind in ('llt','naive'):
  mt,tokens=model_fixture(ops,width=128,layers=2,loops=2,vocab=128,sequence=33)
  mt.c.kind=kind
  # Recreate since parameter structure differs with kind.
  from model import Config
  from tensor_model import BackendTransformer
  c=copy.copy(mt.c);m=BackendTransformer(c,ops).cuda();ref=BackendTransformer(c).cuda();ref.load_state_dict(m.state_dict())
  targets=tokens.roll(-1,1)
  with torch.autocast('cuda',dtype=torch.bfloat16):l=m.loss(tokens,targets);lr=ref.loss(tokens,targets)
  l.backward();lr.backward();errors={}
  for (name,p),(name2,q) in zip(m.named_parameters(),ref.named_parameters()):
   assert name==name2 and p.grad is not None and q.grad is not None
   rel=((p.grad-q.grad).norm()/q.grad.norm().clamp_min(1e-12)).item();assert rel<.08,(name,rel);errors[name]=rel
  rows.append(dict(kind=kind,loss=l.item(),reference_loss=lr.item(),gradient_relative_l2=errors))
 save('optimized-checks',dict(status='passed',checks=rows,coverage=ops.report))
 print('full models passed',flush=True)

if __name__=='__main__':main()
