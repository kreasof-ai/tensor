"""Independent production-dispatch GEMM rechecks versus legacy and Torch."""
import torch
from .optimize import BUILD, save, graph_measure
from tensor_torch.llt import Operators

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 rows=[]
 for i,(m,k,n,ta,tb) in enumerate([(1,3072,768,False,True),(1,768,3072,False,True),(1024,768,3072,False,True),(1024,3072,768,False,False),(3072,1024,768,True,False),(1024,768,50304,False,True)]):
  torch.manual_seed(9510)
  x=torch.randn((k,m) if ta else (m,k),device='cuda',dtype=torch.bfloat16)*.05
  y=torch.randn((n,k) if tb else (k,n),device='cuda',dtype=torch.bfloat16)*.05
  expected=(x.float().T if ta else x.float())@(y.float().T if tb else y.float())
  ops={name:Operators(BUILD/'artifacts',gemm_profile=name) for name in ('legacy','optimized')}
  fns={name:(lambda op=op:op.gemm(x,y,ta,tb)) for name,op in ops.items()}
  fns['torch']=lambda:torch.matmul(x.T if ta else x,y.T if tb else y)
  row=dict(shape=[m,k,n],ta=ta,tb=tb,timings={})
  order=['torch','legacy','optimized'];order=order[i%3:]+order[:i%3]
  for name in order:
   actual=fns[name]();torch.testing.assert_close(actual.float(),expected,atol=.005,rtol=.035)
   row['timings'][name]=graph_measure(fns[name])
  row['speedup']=row['timings']['legacy']['median_ms']/row['timings']['optimized']['median_ms']
  row['coverage']={name:op.report for name,op in ops.items()}
  rows.append(row);save('finalists',dict(status='running',rows=rows))
  print(row['shape'],row['speedup'],flush=True)
 save('finalists',dict(status='passed',rows=rows,protocol='production dispatch; independently measured after search; rotating backend order; hot-L2 isolated kernels, not full-model claims'))

if __name__=='__main__':main()
