"""Validate installed adapter wheel and backend discovery without a compiler/GPU."""
import argparse
import builtins
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifacts',type=Path)
parser.add_argument('--gpu-cache',type=Path)
parser.add_argument('--gpu-profiles',type=Path)
opts=parser.parse_args()

original_import = builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in {'tilelang','tvm','tvm_ffi'}:
        raise ImportError('compiler imports prohibited: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__ = guarded_import

import torch
import tensor_torch as tt
from torch._dynamo.backends.registry import lookup_backend

assert lookup_backend('tensor') is tt.backend
from tensor_torch.bridge import _submit
assert _submit is not None, 'CI must produce the native submission shim'
with torch.inference_mode():
    args=(torch.randn(129),torch.randn(129))
    f=torch.compile(lambda a,b:torch.relu(a*2+b),backend='tensor',fullgraph=True)
    torch.testing.assert_close(f(*args),torch.relu(args[0]*2+args[1]))
assert tt.reports()[-1]['fallback_nodes']
assert not tt.reports()[-1]['regions']
from torch._subclasses.fake_tensor import FakeTensorMode
kernel=tt.load(opts.artifacts/'elementwise.tbin')
with FakeTensorMode():
    a,b=torch.empty(129,device='cuda'),torch.empty(129,device='cuda')
    output=kernel(a,b)
    assert output.shape==(129,) and output.dtype==torch.float32 and output.device.type=='cuda'
gpu_cases=[]
if opts.gpu_cache is not None:
    from phase4_benchmark import cases
    with torch.inference_mode():
        for name,function,args in cases(False):
            backend=tt.Backend(cache_dir=opts.gpu_cache)
            result=torch.compile(function,backend=backend,fullgraph=True,dynamic=False)(*args)
            torch.testing.assert_close(result,function(*args),atol=.01 if name.startswith('mlp') else .002,rtol=.02)
            specs=[s for r in backend.report['regions'] for s in r['specializations']]
            assert specs and all(s['cache_hit'] for s in specs)
            gpu_cases.append(name)
if opts.gpu_profiles is not None:
    from phase4_producer import affine,linear,attention
    profiles=[('affine',affine,[(129,),(129,)],torch.float32),
              ('linear',linear,[(33,64),(65,64),(65,)],torch.float16),
              ('attention',attention,[(1,2,129,64)]*3,torch.float16)]
    with torch.inference_mode():
        for name,function,shapes,dtype in profiles:
            args=[torch.randn(shape,device='cuda',dtype=dtype) for shape in shapes]
            result=tt.load(opts.gpu_profiles/(name+'.tbin'))(*args)
            torch.testing.assert_close(result,function(*args),atol=.002,rtol=.02)
            gpu_cases.append('transferred-'+name)
assert not {'tilelang','tvm','tvm_ffi'} & sys.modules.keys()
installed={d.metadata['Name'].lower() for d in metadata.distributions()}
assert not installed & {'tilelang','apache-tvm-ffi'}
print(json.dumps({'torch':torch.__version__,'adapter':tt.__version__,
                  'native_submission':True,'entry_point':True,'cpu_fallback':True,'fake_cuda_custom_op':True,
                  'compiler_packages':False,'gpu_cases':gpu_cases,
                  'gpu_execution':'passed' if gpu_cases else 'not available on CI'}))
