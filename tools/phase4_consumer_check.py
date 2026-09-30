"""Validate installed adapter wheel and backend discovery without a compiler/GPU."""
import builtins
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

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
kernel=tt.load(Path(sys.argv[1])/'elementwise.tbin')
with FakeTensorMode():
    a,b=torch.empty(129,device='cuda'),torch.empty(129,device='cuda')
    output=kernel(a,b)
    assert output.shape==(129,) and output.dtype==torch.float32 and output.device.type=='cuda'
assert not {'tilelang','tvm','tvm_ffi'} & sys.modules.keys()
installed={d.metadata['Name'].lower() for d in metadata.distributions()}
assert not installed & {'tilelang','apache-tvm-ffi'}
print(json.dumps({'torch':torch.__version__,'adapter':tt.__version__,
                  'native_submission':True,'entry_point':True,'cpu_fallback':True,'fake_cuda_custom_op':True,
                  'compiler_packages':False,'gpu_execution':'not available on CI'}))
