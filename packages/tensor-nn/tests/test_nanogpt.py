"""Training bundle coverage, compiler-free imports and opt-in manual training."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tensor_nn.nanogpt import GPTConfig,requirements,parameter_shapes,initial_weights
from tensor_nn.kernels import source


def test_training_templates_and_tied_parameter_identity():
    for config in [GPTConfig.diagnostic(),GPTConfig()]:
        requirements_=requirements(config)
        assert len(requirements_)==len(set(requirements_))
        for kind,p in requirements_.values():ast.parse(source(kind,p))
        weights=initial_weights(config)
        assert {name:value.shape for name,value in weights.items()}==parameter_shapes(config)
        assert 'head' not in weights
        assert all(value.dtype.name=='float32' for value in weights.values())


@pytest.mark.parametrize('parameters',[{'heads':0},{'layers':True},{'width':31},{'vocab':257},{'loss_scale':float('nan')},{'learning_rate':0}])
def test_invalid_training_configuration(parameters):
    with pytest.raises(ValueError):GPTConfig(**parameters)


def test_training_runtime_imports_without_framework_or_compiler():
    script='''import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tilelang','tvm','tvm_ffi','triton'}:raise ImportError(fullname)
sys.meta_path.insert(0,Guard())
from tensor_nn import NanoGPT,GPTConfig,ManualFunction
assert GPTConfig.diagnostic().width==32
'''
    subprocess.run([sys.executable,'-c',script],check=True)


def test_bundle_schema_and_two_distribution_fingerprints_are_enforced(tmp_path):
    import json
    from types import SimpleNamespace
    from tensor_nn.nanogpt import KernelLibrary
    from tensor_nn.provenance import implementation_hashes
    hashes=implementation_hashes()
    assert 'tensor.runtime.manual' in hashes and 'tensor_nn.nanogpt' in hashes
    device=SimpleNamespace(info={'provider':'cuda'})
    path=tmp_path/'training.json'
    path.write_text(json.dumps({'schema':'tensor.manual-nanogpt.v1'}))
    with pytest.raises(ValueError,match='rebuild'):
        KernelLibrary(tmp_path,device)
    path.write_text(json.dumps({'schema':'tensor.manual-nanogpt.v2',
                               'config':vars(GPTConfig.diagnostic()),'implementation_sha256':{}}))
    with pytest.raises(ValueError,match='matching Tensor and tensor-nn'):
        KernelLibrary(tmp_path,device)


@pytest.mark.skipif(os.environ.get('TENSOR_PHASE6')!='1',reason='set TENSOR_PHASE6=1 for CUDA training validation')
def test_ten_manual_updates_match_pytorch(tmp_path):
    from benchmarks.nanogpt.producer import produce
    from benchmarks.nanogpt.validate import validate
    directory=Path(os.environ.get('TENSOR_PHASE6_DIAGNOSTIC_BUNDLE',''))
    if not (directory/'training.json').is_file():
        directory=tmp_path/'bundle';produce(directory,GPTConfig.diagnostic())
    report=validate(directory,steps=10,out=tmp_path/'result.json')
    assert report['status']=='passed' and len(report['steps'])==10
