"""Training bundle coverage, compiler-free imports and opt-in manual training."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tensor.nn.nanogpt import GPTConfig,requirements,parameter_shapes,initial_weights
from tensor.nn.kernels import source


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
from tensor.nn import NanoGPT,GPTConfig,ManualFunction
assert GPTConfig.diagnostic().width==32
'''
    subprocess.run([sys.executable,'-c',script],check=True)


@pytest.mark.skipif(os.environ.get('TENSOR_PHASE6')!='1',reason='set TENSOR_PHASE6=1 for CUDA training validation')
def test_ten_manual_updates_match_pytorch(tmp_path):
    from tools.phase6_producer import produce
    from tools.phase6_validate import validate
    directory=Path(os.environ.get('TENSOR_PHASE6_DIAGNOSTIC_BUNDLE',''))
    if not (directory/'training.json').is_file():
        directory=tmp_path/'bundle';produce(directory,GPTConfig.diagnostic())
    report=validate(directory,steps=10,out=tmp_path/'result.json')
    assert report['status']=='passed' and len(report['steps'])==10
