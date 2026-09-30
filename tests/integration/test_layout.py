"""Compatibility imports retain shared module globals and a lightweight core."""
import importlib
import subprocess
import sys
from pathlib import Path


def test_compatibility_modules_share_objects_and_mutations(monkeypatch):
    import tensor
    pairs = {
        'abi': 'runtime.abi', 'signature': 'runtime.signature', 'dlpack': 'runtime.dlpack',
        'manual': 'runtime.manual', 'cuda': 'providers.cuda', 'cpu': 'providers.cpu',
        'webgpu': 'providers.webgpu', 'webgpu_contract': 'providers.webgpu_contract',
        'artifact': 'artifacts.format', 'modules': 'artifacts.modules',
        'registry': 'artifacts.registry', 'portable': 'artifacts.portable',
        'build': 'compiler.build', 'nvrtc': 'compiler.nvrtc', 'lowering': 'compiler.lowering',
        'cpu_build': 'compiler.cpu', 'webgpu_build': 'compiler.webgpu',
        'webgpu_lowering': 'compiler.webgpu_lowering', 'tuning': 'compiler.tuning',
        'doctor': 'cli.doctor', 'inspect': 'cli.inspect', 'commands': 'cli.commands',
    }
    for old, new in pairs.items():
        original = importlib.import_module('tensor.' + old)
        canonical = importlib.import_module('tensor.' + new)
        assert original is canonical
    marker = object()
    monkeypatch.setattr('tensor.tuning.measure_cuda', marker)
    assert importlib.import_module('tensor.compiler.tuning').measure_cuda is marker
    assert callable(tensor.build)


def test_core_and_cli_do_not_require_optional_training_or_compiler_packages():
    code = '''import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'tensor_nn','tensor_torch','torch','tilelang','tvm','tvm_ffi','triton','wgpu'}:
   raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
sys.meta_path.insert(0, Guard())
import tensor
from tensor.cli import main
from tensor.nn import ManualFunction
assert ManualFunction is tensor.ManualFunction
assert callable(tensor.build)
try:
 from tensor.nn import NanoGPT
except ImportError as error:
 assert 'Install tensor-nn' in str(error)
else:
 raise AssertionError('optional training unexpectedly imported')
assert not {'tensor_nn','torch','tilelang','tvm','wgpu'} & set(sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_legacy_script_entry_points_work_outside_the_repository(tmp_path):
    root=Path(__file__).resolve().parents[2]
    for name in ('phase6_consumer.py','webgpu_validation.py'):
        result=subprocess.run([sys.executable,str(root/'tools'/name),'--help'],
                              cwd=tmp_path,capture_output=True,text=True,check=True)
        assert 'usage:' in result.stdout
