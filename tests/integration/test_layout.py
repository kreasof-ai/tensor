"""Canonical imports preserve the public API and a lightweight core."""
import importlib
import subprocess
import sys
from pathlib import Path


def test_canonical_modules_are_available_without_retired_aliases():
    import tensor
    canonical = (
        'runtime.buffer', 'runtime.executable', 'runtime.session', 'runtime.event',
        'runtime.abi', 'runtime.signature', 'runtime.dlpack', 'runtime.manual',
        'providers.cuda.device', 'providers.cuda.driver', 'providers.cuda.graph',
        'providers.cpu', 'providers.webgpu.device', 'providers.webgpu.plan',
        'artifacts.format', 'artifacts.webgpu', 'artifacts.portable',
        'modules.project', 'modules.module', 'modules.resolver', 'modules.registry',
        'compiler.build', 'compiler.cuda.nvrtc', 'compiler.frontend.signature',
        'compiler.cpu', 'compiler.webgpu.build', 'compiler.webgpu.passes.matmul',
        'compiler.tuning', 'diagnostics', 'cli.main', 'cli.inspect', 'cli.run',
    )
    for name in canonical:
        importlib.import_module('tensor.' + name)
    retired = ('artifacts.modules', 'artifacts.registry', 'compiler.webgpu_lowering',
               'compiler.webgpu_templates', 'compiler.lowering', 'compiler.entry',
               'compiler.nvrtc', 'providers.webgpu_contract', 'cli.commands')
    for name in retired:
        assert importlib.util.find_spec('tensor.' + name) is None
    assert importlib.util.find_spec('tensor.nn') is None
    assert callable(tensor.build)


def test_core_and_cli_do_not_require_optional_training_or_compiler_packages():
    code = '''import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'tensor.compiler' or fullname.startswith('tensor.compiler.'):
   raise ModuleNotFoundError(fullname, name=fullname)
  if fullname.split('.')[0] in {'tensor_nn','tensor_torch','torch','tilelang','tvm','tvm_ffi','triton','wgpu'}:
   raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
sys.meta_path.insert(0, Guard())
import tensor
from tensor.cli import main
from tensor.runtime.manual import ManualFunction
assert ManualFunction is tensor.ManualFunction
assert callable(tensor.build)
assert not {'tensor_nn','torch','tilelang','tvm','wgpu'} & set(sys.modules)
assert not any(name == 'tensor.compiler' or name.startswith('tensor.compiler.') for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_grouped_script_entry_points_work_outside_the_repository(tmp_path):
    root=Path(__file__).resolve().parents[2]
    for name in ('benchmarks/training/nanogpt/consumer.py', 'tools/validation/webgpu_validation.py'):
        result=subprocess.run([sys.executable,str(root/name),'--help'],
                              cwd=tmp_path,capture_output=True,text=True,check=True)
        assert 'usage:' in result.stdout
