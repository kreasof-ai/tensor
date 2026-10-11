"""Client packaging, FX contracts, and opt-in actual CUDA execution."""

import importlib.abc
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
import tensor_torch as tt

GPU = pytest.mark.skipif(
    os.environ.get("TENSOR_P4_CUDA") != "1", reason="set TENSOR_P4_CUDA=1"
)


def test_core_does_not_import_adapter_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','tensor_torch','tilelang','tvm','tvm_ffi'}:
   raise ImportError(fullname)
sys.meta_path.insert(0,Guard())
import tensor
assert not {'torch','tensor_torch','tilelang','tvm','tvm_ffi'} & sys.modules.keys()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
