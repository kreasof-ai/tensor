"""The package's producer entry files select actual lazy DSL factories."""

import subprocess
import sys
import pytest


@pytest.mark.parametrize(
    "profile",
    [
        "portable",
        "subgroup",
        "decode_searched",
        "decode_fused",
        "prefill_unrolled",
        "prefill_outer",
        "prefill_chunked",
        "quant_searched",
        "prefill_q16",
        "prefill_mixed",
    ],
)
def test_webgpu_diagnostic_profile_constructs_tirx(profile, tmp_path):
    pytest.importorskip("tilelang")
    import tvm
    from benchmarks.lfm2.diagnostic import fixture
    from tensor_llm import GGUF
    from tensor_llm.model import requirements
    from tensor_llm.webgpu_kernels import make_kernel

    path = tmp_path / "model.gguf"
    fixture(path)
    specs = requirements(GGUF(path), 448, (1, 32), provider="webgpu", webgpu_profile=profile)
    for kind, parameters in specs.values():
        kernel = make_kernel(kind, parameters)
        assert isinstance(kernel, tvm.tirx.PrimFunc)
        assert len(kernel.params) > 0


def test_factory_modules_import_without_frontend_or_core_compiler():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in {'tilelang','tvm','tvm_ffi','torch','triton'} or name.startswith('tensor.compiler'):
   raise ImportError(name)
sys.meta_path.insert(0, Guard())
from tensor_llm.kernels import make_kernel, identity
from tensor_llm.cuda_kernels import CUDA_PROFILES, make_kernel
from tensor_llm.webgpu_kernels import make_kernel
assert len(identity('linear', {'k':256})) == 24
""",
        ],
        check=True,
    )
