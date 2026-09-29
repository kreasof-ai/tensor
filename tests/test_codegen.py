"""Pinned compiler checks: real workload lowering, including partial tiles."""

import pytest


@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_100"])
def test_workloads_lower_without_uninitialized_buffers(arch, capfd):
    from experiments.p0 import kernels as K
    from tilelang.tools.compile_only import compile_kernel_source

    functions = [K.fused_elementwise(17, 129), K.gemm_relu(65, 67, 33),
                 K.row_sum(3, 129), K.gather_rows(17, 19, 129),
                 K.flash_attention(65, 2, 64)]
    for func in functions:
        source = compile_kernel_source(func, {"kind": "cuda", "arch": arch})
        assert '__global__' in source
    captured = capfd.readouterr()
    assert "read before initialization" not in captured.err + captured.out


def test_real_source_bundle_includes_headers_and_can_reload(tmp_path):
    from experiments.p0.artifact_build import prepare
    from experiments.p0.artifact_format import read_bundle

    path = tmp_path / "source.zip"
    prepare(path, size=129, arch="sm_80")
    manifest, files = read_bundle(path, kind="source")
    assert manifest["size"] == 129
    assert "include/tl_templates/cuda/common.h" in files
    assert "include/cute/numeric/numeric_types.hpp" in files
    assert "include/cutlass/bfloat16.h" in files
    assert any(name.startswith("licenses/") for name in files)
    assert "kernels.py" in files["kernel.cu"].decode()
    assert str(tmp_path) not in files["kernel.cu"].decode()
