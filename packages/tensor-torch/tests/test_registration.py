"""Client packaging, FX contracts, and opt-in actual CUDA execution."""

import importlib.abc
import json
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
import tensor_torch as tt

GPU = pytest.mark.skipif(
    os.environ.get("TENSOR_P4_CUDA") != "1", reason="set TENSOR_P4_CUDA=1"
)


@GPU
def test_custom_op_fake_dynamic_mutation_and_module_loading(tmp_path):
    import tensor

    artifact = tmp_path / "affine.tbin"
    tensor.build(
        Path(__file__).parents[3] / "examples/dynamic_affine.py",
        artifact,
        compiler="nvrtc",
    )
    (tmp_path / "tensor.json").write_text(
        json.dumps(
            {
                "formatVersion": 1,
                "name": "torch-test",
                "version": "0.1.0",
                "tensorAbi": 1,
                "exports": {"affine": {"artifacts": ["affine.tbin"]}},
            }
        )
    )
    from tensor.modules.resolver import install

    install(tmp_path, cache_dir=tmp_path / "module-cache")
    kernel = tt.load(
        "torch-test::affine", project=tmp_path, module_cache=tmp_path / "module-cache"
    )
    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    assert all(
        v == "SUCCESS"
        for v in torch.library.opcheck(kernel._op, ([a, b], [2.0])).values()
    )
    output = torch.empty_like(a)
    assert all(
        v == "SUCCESS"
        for v in torch.library.opcheck(
            kernel._into_op, ([a, b], [2.0], [output])
        ).values()
    )
    with torch.inference_mode():
        compiled = torch.compile(
            lambda a, b: kernel(a, b, 2.0),
            backend="aot_eager",
            dynamic=True,
            fullgraph=True,
        )
        for size in (129, 257):
            aa, bb = (torch.randn(size, device="cuda") for _ in range(2))
            torch.testing.assert_close(compiled(aa, bb), torch.relu(aa * 2 + bb))
        kernel.into(a, b, 2.0, outputs=[output])
        torch.testing.assert_close(output, torch.relu(a * 2 + b))
        with pytest.raises(ValueError, match="alias"):
            kernel.into(a, b, 2.0, outputs=[a])
        with pytest.raises(ValueError, match="contiguous"):
            kernel(a[::2], b[::2], 2.0)
        with pytest.raises(ValueError, match="dtype"):
            kernel(a.half(), b.half(), 2.0)
