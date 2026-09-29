"""Opt-in A10G check of the product producer and compiler-free workbench."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TENSOR_P1_CUDA"), reason="set TENSOR_P1_CUDA=1")


def test_product_build_load_launch_and_prototype(tmp_path):
    import numpy as np
    import tensor as tx
    from tensor.build import build_artifact

    source = Path(__file__).resolve().parents[1] / "examples" / "elementwise.py"
    artifact = tmp_path / "elementwise.tbin"
    with tx.Device() as device:
        target = device.info["arch"]
    cold = build_artifact(source, artifact, target=target, cache_dir=tmp_path / "cache")
    warm = build_artifact(source, tmp_path / "warm.tbin", target=target, cache_dir=tmp_path / "cache")
    assert not cold["cache_hit"] and warm["cache_hit"]
    assert cold["cache_key"] == warm["cache_key"]
    (tmp_path / "cache" / f"{cold['cache_key']}.cubin").write_bytes(b"corrupt")
    repaired = build_artifact(source, tmp_path / "repaired.tbin", target=target,
                              cache_dir=tmp_path / "cache")
    assert not repaired["cache_hit"]
    rng = np.random.default_rng(2)
    a = rng.standard_normal(129).astype("float32")
    b = rng.standard_normal(129).astype("float32")
    with tx.Device() as device:
        kernel = device.load(artifact)
        da = device.from_numpy(a)
        db = device.from_dlpack(b)
        dc = kernel(da, db)
        tx.assert_close(dc, np.maximum(2*a+b, 0), rtol=1e-6, atol=1e-6)
        assert dc.shape == (129,) and dc.strides == (4,) and len(dc.to_bytes()) == 516
        assert tx.bench(kernel, (da, db, dc), warmup=2, iters=5)["median_launch_and_sync_seconds"] > 0
        np.testing.assert_array_equal(device.zeros((3,)).to_numpy(), [0, 0, 0])
        np.testing.assert_array_equal(device.ones((3,)).to_numpy(), [1, 1, 1])
        np.testing.assert_array_equal(device.arange(3).to_numpy(), [0, 1, 2])
        np.testing.assert_array_equal(device.full((3,), 7).to_numpy(), [7, 7, 7])
        assert device.randn((3,), seed=0).shape == (3,)
    consumer = subprocess.run(
        [sys.executable, "-c", """
import sys
import numpy as np
import tensor as tx
assert not {'tilelang', 'tvm', 'torch'} & sys.modules.keys()
a = np.arange(129, dtype='float32')
with tx.Device() as device:
    kernel = device.load(sys.argv[1])
    output = kernel(device.from_numpy(a), device.from_numpy(a))
    tx.assert_close(output, np.maximum(3*a, 0))
assert not {'tilelang', 'tvm', 'torch'} & sys.modules.keys()
""", str(artifact)], capture_output=True, text=True, timeout=30,
    )
    assert consumer.returncode == 0, consumer.stderr


def test_product_gemm_uses_lowered_dynamic_shared_memory(tmp_path):
    import numpy as np
    import tensor as tx
    from tensor.build import build_artifact
    from tensor.artifact import read_artifact

    source = Path(__file__).resolve().parents[1] / "examples" / "gemm_relu.py"
    with tx.Device() as device:
        target = device.info["arch"]
    artifact = tmp_path / "gemm.tbin"
    build_artifact(source, artifact, target=target, cache_dir=tmp_path / "cache")
    manifest, _ = read_artifact(artifact)
    assert manifest["launch"]["shared_memory_bytes"] == 24576
    rng = np.random.default_rng(4)
    a = rng.normal(0, .2, (64, 64)).astype("float16")
    b = rng.normal(0, .2, (64, 64)).astype("float16")
    bias = rng.normal(0, .2, 64).astype("float16")
    expected = np.maximum(a.astype("float32") @ b.astype("float32")
                          + bias.astype("float32"), 0).astype("float16")
    with tx.Device() as device:
        kernel = device.load(artifact)
        output = kernel(device.from_numpy(a), device.from_numpy(b), device.from_numpy(bias))
        tx.assert_close(output, expected, rtol=1e-2, atol=2e-2)
