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


def test_one_dynamic_binary_handles_boundaries_scalars_and_cli(tmp_path):
    import numpy as np
    import tensor as tx
    from tensor.build import build_artifact
    from tensor.commands import benchmark, run

    root = Path(__file__).resolve().parents[1]
    artifact = tmp_path / "dynamic.tbin"
    with tx.Device() as device:
        target = device.info["arch"]
    build_artifact(root / "examples/dynamic_affine.py", artifact, target=target, cache_dir=tmp_path / "cache")
    rng = np.random.default_rng(8)
    with tx.Device() as device:
        kernel = device.load(artifact)
        for size in (1, 127, 128, 129, 1025):
            a, b = (rng.standard_normal(size).astype("float32") for _ in range(2))
            da, db = device.from_numpy(a), device.from_numpy(b)
            output = kernel(da, db, scale=2.5)
            tx.assert_close(output, np.maximum(2.5*a+b, 0), rtol=1e-6, atol=1e-6)
            assert output.shape == (size,)
            with pytest.raises(ValueError, match="dimension size mismatch"):
                kernel(da, db, scale=2.5, size=size+1)
            output.release()
            da.release()
            db.release()
        a = device.ones((129,))
        with pytest.raises(ValueError, match="missing kernel arguments"):
            kernel(a, a)
        with pytest.raises(ValueError, match="dimension size mismatch"):
            kernel(a, device.ones((128,)), scale=2)
        with pytest.raises(ValueError, match="finite"):
            kernel(a, a, scale=float("inf"))
    np.save(tmp_path / "a.npy", np.arange(129, dtype="float32"))
    np.save(tmp_path / "b.npy", np.ones(129, dtype="float32"))
    inputs = [f"a={tmp_path}/a.npy", f"b={tmp_path}/b.npy"]
    run(artifact, inputs, tmp_path / "result", scalar_values=["scale=2.5"])
    np.testing.assert_array_equal(np.load(tmp_path / "result/c.npy"), 2.5*np.arange(129)+1)
    assert benchmark(artifact, inputs, scalar_values=["scale=2.5"], warmup=1, iters=2)["status"] == "passed"
    consumer = subprocess.run([sys.executable, "-c", """
import sys
import numpy as np
import tensor as tx
with tx.Device() as device:
    kernel = device.load(sys.argv[1])
    for size in (1, 129, 1025):
        a = np.arange(size, dtype='float32')
        out = kernel(device.from_numpy(a), device.from_numpy(a), scale=3.0)
        tx.assert_close(out, 4*a)
assert not {'tilelang', 'tvm', 'torch'} & sys.modules.keys()
""", str(artifact)], capture_output=True, text=True, timeout=30)
    assert consumer.returncode == 0, consumer.stderr


def test_int64_scalar_preserves_width_and_rejects_overflow(tmp_path):
    import numpy as np
    import tensor as tx
    from tensor.build import build_artifact

    source = Path(__file__).resolve().parents[1] / "examples/scalar_offset.py"
    artifact = tmp_path / "scalar.tbin"
    with tx.Device() as device:
        target = device.info["arch"]
    build_artifact(source, artifact, target=target, cache_dir=tmp_path / "cache")
    with tx.Device() as device:
        kernel = device.load(artifact)
        a = np.arange(129, dtype="int64")
        da = device.from_numpy(a)
        output = kernel(da, delta=(1 << 40)+3)
        np.testing.assert_array_equal(output.to_numpy(), a+(1 << 40)+3)
        with pytest.raises(ValueError, match="int64 range"):
            kernel(da, delta=1 << 63)
        with pytest.raises(ValueError, match="integer"):
            kernel(da, delta=1.25)


def test_dynamic_gemm_one_binary_for_five_row_counts(tmp_path):
    import numpy as np
    import tensor as tx
    from tensor.build import build_artifact

    source = Path(__file__).resolve().parents[1] / "examples/dynamic_gemm.py"
    artifact = tmp_path / "gemm.tbin"
    with tx.Device() as device:
        target = device.info["arch"]
    build_artifact(source, artifact, target=target, cache_dir=tmp_path / "cache")
    rng = np.random.default_rng(9)
    with tx.Device() as device:
        kernel = device.load(artifact)
        for rows in (1, 31, 32, 33, 65):
            a = rng.normal(0, .2, (rows, 32)).astype("float16")
            b = rng.normal(0, .2, (32, 32)).astype("float16")
            output = kernel(device.from_numpy(a), device.from_numpy(b))
            expected = (a.astype("float32") @ b.astype("float32")).astype("float16")
            tx.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_gpu_dlpack_orders_foreign_streams_and_preserves_ownership(tmp_path):
    import ctypes
    import torch
    import tensor as tx
    from tensor.build import build_artifact

    root = Path(__file__).resolve().parents[1]
    artifact = tmp_path / "dynamic.tbin"
    torch.cuda.init()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    torch.empty(1, device="cuda")
    with tx.Device() as device:
        target = device.info["arch"]
    build_artifact(root / "examples/dynamic_affine.py", artifact, target=target, cache_dir=tmp_path / "cache")
    for borrowed_stream in (False, True):
        with tx.Device(stream=consumer.cuda_stream if borrowed_stream else None) as device:
            before = ctypes.c_void_p()
            device.driver.call("cuCtxGetCurrent", ctypes.byref(before))
            kernel = device.load(artifact)
            a = torch.arange(1025, dtype=torch.float32, device="cuda")
            b = torch.ones_like(a)
            output = torch.full_like(a, float("nan"))
            expected = [2.5*(a+2)+b, 2.5*(a+4)+b]
            scratch, error = torch.empty_like(a), torch.empty((), device="cuda")
            # Allocate and load modules before the ordering probe, avoiding
            # allocator/module-load synchronizations that could hide missing waits.
            torch.sub(output, expected[0], out=scratch)
            torch.abs(scratch, out=scratch)
            torch.amax(scratch, dim=0, out=error)
            torch.cuda.synchronize()
            for stage in range(2):
                with torch.cuda.stream(producer):
                    torch.cuda._sleep(5_000_000)
                    a.add_(2)
                if stage == 0:
                    # This launch relies on DLPack's stream handshake alone.
                    with torch.cuda.stream(producer):
                        da, db, dc = (device.from_dlpack(value) for value in (a, b, output))
                    assert [da.pointer, db.pointer, dc.pointer] == [a.data_ptr(), b.data_ptr(), output.data_ptr()]
                    assert not any(value.owned for value in (da, db, dc))
                else:
                    # A later mutation requires an explicit producer wait.
                    device.wait_for(producer.cuda_stream)
                kernel.launch(da, db, dc, scale=2.5)
                device.handoff(consumer.cuda_stream)
                with torch.cuda.stream(consumer):
                    torch.sub(output, expected[stage], out=scratch)
                    torch.abs(scratch, out=scratch)
                    torch.amax(scratch, dim=0, out=error)
                # No Tensor synchronization precedes these foreign operations.
                consumer.synchronize()
                assert error.item() == 0
            with pytest.raises(BufferError, match="contiguous"):
                device.from_dlpack(a[::2])
        after = ctypes.c_void_p()
        device.driver.call("cuCtxGetCurrent", ctypes.byref(after))
        assert before.value == after.value
        with torch.cuda.stream(consumer):
            usable = torch.ones(1, device="cuda")+1
        consumer.synchronize()
        assert usable.item() == 2
        assert da._released and db._released and dc._released
