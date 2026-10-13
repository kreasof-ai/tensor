"""Borrowed views preserve storage and never release their base allocation."""

import os

import numpy as np
import pytest

import tensor


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_reshape_view_borrows_storage_and_checks_base_lifetime(provider, tmp_path):
    if os.environ.get("TENSOR_"+provider.upper()) != "1":
        pytest.skip("enable provider to test borrowed storage")
    source = tmp_path / "view.py"
    source.write_text('''import tilelang.language as T
@T.prim_func
def kernel(x:T.Tensor((2,3), "float32"), out:T.Tensor((6,), "float32")):
    with T.Kernel(1, threads=32):
        for i in T.Parallel(32):
            if i < 6:
                out[i] = x[i//3,i%3] * 2
def tensor_export(): return {"kernel":kernel}
''')
    artifact = tmp_path / "view.tbin"
    tensor.build(source, artifact, provider=provider)
    with tensor.Device(provider=provider) as device:
        base = device.from_numpy(np.arange(6,dtype="float32"))
        view = base.view((2,3))
        out = device.empty((6,),"float32")
        kernel = device.load(artifact)
        kernel.launch(view,out)
        np.testing.assert_array_equal(out.to_numpy(), np.arange(6)*2)
        np.testing.assert_array_equal(view.to_numpy(), np.arange(6).reshape(2,3))
        view.release()
        assert not base._released
        np.testing.assert_array_equal(base.to_numpy(),np.arange(6))
        view = base.view((3,2))
        base.release()
        with pytest.raises(RuntimeError,match="released"):
            view.to_numpy()
        view.release()
        view.release()


@pytest.mark.parametrize("shape", [(0,6), (7,), (2,True)])
def test_view_rejects_invalid_shape_before_registering_descriptor(shape):
    from types import SimpleNamespace

    from tensor.runtime.buffer import Buffer
    device = SimpleNamespace(_generation=1, device_type=2, ordinal=0, _check=lambda:None, _buffers=set())
    base = Buffer(device,256,(6,),np.dtype("float32"))
    with pytest.raises(ValueError):
        base.view(shape)
    assert not device._buffers


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("dtype,count", [("float32", 12), ("float16", 3)])
def test_device_copy_preserves_storage_and_borrowed_views(provider, dtype, count):
    if os.environ.get("TENSOR_"+provider.upper()) != "1":
        pytest.skip("enable provider for device copies")
    expected = np.arange(count, dtype=dtype)
    with tensor.Device(provider=provider) as device:
        source = device.from_numpy(expected)
        destination = device.empty((count,), dtype)
        view = destination.view((1, count))
        assert view.copy_from(source) is view
        np.testing.assert_array_equal(destination.to_numpy(), expected)
        # Copying the same storage through a reshape is an ordered no-op.
        source.view((1, count)).copy_from(source)
        wrong = device.empty((count + 1,), dtype)
        with pytest.raises(ValueError, match="dtype and storage size"):
            wrong.copy_from(source)
        view.release()
        np.testing.assert_array_equal(destination.to_numpy(), expected)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int32", "uint32"])
def test_device_zeros_uses_no_host_fill_and_covers_alignment(provider, dtype, monkeypatch):
    if os.environ.get("TENSOR_" + provider.upper()) != "1":
        pytest.skip("enable native device allocation qualification")
    def host_fill(*args, **kwargs):
        raise AssertionError("zeros must not allocate a host fill")
    with tensor.Device(provider=provider) as device:
        monkeypatch.setattr(device, "full", host_fill)
        buffer = device.zeros((3,), dtype)
        try:
            np.testing.assert_array_equal(buffer.to_numpy(), np.zeros(3, dtype=dtype))
        finally:
            buffer.release()
