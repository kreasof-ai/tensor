"""WGSL envelope, producer boundaries and native opaque-buffer execution."""
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

import tensor as tx
from tensor.runtime.abi import BoundCall, check_requirement
from tensor.artifacts.format import ArtifactError, read_artifact, validate_manifest
from tensor.compiler.build import BuildError

ROOT = Path(__file__).resolve().parents[2]
NATIVE = pytest.mark.skipif(os.environ.get("TENSOR_WEBGPU") != "1", reason="set TENSOR_WEBGPU=1 for native adapter tests")


@pytest.mark.parametrize("size", [True, 0, 3, 4.0])
def test_invalid_buffer_limit_is_rejected_before_adapter_creation(size):
    with pytest.raises(ValueError, match="max_buffer_size"):
        tx.Device(provider="webgpu", max_buffer_size=size)


def test_buffer_limit_option_is_provider_specific():
    with pytest.raises(ValueError, match="only supported by the WebGPU"):
        tx.Device(provider="cpu", max_buffer_size=268435456)


@NATIVE
def test_native_buffer_limit_opt_in():
    from tensor.runtime import TensorRuntimeError
    from tensor.providers.webgpu import probe
    default = 134217728
    info = probe()
    maximum = min(info['limits']['max-buffer-size'], info['limits']['max-storage-buffer-binding-size'])
    with tx.Device(provider="webgpu") as device:
        assert device.info['limits']['max-storage-buffer-binding-size'] <= default
        with pytest.raises(ValueError, match="limit"):
            device.empty((default//4+1,))
    with pytest.raises(TensorRuntimeError, match="exceeds adapter limit"):
        with tx.Device(provider="webgpu", max_buffer_size=maximum+1):
            pass
    if maximum > default:
        with tx.Device(provider="webgpu", max_buffer_size=default+4) as device:
            buffer = device.empty((default//4+1,))
            assert buffer.nbytes == default+4
            buffer.release()


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("webgpu")
    files = {}
    for name in ("elementwise", "dynamic_affine", "webgpu_gemm"):
        path = directory / f"{name}.tbin"
        tx.build(ROOT / f"examples/{name}.py", path, provider="webgpu", cache_dir=directory / "cache")
        files[name] = path
    return files


def test_gpu_free_build_and_portable_roundtrip(artifacts, tmp_path):
    original = artifacts["webgpu_gemm"]
    manifest, files = read_artifact(original)
    assert manifest["kind"] == "wgsl" and manifest["provider"] == "webgpu"
    assert manifest["runtime_abi"]["minor"] == 2
    assert "opaque_buffer_handles" in manifest["runtime_abi"]["required_capabilities"]
    assert b"workgroupBarrier" in files["kernel.wgsl"]
    rebuilt = tmp_path / "rebuilt.tbin"
    tx.build(original, rebuilt, provider="webgpu", cache_dir=tmp_path / "cache")
    other, payloads = read_artifact(rebuilt)
    assert other["arguments"] == manifest["arguments"]
    assert payloads["kernel.tirx.json"] == files["kernel.tirx.json"]
    assert payloads["kernel.wgsl"] == files["kernel.wgsl"]
    repeated = tx.build(original, tmp_path / "cached.tbin", provider="webgpu", cache_dir=tmp_path / "cache")
    assert repeated["cache_hit"]


@pytest.mark.parametrize("change", ["binding", "dtype", "feature", "storage", "abi", "target"])
def test_rejects_invalid_webgpu_contract(artifacts, change):
    manifest, _ = read_artifact(artifacts["webgpu_gemm"])
    if change == "binding":
        manifest["webgpu"]["bindings"][0]["binding"] = True
    elif change == "dtype":
        manifest["abi"][0]["dtype"] = "float64"
    elif change == "feature":
        manifest["webgpu"]["required_features"] = []
    elif change == "storage":
        manifest["webgpu"]["workgroup_storage_bytes"] = 65536
    elif change == "abi":
        manifest["runtime_abi"]["minor"] = 1
    else:
        manifest["target"] = "sm_86"
    with pytest.raises(ArtifactError):
        validate_manifest(manifest)


def test_hash_verified_shader_still_must_match_metadata(artifacts, tmp_path):
    import hashlib
    manifest, files = read_artifact(artifacts["elementwise"])
    original = f"@workgroup_size({manifest['launch']['block'][0]},".encode()
    files["kernel.wgsl"] = files["kernel.wgsl"].replace(original, b"@workgroup_size(1,")
    manifest["files"]["kernel.wgsl"] = hashlib.sha256(files["kernel.wgsl"]).hexdigest()
    path = tmp_path / "wrong-workgroup.tbin"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json",json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name,data)
    with pytest.raises(ArtifactError,match="workgroup"):
        read_artifact(path)


def test_rejects_vendor_compiler_and_unsupported_integer_profile(tmp_path):
    with pytest.raises(BuildError,match="compiler wgsl"):
        tx.build(ROOT / "examples/elementwise.py",tmp_path / "bad.tbin",provider="webgpu",compiler="nvrtc")
    with pytest.raises(BuildError,match="unsupported|Do not support"):
        tx.build(ROOT / "examples/scalar_offset.py",tmp_path / "bad64.tbin",provider="webgpu")
    assert not (tmp_path / "bad64.tbin").exists()
    with pytest.raises(ValueError,match="ABI 1.2"):
        check_requirement({"major":1,"minor":1,"required_capabilities":["opaque_buffer_handles"]})
    with pytest.raises(ValueError,match="lacks capabilities"):
        tx.Device(provider="webgpu",require_capabilities=["dlpack"])


def test_nonuniform_barriers_fail_before_native_execution():
    import tilelang
    from tilelang import tvm
    from tensor.compiler.webgpu_lowering import verify_uniform_barriers
    ir = tvm.tirx
    lane = ir.Var("lane", "int32")
    binding = ir.IterVar(tvm.ir.Range(0,128),lane,ir.IterVar.ThreadIndex,"threadIdx.x")
    barrier = ir.Evaluate(ir.call_intrin("int32","tirx.tvm_storage_sync","shared"))
    body = ir.AttrStmt(binding,"thread_extent",128,ir.IfThenElse(lane<8,barrier,None))
    with pytest.raises(ValueError,match="nonuniform"):
        verify_uniform_barriers(ir.PrimFunc([],body))
    verify_uniform_barriers(ir.PrimFunc([],ir.AttrStmt(binding,"thread_extent",128,barrier)))


@NATIVE
def test_native_symbolic_dispatch_and_session_handles(artifacts):
    from tensor.runtime import TensorRuntimeError
    with tx.Device(provider="webgpu") as device, tx.Device(provider="webgpu") as other:
        kernel = device.load(artifacts["dynamic_affine"])
        for size in (1,127,128,129,4097):
            a,b,out = device.arange(size), device.ones((size,)), device.empty((size,))
            with pytest.raises(BufferError,match="opaque"):
                _ = a.pointer
            values,symbols,launch = kernel._bind((a,b,out),{"scale":2.5},include_outputs=True)
            call = BoundCall(device,kernel.manifest,values,symbols,launch)
            arg = next(arg for arg in call.arguments if arg.kind == 3)
            assert arg.buffer.address == 0 and arg.scalar != 0
            device._launch(kernel,call)
            np.testing.assert_allclose(out.to_numpy(),2.5*np.arange(size,dtype="float32")+1)
            saved_handle = arg.scalar
            arg.scalar = (1<<63)
            with pytest.raises(TensorRuntimeError,match="handle"):
                device._launch(kernel,call)
            arg.scalar = saved_handle
            arg.buffer.byte_size += 4
            with pytest.raises(TensorRuntimeError,match="metadata"):
                device._launch(kernel,call)
            with pytest.raises(ValueError,match="different device"):
                kernel(a,other.ones((size,)),scale=2.5)
            for buffer in (a,b,out):
                buffer.release()
            arg.buffer.byte_size -= 4
            with pytest.raises(TensorRuntimeError,match="handle"):
                device._launch(kernel,call)
        with pytest.raises(BufferError,match="DLPack"):
            device.from_dlpack(np.arange(4))
        event = device.record_event()
        assert event.descriptor.flags == 0
        other.wait(event)
        event.release()
    with device:
        with pytest.raises(TensorRuntimeError,match="previous"):
            a.to_numpy()


@NATIVE
def test_native_fp16_tail_and_adapter_feature_gate(artifacts):
    from types import SimpleNamespace
    from tensor.providers.webgpu import Device
    with tx.Device(provider="webgpu") as device:
        if "shader-f16" not in device.info["features"]:
            pytest.skip("adapter has no shader-f16")
        rng = np.random.default_rng(5)
        a,b,bias = [rng.standard_normal(shape).astype("float16") for shape in ((33,37),(37,65),(65,))]
        result = device.load(artifacts["webgpu_gemm"])(*[device.from_numpy(x) for x in (a,b,bias)])
        expected = np.maximum(a.astype("float32")@b.astype("float32")+bias,0).astype("float16")
        np.testing.assert_allclose(result.to_numpy(),expected,atol=.015,rtol=.01)
        short = np.array([1,2,3],dtype="float16")
        np.testing.assert_array_equal(device.from_numpy(short).to_numpy(),short)
    fake = Device.__new__(Device)
    fake._open = True
    fake._gpu = SimpleNamespace(features=set())
    with pytest.raises(ArtifactError,match="required features"):
        fake.load(artifacts["webgpu_gemm"])


@NATIVE
def test_packed_x_dispatch_masks_last_workgroup_plane(artifacts):
    with tx.Device(provider="webgpu") as device:
        kernel = device.load(artifacts["dynamic_affine"])
        # Exercise x/z packing with a small enforced dispatch dimension so the
        # real native adapter can test an incomplete final plane cheaply.
        limit = device._gpu.limits["max-compute-workgroups-per-dimension"]
        device._gpu.limits["max-compute-workgroups-per-dimension"] = 4
        try:
            size = 128*10+1  # eleven x groups packed into three planes of four
            output = kernel(device.arange(size),device.ones((size,)),scale=2.5)
            np.testing.assert_allclose(output.to_numpy(),2.5*np.arange(size,dtype="float32")+1)
        finally:
            device._gpu.limits["max-compute-workgroups-per-dimension"] = limit
