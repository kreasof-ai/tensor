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


@NATIVE
@pytest.mark.parametrize('encoding',['python','native'])
def test_prepared_plan_replay_scalar_isolation_and_released_resources(tmp_path,encoding):
    if encoding=='native':pytest.importorskip('tensor.providers._webgpu_native')
    from tensor.runtime import TensorRuntimeError
    source=tmp_path/'affine.py'
    source.write_text('''import tilelang.language as T
@T.prim_func
def kernel(x:T.Tensor((4,), "float32"), y:T.Tensor((4,), "float32"), scale:T.float32):
    with T.Kernel(1, threads=4):
        for i in T.Parallel(4):
            y[i]=x[i]*scale
def tensor_export():return {"kernel":kernel}
''')
    artifact=tmp_path/'affine.tbin';tx.build(source,artifact,provider='webgpu')
    with tx.Device(provider='webgpu') as device:
        kernel=device.load(artifact);x=device.from_numpy(np.arange(4,dtype=np.float32));y=device.zeros(4);z=device.zeros(4)
        calls=[]
        for inputs in ((x,y,2.),(y,z,3.)):
            values,symbols,launch=kernel._bind(inputs,{},include_outputs=True)
            calls.append((kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True)))
        plan=device.prepare_plan(calls)
        if encoding=='python':plan._encode=None
        else:assert plan._encode is not None
        for multiplier in (1.,4.):
            device.write(x,np.arange(4,dtype=np.float32)*multiplier);plan.launch()
            np.testing.assert_array_equal(z.to_numpy(),np.arange(4,dtype=np.float32)*multiplier*6)
        x.release()
        with pytest.raises(TensorRuntimeError,match='released'):plan.launch()
        plan.close();plan.close()
        with pytest.raises(TensorRuntimeError,match='closed'):plan.launch()


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
def test_cached_readback_reuses_staging_and_returns_owned_arrays():
    with tx.Device(provider="webgpu") as device:
        buffer=device.arange(129)
        first=buffer.to_numpy()
        staging=buffer._readback
        device.write(buffer,np.full(129,7,dtype=np.float32))
        second=buffer.to_numpy()
        assert buffer._readback is staging
        np.testing.assert_array_equal(first,np.arange(129,dtype=np.float32))
        np.testing.assert_array_equal(second,np.full(129,7,dtype=np.float32))
        buffer.release()
        assert buffer._readback is None


@NATIVE
def test_native_prepared_encoder_captures_validation_errors(artifacts):
    import struct,wgpu
    pytest.importorskip('tensor.providers._webgpu_native')
    with tx.Device(provider='webgpu') as device:
        kernel=device.load(artifacts['dynamic_affine'])
        x=device.arange(128);y=device.ones(128);out=device.zeros(128)
        values,symbols,launch=kernel._bind((x,y,out),{'scale':2.},include_outputs=True)
        plan=device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])
        assert plan._encode is not None and device.info['prepared_encoding']=='native'
        saved=plan._records
        bad=bytearray(saved);struct.pack_into('<I',bad,16,device._gpu.limits['max-compute-workgroups-per-dimension']+1)
        plan._records=bytes(bad)
        with pytest.raises(wgpu.GPUValidationError):plan.launch()
        plan._records=saved;plan.launch()
        np.testing.assert_array_equal(out.to_numpy(),np.arange(128,dtype=np.float32)*2+1)
        plan.close()


def test_native_prepared_encoder_rejects_malformed_records():
    native=pytest.importorskip('tensor.providers._webgpu_native')
    for records in (b'x',bytes(32)):
        with pytest.raises(ValueError,match='native WebGPU'):
            native.encode(1,records,1,1,1)


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
