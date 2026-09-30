"""Runtime version gates, descriptor semantics and cross-provider execution."""

import copy
import ctypes as c
from pathlib import Path
import shutil
import subprocess
import json
import zipfile

import numpy as np
import pytest

import tensor as tx
from tensor.runtime.abi import (Argument, BoundCall, BufferDescriptor, CallDescriptor, DTYPES,
                        ErrorDescriptor, EventDescriptor, ExecutableDescriptor, StreamDescriptor,
                        WorkspaceRequirements, check_requirement)
from tensor.artifacts.format import ArtifactError, read_artifact, validate_manifest
from tensor.runtime import TensorRuntimeError

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_version_and_capability_negotiation():
    requirement = {"major": 1, "minor": 0, "required_capabilities": ["contiguous", "scalars"]}
    check_requirement(requirement)
    for field, value in (("major", 2), ("major", True), ("minor", 3), ("minor", -1)):
        incompatible = {**requirement, field: value}
        with pytest.raises(ValueError, match="version"):
            check_requirement(incompatible)
    with pytest.raises(ValueError, match="capabilities"):
        check_requirement({**requirement, "required_capabilities": ["unknown"]})


def test_c_abi_layout_and_64_bit_scalar_payload():
    assert [c.sizeof(t) for t in (BufferDescriptor, Argument, StreamDescriptor, CallDescriptor, ErrorDescriptor)] == [48,64,16,72,512]
    assert c.sizeof(WorkspaceRequirements) == 24 and c.sizeof(ExecutableDescriptor) == 64
    assert c.sizeof(EventDescriptor) == 40
    with tx.Device(provider="cpu") as device:
        manifest = {"arguments": [{"kind": "scalar", "name": "delta", "dtype": "int64"}],
                    "abi": [{"kind": "scalar", "name": "delta", "dtype": "int64"}]}
        call = BoundCall(device, manifest, {"delta": -(1<<40)-3}, {},
                         {"grid": [1,1,1], "block": [1,1,1], "shared_memory_bytes": 0})
        assert c.c_int64(call.arguments[0].scalar).value == -(1<<40)-3
        assert call.descriptor.stream.device_type == 1


@pytest.fixture(scope="module")
def cpu_artifacts(tmp_path_factory):
    if not shutil.which("c++"):
        pytest.skip("CPU producer needs c++")
    import platform
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        pytest.skip("Linux x86-64 CPU profile")
    root = tmp_path_factory.mktemp("cpu-artifacts")
    artifacts = {}
    for name in ("elementwise", "dynamic_affine", "scalar_offset"):
        path = root / f"{name}.tbin"
        tx.build(ROOT / f"examples/{name}.py", path, provider="cpu", cache_dir=root / "cache")
        artifacts[name] = path
    return artifacts


def test_cpu_native_image_and_shared_workbench(cpu_artifacts, tmp_path):
    with tx.Device(provider="cpu") as device:
        kernel = device.load(cpu_artifacts["dynamic_affine"])
        for size in (1,127,128,129,1025):
            a = device.arange(size)
            b = device.ones((size,))
            tx.assert_close(kernel(a,b,scale=2.5), 2.5*a.to_numpy()+1)
        event = device.record_event()
        descriptor = event.descriptor
        assert descriptor.flags == 1 and device.get_event(descriptor) is event
        device.wait(event)
        event.release()
        with pytest.raises(TensorRuntimeError, match="released"):
            device.get_event(descriptor)
        with pytest.raises(TensorRuntimeError, match="released"):
            device.wait(event)
        with pytest.raises(ValueError, match="mismatch"):
            kernel(a,device.ones((128,)),scale=1.)
        offset = device.load(cpu_artifacts["scalar_offset"])
        values = np.arange(129,dtype="int64")
        for delta in ((1<<40)+3,-(1<<40)-3):
            np.testing.assert_array_equal(offset(device.from_numpy(values),delta=delta).to_numpy(),values+delta)
    with device:
        with pytest.raises(TensorRuntimeError, match="previous"):
            kernel(device.arange(129),device.ones((129,)),scale=2.)


def test_native_cpu_rejects_descriptor_before_touching_buffers(cpu_artifacts):
    with tx.Device(provider="cpu") as device:
        kernel = device.load(cpu_artifacts["elementwise"])
        a,b,out = device.arange(129),device.ones((129,)),device.empty((129,))
        values,symbols,launch = kernel._bind((a,b,out),{},include_outputs=True)
        call = BoundCall(device,kernel.manifest,values,symbols,launch)
        error = ErrorDescriptor()
        call.arguments[0].buffer.byte_size = 4
        assert kernel.function(c.byref(call.descriptor),c.byref(error)) == 2
        assert b"insufficient buffer" in error.message
        call.arguments[0].buffer.byte_size = a.nbytes
        call.arguments[0].buffer.strides[0] = 8
        assert kernel.function(c.byref(call.descriptor),c.byref(error)) == 2
        call.arguments[0].buffer.strides[0] = 4
        call.descriptor.struct_size = 8
        assert kernel.function(c.byref(call.descriptor),c.byref(error)) == 1


def test_artifact_abi_is_independent_of_compiler_versions(cpu_artifacts):
    manifest,_ = read_artifact(cpu_artifacts["elementwise"])
    changed = copy.deepcopy(manifest)
    changed["tilelang_version"] = "future-producer"
    changed["tvm_ffi_version"] = "future-producer"
    validate_manifest(changed)
    changed["runtime_abi"]["major"] = 2
    with pytest.raises(ArtifactError, match="runtime ABI"):
        validate_manifest(changed)


def test_executable_descriptor_release_and_session_identity(cpu_artifacts):
    with tx.Device(provider="cpu") as device, tx.Device(provider="cpu") as other:
        kernel = device.load(cpu_artifacts["elementwise"])
        descriptor = kernel.descriptor
        assert descriptor.workspace.byte_size == 0 and descriptor.workspace.alignment == 1
        assert descriptor.workspace.device_type == 1 and descriptor.argument_count == 3
        assert device.get_executable(descriptor) is kernel
        with pytest.raises(TensorRuntimeError, match="different device session"):
            other.get_executable(descriptor)
        changed = ExecutableDescriptor.from_buffer_copy(descriptor)
        changed.workspace.byte_size = 4
        with pytest.raises(TensorRuntimeError, match="metadata"):
            device.get_executable(changed)
        surviving = device.load(cpu_artifacts["elementwise"])
        kernel.release()
        kernel.release()
        with pytest.raises(TensorRuntimeError, match="released"):
            device.get_executable(descriptor)
        with pytest.raises(TensorRuntimeError, match="released"):
            kernel.descriptor
        replacement = device.load(cpu_artifacts["elementwise"])
        assert replacement.descriptor.handle != descriptor.handle
        for executable in (surviving, replacement):
            tx.assert_close(executable(device.arange(129),device.ones((129,))),2*np.arange(129,dtype="float32")+1)
    with device:
        with pytest.raises(TensorRuntimeError, match="different device session"):
            device.get_executable(descriptor)


def test_workspace_contract_rejects_unsupported_scratch(cpu_artifacts):
    manifest,_ = read_artifact(cpu_artifacts["elementwise"])
    assert manifest["runtime_abi"]["minor"] == 1
    for workspace in (None, {"bytes":16,"alignment":1}, {"bytes":0,"alignment":64}, {"bytes":False,"alignment":1}):
        changed = copy.deepcopy(manifest)
        changed["workspace"] = workspace
        with pytest.raises(ArtifactError, match="workspace"):
            validate_manifest(changed)


def test_buffer_metadata_cannot_change_after_descriptor_creation():
    with tx.Device(provider="cpu") as device:
        buffer = device.arange(129)
        for name, value in (("pointer", 0), ("shape", (1,)), ("strides", (8,)), ("dtype", np.dtype('int64')), ("nbytes", 4)):
            with pytest.raises(AttributeError):
                setattr(buffer, name, value)


def test_failed_cpu_entrypoint_load_leaves_no_resources(cpu_artifacts, tmp_path, monkeypatch):
    manifest,files = read_artifact(cpu_artifacts["elementwise"])
    manifest["entrypoint"] = "missing_entrypoint"
    path = tmp_path / "missing.tbin"
    with zipfile.ZipFile(path,"w") as bundle:
        for name,content in files.items():
            bundle.writestr(name,content)
        bundle.writestr("manifest.json",json.dumps(manifest))
    with tx.Device(provider="cpu") as device:
        with pytest.raises(ArtifactError,match="entrypoint unavailable"):
            device.load(path)
        assert not device._modules and not device._executables
        assert not list(Path(device._directory.name).iterdir())
        def unavailable_image(*args, **kwargs):
            raise OSError("missing shared-library dependency")
        monkeypatch.setattr(c,"CDLL",unavailable_image)
        with pytest.raises(ArtifactError,match="image unavailable"):
            device.load(cpu_artifacts["elementwise"])
        assert not device._modules and not device._executables
        assert not list(Path(device._directory.name).iterdir())


def test_cpp_host_uses_public_header_and_cpu_image(cpu_artifacts,tmp_path):
    host=tmp_path / "native-host"
    subprocess.run(["c++","-std=c++17","-O2","-I",str(ROOT / "src/tensor/include"),
                    str(ROOT / "src/tensor/native/host.cpp"),"-ldl","-o",str(host)],check=True,capture_output=True)
    _,files=read_artifact(cpu_artifacts["elementwise"])
    image=tmp_path / "kernel.so";image.write_bytes(files["kernel.so"])
    result=subprocess.run([str(host),"cpu",str(image)],check=True,capture_output=True,text=True)
    assert '"negative_checks":5' in result.stdout
