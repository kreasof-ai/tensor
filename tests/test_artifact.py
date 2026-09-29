"""Artifact corruption, compiler isolation, and driver argument/cleanup tests.

The fake driver tests exercise host-side contracts, not GPU correctness.
"""

import ctypes as C
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from experiments.p0 import artifact_run
from experiments.p0.artifact_format import ArtifactError, read_bundle, write_bundle


def manifest(size=129, kind="cubin"):
    return {"format": "tensor.p0.cuda", "format_version": 1, "kind": kind,
            "operation": "relu_2a_plus_b", "dtype": "float32", "size": size,
            "arch": "sm_80", "entrypoint": "elementwise_kernel", "arguments": ["a", "b", "c"],
            "launch": {"grid": [(size + 127) // 128, 1, 1], "block": [128, 1, 1], "shared_memory_bytes": 0}}


def executable(tmp_path, size=129):
    path = tmp_path / "kernel.tbin"
    write_bundle(path, manifest(size), {"kernel.cubin": b"\x7fELFfake-test-payload"})
    return path


def rewrite(path, mutate):
    with zipfile.ZipFile(path) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    mutate(members)
    with zipfile.ZipFile(path, "w") as archive:
        for n, data in members.items():
            archive.writestr(n, data)


def test_rejects_corruption_before_driver_load(tmp_path):
    path = executable(tmp_path)
    rewrite(path, lambda files: files.update({"kernel.cubin": b"\x7fELFcorrupted"}))
    with pytest.raises(ArtifactError, match="hash mismatch"):
        read_bundle(path)


@pytest.mark.parametrize("field,value", [("format_version", 2), ("size", 0), ("arch", "auto"),
                                         ("entrypoint", "other"), ("arguments", ["a", "b"]),
                                         ("launch", {"grid": [1, 1, 1]})])
def test_rejects_incompatible_contract(tmp_path, field, value):
    path = executable(tmp_path)
    def mutate(files):
        data = json.loads(files["manifest.json"])
        data[field] = value
        files["manifest.json"] = json.dumps(data).encode()
    rewrite(path, mutate)
    with pytest.raises(ArtifactError):
        read_bundle(path)


def test_source_is_not_executable_and_paths_cannot_escape(tmp_path):
    path = tmp_path / "source.zip"
    write_bundle(path, manifest(kind="source"), {"kernel.cu": b"source"})
    with pytest.raises(ArtifactError, match="expected cubin"):
        read_bundle(path, kind="cubin")
    with pytest.raises(ArtifactError, match="invalid payload path"):
        write_bundle(tmp_path / "bad.zip", manifest(kind="source"),
                     {"kernel.cu": b"source", "include/../../escape.h": b"bad"})


def test_extra_archive_members_are_rejected(tmp_path):
    path = executable(tmp_path)
    rewrite(path, lambda files: files.update({"unexpected": b"extra"}))
    with pytest.raises(ArtifactError, match="members do not match"):
        read_bundle(path)


def test_previous_artifact_is_preserved(tmp_path):
    path = executable(tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_bundle(path, manifest(), {"kernel.cubin": b"\x7fELFnew"})
    assert path.read_bytes() == before


def test_consumer_imports_and_guard_in_fresh_process():
    # Fresh interpreter avoids contamination from codegen tests importing TileLang.
    code = """
import sys
from experiments.p0.artifact_run import NoCompilerImports, assert_no_compiler_imports
assert_no_compiler_imports()
sys.meta_path.insert(0, NoCompilerImports())
import numpy
for name in ('tilelang', 'tvm', 'tvm_ffi', 'torch'):
    try:
        __import__(name)
    except ImportError as exc:
        assert 'forbids importing' in str(exc)
    else:
        raise AssertionError(name)
assert_no_compiler_imports()
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


class FakeDriver:
    """Stores device bytes and checks actual ctypes packing and cleanup order."""
    def __init__(self):
        self.memory = {}
        self.calls = []
        self.upload_complete = False

    def device_info(self, ordinal):
        return 0, {"arch": "sm_80", "name": "test double"}

    def call(self, name, *args):
        self.calls.append(name)
        if name in ("cuCtxGetCurrent", "cuCtxCreate_v2", "cuStreamCreate", "cuModuleLoadData", "cuModuleGetFunction"):
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = 123
        elif name == "cuMemAlloc_v2":
            address = 1000 + len(self.memory)
            C.cast(args[0], C.POINTER(C.c_uint64))[0] = address
            self.memory[address] = bytearray(args[1])
        elif name == "cuMemcpyHtoD_v2":
            self.memory[args[0].value][:] = C.string_at(args[1], args[2])
        elif name == "cuLaunchKernel":
            assert self.upload_complete, "upload stream must complete before a non-blocking launch"
            assert args[1:7] == (2, 1, 1, 128, 1, 1)
            assert args[7] == 0
            params = args[9]
            addresses = [C.cast(params[i], C.POINTER(C.c_uint64))[0] for i in range(3)]
            a, b, out = [np.frombuffer(self.memory[addr], dtype=np.float32) for addr in addresses]
            out[:] = np.maximum(2 * a + b, 0)
        elif name == "cuMemcpyDtoH_v2":
            C.memmove(args[0], bytes(self.memory[args[1].value]), args[2])
        elif name == "cuMemFree_v2":
            del self.memory[args[0].value]
        elif name == "cuStreamSynchronize" and args[0] is None:
            self.upload_complete = True


def test_runtime_packs_pointer_values_and_releases_resources(tmp_path, monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(artifact_run, "Driver", lambda: driver)
    # Other test modules may import the compiler; isolation has its own process test.
    monkeypatch.setattr(artifact_run, "assert_no_compiler_imports", lambda: None)
    result = artifact_run.validate(executable(tmp_path), iters=2)
    assert result["max_abs_error"] == 0
    assert driver.memory == {}
    assert driver.calls[-3:] == ["cuStreamDestroy_v2", "cuCtxDestroy_v2", "cuCtxSetCurrent"]


def test_validation_failure_still_releases_resources(tmp_path, monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(artifact_run, "Driver", lambda: driver)
    monkeypatch.setattr(artifact_run, "assert_no_compiler_imports", lambda: None)
    original = driver.call
    def fail(name, *args):
        if name == "cuLaunchKernel":
            raise artifact_run.CudaError("injected launch failure")
        return original(name, *args)
    driver.call = fail
    with pytest.raises(artifact_run.CudaError, match="launch failure"):
        artifact_run.validate(executable(tmp_path), iters=2)
    assert driver.memory == {}
    assert "cuModuleUnload" in driver.calls
    assert "cuCtxDestroy_v2" in driver.calls


def test_target_mismatch_fails_before_loading_module(tmp_path, monkeypatch):
    driver = FakeDriver()
    driver.device_info = lambda ordinal: (0, {"arch": "sm_90"})
    monkeypatch.setattr(artifact_run, "Driver", lambda: driver)
    monkeypatch.setattr(artifact_run, "assert_no_compiler_imports", lambda: None)
    with pytest.raises(ArtifactError, match="target mismatch"):
        artifact_run.validate(executable(tmp_path), iters=2)
    assert "cuModuleLoadData" not in driver.calls
    assert "cuCtxDestroy_v2" in driver.calls
