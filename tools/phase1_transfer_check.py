"""Validate a downloaded Phase 1 product bundle using only Tensor and NumPy.

Run this script with the clean consumer interpreter after installing the
downloaded wheel. The bundle must contain producer.json, five .tbin files,
and the wheel from the Phase 1 GitHub Actions producer.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes as c
import hashlib
import importlib.abc
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


class NoCompilerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"tilelang", "tvm", "tvm_ffi", "torch"}:
            raise AssertionError(f"consumer tried to import {fullname}")


class GPUProducer:
    """A small framework-free DLPack producer for a Tensor-owned allocation."""

    def __init__(self, buffer, *, versioned):
        from tensor.dlpack import (DLDataType, DLDevice, DLManagedTensor,
                                   DLManagedTensorVersioned, DLPackVersion, DLTensor)

        self.buffer, self.versioned = buffer, versioned
        self.deletes, self.streams = 0, []
        self.shape = (c.c_int64*len(buffer.shape))(*buffer.shape)
        self.strides = (c.c_int64*len(buffer.shape))(*(value//buffer.dtype.itemsize for value in buffer.strides))
        self.deleter = c.CFUNCTYPE(None, c.c_void_p)(self._release)
        tensor = DLTensor(buffer.pointer, DLDevice(2, buffer.device.ordinal), len(buffer.shape),
                          DLDataType(2, 32, 1), self.shape, self.strides, 0)
        pointer = c.cast(self.deleter, c.c_void_p)
        self.tensor = (DLManagedTensorVersioned(DLPackVersion(1, 0), None, pointer, 0, tensor)
                       if versioned else DLManagedTensor(tensor, None, pointer))

    def _release(self, address):
        assert address == c.addressof(self.tensor)
        self.deletes += 1

    def __dlpack_device__(self):
        return 2, self.buffer.device.ordinal

    def __dlpack__(self, *, stream, max_version=None):
        self.streams.append(stream)
        self.buffer._check()
        # The consumer uses this same session's stream; no extra wait is needed.
        assert stream == (self.buffer.device.stream.value or 1)
        create = c.pythonapi.PyCapsule_New
        create.argtypes, create.restype = [c.c_void_p, c.c_char_p, c.c_void_p], c.py_object
        return create(c.addressof(self.tensor), b"dltensor_versioned" if self.versioned else b"dltensor", None)


def check(bundle: Path, *, require_two_hosts: bool = True, legacy_artifact: Path | None = None):
    started = time.perf_counter()
    sys.meta_path.insert(0, NoCompilerImports())
    import numpy as np
    import tensor as tx
    from tensor.artifact import read_artifact
    from tensor.cli import main as cli

    packages = {item.metadata["Name"].lower(): item.version for item in metadata.distributions()}
    if packages != {"numpy": "2.5.3", "tensor-workspace": "0.1.0"}:
        raise ValueError(f"use a clean Tensor/NumPy consumer environment: {packages}")
    provenance_paths = list(bundle.rglob("producer.json"))
    if len(provenance_paths) != 1:
        raise ValueError("bundle must contain exactly one producer.json")
    provenance_path = provenance_paths[0]
    producer = json.loads(provenance_path.read_text())
    checkout = Path(__file__).resolve().parents[1]
    if producer["lock_sha256"] != hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest():
        raise ValueError("producer and consumer dependency locks differ")
    wheels = list(bundle.rglob(producer["wheel"]))
    if len(wheels) != 1 or hashlib.sha256(wheels[0].read_bytes()).hexdigest() != producer["wheel_sha256"]:
        raise ValueError("downloaded wheel hash mismatch")
    expected = {"elementwise", "gemm_relu", "dynamic_affine", "dynamic_gemm", "scalar_offset"}
    if set(producer["artifacts"]) != {f"{name}.tbin" for name in expected}:
        raise ValueError("the complete five-artifact acceptance matrix is required")
    artifacts = {}
    for name, digest in producer["artifacts"].items():
        artifact = provenance_path.parent / name
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
        manifest, _ = read_artifact(artifact)
        if manifest["format_version"] != 2 or manifest["target"] != producer["target"]:
            raise ValueError(f"artifact metadata mismatch: {name}")
        if manifest["source_sha256"] != hashlib.sha256((checkout / "examples" / f"{artifact.stem}.py").read_bytes()).hexdigest():
            raise ValueError(f"producer and consumer source hashes differ: {name}")
        artifacts[artifact.stem] = artifact
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != producer["commit"]:
        raise ValueError("consumer checkout must match the producer commit")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if require_two_hosts and (producer["git_dirty"] is not False or dirty):
        raise ValueError("two-host verification requires clean producer and consumer checkouts")
    two_hosts = producer["host"] != socket.gethostname()
    if require_two_hosts and not two_hosts:
        raise ValueError("producer and consumer must be different hosts")
    rng = np.random.default_rng(2026)
    records, diagnostics = [], []

    def compare(label, output, expected, **tolerances):
        actual = output.to_numpy()
        np.testing.assert_allclose(actual, expected, **tolerances)
        records.append({"case": label, "shape": list(actual.shape),
                        "max_abs_error": float(np.max(np.abs(actual.astype("float64")-expected.astype("float64"))))})

    def reject(label, operation, expected_cause):
        try:
            operation()
        except (ValueError, TypeError, BufferError) as exc:
            if expected_cause not in str(exc):
                raise AssertionError(f"{label}: wrong cause: {exc}") from exc
            diagnostics.append({"case": label, "cause": str(exc)})
        else:
            raise AssertionError(f"{label}: invalid input was accepted")

    with tx.Device() as device:
        kernels = {name: device.load(path) for name, path in artifacts.items()}
        a = np.arange(129, dtype="float32")
        da = device.from_numpy(a)
        compare("static_elementwise", kernels["elementwise"](da, da), 3*a)
        first_kernel = time.perf_counter()-started
        for size in (1, 127, 128, 129, 1025):
            for scale in (2.5, -.75):
                a, b = (rng.standard_normal(size).astype("float32") for _ in range(2))
                output = kernels["dynamic_affine"](device.from_numpy(a), device.from_numpy(b), scale=scale)
                compare(f"dynamic_affine_{size}_{scale}", output, np.maximum(scale*a+b, 0), rtol=1e-6, atol=1e-6)
        for rows in (1, 31, 32, 33, 65):
            a = rng.normal(0, .2, (rows, 32)).astype("float16")
            b = rng.normal(0, .2, (32, 32)).astype("float16")
            expected = (a.astype("float32")@b.astype("float32")).astype("float16")
            compare(f"dynamic_gemm_{rows}", kernels["dynamic_gemm"](device.from_numpy(a), device.from_numpy(b)),
                    expected, rtol=1e-2, atol=1e-2)
        a, b = (rng.normal(0, .2, (64, 64)).astype("float16") for _ in range(2))
        bias = rng.normal(0, .2, 64).astype("float16")
        expected = np.maximum(a.astype("float32")@b.astype("float32")+bias.astype("float32"), 0).astype("float16")
        compare("static_gemm_relu", kernels["gemm_relu"](device.from_numpy(a), device.from_numpy(b), device.from_numpy(bias)),
                expected, rtol=1e-2, atol=1e-2)
        a = np.arange(129, dtype="int64")
        for delta in ((1 << 40)+3, -(1 << 40)+3):
            compare(f"int64_delta_{delta}", kernels["scalar_offset"](device.from_numpy(a), delta=delta), a+delta,
                    rtol=0, atol=0)
        da = device.arange(129)
        for versioned in (False, True):
            source = GPUProducer(da, versioned=versioned)
            imported = device.from_dlpack(source)
            assert imported.pointer == da.pointer and not imported.owned
            compare(f"gpu_dlpack_versioned_{versioned}", kernels["dynamic_affine"](imported, imported, scale=2),
                    3*np.arange(129, dtype="float32"))
            imported.release()
            assert source.deletes == 1
        reject("missing_scalar", lambda: kernels["dynamic_affine"](da, da), "missing kernel arguments")
        reject("shape_conflict", lambda: kernels["dynamic_affine"](da, device.ones((128,)), scale=2), "dimension size mismatch")
        reject("explicit_dimension_conflict", lambda: kernels["dynamic_affine"](da, da, scale=2, size=128), "dimension size mismatch")
        reject("nonfinite_scalar", lambda: kernels["dynamic_affine"](da, da, scale=float("inf")), "finite")
        reject("wrong_dtype", lambda: kernels["dynamic_affine"](device.ones((129,), "float16"), da, scale=2), "dtype float32")
        reject("integer_overflow", lambda: kernels["scalar_offset"](device.arange(129, "int64"), delta=1 << 63), "int64 range")
        reject("noninteger_scalar", lambda: kernels["scalar_offset"](device.arange(129, "int64"), delta=1.25), "integer")
        reject("unknown_argument", lambda: kernels["dynamic_affine"](da, da, scale=2, other=3), "unknown kernel argument")
        if legacy_artifact:
            legacy_manifest, _ = read_artifact(legacy_artifact)
            assert legacy_manifest["format_version"] == 1
            compare("v1_artifact_compatibility", device.load(legacy_artifact)(da, da), 3*np.arange(129, dtype="float32"))
        arguments, dimensions, _ = kernels["dynamic_affine"].prepare(da, da, scale=2.5)
        benchmark = tx.bench(kernels["dynamic_affine"], arguments, iters=100, **dimensions)
        device_info = device.info
    with tempfile.TemporaryDirectory(prefix="tensor-transfer-cli-") as directory:
        root = Path(directory)
        a = np.arange(129, dtype="float32")
        np.save(root / "a.npy", a)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = cli(["run", str(artifacts["dynamic_affine"]), "--input", f"a={root}/a.npy",
                          "--input", f"b={root}/a.npy", "--scalar", "scale=2.5", "--out-dir", str(root / "out")])
        assert status == 0
        cli_report = json.loads(output.getvalue())
        np.testing.assert_array_equal(np.load(root / "out/c.npy"), 3.5*a)
    imports = sorted({name.split(".")[0] for name in sys.modules} & {"tilelang", "tvm", "tvm_ffi", "torch"})
    assert not imports
    return {"status": "passed", "producer": producer, "consumer_host": socket.gethostname(),
            "commit": revision, "two_hosts": two_hosts, "packages": packages, "compiler_imports": imports,
            "compiler_import_guard": True, "device": device_info, "records": records, "diagnostics": diagnostics,
            "clean_matching_checkouts": not dirty and producer["git_dirty"] is False,
            "legacy_artifact_sha256": hashlib.sha256(legacy_artifact.read_bytes()).hexdigest() if legacy_artifact else None,
            "first_kernel_seconds_from_script_check_entry": first_kernel,
            "benchmark": benchmark, "cli_first_result_seconds": cli_report["timings"]["first_result_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-same-host", action="store_true")
    parser.add_argument("--legacy-artifact", type=Path)
    args = parser.parse_args()
    report = check(args.bundle, require_two_hosts=not args.allow_same_host, legacy_artifact=args.legacy_artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "cases": len(report["records"]),
                      "diagnostics": len(report["diagnostics"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
