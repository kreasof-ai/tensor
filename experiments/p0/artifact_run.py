"""P0 compiler-free consumer. Only NumPy and the NVIDIA driver are needed.

inspect reads manifests without a device; doctor probes the driver; validate
loads a prebuilt cubin in this fresh process and checks it against NumPy.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse
import ctypes as C
import importlib.abc
import json
import platform
import socket
import sys
from pathlib import Path

from experiments.p0.artifact_format import ArtifactError, read_bundle, sha256
from experiments.p0.cuda_driver import CudaError, CudaUnavailable, Driver, Session
from experiments.p0.provenance import snapshot

FORBIDDEN = {"tilelang", "tvm", "tvm_ffi", "torch"}


class NoCompilerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ImportError(f"compiler-free consumer forbids importing {fullname}")
        return None


def assert_no_compiler_imports():
    loaded = sorted({name.split(".")[0] for name in sys.modules} & FORBIDDEN)
    if loaded:
        raise ArtifactError(f"compiler modules already loaded: {loaded}")


def validate(path: Path, *, device=0, seed=0, iters=20) -> dict:
    assert_no_compiler_imports()
    gate = NoCompilerImports()
    sys.meta_path.insert(0, gate)
    try:
        import numpy as np

        manifest, files = read_bundle(path, kind="cubin")
        rng = np.random.default_rng(seed)
        size = manifest["size"]
        a = rng.standard_normal(size).astype(np.float32)
        b = rng.standard_normal(size).astype(np.float32)
        expected = np.maximum(2 * a + b, 0)
        if iters <= 0:
            raise ArtifactError("iters must be positive")
        actual = np.empty_like(a)
        timings = {}
        started = time.perf_counter()
        driver = Driver()
        with Session(driver, device) as session:
            if session.info["arch"] != manifest["arch"]:
                raise ArtifactError(f"target mismatch: artifact {manifest['arch']}, device {session.info['arch']}; "
                                    "this experiment requires an exact SM match")
            timings["driver_context_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            function = session.load(files["kernel.cubin"], manifest["entrypoint"])
            timings["module_load_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            da, db, dc = [session.allocate(a.nbytes) for _ in range(3)]
            driver.call("cuMemcpyHtoD_v2", da, C.c_void_p(a.ctypes.data), a.nbytes)
            driver.call("cuMemcpyHtoD_v2", db, C.c_void_p(b.ctypes.data), b.nbytes)
            # NaNs reveal unwritten outputs (including missed tail elements).
            actual.fill(np.nan)
            driver.call("cuMemcpyHtoD_v2", dc, C.c_void_p(actual.ctypes.data), actual.nbytes)
            # Pageable HtoD may return after staging, before DMA finishes.
            # Finish the default copy stream before our non-blocking launch
            # stream uses those buffers; do not rely on implicit ordering.
            driver.call("cuStreamSynchronize", None)
            timings["allocate_upload_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            session.launch(function, [da, db, dc], manifest["launch"])
            session.synchronize()
            timings["first_launch_and_sync_seconds"] = time.perf_counter() - started
            driver.call("cuMemcpyDtoH_v2", C.c_void_p(actual.ctypes.data), dc, actual.nbytes)
            timings["process_entry_to_first_result_seconds"] = time.perf_counter() - PROCESS_START
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            samples = []
            for _ in range(iters):
                started = time.perf_counter()
                session.launch(function, [da, db, dc], manifest["launch"])
                session.synchronize()
                samples.append(time.perf_counter() - started)
            timings["warm_launch_and_sync_median_seconds"] = float(np.median(samples))
            device_info = session.info
        assert_no_compiler_imports()
        return {"status": "passed", "numerics": "numpy_allclose", "rtol": 1e-6, "atol": 1e-6,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "size": size, "seed": seed, "iterations": iters, "timings": timings,
                "artifact_sha256": sha256(path.read_bytes()), "producer": manifest.get("producer"),
                "consumer": {**snapshot(), "host": platform.platform(), "python": platform.python_version(),
                             "hostname": socket.gethostname(),
                             "numpy": np.__version__, "device": device_info},
                "compiler_imports": [], "compiler_import_guard": True}
    finally:
        sys.meta_path.remove(gate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("inspect")
    cmd.add_argument("artifact", type=Path)
    sub.add_parser("doctor")
    cmd = sub.add_parser("validate")
    cmd.add_argument("artifact", type=Path)
    cmd.add_argument("--device", type=int, default=0)
    cmd.add_argument("--seed", type=int, default=0)
    cmd.add_argument("--iters", type=int, default=20)
    cmd.add_argument("--report", type=Path)
    args = parser.parse_args()
    code = 0
    try:
        if args.command == "inspect":
            result, _ = read_bundle(args.artifact)
        elif args.command == "doctor":
            _, info = Driver().device_info()
            result = {"status": "available", "device": info}
        else:
            if args.iters <= 0:
                raise ArtifactError("iters must be positive")
            result = validate(args.artifact, device=args.device, seed=args.seed, iters=args.iters)
    except CudaUnavailable as exc:
        result, code = {"status": "skipped", "reason": str(exc), "gpu_execution": "unverified"}, 2
    except (ArtifactError, CudaError, OSError, AssertionError, ImportError) as exc:
        result, code = {"status": "failed", "error": str(exc)}, 1
    if getattr(args, "report", None):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
