"""Check the compiler and CUDA environment without importing it at CLI startup."""

from __future__ import annotations

import ctypes as c
import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PINNED_PACKAGES = {"tilelang": "0.1.14", "apache-tvm-ffi": "0.1.12"}
RUNTIME_PACKAGES = {"numpy": "2.5.3"}
TARGET = re.compile(r"sm_[0-9]{2,3}\Z")


def _check(status: str, detail: str, *, hint: str | None = None, **data: object) -> dict:
    result = {"status": status, "detail": detail, **data}
    if hint:
        result["hint"] = hint
    return result


def check_packages() -> dict:
    if sys.version_info[:2] != (3, 12):
        return _check("error", f"Python {sys.version_info.major}.{sys.version_info.minor}; Python 3.12 is required",
                      hint="Use Python 3.12 and reinstall Tensor.")
    installed = {}
    for name, expected in PINNED_PACKAGES.items():
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = None
    wrong = [f"{name}=={installed[name] or 'missing'} (need {expected})"
             for name, expected in PINNED_PACKAGES.items() if installed[name] != expected]
    if wrong:
        return _check("error", "; ".join(wrong),
                      hint="Install tilelang==0.1.14 and apache-tvm-ffi==0.1.12 (or run uv sync --locked in this checkout).",
                      installed=installed)
    return _check("ok", "pinned compiler packages installed", installed=installed)


def check_runtime_packages() -> dict:
    if sys.version_info[:2] != (3, 12):
        return _check("error", "runtime requires Python 3.12")
    try:
        installed = version("numpy")
    except PackageNotFoundError:
        installed = None
    if installed != RUNTIME_PACKAGES["numpy"]:
        return _check("error", f"numpy=={installed or 'missing'} (need 2.5.3)",
                      hint="Install the Tensor wheel with its NumPy dependency.")
    return _check("ok", "NumPy runtime installed", numpy=installed)


def check_provider() -> dict:
    try:
        from tilelang.backend import list_backends

        providers = list_backends()
        cuda = providers.get("cuda")
        if cuda is None or "cuda" not in cuda.target_kinds:
            return _check("error", "TileLang CUDA backend is not registered",
                          hint="Check the TileLang installation and its backend manifest.",
                          registered=sorted(providers))
        return _check("ok", "TileLang CUDA backend registered", registered=sorted(providers))
    except Exception as exc:
        return _check("error", f"TileLang backend import failed: {type(exc).__name__}: {exc}",
                      hint="Check the pinned TileLang installation and native library dependencies.")


def _cuda_call(lib: object, name: str, argtypes: list, *args: object) -> None:
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = c.c_int
    code = fn(*args)
    if code:
        raise RuntimeError(f"{name} returned CUDA error {code}")


def check_device(ordinal: int) -> dict:
    if ordinal < 0:
        return _check("error", "device ordinal must be non-negative")
    if c.sizeof(c.c_void_p) != 8:
        return _check("error", "CUDA driver probing requires 64-bit Python")
    try:
        lib = c.WinDLL("nvcuda.dll") if os.name == "nt" else c.CDLL("libcuda.so.1")
        integer, uint, pointer = c.c_int, c.c_uint, c.c_void_p
        _cuda_call(lib, "cuInit", [uint], 0)
        count = integer()
        _cuda_call(lib, "cuDeviceGetCount", [c.POINTER(integer)], c.byref(count))
        if ordinal >= count.value:
            return _check("unavailable", f"CUDA device {ordinal} unavailable ({count.value} detected)",
                          hint="Connect an NVIDIA GPU, or pass --target sm_XX to check a build host.")
        device = integer()
        _cuda_call(lib, "cuDeviceGet", [c.POINTER(integer), integer], c.byref(device), ordinal)
        major, minor, driver = integer(), integer(), integer()
        for attribute, output in ((75, major), (76, minor)):
            _cuda_call(lib, "cuDeviceGetAttribute", [c.POINTER(integer), integer, integer],
                       c.byref(output), attribute, device)
        _cuda_call(lib, "cuDriverGetVersion", [c.POINTER(integer)], c.byref(driver))
        name = c.create_string_buffer(256)
        _cuda_call(lib, "cuDeviceGetName", [pointer, integer, integer],
                   name, len(name), device)
        return _check("ok", f"{name.value.decode(errors='replace')} (sm_{major.value}{minor.value})",
                      arch=f"sm_{major.value}{minor.value}", ordinal=ordinal,
                      driver_version=driver.value)
    except (OSError, AttributeError, RuntimeError) as exc:
        return _check("unavailable", f"CUDA driver unavailable: {exc}",
                      hint="Install a working NVIDIA driver, or pass --target sm_XX to check a build host.")


def _resolve_nvcc(explicit: str | None) -> str:
    if explicit is not None:
        candidate = explicit
    elif root := os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        candidate = str(Path(root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))
    else:
        candidate = "nvcc"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(f"CUDA compiler {candidate!r} was not found")
    return str(Path(resolved).resolve())


def check_toolchain(target: str | None, nvcc: str | None) -> dict:
    if target is None:
        return _check("skipped", "no CUDA target selected",
                      hint="Connect a GPU or supply --target sm_XX for a GPU-free build host.")
    try:
        compiler = _resolve_nvcc(nvcc)
        info = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                              timeout=30, check=True)
        with tempfile.TemporaryDirectory(prefix="tensor-doctor-") as directory:
            source = Path(directory) / "probe.cu"
            output = Path(directory) / "probe.cubin"
            source.write_text(
                "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n"
                "#include <cuda_fp8.h>\n#include <nv/target>\n"
                'extern "C" __global__ void probe(float* out) { out[threadIdx.x] = float(threadIdx.x); }\n',
                encoding="utf-8",
            )
            command = [compiler, "--cubin", "-std=c++20", f"-arch={target}",
                       str(source), "-o", str(output)]
            if os.name == "nt":
                command += ["-Xcompiler", "/Zc:preprocessor /Zc:__cplusplus"]
            built = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if built.returncode:
                diagnostic = (built.stderr or built.stdout).strip()
                return _check("error", f"nvcc could not compile a CUDA probe: {diagnostic}",
                              hint="Install the full CUDA toolkit, including runtime/CCCL headers and a host C++ compiler.",
                              path=compiler, target=target)
            if not output.is_file() or not output.read_bytes().startswith(b"\x7fELF"):
                return _check("error", "nvcc completed without producing an ELF cubin",
                              path=compiler, target=target)
        return _check("ok", f"CUDA cubin compiled for {target}", path=compiler,
                      target=target, nvcc=info.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        return _check("error", f"CUDA toolchain unavailable: {exc}",
                      hint="Install a full CUDA toolkit and set CUDA_HOME, or pass --nvcc.")


def check_nvrtc(target: str | None, home: str | Path | None = None) -> dict:
    if target is None:
        return _check("skipped", "no CUDA target selected", hint="Supply --target sm_XX.")
    try:
        from tensor.compiler.nvrtc import NvrtcCompiler
        compiler = NvrtcCompiler(home)
        # Check real code generation, not just loading a library or reading its version.
        compiler.compile('extern "C" __global__ void probe(float* out) { out[threadIdx.x] = 1; }',
                         target, ())
        return _check("ok", f"NVRTC {compiler.version} compiled a cubin for {target}",
                      compiler="nvrtc", root=str(compiler.root), target=target)
    except (OSError, ValueError) as exc:
        return _check("error", str(exc), hint="Install the pinned bundle with tools/bootstrap_nvrtc.py "
                      "and set TENSOR_NVRTC_HOME; --compiler nvcc selects the offline compiler.")


def diagnose(*, target: str | None = None, device: int = 0, nvcc: str | None = None,
             compiler: str | None = None, nvrtc_home: str | Path | None = None) -> dict:
    if target is not None and not TARGET.fullmatch(target):
        return {"status": "needs_setup", "target": target,
                "checks": {"target": _check("error", "target must be an exact CUDA SM, e.g. sm_86")}}
    if device < 0:
        return {"status": "needs_setup", "target": target,
                "checks": {"device": _check("error", "device ordinal must be non-negative")}}
    runtime = check_runtime_packages()
    packages = check_packages()
    provider = check_provider() if packages["status"] == "ok" else _check(
        "skipped", "compiler packages must be fixed before checking providers")
    detected = check_device(device)
    selected = target or detected.get("arch")
    chosen = compiler or ("nvcc" if nvcc else "nvrtc")
    if chosen == "nvrtc" and nvcc or chosen == "nvcc" and nvrtc_home:
        toolchain = _check("error", "compiler selection conflicts with --nvcc or --nvrtc-home")
    elif chosen == "nvrtc":
        toolchain = check_nvrtc(selected, nvrtc_home)
    elif chosen == "nvcc":
        toolchain = check_toolchain(selected, nvcc)
    else:
        toolchain = _check("error", "compiler must be nvrtc or nvcc")
    build_ready = all(check["status"] == "ok" for check in (runtime, packages, provider, toolchain))
    device_ready = detected["status"] == "ok" and runtime["status"] == "ok"
    if build_ready and device_ready and selected == detected["arch"]:
        status = "ready"
    elif build_ready:
        status = "build_ready"
    elif device_ready:
        status = "run_ready"
    else:
        status = "needs_setup"
    return {"status": status, "target": selected,
            "checks": {"runtime": runtime, "packages": packages, "provider": provider,
                       "device": detected, "toolchain": toolchain}}


def render(report: dict) -> str:
    lines = [f"Tensor doctor: {report['status'].replace('_', ' ')}"]
    if report.get("target"):
        lines.append(f"Target: {report['target']}")
    for name, check in report["checks"].items():
        lines.append(f"{name:9} {check['status']:11} {check['detail']}")
        if check.get("hint"):
            lines.append(f"          → {check['hint']}")
    return "\n".join(lines)
