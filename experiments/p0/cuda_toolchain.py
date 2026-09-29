"""Compiler-only CUDA checks, usable without TileLang or an NVIDIA device."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from experiments.p0.artifact_format import ArtifactError


def resolve_nvcc(nvcc: str | None = None) -> str:
    """An explicit compiler wins, then CUDA_HOME/CUDA_PATH, then PATH."""
    if nvcc is None:
        root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if root:
            name = "nvcc.exe" if os.name == "nt" else "nvcc"
            nvcc = str(Path(root) / "bin" / name)
        else:
            nvcc = "nvcc"
    compiler = shutil.which(nvcc)
    if compiler is None:
        raise ArtifactError(
            f"CUDA compiler unavailable: {nvcc}. Install a full CUDA toolkit and "
            "set CUDA_HOME (or CUDA_PATH), add its bin directory to PATH, or use --nvcc. "
            "Use prepare for source-only work; source bundles are not executables."
        )
    return str(Path(compiler).absolute())


def compile_cubin(source: Path, output: Path, *, arch: str,
                  include_dirs: tuple[Path, ...] = (), nvcc: str | None = None) -> dict:
    compiler = resolve_nvcc(nvcc)
    if not re.fullmatch(r"sm_[0-9]{2,3}", arch):
        raise ArtifactError("arch must be a generic exact SM target, e.g. sm_80")
    started = time.perf_counter()
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                             timeout=30, check=True)
    options = ["--cubin", "-std=c++20", "-O3", "-lineinfo", f"-arch={arch}"]
    if platform.system() == "Windows":
        options += ["-Xcompiler", "/Zc:preprocessor /Zc:__cplusplus"]
    includes = [part for directory in include_dirs for part in ("-I", str(directory))]
    proc = subprocess.run([compiler, *options, *includes, str(source), "-o", str(output)],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise ArtifactError(
            f"nvcc failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}\n"
            "Check the full CUDA toolkit, including runtime and CCCL headers, and the "
            "host C++ compiler. Select it with CUDA_HOME/CUDA_PATH or --nvcc. "
            "Run python -m experiments.p0.artifact_build doctor to check compilation."
        )
    if not output.is_file() or not output.read_bytes().startswith(b"\x7fELF"):
        raise ArtifactError("nvcc did not produce an ELF cubin")
    return {"path": compiler, "nvcc": version.stdout.strip(), "host": platform.platform(),
            "options": options, "seconds": time.perf_counter() - started}


def doctor(*, nvcc: str | None = None, arch: str = "sm_80") -> dict:
    # Actually compile: an nvcc executable alone does not establish that its
    # headers, device compiler, assembler and host compiler are usable together.
    with tempfile.TemporaryDirectory(prefix="tensor-p0-doctor-") as directory:
        root = Path(directory)
        source = root / "probe.cu"
        source.write_text(
            "#include <cuda_runtime.h>\n"
            "#include <cuda_fp16.h>\n"
            "#include <cuda_fp8.h>\n"
            "#include <nv/target>\n"
            'extern "C" __global__ void probe(float* out) {\n'
            "  out[threadIdx.x] = float(threadIdx.x);\n"
            "}\n", encoding="utf-8",
        )
        info = compile_cubin(source, root / "probe.cubin", arch=arch, nvcc=nvcc)
    return {"status": "available", "arch": arch, "compiler": info,
            "check": "cuda_cubin_compilation", "gpu_required": False}
