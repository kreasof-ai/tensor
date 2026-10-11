"""Resolve and select CUDA compiler toolchains."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tensor.compiler.cuda.nvrtc import NvrtcCompiler, NvrtcError


def _resolve_nvcc(explicit: str | None) -> str:
    if explicit is not None:
        candidate = explicit
    elif root := os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        candidate = str(
            Path(root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        )
    else:
        candidate = "nvcc"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(f"CUDA compiler {candidate!r} was not found")
    return str(Path(resolved).resolve())


def select_compiler(
    name: str | None = None,
    *,
    nvcc: str | None = None,
    nvrtc_home: str | Path | None = None,
):
    # Existing callers supplying --nvcc retain an explicit offline compilation path.
    from tensor.compiler.cuda.nvcc import NvccCompiler

    selected = name or ("nvcc" if nvcc else "nvrtc")
    if selected == "nvrtc":
        if nvcc:
            raise NvrtcError("--nvcc cannot be combined with --compiler nvrtc")
        return NvrtcCompiler(nvrtc_home)
    if selected == "nvcc":
        if nvrtc_home:
            raise ValueError("--nvrtc-home cannot be combined with --compiler nvcc")
        return NvccCompiler(nvcc)
    raise ValueError(f"unsupported compiler {selected!r}; choose nvrtc or nvcc")
