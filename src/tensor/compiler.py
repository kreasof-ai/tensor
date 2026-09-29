"""Executable compiler boundary: target source and headers in, image bytes out.

Compiler selection belongs to the producer; runtime providers never import
this module or load compiler libraries.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

from tensor.nvrtc import NvrtcCompiler, NvrtcError


class NvccCompiler:
    def __init__(self, path: str | None = None):
        from tensor.doctor import _resolve_nvcc
        self.path = _resolve_nvcc(path)
        try:
            self.version = subprocess.run([self.path, "--version"], capture_output=True, text=True,
                                          check=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"CUDA compiler version check failed: {exc}") from exc

    def options(self, target: str, includes: tuple[Path, ...]) -> list[str]:
        options = ["--cubin", "-std=c++20", "-O3", "-lineinfo", f"-arch={target}"]
        if os.name == "nt":
            options += ["-Xcompiler", "/Zc:preprocessor /Zc:__cplusplus"]
        for root in includes:
            options += ["-I", str(root)]
        return options

    def identity(self, target: str, includes: tuple[Path, ...]) -> dict:
        return {"name": "nvcc", "path": self.path, "version": self.version,
                "options": self.options(target, includes)}

    def compile(self, source: str, target: str, includes: tuple[Path, ...]) -> bytes:
        with tempfile.TemporaryDirectory(prefix="tensor-nvcc-") as directory:
            root = Path(directory)
            cuda, image = root / "kernel.cu", root / "kernel.cubin"
            cuda.write_text(source, encoding="utf-8")
            try:
                result = subprocess.run([self.path, *self.options(target, includes), str(cuda), "-o", str(image)],
                                        capture_output=True, text=True, timeout=300)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"nvcc failed to start or timed out: {exc}") from exc
            if result.returncode:
                raise ValueError(f"nvcc failed ({result.returncode}):\n{result.stderr or result.stdout}")
            binary = image.read_bytes() if image.is_file() else b""
            if not binary.startswith(b"\x7fELF"):
                raise ValueError("nvcc did not produce an ELF cubin")
            return binary


def select_compiler(name: str | None = None, *, nvcc: str | None = None,
                    nvrtc_home: str | Path | None = None):
    # Existing callers supplying --nvcc retain an explicit offline compilation path.
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
