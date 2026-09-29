"""CUDA C++ to cubin through a pinned, locally bundled NVRTC C API.

No framework imports, CUDA driver calls, subprocesses, host compiler, or
implicit system header paths. TileLang's nvrtc_std shim supplies the device
standard-library definitions needed by its templates.
"""

from __future__ import annotations

import ctypes as c
import hashlib
import os
from pathlib import Path


class NvrtcError(ValueError):
    pass


def bundle_root(override: str | Path | None = None) -> Path:
    return Path(override or os.environ.get("TENSOR_NVRTC_HOME") or
                (Path.home() / ".cache" / "tensor" / "nvrtc-12.9")).resolve()


class NvrtcCompiler:
    def __init__(self, root: str | Path | None = None):
        self.root = bundle_root(root)
        library = self.root / "lib" / ("nvrtc64_120_0.dll" if os.name == "nt" else "libnvrtc.so.12")
        if not library.is_file():
            raise NvrtcError(f"NVRTC bundle missing at {self.root}; run python tools/bootstrap_nvrtc.py "
                             f"--out {self.root}, or set TENSOR_NVRTC_HOME")
        self._dll_directory = os.add_dll_directory(str(library.parent)) if os.name == "nt" else None
        try:
            builtins = library.parent / ("nvrtc-builtins64_129.dll" if os.name == "nt" else "libnvrtc-builtins.so.12.9")
            # NVRTC opens builtins by basename lazily. Preload the exact sibling
            # library so relocation works without PATH/LD_LIBRARY_PATH or CUDA.
            self.builtins = c.CDLL(str(builtins))
            self.lib = c.CDLL(str(library))
            signatures = {
                "nvrtcVersion": [c.POINTER(c.c_int), c.POINTER(c.c_int)],
                "nvrtcCreateProgram": [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p, c.c_int,
                                       c.POINTER(c.c_char_p), c.POINTER(c.c_char_p)],
                "nvrtcCompileProgram": [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)],
                "nvrtcGetProgramLogSize": [c.c_void_p, c.POINTER(c.c_size_t)],
                "nvrtcGetProgramLog": [c.c_void_p, c.c_void_p],
                "nvrtcGetCUBINSize": [c.c_void_p, c.POINTER(c.c_size_t)],
                "nvrtcGetCUBIN": [c.c_void_p, c.c_void_p],
                "nvrtcDestroyProgram": [c.POINTER(c.c_void_p)],
            }
            for name, parameters in signatures.items():
                function = getattr(self.lib, name)
                function.argtypes, function.restype = parameters, c.c_int
            self.lib.nvrtcGetErrorString.argtypes = [c.c_int]
            self.lib.nvrtcGetErrorString.restype = c.c_char_p
            major, minor = c.c_int(), c.c_int()
            self._call("nvrtcVersion", c.byref(major), c.byref(minor))
            self.version = f"{major.value}.{minor.value}"
            if self.version != "12.9":
                raise NvrtcError(f"NVRTC {self.version} found; this profile requires 12.9")
        except (OSError, AttributeError) as exc:
            raise NvrtcError(f"NVRTC library unavailable: {exc}") from exc

    def _call(self, name, *arguments):
        status = getattr(self.lib, name)(*arguments)
        if status:
            label = self.lib.nvrtcGetErrorString(status).decode(errors="replace")
            raise NvrtcError(f"{name}: {label} ({status})")

    def options(self, target: str, includes: tuple[Path, ...]) -> list[str]:
        from tensor.artifact import TARGET
        if not TARGET.fullmatch(target):
            raise NvrtcError("NVRTC target must be an exact SM, e.g. sm_86")
        roots = (*includes, self.root / "include")
        if any(not root.is_dir() for root in roots):
            raise NvrtcError("NVRTC or TileLang compiler headers are missing")
        return [f"--gpu-architecture={target}", "--std=c++20", "--device-as-default-execution-space",
                "--generate-line-info", *[f"--include-path={root.resolve()}" for root in roots]]

    def identity(self, target: str, includes: tuple[Path, ...]) -> dict:
        # Hash the actual loaded library and builtins, including patch versions.
        libraries = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((self.root / "lib").iterdir()) if path.is_file() and not path.is_symlink()}
        return {"name": "nvrtc", "version": self.version, "libraries": libraries,
                "options": self.options(target, includes)}

    def compile(self, source: str, target: str, includes: tuple[Path, ...]) -> bytes:
        options = self.options(target, includes)
        shim = '#include <tl_templates/cuda/nvrtc_std.h>\n' if includes else ""
        program = c.c_void_p()
        self._call("nvrtcCreateProgram", c.byref(program), (shim + source).encode(),
                   b"tensor_kernel.cu", 0, None, None)
        try:
            encoded = [option.encode() for option in options]
            pointers = (c.c_char_p * len(encoded))(*encoded)
            status = self.lib.nvrtcCompileProgram(program, len(encoded), pointers)
            if status:
                size = c.c_size_t()
                self._call("nvrtcGetProgramLogSize", program, c.byref(size))
                log = c.create_string_buffer(size.value)
                self._call("nvrtcGetProgramLog", program, log)
                raise NvrtcError(f"NVRTC compilation failed ({status}):\n{log.value.decode(errors='replace')}")
            size = c.c_size_t()
            self._call("nvrtcGetCUBINSize", program, c.byref(size))
            if not size.value:
                raise NvrtcError("NVRTC returned an empty cubin")
            image = c.create_string_buffer(size.value)
            self._call("nvrtcGetCUBIN", program, image)
            binary = image.raw
            if not binary.startswith(b"\x7fELF"):
                raise NvrtcError("NVRTC did not produce an ELF cubin")
            return binary
        finally:
            # Release the program on compilation, log retrieval, and extraction failures.
            self._call("nvrtcDestroyProgram", c.byref(program))
