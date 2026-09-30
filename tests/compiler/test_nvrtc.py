"""NVRTC dependency diagnostics and exactly-once program cleanup."""

import ctypes as c
from types import SimpleNamespace

import pytest

from tensor.compiler import select_compiler
from tensor.compiler.nvrtc import NvrtcCompiler, NvrtcError


def test_missing_bundle_and_conflicting_compiler_selection(tmp_path):
    with pytest.raises(NvrtcError,match="bundle missing"):
        NvrtcCompiler(tmp_path)
    with pytest.raises(NvrtcError,match="cannot be combined"):
        select_compiler("nvrtc",nvcc="/some/nvcc")


def test_nvrtc_compile_failure_keeps_log_and_releases_program():
    compiler = NvrtcCompiler.__new__(NvrtcCompiler)
    compiler.options = lambda target, includes: []
    calls = []
    def invoke(name,*args):
        calls.append(name)
        if name == "nvrtcCreateProgram":
            c.cast(args[0],c.POINTER(c.c_void_p))[0]=42
        elif name == "nvrtcGetProgramLogSize":
            c.cast(args[1],c.POINTER(c.c_size_t))[0]=len(b"header missing\0")
        elif name == "nvrtcGetProgramLog":
            c.memmove(args[1],b"header missing\0",len(b"header missing\0"))
    compiler._call = invoke
    compiler.lib = SimpleNamespace(nvrtcCompileProgram=lambda *args: 6)
    with pytest.raises(NvrtcError,match="header missing"):
        compiler.compile("invalid","sm_86",())
    assert calls.count("nvrtcDestroyProgram") == 1
