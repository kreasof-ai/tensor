"""Version-pinned observational hooks for P0, not a public Tensor adapter.

Private compiler/cache/adapter touchpoints used by the new experiments live
here. Hooks are restored after each observation and do not alter installed code.
"""

from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import version
import time


def check_versions():
    for package, expected in (("tilelang", "0.1.14"), ("apache-tvm-ffi", "0.1.12")):
        if version(package) != expected:
            raise RuntimeError(f"P0 requires {package}=={expected}")


@contextmanager
def observe_compile(forbid=False):
    check_versions()
    lower_module = import_module("tilelang.engine.lower")
    from tilelang.contrib import nvcc
    stats = {"lower_calls": 0, "lower_seconds": 0.0,
             "cuda_compile_calls": 0, "cuda_compile_seconds": 0.0}
    original_lower = lower_module.lower_with_context
    original_cuda = nvcc.compile_cuda

    def wrap(original, prefix):
        def observed(*args, **kwargs):
            stats[prefix + "_calls"] += 1
            if forbid:
                raise AssertionError(f"a cache hit unexpectedly invoked {prefix}")
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stats[prefix + "_seconds"] += time.perf_counter() - started
        return observed

    lower_module.lower_with_context = wrap(original_lower, "lower")
    nvcc.compile_cuda = wrap(original_cuda, "cuda_compile")
    try:
        yield stats
    finally:
        lower_module.lower_with_context = original_lower
        nvcc.compile_cuda = original_cuda


def materialize(kernel):
    """Realize the lazy TVM FFI host executable before calling it compiled."""
    import tvm_ffi
    executable = kernel.adapter.get_exportable_executable()
    return executable if isinstance(executable, tvm_ffi.Module) else executable.jit()


def cache_identity(kernel):
    return {"key": kernel._tilelang_cache_key, "path": kernel._tilelang_cache_path}


def cache_key_matrix(func, target):
    import tilelang.cache as cache_module
    from tilelang.backend import create_backend_context
    context = create_backend_context(target, execution_backend="tvm_ffi")
    cache = cache_module._dispatch_map["tvm_ffi"]
    args = dict(func=func, out_idx=None, args=(), target=context.target,
                target_host=context.target_host, execution_backend="tvm_ffi",
                pass_configs={}, compile_flags=None)
    baseline = cache._generate_key(**args)
    variants = {
        "function_ir": {"func": func.with_attr("global_symbol", "renamed")},
        "target": {"target": "cuda -arch=sm_80"},
        "target_host": {"target_host": "llvm"},
        "execution_backend": {"execution_backend": "cython"},
        "out_idx": {"out_idx": [2]},
        "pass_configs": {"pass_configs": {"tirx.disable_vectorize": True}},
        "compile_flags": {"compile_flags": ["--use_fast_math"]},
    }
    return {"base_key_fields": cache._get_base_key(), "baseline": baseline,
            "variants": {name: cache._generate_key(**{**args, **change})
                         for name, change in variants.items()}}


def cpu_library(kernel):
    return kernel.adapter.lib_generator.get_lib_path()


def cpu_registration_probe():
    from dataclasses import replace
    from tilelang.backend import get_backend, register_backend
    cpu = get_backend("cpu")
    assert register_backend(cpu) is cpu
    try:
        register_backend(replace(cpu, name="p0_cpu", supports_target=lambda target: True))
    except ValueError as error:
        return {"manifest_reregistration": True, "duplicate_target_error": str(error)}
    raise AssertionError("pinned registry unexpectedly accepted an overlapping CPU provider")


@contextmanager
def capture_tile_op(captured):
    """Use the pinned compilation-tool extension to observe nested pass contexts."""
    check_versions()
    import tilelang
    from tilelang.instrumentation import (PassInstrumentationTool,
        register_pass_instrumentation_tool, unregister_pass_instrumentation_tool)
    @tilelang.tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name.rsplit(".",1)[-1] == "LowerTileOp":
                captured.append(tilelang.tvm.ir.save_json(mod))
    class Tool(PassInstrumentationTool):
        def create_pass_instrument(self):
            return Capture()
    register_pass_instrumentation_tool("tensor-p0-capture", Tool)
    try:
        yield
    finally:
        unregister_pass_instrumentation_tool("tensor-p0-capture")
