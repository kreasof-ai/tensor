"""Independent P0 CPU manifest using TVM's unclaimed CPU `test` target.

Reuses the registered CPU compiler components through their manifest methods.
Owns native compilation/execution; does not replace a built-in or patch a registry.
This experiment is not a general TileLang JIT adapter for custom targets.
"""

import ctypes
from pathlib import Path
import subprocess


def register():
    from experiments.p0.upstream_probe import check_versions
    check_versions()
    import tilelang
    from tilelang.backend import BackendModule, create_backend_context, get_backend, register_backend, list_backends
    from tilelang.backend.device_codegen import DeviceCodegen
    from tilelang.backend.execution_backend import ExecutionBackendSpec
    from tilelang.backend.pass_pipeline import PassPipeline
    before = list_backends()
    assert all("test" not in backend.target_kinds for backend in before.values())
    cpu = get_backend("cpu")
    calls = {"lower": 0, "codegen": 0, "compile": 0}
    c_target = create_backend_context("c", target_host="c", execution_backend="cython").target

    def lower(mod, target):
        calls["lower"] += 1
        with c_target:
            return cpu.lower(mod, c_target)

    def codegen(mod, target):
        calls["codegen"] += 1
        return cpu.codegen_device(mod, c_target, compile_device=False)

    def compile_source(source, output):
        calls["compile"] += 1
        include = Path(tilelang.__file__).parent / "src"
        subprocess.run(["g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-I"+str(include),
                        source, "-o", output], check=True, capture_output=True, text=True, timeout=120)
        return output

    manifest = BackendModule(name="p0_cpu", target_kinds=("test",),
        pipelines={"test": PassPipeline("test", lower)},
        device_codegens={"test": DeviceCodegen("p0_cpu_c", build_without_compile=codegen)},
        execution_backends=(ExecutionBackendSpec("p0_native"),),
        callbacks={"tensor.p0.cpu.compile": compile_source})
    assert register_backend(manifest) is manifest
    assert register_backend(manifest) is manifest
    assert get_backend("cpu") is cpu
    return manifest, calls


def build(context, func, root):
    import tilelang
    import tvm_ffi
    from tilelang.engine.lower import get_device_call, device_codegen_without_compile
    with context.target, tilelang.transform.PassContext(config={"tirx.disable_vectorize": True}):
        mod = context.lower(tilelang.tvm.IRModule({str(func.attrs["global_symbol"]): func}))
        device = tilelang.tvm.tirx.transform.Filter(get_device_call(is_device_c=True))(mod)
        assert len(device.functions) == 1
        source = str(device_codegen_without_compile(device, context).inspect_source())
    path = root / "kernel.cpp"
    path.write_text(source)
    library = root / "kernel.so"
    tvm_ffi.get_global_func("tensor.p0.cpu.compile")(str(path), str(library))
    handle = ctypes.CDLL(str(library.resolve()))
    run = handle.elementwise_kernel
    run.argtypes = [ctypes.c_void_p]*3
    run.restype = ctypes.c_int32
    return handle, run


def probe(root):
    import numpy as np
    from tilelang.backend import create_backend_context
    from experiments.p0.kernels import artifact_elementwise
    manifest, calls = register()
    context = create_backend_context("test", target_host="c", execution_backend="p0_native")
    assert context.module is manifest and context.name == "p0_cpu"
    rows = []
    for size in (1,127,128,129,1025):
        directory = root / str(size)
        directory.mkdir()
        handle, run = build(context, artifact_elementwise(size), directory)
        for seed in range(3):
            rng = np.random.default_rng(seed)
            a,b = [rng.standard_normal(size).astype("float32") for _ in range(2)]
            out = np.full(size,np.nan,dtype="float32")
            assert run(a.ctypes.data,b.ctypes.data,out.ctypes.data) == 0
            expected = np.maximum(2*a+b,0)
            np.testing.assert_allclose(out,expected,rtol=1e-6,atol=1e-6)
            rows.append({"size":size,"seed":seed,"max_abs_error":float(np.max(np.abs(out-expected)))})
    assert calls == {"lower":5,"codegen":5,"compile":5}
    return {"provider":context.name,"target_kind":"test","physical_device":"CPU",
            "builtin_cpu_preserved":True,"registration":"public BackendModule API; no registry patch",
            "compiler_components":"delegated to existing CPU manifest with C target",
            "execution":"provider-owned g++ shared library + borrowed NumPy pointers",
            "jit_adapter":"p0_native is not supported by tilelang.compile; context path exercised",
            "calls":calls,"rows":rows}
