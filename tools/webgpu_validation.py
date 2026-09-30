"""Build/transfer the Phase 5 inference suite, then validate with a clean consumer.

Producer: python tools/webgpu_validation.py --build build/webgpu-transfer
Consumer: python tools/webgpu_validation.py --consume build/webgpu-transfer --out report.json
The consume path imports NumPy, Tensor and wgpu only, including for references.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions, version
import json
from pathlib import Path
import platform
import re
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONSUMER_FILES = ("webgpu.py", "webgpu_contract.py", "abi.py", "artifact.py", "runtime.py", "modules.py", "providers.py")


def source_hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CONSUMER_FILES}


def specialize(source, constants):
    for name, value in constants.items():
        source, count = re.subn(r"^"+re.escape(name)+r" = .*$", name+" = "+repr(value), source, flags=re.M)
        if count != 1:
            raise ValueError(f"specialization constant {name} missing or ambiguous")
    return source


def produce(directory):
    import tensor as tx
    from tensor.modules import pack
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases, exports = [], {}
    module = directory / "module"
    (module / "src").mkdir(parents=True)
    (module / "artifacts").mkdir()

    def artifact(name, example, constants=None, source=None):
        text = source or (ROOT / "examples" / example).read_text()
        text = specialize(text, constants or {})
        src = module / "src" / f"{name}.py"
        src.write_text(text, encoding="utf-8")
        out = module / "artifacts" / f"{name}.tbin"
        result = tx.build(src, out, provider="webgpu", cache_dir=directory / "compiler-cache")
        exports[name] = {"source": f"src/{name}.py", "portable": f"artifacts/{name}.tbin",
                         "artifacts": [f"artifacts/{name}.tbin"]}
        return {"artifact": str(out.relative_to(directory)), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "export": name, "build_seconds": result["seconds"]}

    affine = artifact("affine", "dynamic_affine.py")
    for size in (1, 127, 128, 129, 4097, 1048576):
        cases.append({"name": f"affine_{size}", "profile": "affine", "size": size, **affine})
    half = artifact("elementwise_f16", "elementwise.py", source=(ROOT / "examples/elementwise.py").read_text().replace('"float32"', '"float16"'))
    cases.append({"name": "elementwise_f16_tail", "profile": "elementwise", "size": 129, **half})
    variants = [
        ("gemm_f16", {"USE_BIAS": False, "RELU": False}),
        ("linear_f16", {}),
        ("linear_transpose", {"TRANSPOSE_B": True}),
        ("linear_f32", {"DTYPE": "float32", "OUTPUT_DTYPE": "float32"}),
        ("linear_f16_acc32", {"OUTPUT_DTYPE": "float32"}),
        ("mlp_second", {"K": 65, "N": 29}),
        ("gemm_256", {"M": 256, "N": 256, "K": 256, "USE_BIAS": False, "RELU": False}),
        ("gemm_512", {"M": 512, "N": 512, "K": 512, "USE_BIAS": False, "RELU": False}),
    ]
    for name, overrides in variants:
        constants = {"M": 33, "N": 65, "K": 37, "DTYPE": "float16", "OUTPUT_DTYPE": "float16",
                     "USE_BIAS": True, "RELU": True, "TRANSPOSE_B": False, **overrides}
        compiled = artifact(name, "webgpu_gemm.py", constants)
        cases.append({"name": name, "profile": "gemm", "constants": constants, **compiled})
    dynamic = artifact("dynamic_gemm", "dynamic_gemm.py")
    for rows in (1, 31, 32, 33, 65):
        cases.append({"name": f"dynamic_gemm_{rows}", "profile": "dynamic_gemm", "rows": rows, **dynamic})
    for depth in (64, 128):
        for causal in (False, True):
            for sequence in (65, 129, 512):
                name = f"attention_d{depth}_s{sequence}_{'causal' if causal else 'full'}"
                constants = {"BATCH": 1 if sequence == 512 else 2, "HEADS": 1 if sequence == 512 else 2,
                             "SEQ_LEN": sequence, "HEAD_DIM": depth,
                             "IS_CAUSAL": causal, "BLOCK_M": 8, "BLOCK_N": 16}
                compiled = artifact(name, "flash_attention.py", constants)
                cases.append({"name": name, "profile": "attention", "constants": constants, **compiled})
    manifest = {"formatVersion": 1, "name": "tensor-webgpu-validation", "version": "0.1.0",
                "tensorAbi": 1, "capabilities": ["contiguous", "opaque_buffer_handles"], "exports": exports}
    (module / "tensor.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    packed = pack(module, directory / "validation.tpack", cache_dir=directory / "module-cache")
    report = {"schema": "tensor.webgpu-validation.v1", "cases": cases, "module_sha256": packed["sha256"],
              "consumer_source_sha256": source_hashes(ROOT / "src/tensor"),
              "producer": {"tilelang": version("tilelang"), "tvm_ffi": version("apache-tvm-ffi"),
                           "hostname": socket.gethostname(),
                           "lowering_sha256": hashlib.sha256((ROOT / "src/tensor/webgpu_lowering.py").read_bytes()).hexdigest()}}
    (directory / "suite.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def inputs_reference(case, rng):
    import numpy as np
    profile = case["profile"]
    if profile in ("affine", "elementwise"):
        dtype = "float16" if profile == "elementwise" else "float32"
        a, b = [rng.standard_normal(case["size"]).astype(dtype) for _ in range(2)]
        scale = 2.5 if profile == "affine" else 2
        reference = np.maximum(scale*a.astype("float32")+b.astype("float32"), 0).astype(dtype)
        return [a,b], {"scale": scale} if profile == "affine" else {}, reference
    if profile in ("gemm", "dynamic_gemm"):
        c = case.get("constants", {"M": case.get("rows"), "N": 32, "K": 32, "DTYPE": "float16", "OUTPUT_DTYPE": "float16"})
        a = rng.standard_normal((c["M"],c["K"])).astype(c["DTYPE"])
        b = rng.standard_normal((c["N"],c["K"]) if c.get("TRANSPOSE_B") else (c["K"],c["N"])).astype(c["DTYPE"])
        reference = a.astype("float32") @ (b.T if c.get("TRANSPOSE_B") else b).astype("float32")
        values = [a,b]
        if profile == "gemm":
            if c["USE_BIAS"]:
                bias = rng.standard_normal(c["N"]).astype(c["DTYPE"])
                values.append(bias)
                reference += bias
            if c["RELU"]:
                reference = np.maximum(reference, 0)
        return values, {}, reference.astype(c["OUTPUT_DTYPE"])
    c = case["constants"]
    shape = (c["BATCH"],c["HEADS"],c["SEQ_LEN"],c["HEAD_DIM"])
    q,k,v = [rng.standard_normal(shape).astype("float16") for _ in range(3)]
    scores = (q.astype("float32") @ k.astype("float32").swapaxes(-1,-2)) * c["HEAD_DIM"]**-0.5
    if c["IS_CAUSAL"]:
        scores = np.where(np.arange(c["SEQ_LEN"])[None,:] <= np.arange(c["SEQ_LEN"])[:,None], scores, -np.inf)
    probability = np.exp(scores - scores.max(-1,keepdims=True))
    probability /= probability.sum(-1,keepdims=True)
    return [q,k,v], {}, (probability @ v.astype("float32")).astype("float16")


@contextmanager
def compiler_guard():
    import importlib.abc
    blocked = {"tilelang", "tvm", "tvm_ffi", "torch", "triton"}
    if blocked & set(sys.modules):
        raise RuntimeError("consumer must start without compiler/framework imports")
    class CompilerGuard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError(f"compiler-free WebGPU consumer attempted to import {fullname}")
            return None
    guard = CompilerGuard()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


@compiler_guard()
def consume(directory, *, ordinal=0, iters=5, require_hardware=False):
    import numpy as np
    import tensor as tx
    from tensor.modules import add, install
    import tempfile
    from wgpu.backends import wgpu_native
    directory = Path(directory)
    suite = json.loads((directory / "suite.json").read_text())
    if suite["schema"] != "tensor.webgpu-validation.v1":
        raise ValueError("unknown suite schema")
    consumer_hashes = source_hashes(Path(tx.__file__).parent)
    if consumer_hashes != suite["consumer_source_sha256"]:
        raise ValueError("installed Tensor consumer differs from the producer; install the supplied matching wheel")
    package = directory / "validation.tpack"
    if hashlib.sha256(package.read_bytes()).hexdigest() != suite["module_sha256"]:
        raise ValueError("module checksum mismatch")
    cases = []
    with tempfile.TemporaryDirectory(prefix="tensor-webgpu-validate-") as temporary, tx.Device(ordinal,provider="webgpu") as device:
        info = device.info
        hardware = info["adapter"]["adapter_type"] in ("DiscreteGPU", "IntegratedGPU")
        vendor = str(info["adapter"].get("vendor", "")).lower()
        second_gpu = hardware and (info["adapter"].get("vendor_id") in (0x1002, 0x106b) or "amd" in vendor or "apple" in vendor)
        if require_hardware and not second_gpu:
            raise RuntimeError("acceptance requires a physical AMD or Apple adapter")
        project = Path(temporary) / "project"
        project.mkdir()
        (project / "tensor.json").write_text(json.dumps({"formatVersion":1,"name":"consumer","version":"0.1.0","tensorAbi":1,"exports":{}}))
        cache = Path(temporary) / "modules"
        add(package,project,cache_dir=cache)
        install(project,cache_dir=cache,frozen=True)
        module = tx.Project(project,cache_dir=cache).module("tensor-webgpu-validation")
        rng = np.random.default_rng(20260930)
        executables = {}
        for case in suite["cases"]:
            path = directory / case["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
                raise ValueError(f"artifact checksum mismatch: {case['name']}")
            if case["export"] not in executables:
                started = time.perf_counter()
                kernel = module.load(case["export"],device)
                executables[case["export"]] = kernel, time.perf_counter()-started
            kernel, pipeline_seconds = executables[case["export"]]
            values, bindings, reference = inputs_reference(case,rng)
            uploaded = [device.from_numpy(value) for value in values]
            for original, buffer in zip(values, uploaded):
                np.testing.assert_array_equal(buffer.to_numpy(),original)
            ordered, dimensions, outputs = kernel.prepare(*uploaded,**bindings)
            try:
                kernel.launch(*ordered,**dimensions)
                actual = next(iter(outputs.values())).to_numpy()
                # FP32 references accumulate without the shader's FMA/order;
                # FP16 attention also rounds probability tiles before P@V.
                atol = 3e-5 if reference.dtype == np.dtype("float32") else 0.015
                rtol = 2e-4 if reference.dtype == np.dtype("float32") else 0.01
                np.testing.assert_allclose(actual,reference,rtol=rtol,atol=atol)
                timing = tx.bench(kernel,ordered,warmup=2,iters=iters,**dimensions)
                cases.append({"name":case["name"],"profile":case["profile"],"status":"passed",
                              "artifact_sha256":case["sha256"],"pipeline_create_seconds":pipeline_seconds,
                              "maximum_absolute_error":float(np.max(np.abs(actual.astype("float32")-reference.astype("float32")))),
                              "rtol":rtol,"atol":atol,"timing":timing})
                print(f"passed {case['name']}", file=sys.stderr, flush=True)
            finally:
                for buffer in [*uploaded,*outputs.values()]:
                    buffer.release()
        # Compose two opaque executables without a host round-trip of their intermediate.
        first, second = executables["linear_f16"][0], executables["mlp_second"][0]
        a,b,bias = [device.randn(shape,"float16",seed=i) for i,shape in enumerate(((33,37),(37,65),(65,)))]
        w2,b2 = device.randn((65,29),"float16",seed=3),device.randn((29,),"float16",seed=4)
        middle = first(a,b,bias)
        final = second(middle,w2,b2)
        middle_reference = np.maximum(a.to_numpy().astype("float32") @ b.to_numpy().astype("float32")+bias.to_numpy(),0).astype("float16")
        expected = np.maximum(middle_reference.astype("float32") @ w2.to_numpy().astype("float32")+b2.to_numpy(),0).astype("float16")
        np.testing.assert_allclose(final.to_numpy(),expected,rtol=.01,atol=.03)
        event = device.record_event()
        device.wait(device.get_event(event.descriptor))
        event.release()
        cases.append({"name":"mlp_chain","profile":"composition","status":"passed"})
    compiler_imports = sorted(name for name in sys.modules if name in ("tilelang","tvm","torch","triton"))
    return {"schema":"tensor.webgpu-result.v1","status":"passed","timestamp":datetime.now(timezone.utc).isoformat(),
            "platform":platform.platform(),"python":platform.python_version(),"adapter":info,
            "hostname":socket.gethostname(),"suite_sha256":hashlib.sha256((directory / "suite.json").read_bytes()).hexdigest(),
            "consumer_source_sha256":consumer_hashes,
            "physical_second_gpu":second_gpu,"software_adapter":info["adapter"]["adapter_type"] == "CPU","compiler_imports":compiler_imports,
            "compiler_import_guard":True,"packages":sorted(dist.metadata["Name"] for dist in distributions()),
            "versions":{**{name:version(name) for name in ("numpy","wgpu","tensor-workspace")},"wgpu-native":wgpu_native.__version__},
            "module_sha256":suite["module_sha256"],"producer":suite["producer"],"cases":cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build",type=Path)
    action.add_argument("--consume",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--device",type=int,default=0)
    parser.add_argument("--iters",type=int,default=5)
    parser.add_argument("--require-second-gpu",action="store_true")
    args=parser.parse_args()
    if args.iters < 1:
        parser.error("iters must be positive")
    report=produce(args.build) if args.build else consume(args.consume,ordinal=args.device,iters=args.iters,require_hardware=args.require_second_gpu)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report.get("status","built"),"cases":len(report["cases"]),"physical_second_gpu":report.get("physical_second_gpu")}))


if __name__=="__main__":
    main()
