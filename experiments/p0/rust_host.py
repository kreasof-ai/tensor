"""Compile and run the Rust C-ABI host against the independent P0 CPU provider."""

import json
import os
from pathlib import Path
import shutil
import subprocess


def probe(root):
    import tilelang
    import tvm_ffi
    from experiments.p0.validation_kernels import dynamic_elementwise

    rustc = os.environ.get("TENSOR_P0_RUSTC") or shutil.which("rustc")
    if not rustc:
        raise RuntimeError("Rust toolchain absent; run tools/bootstrap_rust.py and set TENSOR_P0_RUSTC")
    version = subprocess.run([rustc,"--version"],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    assert version.startswith("rustc 1.98.1 "), version
    provider = root.parent / "cpu_provider" / "129" / "kernel.so"
    assert provider.is_file(), "independent P0 CPU provider did not produce size-129 library"
    shutil.copy2(provider,root / "kernel.so")
    ffi = Path(tvm_ffi.libinfo.find_libtvm_ffi()).parent
    include = tvm_ffi.libinfo.include_paths()[0]
    source = Path(__file__).parent.resolve()
    commands = [
        ["g++","-std=c++17","-O2","-fPIC","-shared","-I"+include,
         str(source / "native_module.cpp"),"-L"+str(root),"-l:kernel.so",
         "-L"+str(ffi),"-ltvm_ffi","-Wl,-rpath,$ORIGIN",
         "-Wl,-rpath,"+str(ffi),"-o",str(root / "libp0_ffi.so")],
        [rustc,"--edition=2021","-C","opt-level=2","-L","native="+str(ffi),
         "-C","link-arg=-Wl,-rpath,"+str(ffi),str(source / "native_host.rs"),
         "-o",str(root / "native_host")]]
    for command in commands:
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
    ir=tilelang.tvm.ir.save_json(tilelang.tvm.IRModule({"main":dynamic_elementwise()}))
    (root / "dynamic.json").write_text(ir)
    compiler=Path(tilelang.__file__).parent / "lib"
    cases={"cpu_run":["run",str(root / "libp0_ffi.so")],
           "native_ir":["ir",str(compiler / "libtvm_compiler.so"),
                        str(compiler / "libtilelang.so"),str(root / "dynamic.json")]}
    results={}
    for name,args in cases.items():
        proc=subprocess.run([str(root / "native_host"),*args],check=True,
                            capture_output=True,text=True,timeout=60)
        results[name]=json.loads(proc.stdout)
    linked=subprocess.run(["ldd",str(root / "native_host")],check=True,
                          capture_output=True,text=True,timeout=10).stdout
    assert all(name not in linked for name in ("libpython","libtorch","libc10"))
    return {"rustc":version,"cpu_run":results["cpu_run"],"native_ir":results["native_ir"],
            "host_dependencies":linked,"commands":commands,
            "native_compilation_pipeline":"not exercised; native IR load and executable call only",
            "ownership":"Rust owns FFI object refs and CPU Vec storage; module borrows DLTensor views"}
