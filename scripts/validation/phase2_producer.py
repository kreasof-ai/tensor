"""Build the NVRTC acceptance bundle; optionally require a tool/GPU-free host.

Run with the pinned compiler packages, a Tensor wheel/source, and the local
NVRTC bundle. Numerical execution happens separately in a compiler-free host.
"""

from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import ntpath
import os
from pathlib import Path
import re
import shutil
import platform
import socket
import sys
import time

EXAMPLES = ("elementwise", "gemm_relu", "dynamic_affine", "dynamic_gemm", "scalar_offset")
TOOLS = {"nvcc", "ptxas", "gcc", "g++", "cc", "c++", "cl", "clang", "clang++"}


@contextmanager
def compiler_audit():
    """Reject external compilation without replacing subprocess.Popen's class.

    Windows asyncio subclasses Popen during import. A process audit hook also
    catches calls through such subclasses. Hooks cannot be removed, so disable
    this hook when the scoped producer check ends.
    """
    active = True
    calls = []

    def observe(event, arguments):
        if active and event == "subprocess.Popen":
            executable, command = arguments[:2]
            # Windows can pass executable=None after converting argv to a
            # quoted command line. POSIX usually supplies the executable.
            if executable is None:
                if isinstance(command, (list, tuple)):
                    executable = command[0]
                else:
                    match = re.match(r'^(?:"([^"]+)"|(\S+))', os.fsdecode(command).lstrip())
                    executable = (match.group(1) or match.group(2)) if match else ""
            executable = ntpath.basename(os.fsdecode(executable)).lower()
            if executable.removesuffix(".exe") in TOOLS:
                calls.append(str(arguments[1]))
                raise RuntimeError(f"external compilation prohibited: {arguments[1]}")

    sys.addaudithook(observe)
    try:
        yield calls
    finally:
        active = False


def produce(source_root: Path, out: Path, target: str, isolated: bool = False, identity: dict | None = None):
    available = {name: shutil.which(name) for name in sorted(TOOLS)}
    try:
        ctypes.CDLL("libcuda.so.1")
        driver_present = True
    except OSError:
        driver_present = False
    if isolated and (driver_present or any(available.values())):
        raise RuntimeError(f"producer must have no CUDA driver or compiler tools: {available}")
    # Audit and reject external compilation, even when the host has those tools.
    with compiler_audit() as tool_calls:
        from tensor.compiler.build import build_artifact
        from tensor.artifacts.format import read_artifact
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for name in EXAMPLES:
            source = source_root / "examples" / f"{name}.py"
            cold = build_artifact(source,out / f"{name}.tbin",target=target,cache_dir=out / "cache",compiler="nvrtc")
            warm_path = out / f"warm-{name}.tbin"
            warm = build_artifact(source,warm_path,target=target,cache_dir=out / "cache",compiler="nvrtc")
            if cold["cache_hit"] or not warm["cache_hit"] or cold["cache_key"] != warm["cache_key"]:
                raise RuntimeError("cache validation failed; use a fresh output directory")
            warm_path.unlink()
            manifest,_ = read_artifact(out / f"{name}.tbin")
            records.append({"example":name,"cold_seconds":cold["seconds"],"warm_seconds":warm["seconds"],
                            "compile_seconds":cold["compile_seconds"],"artifact_bytes":cold["bytes"],
                            "format_version":manifest["format_version"],"runtime_abi":manifest["runtime_abi"],
                            "compiler":manifest["compiler"]})
        # Test recovery from a corrupt cached image under the same compiler identity.
        key = cold["cache_key"]
        (out / "cache" / f"{key}.cubin").write_bytes(b"corrupt")
        recovered_path = out / "recovered.tbin"
        repaired = build_artifact(source,recovered_path,target=target,cache_dir=out / "cache",compiler="nvrtc")
        if repaired["cache_hit"]:
            raise RuntimeError("corrupt cache entry was accepted")
        recovered_path.unlink()
        report = {"status":"passed","target":target,"hostname":socket.gethostname(),
                  "isolated":isolated,"driver_present":driver_present,"available_tools":available,
                  "python":platform.python_version(),"platform":platform.platform(),
                  "external_compiler_calls":tool_calls,"cache_corruption_recovered":True,
                  "packages":{d.metadata["Name"]:d.version for d in metadata.distributions()},
                  "records":records,
                  "artifacts":{f"{name}.tbin":hashlib.sha256((out / f"{name}.tbin").read_bytes()).hexdigest() for name in EXAMPLES}}
        if identity:
            report["checkout"] = identity
        (out / "nvrtc-producer.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--target",default="sm_86")
    parser.add_argument("--require-isolated",action="store_true")
    parser.add_argument("--identity",type=Path)
    args=parser.parse_args()
    report=produce(args.source_root,args.out,args.target,args.require_isolated,
                   json.loads(args.identity.read_text()) if args.identity else None)
    print(json.dumps({"status":report["status"],"out":str(args.out),"artifacts":list(report["artifacts"])}))
