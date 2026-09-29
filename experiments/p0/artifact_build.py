"""P0 opaque-artifact producer: prepare source, compile with nvcc, or do both.

prepare needs TileLang; compile needs only Python's stdlib and nvcc. Neither
requires a GPU. Outputs use an experimental manifest, not a stable module ABI.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import tempfile
import time
from importlib.metadata import distribution, version
from pathlib import Path

from experiments.p0.artifact_format import ArtifactError, read_bundle, write_bundle
from experiments.p0.cuda_toolchain import compile_cubin, doctor, resolve_nvcc
from experiments.p0.provenance import snapshot


def prepare(path: Path, *, size: int, arch: str) -> dict:
    started = time.perf_counter()
    if version("tilelang") != "0.1.14" or version("apache-tvm-ffi") != "0.1.12":
        raise ArtifactError("producer requires tilelang==0.1.14 and apache-tvm-ffi==0.1.12; run uv sync --locked")
    # All compiler imports live on this path, never in the consumer.
    import tilelang
    from tilelang.tools.compile_only import compile_kernel_source
    from experiments.p0.kernels import artifact_elementwise

    if not re.fullmatch(r"sm_[0-9]{2,3}", arch):
        raise ArtifactError("arch must be a generic exact SM target, e.g. sm_80")
    emit_started = time.perf_counter()
    source = compile_kernel_source(artifact_elementwise(size), {"kind": "cuda", "arch": arch})
    emission_seconds = time.perf_counter() - emit_started
    # Paths in #line are diagnostic metadata, not part of kernel semantics.
    # Normalize them so a source bundle does not depend on the producer's checkout path.
    source = re.sub(r'^#line (\d+) ".*"$', r'#line \1 "kernels.py"', source, flags=re.MULTILINE)
    signature = re.search(r'elementwise_kernel\(([^)]*)\)\s*\{', source)
    if not signature or len(signature[1].split(",")) != 3 or "__shared__" in source:
        raise ArtifactError("generated kernel no longer matches the three-pointer/no-shared-memory contract")
    root = Path(tilelang.__file__).parent
    files = {"kernel.cu": source.encode()}
    # The generated includes pull in CUTLASS/CuTe as well as tl_templates.
    for directory, prefix in ((root / "src" / "tl_templates" / "cuda", "include/tl_templates/cuda"),
                              (root / "3rdparty" / "cutlass" / "include", "include")):
        if not directory.is_dir():
            raise ArtifactError(f"required compiler headers missing: {directory}")
        for header in sorted(directory.rglob("*")):
            if header.is_file():
                files[f"{prefix}/{header.relative_to(directory).as_posix()}"] = header.read_bytes()
    # Preserve upstream redistribution notices alongside bundled headers.
    license_candidates = [root / "3rdparty" / "cutlass" / "LICENSE"]
    dist = distribution("tilelang")
    license_candidates += [Path(dist.locate_file(f)) for f in (dist.files or []) if "licenses" in f.parts]
    for i, license_path in enumerate(license_candidates):
        if license_path.is_file():
            files[f"licenses/{i}-{license_path.name}"] = license_path.read_bytes()
    # Some wheels omit CUTLASS's root LICENSE. Its public header carries the
    # complete BSD notice, which must also accompany derived binaries.
    cutlass_header = root / "3rdparty" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    notice = re.match(r"/\*.*?\*/", cutlass_header.read_text(encoding="utf-8"), re.DOTALL)
    if not notice or "SPDX-License-Identifier: BSD-3-Clause" not in notice[0]:
        raise ArtifactError("CUTLASS redistribution notice missing")
    files["licenses/cutlass-notice.txt"] = notice[0].encode()
    if not any(name.startswith("licenses/") for name in files):
        raise ArtifactError("upstream license files missing from the installed wheel")
    manifest = {
        "format": "tensor.p0.cuda", "format_version": 1, "kind": "source",
        "operation": "relu_2a_plus_b", "dtype": "float32", "size": size,
        "arch": arch, "entrypoint": "elementwise_kernel", "arguments": ["a", "b", "c"],
        "launch": {"grid": [(size + 127) // 128, 1, 1], "block": [128, 1, 1], "shared_memory_bytes": 0},
        "producer": {**snapshot(), "host": platform.platform()},
        "source_emission_seconds": emission_seconds,
    }
    write_bundle(path, manifest, files)
    return {"status": "source_prepared", "path": str(path.resolve()), "bytes": path.stat().st_size,
            "seconds": time.perf_counter() - started, "gpu_execution": "unverified"}


def compile_bundle(source_path: Path, output: Path, *, nvcc: str | None = None) -> dict:
    manifest, files = read_bundle(source_path, kind="source")
    compiler = resolve_nvcc(nvcc)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tensor-p0-build-") as directory:
        root = Path(directory)
        for name, data in files.items():
            dest = root / name  # read_bundle already checked all paths
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        binary = root / "kernel.cubin"
        compiler_info = compile_cubin(root / "kernel.cu", binary, arch=manifest["arch"],
                                      include_dirs=(root / "include",), nvcc=compiler)
        cubin = binary.read_bytes()
    manifest = {**manifest, "kind": "cubin",
                "compiler": compiler_info}
    write_bundle(output, manifest, {"kernel.cubin": cubin,
                                    **{name: data for name, data in files.items() if name.startswith("licenses/")}})
    return {"status": "executable_built", "path": str(output.resolve()), "bytes": output.stat().st_size,
            "seconds": time.perf_counter() - started, "gpu_execution": "unverified"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("doctor", help="compile a CUDA probe; no GPU or TileLang needed")
    cmd.add_argument("--nvcc")
    cmd.add_argument("--arch", default="sm_80")
    for name in ("prepare", "build"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--size", type=int, default=129)
        cmd.add_argument("--arch", default="sm_80")
        cmd.add_argument("--out", type=Path, required=True)
        if name == "build":
            cmd.add_argument("--nvcc")
    cmd = sub.add_parser("compile")
    cmd.add_argument("source", type=Path)
    cmd.add_argument("--out", type=Path, required=True)
    cmd.add_argument("--nvcc")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(nvcc=args.nvcc, arch=args.arch)
        elif args.command == "prepare":
            result = prepare(args.out, size=args.size, arch=args.arch)
        elif args.command == "compile":
            result = compile_bundle(args.source, args.out, nvcc=args.nvcc)
        else:
            compiler = resolve_nvcc(args.nvcc)
            with tempfile.TemporaryDirectory(prefix="tensor-p0-source-") as directory:
                source = Path(directory) / "source.zip"
                prepare(source, size=args.size, arch=args.arch)
                result = compile_bundle(source, args.out, nvcc=compiler)
        print(json.dumps(result, indent=2))
        return 0
    except (ArtifactError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
