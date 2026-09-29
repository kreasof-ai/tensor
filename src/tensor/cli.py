"""Command line entry point. Compiler imports stay inside the selected command."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensor", description="Tensor kernel tooling")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check the CUDA build and device environment")
    doctor.add_argument("--target", help="explicit CUDA target, e.g. sm_86; permits a GPU-free build host")
    doctor.add_argument("--device", type=int, default=0, help="CUDA device ordinal (default: 0)")
    doctor.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    doctor.add_argument("--json", action="store_true", help="print a machine-readable report")
    build = commands.add_parser("build", help="compile a TileLang source file to a CUDA artifact")
    build.add_argument("source", help="Python source exporting tensor_export()")
    build.add_argument("--target", help="exact CUDA target, e.g. sm_86 (detected from device 0 by default)")
    build.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    build.add_argument("--cache-dir", help="override the content-addressed build cache")
    build.add_argument("--out", required=True, help="new output .tbin path")
    cache = commands.add_parser("cache", help="show build-cache contents")
    cache.add_argument("--cache-dir", help="override the content-addressed build cache")
    inspect = commands.add_parser("inspect", help="inspect an artifact or source lowering")
    inspect.add_argument("path", help=".tbin artifact or TileLang Python source")
    inspect.add_argument("--stage", choices=("manifest", "tirx", "target", "passes"),
                         default="manifest")
    inspect.add_argument("--target", help="CUDA target for source lowering")
    inspect.add_argument("--out", help="new directory for pass trace files")
    run = commands.add_parser("run", help="execute a cubin artifact with .npy inputs")
    run.add_argument("artifact")
    run.add_argument("--input", action="append", default=[], metavar="NAME=FILE.npy")
    run.add_argument("--scalar", action="append", default=[], metavar="NAME=NUMBER")
    run.add_argument("--out-dir", required=True, help="directory for named .npy outputs")
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--target", help="target used when the input is Python source")
    run.add_argument("--nvcc", help="CUDA compiler used when the input is Python source")
    run.add_argument("--cache-dir", help="cache used when the input is Python source")
    benchmark = commands.add_parser("bench", help="measure artifact launch plus synchronization")
    benchmark.add_argument("artifact")
    benchmark.add_argument("--input", action="append", default=[], metavar="NAME=FILE.npy")
    benchmark.add_argument("--scalar", action="append", default=[], metavar="NAME=NUMBER")
    benchmark.add_argument("--device", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--iters", type=int, default=100)
    benchmark.add_argument("--target", help="target used when the input is Python source")
    benchmark.add_argument("--nvcc", help="CUDA compiler used when the input is Python source")
    benchmark.add_argument("--cache-dir", help="cache used when the input is Python source")
    for command in (doctor, build, run, benchmark):
        command.add_argument("--compiler", choices=("nvrtc", "nvcc") if command is doctor else ("nvrtc", "nvcc", "native"),
                             help="executable compiler (default: nvrtc; --nvcc selects nvcc)")
        command.add_argument("--nvrtc-home", help="NVRTC library/header bundle (or TENSOR_NVRTC_HOME)")
    for command in (build, run, benchmark):
        command.add_argument("--provider", choices=("cuda", "cpu"),
                             help="provider (default: cuda for source; inferred for artifacts)")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from tensor.doctor import diagnose, render

        report = diagnose(target=args.target, device=args.device, nvcc=args.nvcc,
                          compiler=args.compiler, nvrtc_home=args.nvrtc_home)
        print(json.dumps(report, indent=2) if args.json else render(report))
        return 0 if report["status"] in ("ready", "build_ready", "run_ready") else 1
    if args.command == "build":
        from pathlib import Path
        from tensor.build import BuildError, build_artifact

        try:
            report = build_artifact(Path(args.source), Path(args.out), target=args.target, nvcc=args.nvcc,
                                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                                    compiler=args.compiler, nvrtc_home=args.nvrtc_home,
                                    provider=args.provider or "cuda")
        except (BuildError, OSError, ValueError) as exc:
            parser.exit(1, f"tensor build: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "cache":
        from pathlib import Path
        from tensor.build import cache_info

        print(json.dumps(cache_info(Path(args.cache_dir) if args.cache_dir else None), indent=2))
        return 0
    if args.command == "inspect":
        from pathlib import Path
        from tensor.artifact import ArtifactError
        from tensor.commands import inspect_artifact
        from tensor.inspect import inspect_source

        try:
            path = Path(args.path)
            if path.suffix == ".py":
                result = inspect_source(path, stage=args.stage, target=args.target,
                                        trace_dir=Path(args.out) if args.out else None)
            else:
                result = inspect_artifact(path, args.stage)
        except (ArtifactError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"tensor inspect: {exc}\n")
        print(result)
        return 0
    if args.command in ("run", "bench"):
        import tempfile
        from pathlib import Path
        from tensor.artifact import ArtifactError
        from tensor.build import BuildError, build_artifact
        from tensor.commands import benchmark, run
        from tensor.cuda import CudaError
        from tensor.runtime import TensorRuntimeError
        from tensor.doctor import check_device

        try:
            source = Path(args.artifact)
            with tempfile.TemporaryDirectory(prefix="tensor-cli-") as directory:
                artifact = source
                if source.suffix == ".py":
                    provider = args.provider or "cuda"
                    detected = check_device(args.device) if provider == "cuda" else {"status": "ok", "arch": "cpu-linux-x86_64"}
                    if detected["status"] != "ok":
                        raise CudaError(detected["detail"])
                    artifact = Path(directory) / "kernel.tbin"
                    build_artifact(source, artifact, target=args.target or detected["arch"], nvcc=args.nvcc,
                                   cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                                    compiler=args.compiler, nvrtc_home=args.nvrtc_home,
                                    provider=provider)
                if args.command == "run":
                    result = run(artifact, args.input, Path(args.out_dir), ordinal=args.device,
                                 scalar_values=args.scalar, provider=args.provider)
                else:
                    result = benchmark(artifact, args.input, ordinal=args.device,
                                       warmup=args.warmup, iters=args.iters, scalar_values=args.scalar,
                                       provider=args.provider)
                if source.suffix == ".py":
                    result.pop("artifact")
                    result["source"] = str(source.resolve())
        except (ArtifactError, BuildError, CudaError, TensorRuntimeError, OSError, ValueError, TypeError) as exc:
            parser.exit(1, f"tensor {args.command}: {exc}\n")
        print(json.dumps(result, indent=2))
        return 0
    return 1
