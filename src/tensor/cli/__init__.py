"""Command line entry point. Compiler imports stay inside the selected command."""

from __future__ import annotations

import argparse
import json


def _execution_target(provider, ordinal):
    if provider == "webgpu":
        from tensor.providers.webgpu import probe
        return probe(ordinal)
    if provider == "cpu":
        return {"status": "ok", "arch": "cpu-linux-x86_64"}
    from tensor.cli.doctor import check_device
    return check_device(ordinal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensor", description="Tensor kernel tooling")
    from tensor import __version__
    parser.add_argument("--version", action="version", version=f"Tensor {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="add a local module, Tensor wheel or pypi:distribution==version")
    add.add_argument("source")
    add.add_argument("--index-url", help="Simple Index URL for pypi: references (default: PyPI)")
    install = commands.add_parser("install", help="install the project's pinned module graph")
    install.add_argument("--frozen", action="store_true", help="require an unchanged tensor.lock")
    install.add_argument("--offline", action="store_true", help="restore only from local sources and verified cache")
    pack = commands.add_parser("pack", help="create a deterministic .tpack including dependencies")
    pack.add_argument("source", nargs="?", default=".")
    pack.add_argument("--out", required=True)
    publish = commands.add_parser("publish", help="wrap a module closure in a wheel and publish using Twine")
    publish.add_argument("source", nargs="?", default=".")
    publish.add_argument("--out-dir", default="dist", help="directory for the prepared wheel")
    publish.add_argument("--distribution", help="PyPI project name (default: tensor-module-MODULE)")
    publish.add_argument("--dry-run", action="store_true", help="prepare and verify the wheel without uploading")
    repository = publish.add_mutually_exclusive_group()
    repository.add_argument("--repository", choices=("pypi", "testpypi"), default="pypi")
    repository.add_argument("--repository-url", help="custom upload URL; distinct from the Simple Index URL")
    resolve = commands.add_parser("resolve", help="select an installed module-name::export_name")
    resolve.add_argument("reference")
    resolve.add_argument("--target")
    resolve.add_argument("--nvcc", help="explicit compiler used by --compile")
    resolve.add_argument("--cache-dir", help="compiler cache used by explicit compilation")
    doctor = commands.add_parser("doctor", help="check the CUDA build and device environment")
    doctor.add_argument("--target", help="explicit CUDA target, e.g. sm_86; permits a GPU-free build host")
    doctor.add_argument("--device", type=int, default=0, help="CUDA device ordinal (default: 0)")
    doctor.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    doctor.add_argument("--json", action="store_true", help="print a machine-readable report")
    build = commands.add_parser("build", help="build source, portable TIRx or a module export")
    build.add_argument("source", help="Python source, portable .tbin or module-name::export_name")
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
    inspect.add_argument("--nvcc", help="explicit compiler used by --compile")
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
    for command in (doctor, build, run, benchmark, resolve, inspect):
        command.add_argument("--compiler", choices=("nvrtc", "nvcc", "wgsl") if command is doctor else ("nvrtc", "nvcc", "native", "wgsl"),
                             help="executable compiler (default: nvrtc; --nvcc selects nvcc)")
        command.add_argument("--nvrtc-home", help="NVRTC library/header bundle (or TENSOR_NVRTC_HOME)")
    for command in (doctor, build, run, benchmark, resolve, inspect):
        command.add_argument("--provider", choices=("cuda", "cpu", "webgpu"),
                             help="provider (default: cuda for source; inferred for artifacts)")
    for command in (add, install, build, run, benchmark, resolve, inspect):
        command.add_argument("--project", default=".", help="directory containing tensor.json and tensor.lock")
    for command in (add, install, pack, publish, build, run, benchmark, resolve, inspect, cache):
        command.add_argument("--module-cache", help="module cache root (or TENSOR_MODULE_CACHE)")
    for command in (run, benchmark, resolve, inspect):
        command.add_argument("--compile", action="store_true", help="allow source/TIRx compilation when no exact-target artifact exists")
    args = parser.parse_args(argv)

    if args.command in ("add", "install", "pack", "publish", "resolve"):
        from tensor.artifacts.modules import ModuleError, add as add_module, install as install_modules, pack as pack_module, resolve_reference
        from tensor.compiler.build import BuildError
        from tensor.artifacts.format import ArtifactError
        try:
            if args.command == "add":
                result = add_module(args.source, args.project, cache_dir=args.module_cache, index_url=args.index_url)
            elif args.command == "install":
                result = install_modules(args.project, cache_dir=args.module_cache, frozen=args.frozen, offline=args.offline)
            elif args.command == "pack":
                result = pack_module(args.source, args.out, cache_dir=args.module_cache)
            elif args.command == "publish":
                from tensor.artifacts.registry import publish as publish_module
                result = publish_module(args.source, out_dir=args.out_dir, distribution=args.distribution,
                    cache_dir=args.module_cache, dry_run=args.dry_run, repository=args.repository,
                    repository_url=args.repository_url)
            else:
                result = resolve_reference(args.reference, project=args.project, module_cache=args.module_cache,
                    provider=args.provider or "cuda", target=args.target, compile=args.compile,
                    compiler=args.compiler, nvcc=args.nvcc, nvrtc_home=args.nvrtc_home,
                    cache_dir=args.cache_dir)
        except (ModuleError, BuildError, ArtifactError, OSError, ValueError) as exc:
            parser.exit(1, f"tensor {args.command}: {exc}\n")
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "doctor":
        from tensor.cli.doctor import diagnose, render
        if args.provider == "webgpu":
            from tensor.providers.webgpu import probe
            from tensor.runtime import TensorRuntimeError
            try:
                report = probe(args.device)
                report["status"] = "run_ready"
            except (TensorRuntimeError, ValueError) as exc:
                report = {"status": "blocked", "provider": "webgpu", "detail": str(exc)}
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "run_ready" else 1
        report = diagnose(target=args.target, device=args.device, nvcc=args.nvcc,
                          compiler=args.compiler, nvrtc_home=args.nvrtc_home)
        print(json.dumps(report, indent=2) if args.json else render(report))
        return 0 if report["status"] in ("ready", "build_ready", "run_ready") else 1
    if args.command == "build":
        from pathlib import Path
        from tensor.compiler.build import BuildError, build_artifact

        try:
            if "::" in args.source:
                from tensor.artifacts.modules import resolve_reference
                output = Path(args.out)
                if output.exists():
                    raise BuildError(f"output already exists: {output}")
                resolved = resolve_reference(args.source, project=args.project, module_cache=args.module_cache,
                    provider=args.provider or "cuda", target=args.target, compile=True,
                    compiler=args.compiler, nvcc=args.nvcc, nvrtc_home=args.nvrtc_home,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None)
                output.parent.mkdir(parents=True, exist_ok=True)
                contents = Path(resolved["path"]).read_bytes()
                created = False
                try:
                    with output.open("xb") as stream:
                        created = True
                        stream.write(contents)
                except BaseException:
                    if created:
                        output.unlink(missing_ok=True)
                    raise
                report = {**resolved,"status":"built","path":str(output.resolve()),"bytes":output.stat().st_size}
            else:
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
        from tensor.compiler.build import cache_info
        from tensor.artifacts.modules import cache_info as module_cache_info

        print(json.dumps({**cache_info(Path(args.cache_dir) if args.cache_dir else None),
                          "modules": module_cache_info(args.module_cache)}, indent=2))
        return 0
    if args.command == "inspect":
        from pathlib import Path
        from tensor.artifacts.format import ArtifactError
        from tensor.cli.commands import inspect_artifact
        from tensor.cli.inspect import inspect_source

        try:
            path = Path(args.path)
            if "::" in args.path:
                from tensor.artifacts.modules import resolve_reference
                resolved = resolve_reference(args.path, project=args.project, module_cache=args.module_cache,
                    provider=args.provider or "cuda", target=args.target, compile=args.compile,
                    compiler=args.compiler, nvcc=args.nvcc, nvrtc_home=args.nvrtc_home)
                result = inspect_artifact(Path(resolved["path"]), args.stage)
            elif path.is_dir() or path.name == "tensor.json":
                from tensor.artifacts.modules import _directory
                result = json.dumps(_directory(path if path.is_dir() else path.parent).manifest, indent=2)
            elif path.suffix == ".tpack":
                from tensor.artifacts.modules import _archive
                result = json.dumps(_archive(path)[0], indent=2)
            elif path.suffix == ".whl":
                from tensor.artifacts.registry import read_wheel
                result = json.dumps(read_wheel(path)[0], indent=2)
            elif path.suffix == ".py":
                result = inspect_source(path, stage=args.stage, target=args.target,
                                        trace_dir=Path(args.out) if args.out else None, provider=args.provider or "cuda")
            else:
                result = inspect_artifact(path, args.stage)
        except (ArtifactError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"tensor inspect: {exc}\n")
        print(result)
        return 0
    if args.command in ("run", "bench"):
        import tempfile
        from pathlib import Path
        from tensor.artifacts.format import ArtifactError
        from tensor.compiler.build import BuildError, build_artifact
        from tensor.cli.commands import benchmark, run
        from tensor.providers.cuda import CudaError
        from tensor.runtime import TensorRuntimeError
        from tensor.cli.doctor import check_device

        try:
            source = Path(args.artifact)
            with tempfile.TemporaryDirectory(prefix="tensor-cli-") as directory:
                artifact = source
                resolution = None
                if "::" in args.artifact:
                    from tensor.artifacts.modules import resolve_reference
                    provider = args.provider or "cuda"
                    detected = _execution_target(provider, args.device)
                    if detected["status"] != "ok":
                        raise CudaError(detected["detail"])
                    from tensor.runtime.cuda_target import matches_device
                    if args.target and not matches_device(args.target, detected["arch"]):
                        raise ValueError("module execution target must match the selected device")
                    resolution = resolve_reference(args.artifact, project=args.project, module_cache=args.module_cache,
                        provider=provider, target=args.target or detected["arch"], compile=args.compile, compiler=args.compiler,
                        nvcc=args.nvcc, nvrtc_home=args.nvrtc_home, cache_dir=args.cache_dir)
                    artifact = Path(resolution["path"])
                elif source.suffix == ".py":
                    provider = args.provider or "cuda"
                    detected = _execution_target(provider, args.device)
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
                if resolution:
                    result["module"] = {k:v for k,v in resolution.items() if k not in ("path", "build")}
        except (ArtifactError, BuildError, CudaError, TensorRuntimeError, OSError, ValueError, TypeError) as exc:
            parser.exit(1, f"tensor {args.command}: {exc}\n")
        print(json.dumps(result, indent=2))
        return 0
    return 1
