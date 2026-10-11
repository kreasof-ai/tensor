"""Tensor command arguments."""

import argparse


def make_parser():
    parser = argparse.ArgumentParser(prog="tensor", description="Tensor kernel tooling")
    from tensor import __version__

    parser.add_argument("--version", action="version", version=f"Tensor {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser(
        "add", help="add a local module, Tensor wheel or pypi:distribution==version"
    )
    add.add_argument("source")
    add.add_argument(
        "--index-url", help="Simple Index URL for pypi: references (default: PyPI)"
    )
    install = commands.add_parser(
        "install", help="install the project's pinned module graph"
    )
    install.add_argument(
        "--frozen", action="store_true", help="require an unchanged tensor.lock"
    )
    install.add_argument(
        "--offline",
        action="store_true",
        help="restore only from local sources and verified cache",
    )
    pack = commands.add_parser(
        "pack", help="create a deterministic .tpack including dependencies"
    )
    pack.add_argument("source", nargs="?", default=".")
    pack.add_argument("--out", required=True)
    publish = commands.add_parser(
        "publish", help="wrap a module closure in a wheel and publish using Twine"
    )
    publish.add_argument("source", nargs="?", default=".")
    publish.add_argument(
        "--out-dir", default="dist", help="directory for the prepared wheel"
    )
    publish.add_argument(
        "--distribution", help="PyPI project name (default: tensor-module-MODULE)"
    )
    publish.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare and verify the wheel without uploading",
    )
    repository = publish.add_mutually_exclusive_group()
    repository.add_argument(
        "--repository", choices=("pypi", "testpypi"), default="pypi"
    )
    repository.add_argument(
        "--repository-url", help="custom upload URL; distinct from the Simple Index URL"
    )
    resolve = commands.add_parser(
        "resolve", help="select an installed module-name::export_name"
    )
    resolve.add_argument("reference")
    resolve.add_argument("--target")
    resolve.add_argument("--nvcc", help="explicit compiler used by --compile")
    resolve.add_argument(
        "--cache-dir", help="compiler cache used by explicit compilation"
    )
    doctor = commands.add_parser(
        "doctor", help="check the CUDA build and device environment"
    )
    doctor.add_argument(
        "--target",
        help="explicit CUDA target, e.g. sm_86; permits a GPU-free build host",
    )
    doctor.add_argument(
        "--device", type=int, default=0, help="CUDA device ordinal (default: 0)"
    )
    doctor.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    doctor.add_argument(
        "--json", action="store_true", help="print a machine-readable report"
    )
    build = commands.add_parser(
        "build", help="build source, portable TIRx or a module export"
    )
    build.add_argument(
        "source", help="Python source, portable .tbin or module-name::export_name"
    )
    build.add_argument(
        "--target",
        help="exact CUDA target, e.g. sm_86 (detected from device 0 by default)",
    )
    build.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    build.add_argument("--cache-dir", help="override the content-addressed build cache")
    build.add_argument("--out", required=True, help="new output .tbin path")
    cache = commands.add_parser("cache", help="show build-cache contents")
    cache.add_argument("--cache-dir", help="override the content-addressed build cache")
    inspect = commands.add_parser(
        "inspect", help="inspect an artifact or source lowering"
    )
    inspect.add_argument("path", help=".tbin artifact or TileLang Python source")
    inspect.add_argument(
        "--stage", choices=("manifest", "tirx", "target", "passes"), default="manifest"
    )
    inspect.add_argument("--target", help="CUDA target for source lowering")
    inspect.add_argument("--nvcc", help="explicit compiler used by --compile")
    inspect.add_argument("--out", help="new directory for pass trace files")
    run = commands.add_parser("run", help="execute a cubin artifact with .npy inputs")
    run.add_argument("artifact")
    run.add_argument("--input", action="append", default=[], metavar="NAME=FILE.npy")
    run.add_argument("--scalar", action="append", default=[], metavar="NAME=NUMBER")
    run.add_argument(
        "--out-dir", required=True, help="directory for named .npy outputs"
    )
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--target", help="target used when the input is Python source")
    run.add_argument(
        "--nvcc", help="CUDA compiler used when the input is Python source"
    )
    run.add_argument("--cache-dir", help="cache used when the input is Python source")
    benchmark = commands.add_parser(
        "bench", help="measure artifact launch plus synchronization"
    )
    benchmark.add_argument("artifact")
    benchmark.add_argument(
        "--input", action="append", default=[], metavar="NAME=FILE.npy"
    )
    benchmark.add_argument(
        "--scalar", action="append", default=[], metavar="NAME=NUMBER"
    )
    benchmark.add_argument("--device", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--iters", type=int, default=100)
    benchmark.add_argument(
        "--target", help="target used when the input is Python source"
    )
    benchmark.add_argument(
        "--nvcc", help="CUDA compiler used when the input is Python source"
    )
    benchmark.add_argument(
        "--cache-dir", help="cache used when the input is Python source"
    )
    for command in (doctor, build, run, benchmark, resolve, inspect):
        command.add_argument(
            "--compiler",
            choices=("nvrtc", "nvcc", "wgsl")
            if command is doctor
            else ("nvrtc", "nvcc", "native", "wgsl"),
            help="executable compiler (default: nvrtc; --nvcc selects nvcc)",
        )
        command.add_argument(
            "--nvrtc-home", help="NVRTC library/header bundle (or TENSOR_NVRTC_HOME)"
        )
    for command in (doctor, build, run, benchmark, resolve, inspect):
        command.add_argument(
            "--provider",
            choices=("cuda", "cpu", "webgpu"),
            help="provider (default: cuda for source; inferred for artifacts)",
        )
    for command in (add, install, build, run, benchmark, resolve, inspect):
        command.add_argument(
            "--project",
            default=".",
            help="directory containing tensor.json and tensor.lock",
        )
    for command in (
        add,
        install,
        pack,
        publish,
        build,
        run,
        benchmark,
        resolve,
        inspect,
        cache,
    ):
        command.add_argument(
            "--module-cache", help="module cache root (or TENSOR_MODULE_CACHE)"
        )
    for command in (run, benchmark, resolve, inspect):
        command.add_argument(
            "--compile",
            action="store_true",
            help="allow source/TIRx compilation when no exact-target artifact exists",
        )
    return parser
