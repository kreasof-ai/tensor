"""Run and benchmark commands with explicit NumPy inputs."""

from __future__ import annotations

from pathlib import Path
import json
from tensor.artifacts.format import ArtifactError
from tensor.providers import Device
from tensor.runtime.signature import buffer_argument
import time
from tensor.artifacts.format import read_artifact
from tensor.runtime.timing import bench


def _execution_target(provider, ordinal):
    if provider == "webgpu":
        from tensor.providers.webgpu.adapter import probe

        return probe(ordinal)
    if provider == "cpu":
        return {"status": "ok", "arch": "cpu-linux-x86_64"}
    from tensor.diagnostics import check_device

    return check_device(ordinal)


def _input_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("each --input must be a distinct NAME=FILE.npy")
        result[name] = Path(path)
    return result


def _scalar_values(values: list[str]) -> dict:
    result = {}
    for value in values:
        name, separator, literal = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("each --scalar must be a distinct NAME=NUMBER")
        try:
            parsed = json.loads(literal)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{name}: --scalar needs a JSON number or boolean"
            ) from exc
        if type(parsed) not in (int, float, bool):
            raise ValueError(f"{name}: --scalar needs a number or boolean")
        result[name] = parsed
    return result


def _prepare(
    device: Device, artifact: Path, input_values: list[str], scalar_values: list[str]
):
    import numpy as np

    executable = device.load(artifact)
    descriptors = executable.manifest["arguments"]
    outputs = executable.manifest.get("outputs", [])
    if not outputs:
        raise ArtifactError(
            "artifact declares no outputs; rebuild with tensor_export()['outputs']"
        )
    paths = _input_paths(input_values)
    input_names = {
        item["name"]
        for item in descriptors
        if buffer_argument(item) and item["name"] not in outputs
    }
    if paths.keys() != input_names:
        raise ValueError(
            f"inputs must be exactly {sorted(input_names)}; received {sorted(paths)}"
        )
    supplied = _scalar_values(scalar_values)
    allowed = {
        item["name"] for item in descriptors if not buffer_argument(item)
    } | executable.manifest.get("symbols", {}).keys()
    if supplied.keys() - allowed:
        raise ValueError(
            f"unknown scalar arguments: {sorted(supplied.keys() - allowed)}"
        )
    for item in descriptors:
        name = item["name"]
        if buffer_argument(item) and name not in outputs:
            array = np.load(paths[name], allow_pickle=False)
            if not isinstance(array, np.ndarray):
                raise ValueError(f"{name} needs a .npy array")
            supplied[name] = device.from_numpy(array)
    ordered, dimensions, generated = executable.prepare(**supplied)
    return executable, ordered, dimensions, generated


def run(
    artifact: Path,
    input_values: list[str],
    out_dir: Path,
    *,
    ordinal: int = 0,
    scalar_values: list[str] | None = None,
    provider: str | None = None,
) -> dict:
    started = time.perf_counter()
    import numpy as np

    manifest, _ = read_artifact(artifact)
    with Device(
        ordinal, provider=provider or manifest.get("provider", "cuda")
    ) as device:
        device_ready = time.perf_counter()
        executable, buffers, dimensions, outputs = _prepare(
            device, artifact, input_values, scalar_values or []
        )
        prepared = time.perf_counter()
        executable.launch(*buffers, **dimensions)
        results = {name: buffer.to_numpy() for name, buffer in outputs.items()}
        first_result = time.perf_counter()
        info = device.info
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for name, array in results.items():
            path = out_dir / f"{name}.npy"
            with path.open("xb") as stream:
                created.append(path)
                np.save(stream, array, allow_pickle=False)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {
        "status": "passed",
        "artifact": str(artifact.resolve()),
        "device": info,
        "outputs": {name: str((out_dir / f"{name}.npy").resolve()) for name in results},
        "timings": {
            "device_setup_seconds": device_ready - started,
            "artifact_load_and_upload_seconds": prepared - device_ready,
            "first_result_seconds": first_result - started,
        },
    }


def benchmark(
    artifact: Path,
    input_values: list[str],
    *,
    ordinal: int = 0,
    warmup: int = 10,
    iters: int = 100,
    scalar_values: list[str] | None = None,
    provider: str | None = None,
) -> dict:
    manifest, _ = read_artifact(artifact)
    with Device(
        ordinal, provider=provider or manifest.get("provider", "cuda")
    ) as device:
        executable, buffers, dimensions, _ = _prepare(
            device, artifact, input_values, scalar_values or []
        )
        result = bench(executable, buffers, warmup=warmup, iters=iters, **dimensions)
        return {
            "status": "passed",
            "artifact": str(artifact.resolve()),
            "device": device.info,
            "metric": "host_launch_plus_stream_sync",
            **result,
        }


def execute_command(args, parser):
    if args.command in ("run", "bench"):
        import tempfile
        from pathlib import Path
        from tensor.artifacts.format import ArtifactError
        from tensor.compiler.errors import BuildError
        from tensor.compiler.build import build_artifact
        from tensor.cli.run import benchmark, run
        from tensor.providers.cuda.driver import CudaError
        from tensor.runtime.errors import TensorRuntimeError

        try:
            source = Path(args.artifact)
            with tempfile.TemporaryDirectory(prefix="tensor-cli-") as directory:
                artifact = source
                resolution = None
                if "::" in args.artifact:
                    from tensor.modules.project import resolve_reference

                    provider = args.provider or "cuda"
                    detected = _execution_target(provider, args.device)
                    if detected["status"] != "ok":
                        raise CudaError(detected["detail"])
                    from tensor.runtime.cuda_target import matches_device

                    if args.target and not matches_device(
                        args.target, detected["arch"]
                    ):
                        raise ValueError(
                            "module execution target must match the selected device"
                        )
                    resolution = resolve_reference(
                        args.artifact,
                        project=args.project,
                        module_cache=args.module_cache,
                        provider=provider,
                        target=args.target or detected["arch"],
                        compile=args.compile,
                        compiler=args.compiler,
                        nvcc=args.nvcc,
                        nvrtc_home=args.nvrtc_home,
                        cache_dir=args.cache_dir,
                    )
                    artifact = Path(resolution["path"])
                elif source.suffix == ".py":
                    provider = args.provider or "cuda"
                    detected = _execution_target(provider, args.device)
                    if detected["status"] != "ok":
                        raise CudaError(detected["detail"])
                    artifact = Path(directory) / "kernel.tbin"
                    build_artifact(
                        source,
                        artifact,
                        target=args.target or detected["arch"],
                        nvcc=args.nvcc,
                        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                        compiler=args.compiler,
                        nvrtc_home=args.nvrtc_home,
                        provider=provider,
                    )
                if args.command == "run":
                    result = run(
                        artifact,
                        args.input,
                        Path(args.out_dir),
                        ordinal=args.device,
                        scalar_values=args.scalar,
                        provider=args.provider,
                    )
                else:
                    result = benchmark(
                        artifact,
                        args.input,
                        ordinal=args.device,
                        warmup=args.warmup,
                        iters=args.iters,
                        scalar_values=args.scalar,
                        provider=args.provider,
                    )
                if source.suffix == ".py":
                    result.pop("artifact")
                    result["source"] = str(source.resolve())
                if resolution:
                    result["module"] = {
                        k: v
                        for k, v in resolution.items()
                        if k not in ("path", "build")
                    }
        except (
            ArtifactError,
            BuildError,
            CudaError,
            TensorRuntimeError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            parser.exit(1, f"tensor {args.command}: {exc}\n")
        print(json.dumps(result, indent=2))
        return 0
