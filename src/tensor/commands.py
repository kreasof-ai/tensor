"""CLI execution helpers; NumPy is loaded only for run and bench."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tensor.artifact import ArtifactError, read_artifact
from tensor.cuda import Device, bench


def _input_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("each --input must be a distinct NAME=FILE.npy")
        result[name] = Path(path)
    return result


def _prepare(device: Device, artifact: Path, input_values: list[str]):
    import numpy as np

    executable = device.load(artifact)
    descriptors = executable.manifest["arguments"]
    outputs = executable.manifest.get("outputs", [])
    if not outputs:
        raise ArtifactError("artifact declares no outputs; rebuild with tensor_export()['outputs']")
    paths = _input_paths(input_values)
    input_names = {item["name"] for item in descriptors if item["name"] not in outputs}
    if paths.keys() != input_names:
        raise ValueError(f"inputs must be exactly {sorted(input_names)}; received {sorted(paths)}")
    buffers = {}
    for item in descriptors:
        name = item["name"]
        if name in outputs:
            buffers[name] = device.empty(item["shape"], item["dtype"])
        else:
            array = np.load(paths[name], allow_pickle=False)
            if array.shape != tuple(item["shape"]) or str(array.dtype) != item["dtype"]:
                raise ValueError(f"{name} needs shape {item['shape']} and dtype {item['dtype']}")
            buffers[name] = device.from_numpy(array)
    ordered = tuple(buffers[item["name"]] for item in descriptors)
    return executable, ordered, {name: buffers[name] for name in outputs}


def run(artifact: Path, input_values: list[str], out_dir: Path, *, ordinal: int = 0) -> dict:
    started = time.perf_counter()
    import numpy as np

    with Device(ordinal) as device:
        device_ready = time.perf_counter()
        executable, buffers, outputs = _prepare(device, artifact, input_values)
        prepared = time.perf_counter()
        executable.launch(*buffers)
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
    return {"status": "passed", "artifact": str(artifact.resolve()), "device": info,
            "outputs": {name: str((out_dir / f"{name}.npy").resolve()) for name in results},
            "timings": {"device_setup_seconds": device_ready - started,
                        "artifact_load_and_upload_seconds": prepared - device_ready,
                        "first_result_seconds": first_result - started}}


def benchmark(artifact: Path, input_values: list[str], *, ordinal: int = 0,
              warmup: int = 10, iters: int = 100) -> dict:
    with Device(ordinal) as device:
        executable, buffers, _ = _prepare(device, artifact, input_values)
        result = bench(executable, buffers, warmup=warmup, iters=iters)
        return {"status": "passed", "artifact": str(artifact.resolve()),
                "device": device.info, "metric": "host_launch_plus_stream_sync", **result}


def inspect_artifact(path: Path, stage: str) -> str:
    manifest, files = read_artifact(path)
    if stage == "manifest":
        return json.dumps(manifest, indent=2)
    if stage == "tirx":
        return files["kernel.tirx.json"].decode("utf-8")
    raise ValueError("artifact inspection supports manifest or tirx; pass source.py for target or passes")
