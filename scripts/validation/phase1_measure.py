"""Measure the Phase 1 elementwise CLI path across fresh processes.

Example:
    CUDA_HOME=... uv run --locked python scripts/validation/phase1_measure.py \
        --runtime-python build/consumer-venv/bin/python --out build/phase1-metrics.json

The runtime interpreter should contain only Tensor and NumPy. The script keeps
all generated inputs, cubins and caches in a temporary directory.
"""

from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def _invoke(command: list[str], *, env: dict | None = None) -> tuple[dict, float]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=360)
    wall = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}\n{result.stdout}")
    return json.loads(result.stdout), wall


def measure(runtime_python: Path, *, target: str | None = None, runs: int = 3,
            iters: int = 100) -> dict:
    import numpy as np

    if runs < 1 or iters < 1:
        raise ValueError("runs and iters must be positive")
    # Resolving the venv's python symlink would escape the venv and drop its packages.
    runtime_python = runtime_python.absolute()
    if not runtime_python.is_file():
        raise FileNotFoundError(runtime_python)
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="tensor-phase1-") as temporary:
        root = Path(temporary)
        cache, cold, warm = root / "cache", root / "cold.tbin", root / "warm.tbin"
        source = ROOT / "examples" / "elementwise.py"
        common = ["--compiler", "nvcc", "--cache-dir", str(cache)]
        if target:
            common += ["--target", target]
        build_command = [sys.executable, "-m", "tensor", "build", str(source)]
        cold_report, cold_wall = _invoke([*build_command, *common, "--out", str(cold)], env=env)
        warm_report, warm_wall = _invoke([*build_command, *common, "--out", str(warm)], env=env)
        if cold_report["cache_hit"] or not warm_report["cache_hit"]:
            raise RuntimeError("cold/warm build cache states were not observed")
        rng = np.random.default_rng(2026)
        inputs = {}
        for name in ("a", "b"):
            path = root / f"{name}.npy"
            np.save(path, rng.standard_normal(129).astype("float32"))
            inputs[name] = path
        arguments = [part for name, path in inputs.items() for part in ("--input", f"{name}={path}")]
        samples = []
        for index in range(runs):
            output = root / f"result-{index}"
            report, wall = _invoke([str(runtime_python), "-m", "tensor", "run", str(cold),
                                    *arguments, "--out-dir", str(output)])
            actual = np.load(output / "c.npy")
            expected = np.maximum(2*np.load(inputs["a"]) + np.load(inputs["b"]), 0)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            samples.append({"process_wall_seconds": wall,
                            "first_result_seconds": report["timings"]["first_result_seconds"],
                            "max_abs_error": float(np.max(np.abs(actual-expected)))})
        benchmark, bench_wall = _invoke([str(runtime_python), "-m", "tensor", "bench", str(cold),
                                         *arguments, "--warmup", "10", "--iters", str(iters)])
        packages, _ = _invoke([str(runtime_python), "-c",
                               "import json,importlib.metadata as m; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"])
    return {
        "status": "passed", "host": platform.platform(), "python": platform.python_version(),
        "device": benchmark["device"], "source": "examples/elementwise.py", "size": 129,
        "producer": {"cold_build_seconds": cold_report["seconds"],
                     "warm_build_seconds": warm_report["seconds"],
                     "cold_process_wall_seconds": cold_wall,
                     "warm_process_wall_seconds": warm_wall,
                     "cold_nvcc_seconds": cold_report["nvcc_seconds"],
                     "cache_key": cold_report["cache_key"]},
        "consumer_packages": packages, "fresh_process_runs": samples,
        "benchmark": {"process_wall_seconds": bench_wall,
                      "median_host_enqueue_seconds": benchmark["median_host_enqueue_seconds"],
                      "median_launch_and_sync_seconds": benchmark["median_launch_and_sync_seconds"],
                      "iters": iters},
        "same_host": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.runtime_python, target=args.target, runs=args.runs, iters=args.iters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
