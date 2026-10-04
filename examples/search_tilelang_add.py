"""Discover schedules for the addition factory with correctness-gated timings."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import tensor as tx
from tensor.compiler.entry import export_source
from tensor.compiler.search import ScheduleProfile, ScheduleSearch
from tensor.compiler.tuning import measure_cuda

SPACES = {"add": {"block": (64, 128, 256)}}
SEED = {"family": "add", "block": 128}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive finite seconds")
    return number


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for candidates and reports")
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=positive_int, default=65537)
    parser.add_argument("--candidates", type=positive_int, default=3)
    parser.add_argument("--seconds", type=positive_seconds, default=60,
                        help="wall-time budget checked between trials; does not interrupt compilation")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    factory = Path(__file__).with_name("tilelang_add.py")
    # Export sources can be rebuilt independently of the original examples folder.
    shutil.copyfile(factory, out / factory.name)
    rng = np.random.default_rng(431)
    a_host = rng.normal(size=args.size).astype(np.float32)
    b_host = rng.normal(size=args.size).astype(np.float32)
    reference = a_host + b_host
    for name, array in (("a", a_host), ("b", b_host), ("reference", reference)):
        np.save(out / f"{name}.npy", array)
    metric = "median_gpu_seconds" if args.provider == "cuda" else "median_launch_and_sync_seconds"
    report = {
        "operation": "add", "parameters": {"n": args.size, "dtype": "float32"},
        "spaces": SPACES, "seeds": [SEED], "metric": metric, "records": [],
        "limits": {"candidates": args.candidates, "seconds": args.seconds},
        "protocol": {"reference": "NumPy FP32 addition", "rng_seed": 431,
                     "rtol": 1e-5, "atol": 1e-8, "output_initialization": "fresh NaN buffer per trial",
                     "timing_excludes": ["compilation", "allocation", "transfers", "correctness check"]},
        "versions": {name: version(name) for name in ("tensor-workspace", "tilelang", "apache-tvm-ffi", "numpy")},
        "factory_sha256": hashlib.sha256(factory.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.provider == "webgpu":
        report["versions"]["wgpu"] = version("wgpu")
    search = ScheduleSearch([SEED], spaces=SPACES, width=2)
    started = time.perf_counter()
    with tx.Device(args.device, provider=args.provider) as device:
        report["device"] = device.info
        for index in range(args.candidates):
            if time.perf_counter() - started >= args.seconds:
                report["stop_reason"] = "wall-time budget"
                break
            try:
                config = search.next()
            except StopIteration:
                report["stop_reason"] = "space exhausted"
                break
            label = f"candidate-{index}-block-{config['block']}"
            source, artifact = out / f"{label}.py", out / f"{label}.tbin"
            row = {"config": config, "source": source.name, "artifact": artifact.name}
            kernel = output = a = b = None
            try:
                source.write_text(export_source("tilelang_add", "make_add", args.size, config["block"],
                                                dependencies=(), outputs=["out"]), encoding="utf-8")
                tx.build(source, artifact, provider=args.provider,
                         target=device.info["arch"] if args.provider == "cuda" else None)
                kernel = device.load(artifact)
                a, b = device.from_numpy(a_host), device.from_numpy(b_host)
                output = device.full((args.size,), np.nan, "float32")
                kernel.launch(a, b, output)
                actual = output.to_numpy()
                if not np.isfinite(actual).all():
                    raise ValueError("candidate did not produce finite values for every element")
                tx.assert_close(actual, reference, rtol=1e-5, atol=1e-8)
                if args.provider == "cuda":
                    timing = measure_cuda(device, lambda: kernel.launch(a, b, output),
                                          warmup=3, samples=7, repeats=10)
                else:
                    timing = tx.bench(kernel, (a, b, output), warmup=3, iters=7)
                score = timing[metric]
                if not math.isfinite(score) or score <= 0:
                    raise ValueError("candidate timing must be positive and finite")
                search.record(config, score)
                row.update(status="passed", score_seconds=score, timing=timing,
                           maximum_absolute_error=float(np.max(np.abs(actual - reference))))
                print(f"{label}: passed, {metric}={score * 1e6:.2f} us", flush=True)
            except (AssertionError, ValueError) as error:
                row.update(status="rejected", reason=str(error))
                print(f"{label}: rejected: {error}", flush=True)
            finally:
                for resource in (kernel, output, a, b):
                    if resource is not None:
                        resource.release()
                report["records"].append(row)
                report["wall_seconds"] = time.perf_counter() - started
                save(out / "report.json", report)
        else:
            report["stop_reason"] = "candidate budget"
        passed = [row for row in report["records"] if row["status"] == "passed"]
        if not passed:
            save(out / "report.json", report)
            raise ValueError("no candidate passed; inspect report.json")
        best = min(passed, key=lambda row: row["score_seconds"])
        shutil.copyfile(out / best["artifact"], out / "selected.tbin")
        profile = ScheduleProfile({
            "schema": "tensor.schedule-profile.v1", "provider": args.provider,
            "target": device.info["arch"],
            "entries": [{"operation": "add", "parameters": report["parameters"],
                         "schedule": {"block": best["config"]["block"]}}],
            "provenance": {"device": device.info, "versions": report["versions"], "metric": metric,
                           "factory_sha256": report["factory_sha256"], "report": "report.json"},
        })
        save(out / "profile.json", profile.data)
        report.update(selected=best["artifact"], profile_sha256=profile.sha256,
                      selected_artifact_sha256=hashlib.sha256((out / "selected.tbin").read_bytes()).hexdigest())
        save(out / "report.json", report)
        print(f"Selected block={best['config']['block']}; artifact and profile saved in {out}")


if __name__ == "__main__":
    main()
