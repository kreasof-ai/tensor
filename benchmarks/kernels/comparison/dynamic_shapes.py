"""Compare shared dynamic CUDA schedules with matched exact-row compilations.

Uses the compiler's existing matmul operation. Artifacts, producer costs and
per-shape measurements are retained; there are no serving/model kernels here.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensor as tx
from tensor.compiler.search import ScheduleSearch
from tensor.compiler.tuning import measure, tune_shapes, select_regions


def write_module(out, report):
    """Publish the measured bank through the common guarded module format."""
    selection = report.get("portfolio") or select_regions(
        report, symbol="rows", bounds=(report["shapes"][0], report["shapes"][-1]), max_regions=4)
    regions = []
    for region in selection["regions"]:
        config = report["trials"][region["trial"]]["config"]
        artifact = f"dynamic-m{config['block_m']}.tbin"
        regions.append(dict(bounds=region["bounds"], artifacts=[artifact]))
    report["portfolio"] = selection
    manifest = dict(formatVersion=2, name="dynamic-matmul", version="0.1.0", tensorAbi=1,
                    exports={"matmul": {"portfolio": dict(symbols={"rows": "int32"},
                        shape_domain=report["domain"], regions=regions, evidence="results.json")}})
    (out / "tensor.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def run(out, *, rows, columns, depth, budget=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    domain = {"version": 1, "bounds": {"rows": {"min": min(rows), "max": max(rows)}},
              "relations": []}
    started = time.perf_counter()
    builds = []
    with tx.Device() as device:
        rng = np.random.default_rng(29)
        weights = (rng.normal(size=(columns, depth)) * .1).astype("float16")
        values = (rng.normal(size=(max(rows), depth)) * .1).astype("float16")
        x, w = device.from_numpy(values), device.from_numpy(weights)
        output = device.empty((max(rows), columns), "float16")

        def build_image(config, count=None):
            name = ("dynamic" if count is None else f"exact-{count}") + f"-m{config['block_m']}"
            source, artifact = out / f"{name}.py", out / f"{name}.tbin"
            schedule = {k: v for k, v in config.items() if k != "family"}
            m = "T.dynamic('rows')" if count is None else repr(count)
            export_domain = f", 'shape_domain': {domain!r}" if count is None else ""
            source.write_text(
                "import tilelang.language as T\nfrom tensor.compiler.cuda.matmul import matmul\n"
                f"def tensor_export():\n    return {{'kernel': matmul({m}, {columns}, {depth}, "
                f"dtype='float16', schedule={schedule!r}), 'outputs': ['out']{export_domain}}}\n")
            if not artifact.exists():
                build = tx.build(source, artifact, target=device.info["arch"],
                                 cache_dir=out / "compiler-cache")
                builds.append({"name": name, **build})
            return artifact

        @contextmanager
        def build(config):
            path = build_image(config)
            with device.load(path) as executable:
                yield executable

        def evaluate(executable, count):
            # One reference/shape is live at a time. Unused capacity is excluded
            # from both logical descriptors and the launched grid.
            a, result = x.prefix((count, depth)), output.prefix((count, columns))
            try:
                expected = values[:count].astype("float32") @ weights.astype("float32").T
                call = executable.bind(a, w, result)
                call.launch()
                np.testing.assert_allclose(result.to_numpy(), expected, rtol=.02, atol=.003)
                timing = measure(device, call.launch, warmup=3, samples=7, repeats=10)
                return {"correct": True, "seconds": timing["median_seconds"],
                        "timing": timing, "grid": list(call.descriptor.grid)}
            finally:
                result.release()
                a.release()

        search = ScheduleSearch([{"family": "matmul", "block_m": 32}],
                                spaces={"matmul": {"block_m": (16, 32, 64, 128)}})
        report = tune_shapes(search, rows, build, evaluate, budget=budget)
        report["portfolio"] = select_regions(report, symbol="rows", bounds=(min(rows), max(rows)), max_regions=4)
        exact = []
        for index, count in enumerate(rows):
            region = next(r for r in report["portfolio"]["regions"]
                          if r["bounds"]["rows"]["min"] <= count <= r["bounds"]["rows"]["max"])
            trial = region["trial"]
            config = report["trials"][trial]["config"]
            path = build_image(config, count)
            with device.load(path) as executable:
                sample = evaluate(executable, count)
            dynamic = report["trials"][trial]["measurements"][index]
            exact.append({"rows": count, "measurement": sample,
                          "selected_trial": trial,
                          "dynamic_over_exact": dynamic["seconds"] / sample["seconds"]})
        report.update(schema="tensor.dynamic-shape-study.v1", device=device.info,
                      columns=columns, depth=depth, domain=domain, exact=exact,
                      builds=builds, producer_seconds=time.perf_counter() - started,
                      artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in out.glob("*.tbin")})
    write_module(out, report)
    print(json.dumps({"results": str(out / "results.json"), "trials": len(report["trials"]),
                      "ratios": [row["dynamic_over_exact"] for row in exact]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 3, 7, 31, 33, 64, 129, 256])
    parser.add_argument("--columns", type=int, default=1536)
    parser.add_argument("--depth", type=int, default=2048)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()
    if any(v < 1 for v in (*args.rows, args.columns, args.depth)):
        parser.error("dimensions must be positive")
    run(args.out, rows=args.rows, columns=args.columns, depth=args.depth, budget=args.candidates)
