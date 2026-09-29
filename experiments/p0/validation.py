"""Run the remaining Phase 0 probes in fresh processes with isolated caches.

Choose a new output directory for each run. GPU compilation/performance require
the full CUDA toolkit; missing devices are recorded as skipped. Compiler
rejections are measured outcomes in the compile matrix, not successful builds.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.p0.provenance import snapshot


def run(root, architectures, include_perf=True):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = {}

    def worker(mode, directory=None, arch=None, cache=None):
        label = mode if arch is None else f"compile-{arch}"
        output = root / f"{label}.json"
        command = [sys.executable, "-m", "experiments.p0.validation_worker", mode,
                   "--root", str(root / (directory or label)), "--report", str(output)]
        if arch:
            command += ["--arch", arch]
        env = os.environ.copy()
        env.pop("TILELANG_DISABLE_CACHE", None)
        env["TILELANG_CACHE_DIR"] = str(root / (cache or f"{label}-cache"))
        if mode in ("symbolic", "provider", "framework", "fusion", "abi"):
            env["TILELANG_DISABLE_CACHE"] = "1"
        with (root / f"{label}.log").open("w") as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if output.exists():
            records[label] = json.loads(output.read_text())
        else:
            records[label] = {"status": "failed", "error": "worker produced no report", "exit_code": proc.returncode}
        print(f"{label}: {records[label]['status']}", flush=True)
        (root / "results.json").write_text(json.dumps({"records": records}, indent=2)+"\n")
        return records[label]

    cold = worker("cache_cold", directory="cache", cache="cache")
    if cold["status"] == "passed":
        worker("cache_disk", directory="cache", cache="cache")
        path = Path(cold["result"]["identity"]["path"]) / "device_kernel.cu"
        if not path.is_relative_to(root / "cache"):
            raise RuntimeError("refusing to corrupt a cache outside the isolated experiment")
        original = path.read_bytes()
        path.write_bytes(original+b"\n// P0 deliberate isolated-cache corruption\n")
        worker("cache_corrupt", directory="cache", cache="cache")
    worker("provider")
    worker("symbolic")
    if all(records[name]["status"] == "passed" for name in ("provider","symbolic")):
        worker("abi")
    worker("framework")
    worker("fusion")
    for arch in architectures:
        worker("compile", arch=arch)
    if include_perf:
        worker("perf")
    # Each limitation is reported separately; no all-green Phase 0 claim.
    status = "completed_with_limits" if all(r["status"] == "passed" for r in records.values()) else "incomplete"
    result = {"status": status, "provenance": snapshot(), "records": records,
              "limits": ["fusion and rescheduling remain unverified; post-LowerTileOp re-lowering is rejected",
                         "new provider claiming built-in CPU target is rejected",
                         "Rust hosting and native end-to-end compilation remain unverified",
                         "cross-GPU throughput unverified; available hardware is sm_86",
                         "foreign stream, event, signal, and memory ordering contracts remain unverified"],
              "measurement_policy": "workers run sequentially; imports excluded from compile timing; fresh isolated caches"}
    (root / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=["sm_80","sm_86","sm_90","sm_100","sm_90a","sm_100a"])
    parser.add_argument("--skip-perf", action="store_true")
    args = parser.parse_args()
    result = run(args.out, args.architectures, not args.skip_perf)
    print(json.dumps({"status": result["status"], "report": str(args.out / "results.json")}))
    raise SystemExit(0 if result["status"] == "completed_with_limits" else 1)
