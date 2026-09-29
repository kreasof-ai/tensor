"""Run the final scoped Phase 0 gates against prior measured E15 evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.p0.provenance import ROOT, snapshot

MODES = ("cpu_provider", "composition", "rust_host", "foreign_stream", "static_symbolic")


def run(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((ROOT / "docs/research/data/e15-phase0-validation.json").read_text())
    transfer = json.loads((ROOT / "docs/research/data/e15-artifact-transfer.json").read_text())
    assert prior["status"] == "completed_with_limits"
    assert all(row["status"] == "passed" for row in prior["records"].values())
    assert transfer["status"] == "passed" and len(transfer["records"]) == 15
    records = {}
    for mode in MODES:
        command = [sys.executable, "-m", "experiments.p0.validation_worker", mode,
                   "--root", str(out / mode), "--report", str(out / f"{mode}.json")]
        env = os.environ.copy()
        env["TILELANG_DISABLE_CACHE"] = "1"
        env["TILELANG_CACHE_DIR"] = str(out / f"{mode}-cache")
        with (out / f"{mode}.log").open("w") as log:
            process = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        path = out / f"{mode}.json"
        records[mode] = json.loads(path.read_text()) if path.exists() else {
            "status":"failed","error":f"worker did not write report (exit {process.returncode})"}
        print(f"{mode}: {records[mode]['status']}",flush=True)
        (out / "results.json").write_text(json.dumps({"records":records},indent=2)+"\n")
    success = all(row["status"] == "passed" for row in records.values())
    result = {
        "status":"passed" if success else "failed", "provenance":snapshot(), "records":records,
        "prior_e15": {"experiment_revision":prior["provenance"]["git_revision"],
                      "validation_status":prior["status"],
                      "gpu_regression_suite":prior["regression_suite"],
                      "two_host_transfer_status":transfer["status"],
                      "two_host_actions_run_url":transfer["actions_run_url"]},
        "exit_gates":{
            "E4_frontend_representation_and_bounded_composition":records["composition"]["status"] == "passed",
            "E6_independent_non_cuda_cpu_provider":records["cpu_provider"]["status"] == "passed",
            "E9_full_compile_measured":prior["status"] == "completed_with_limits",
            "E11_a10g_static_symbolic_measured":records["static_symbolic"]["status"] == "passed",
            "E12_two_host_executable_transfer":transfer["status"] == "passed",
            "E7_rust_c_abi_and_foreign_cuda_streams":records["rust_host"]["status"] == "passed" and records["foreign_stream"]["status"] == "passed",
        },
        "limits":[
            "cross-GPU throughput deferred by user; only A10G measured",
            "composition supports one fixed pointwise producer/consumer pattern, not general fusion",
            "P0 CPU provider uses public context routing on the unclaimed test target; custom tilelang.compile JIT adapter unsupported",
            "Rust hosts executable calls and IR loading; a Python-free native compiler pipeline was not established",
            "foreign stream/event ordering is CUDA-specific; no provider-neutral or distributed signal ABI is frozen",
            "GPU compile failures for some architecture/workload pairs remain measured restrictions",
        ],
    }
    (out / "results.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.out)
    print(json.dumps({"status":result["status"],"report":str(args.out / "results.json")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
