"""Execute a transferred boundary matrix and verify its two-host provenance."""

import argparse
import json
from pathlib import Path

from experiments.p0.artifact_check import check
from experiments.p0.artifact_format import ArtifactError


def audit(report):
    if report.get("status") != "passed" or not report.get("records"):
        raise ArtifactError("all consumer executions must pass")
    sizes = set()
    for record in report["records"]:
        result = record["result"]
        if record["exit_code"] != 0 or result.get("status") != "passed":
            raise ArtifactError("a consumer execution did not pass")
        producer, consumer = result["producer"], result["consumer"]
        if not producer.get("hostname") or not consumer.get("hostname") or producer["hostname"] == consumer["hostname"]:
            raise ArtifactError("producer and consumer must be different hosts")
        for field in ("git_revision", "source_sha256", "lock_sha256"):
            if not producer.get(field) or producer[field] != consumer.get(field):
                raise ArtifactError(f"producer/consumer {field} mismatch")
        if producer.get("git_dirty") is not False or consumer.get("git_dirty") is not False:
            raise ArtifactError("use clean matching checkouts for the two-host experiment")
        if result.get("compiler_imports") != [] or result.get("compiler_import_guard") is not True:
            raise ArtifactError("compiler import isolation was not verified")
        if any(consumer.get("packages", {}).get(name) is not None
               for name in ("tilelang", "apache-tvm-ffi", "torch")):
            raise ArtifactError("consumer must use the NumPy-only environment")
        sizes.add(result["size"])
    if sizes != {1,127,128,129,1025}:
        raise ArtifactError("the full boundary-size matrix is required")
    return {"status": "passed", "two_hosts": True, "clean_matching_checkouts": True,
            "compiler_packages_absent": True, "sizes": sorted(sizes), "executions": len(report["records"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("runs must be positive")
    result = check(args.artifacts, args.runtime_python, args.runs)
    try:
        result["transfer_verification"] = audit(result)
    except (ArtifactError, KeyError) as error:
        result.update(status="failed", transfer_error=str(error))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "report": str(args.report)}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
