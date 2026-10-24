"""Audit retained measurements and artifact identities before sharing results."""

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    checked_artifacts, checked_timings, failed_trials = 0, 0, 0
    issues = []
    profiles = []
    identical_controls = []
    candidate_rows = []
    for case in result["cases"]:
        if case.get("status") != "passed":
            issues.append(
                {
                    "case": case["id"],
                    "status": case.get("status"),
                    "errors": case.get("errors"),
                }
            )
        for trial in case["candidates"]:
            failed_trials += trial["status"] == "failed"
            candidate_rows.append(
                {
                    "case": case["id"],
                    "status": trial["status"],
                    "config": trial["config"],
                    "search_median_us": trial.get("timing", {}).get("median_us"),
                    "wall_seconds": trial["wall_seconds"],
                    "error": trial.get("error"),
                }
            )
            if trial["status"] != "passed":
                continue
            for path, expected_sha in zip(trial["artifacts"], trial["artifact_sha256"]):
                path = Path(path)
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                assert actual == expected_sha, path
                with zipfile.ZipFile(path) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    for name, digest in manifest["files"].items():
                        assert hashlib.sha256(archive.read(name)).hexdigest() == digest
                    if case["workload"] == "matmul":
                        assert (
                            manifest["cuda_prelowering"]["schedule"] == trial["config"]
                        )
                    assert manifest["target"] == result["environment"]["gpu"]["sm"]
                checked_artifacts += 1
        for measurement in case["measurements"].values():
            xs = measurement["samples_us"]
            assert len(xs) >= 3 and all(math.isfinite(x) and x > 0 for x in xs)
            assert statistics.median(xs) == measurement["median_us"]
            assert min(xs) == measurement["min_us"] and max(xs) == measurement["max_us"]
            checked_timings += 1
        if "selected" in case:

            def image_digest(evidence):
                with zipfile.ZipFile(evidence["artifacts"][0]) as archive:
                    return json.loads(archive.read("manifest.json"))["files"][
                        "kernel.cubin"
                    ]

            default = case["baselines"].get("tensor_default")
            if default and image_digest(default) == image_digest(case["selected"]):
                ratio = (
                    case["measurements"]["tensor_searched"]["median_us"]
                    / case["measurements"]["tensor_default"]["median_us"]
                )
                identical_controls.append(
                    {"case": case["id"], "selected_over_default_latency": ratio}
                )
                if not 0.95 <= ratio <= 1.05:
                    issues.append(
                        {
                            "case": case["id"],
                            "issue": "identical cubin timing differs by more than 5%",
                        }
                    )
            profiles.append(
                {
                    "operation": case["workload"],
                    "parameters": {
                        "dtype": "float32" if case["workload"] == "gguf" else "float16",
                        "out_dtype": "float32"
                        if case["workload"] == "gguf"
                        else "float16",
                        **{
                            key: case[key]
                            for key in (
                                "shape",
                                "causal",
                                "encoding",
                                "qk_dim",
                                "v_dim",
                            )
                            if key in case
                        },
                    },
                    "schedule": case["selected"]["config"],
                }
            )
    audit = {
        "status": "passed" if not issues else "partial",
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": len(result["cases"]),
        "checked_artifacts": checked_artifacts,
        "checked_provider_measurements": checked_timings,
        "failed_search_candidates": failed_trials,
        "identical_cubin_controls": identical_controls,
        "issues": issues,
    }
    (args.run / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.run / "schedule-profile.json").write_text(
        json.dumps(
            {
                "schema": "tensor.schedule-profile.v1",
                "provider": "cuda",
                "target": result["environment"]["gpu"]["sm"],
                "entries": profiles,
            },
            indent=2,
        )
        + "\n"
    )
    (args.run / "candidates.json").write_text(
        json.dumps(candidate_rows, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
