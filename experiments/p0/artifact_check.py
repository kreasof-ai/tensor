"""Run executable artifacts in fresh consumer interpreters and record wall time.

Pass --runtime-python from a NumPy-only venv. No compiler is imported here.
The process wall time includes interpreter startup, imports, validation,
benchmark iterations, reporting and teardown; the consumer separately reports
its time to first result. These two measurements are not interchangeable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def check(artifacts: list[Path], runtime_python: str, runs: int) -> dict:
    records = []
    for path in artifacts:
        for run in range(runs):
            started = time.perf_counter()
            proc = subprocess.run([runtime_python, "-m", "experiments.p0.artifact_run", "validate",
                                   str(path.resolve()), "--seed", str(run)],
                                  cwd=Path(__file__).resolve().parents[2],
                                  capture_output=True, text=True, timeout=120)
            elapsed = time.perf_counter() - started
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError:
                result = {"status": "failed", "error": proc.stdout[-2000:] + proc.stderr[-2000:]}
            records.append({"artifact": str(path.resolve()), "run": run,
                            "process_wall_seconds": elapsed, "exit_code": proc.returncode,
                            "result": result})
    passed = all(row["exit_code"] == 0 and row["result"].get("status") == "passed" for row in records)
    skipped = all(row["result"].get("status") == "skipped" for row in records)
    summary = {}
    for path in artifacts:
        rows = [row for row in records if row["artifact"] == str(path.resolve()) and row["exit_code"] == 0]
        if rows:
            summary[str(path.resolve())] = {
                "fresh_process_wall_median_seconds": statistics.median(row["process_wall_seconds"] for row in rows),
                "process_entry_to_first_result_median_seconds": statistics.median(
                    row["result"]["timings"]["process_entry_to_first_result_seconds"] for row in rows),
            }
    return {"status": "passed" if passed else "skipped" if skipped else "failed",
            "runs_per_artifact": runs, "runtime_python": runtime_python,
            "summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--runtime-python", default=sys.executable)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("runs must be positive")
    try:
        result = check(args.artifacts, args.runtime_python, args.runs)
    except (OSError, subprocess.SubprocessError) as exc:
        result = {"status": "failed", "error": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(args.report.resolve())}, indent=2))
    return 0 if result["status"] == "passed" else 2 if result["status"] == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
