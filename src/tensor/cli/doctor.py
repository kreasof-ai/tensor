"""Environment diagnostic presentation."""

from __future__ import annotations

import json


def render(report: dict) -> str:
    lines = [f"Tensor doctor: {report['status'].replace('_', ' ')}"]
    if report.get("target"):
        lines.append(f"Target: {report['target']}")
    for name, check in report["checks"].items():
        lines.append(f"{name:9} {check['status']:11} {check['detail']}")
        if check.get("hint"):
            lines.append(f"          → {check['hint']}")
    return "\n".join(lines)


def execute_command(args, parser):
    if args.command == "doctor":
        from tensor.diagnostics import diagnose
        from tensor.cli.doctor import render

        if args.provider == "webgpu":
            from tensor.providers.webgpu.adapter import probe
            from tensor.runtime.errors import TensorRuntimeError

            try:
                report = probe(args.device)
                report["status"] = "run_ready"
            except (TensorRuntimeError, ValueError) as exc:
                report = {"status": "blocked", "provider": "webgpu", "detail": str(exc)}
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "run_ready" else 1
        report = diagnose(
            target=args.target,
            device=args.device,
            nvcc=args.nvcc,
            compiler=args.compiler,
            nvrtc_home=args.nvrtc_home,
        )
        print(json.dumps(report, indent=2) if args.json else render(report))
        return 0 if report["status"] in ("ready", "build_ready", "run_ready") else 1
