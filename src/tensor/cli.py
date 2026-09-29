"""Command line entry point. Compiler imports stay inside the selected command."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensor", description="Tensor kernel tooling")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check the CUDA build and device environment")
    doctor.add_argument("--target", help="explicit CUDA target, e.g. sm_86; permits a GPU-free build host")
    doctor.add_argument("--device", type=int, default=0, help="CUDA device ordinal (default: 0)")
    doctor.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    doctor.add_argument("--json", action="store_true", help="print a machine-readable report")
    build = commands.add_parser("build", help="compile a TileLang source file to a CUDA artifact")
    build.add_argument("source", help="Python source exporting tensor_export()")
    build.add_argument("--target", help="exact CUDA target, e.g. sm_86 (detected from device 0 by default)")
    build.add_argument("--nvcc", help="explicit path to the CUDA compiler")
    build.add_argument("--out", required=True, help="new output .tbin path")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from tensor.doctor import diagnose, render

        report = diagnose(target=args.target, device=args.device, nvcc=args.nvcc)
        print(json.dumps(report, indent=2) if args.json else render(report))
        return 0 if report["status"] in ("ready", "build_ready") else 1
    if args.command == "build":
        from pathlib import Path
        from tensor.build import BuildError, build_artifact

        try:
            report = build_artifact(Path(args.source), Path(args.out), target=args.target, nvcc=args.nvcc)
        except (BuildError, OSError, ValueError) as exc:
            parser.exit(1, f"tensor build: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0
    return 1
