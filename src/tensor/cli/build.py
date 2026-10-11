"""Build command presentation."""

from __future__ import annotations

import json


def execute_command(args, parser):
    if args.command == "build":
        from pathlib import Path
        from tensor.compiler.errors import BuildError
        from tensor.compiler.build import build_artifact

        try:
            if "::" in args.source:
                from tensor.modules.project import resolve_reference

                output = Path(args.out)
                if output.exists():
                    raise BuildError(f"output already exists: {output}")
                resolved = resolve_reference(
                    args.source,
                    project=args.project,
                    module_cache=args.module_cache,
                    provider=args.provider or "cuda",
                    target=args.target,
                    compile=True,
                    compiler=args.compiler,
                    nvcc=args.nvcc,
                    nvrtc_home=args.nvrtc_home,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                contents = Path(resolved["path"]).read_bytes()
                created = False
                try:
                    with output.open("xb") as stream:
                        created = True
                        stream.write(contents)
                except BaseException:
                    if created:
                        output.unlink(missing_ok=True)
                    raise
                report = {
                    **resolved,
                    "status": "built",
                    "path": str(output.resolve()),
                    "bytes": output.stat().st_size,
                }
            else:
                report = build_artifact(
                    Path(args.source),
                    Path(args.out),
                    target=args.target,
                    nvcc=args.nvcc,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                    compiler=args.compiler,
                    nvrtc_home=args.nvrtc_home,
                    provider=args.provider or "cuda",
                )
        except (BuildError, OSError, ValueError) as exc:
            parser.exit(1, f"tensor build: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0
