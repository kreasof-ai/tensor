"""Inspect artifacts, modules and frontend lowering from the CLI."""

import json


def execute_command(args, parser):
    if args.command == "inspect":
        from pathlib import Path
        from tensor.artifacts.format import ArtifactError
        from tensor.artifacts.inspect import inspect_artifact
        from tensor.compiler.inspect import inspect_source

        try:
            path = Path(args.path)
            if "::" in args.path:
                from tensor.modules.project import resolve_reference

                resolved = resolve_reference(
                    args.path,
                    project=args.project,
                    module_cache=args.module_cache,
                    provider=args.provider or "cuda",
                    target=args.target,
                    compile=args.compile,
                    compiler=args.compiler,
                    nvcc=args.nvcc,
                    nvrtc_home=args.nvrtc_home,
                )
                result = inspect_artifact(Path(resolved["path"]), args.stage)
            elif path.is_dir() or path.name == "tensor.json":
                from tensor.modules.content import _directory

                result = json.dumps(
                    _directory(path if path.is_dir() else path.parent).manifest,
                    indent=2,
                )
            elif path.suffix == ".tpack":
                from tensor.modules.archive import _archive

                result = json.dumps(_archive(path)[0], indent=2)
            elif path.suffix == ".whl":
                from tensor.modules.registry import read_wheel

                result = json.dumps(read_wheel(path)[0], indent=2)
            elif path.suffix == ".py":
                result = inspect_source(
                    path,
                    stage=args.stage,
                    target=args.target,
                    trace_dir=Path(args.out) if args.out else None,
                    provider=args.provider or "cuda",
                )
            else:
                result = inspect_artifact(path, args.stage)
        except (ArtifactError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"tensor inspect: {exc}\n")
        print(result)
        return 0
