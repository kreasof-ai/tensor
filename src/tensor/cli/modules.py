"""Module installation, packaging and publication commands."""

from __future__ import annotations

import json


def execute_command(args, parser):
    if args.command in ("add", "install", "pack", "publish", "resolve"):
        from tensor.modules.errors import ModuleError
        from tensor.modules.resolver import (
            add as add_module,
            install as install_modules,
        )
        from tensor.modules.archive import pack as pack_module
        from tensor.modules.project import resolve_reference
        from tensor.compiler.errors import BuildError
        from tensor.artifacts.format import ArtifactError

        try:
            if args.command == "add":
                result = add_module(
                    args.source,
                    args.project,
                    cache_dir=args.module_cache,
                    index_url=args.index_url,
                )
            elif args.command == "install":
                result = install_modules(
                    args.project,
                    cache_dir=args.module_cache,
                    frozen=args.frozen,
                    offline=args.offline,
                )
            elif args.command == "pack":
                result = pack_module(args.source, args.out, cache_dir=args.module_cache)
            elif args.command == "publish":
                from tensor.modules.registry import publish as publish_module

                result = publish_module(
                    args.source,
                    out_dir=args.out_dir,
                    distribution=args.distribution,
                    cache_dir=args.module_cache,
                    dry_run=args.dry_run,
                    repository=args.repository,
                    repository_url=args.repository_url,
                )
            else:
                result = resolve_reference(
                    args.reference,
                    project=args.project,
                    module_cache=args.module_cache,
                    provider=args.provider or "cuda",
                    target=args.target,
                    compile=args.compile,
                    compiler=args.compiler,
                    nvcc=args.nvcc,
                    nvrtc_home=args.nvrtc_home,
                    cache_dir=args.cache_dir,
                )
        except (ModuleError, BuildError, ArtifactError, OSError, ValueError) as exc:
            parser.exit(1, f"tensor {args.command}: {exc}\n")
        print(json.dumps(result, indent=2))
        return 0
