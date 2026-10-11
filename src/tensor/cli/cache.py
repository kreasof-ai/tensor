"""Build and module cache reports."""

from __future__ import annotations

import json


def execute_command(args, parser):
    if args.command == "cache":
        from pathlib import Path
        from tensor.compiler.cache import cache_info
        from tensor.modules.cache import cache_info as module_cache_info

        print(
            json.dumps(
                {
                    **cache_info(Path(args.cache_dir) if args.cache_dir else None),
                    "modules": module_cache_info(args.module_cache),
                },
                indent=2,
            )
        )
        return 0
