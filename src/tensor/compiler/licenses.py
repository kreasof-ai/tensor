"""Redistribution notices for compiler-generated artifacts."""

from __future__ import annotations

import re
from importlib.metadata import distribution
from pathlib import Path


def _notices(tilelang_root: Path) -> dict[str, bytes]:
    from tensor.compiler.errors import BuildError

    result = {}
    header = (
        tilelang_root / "3rdparty" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    )
    try:
        match = re.match(r"/\*.*?\*/", header.read_text(encoding="utf-8"), re.S)
    except OSError as exc:
        raise BuildError(f"CUTLASS redistribution notice unavailable: {exc}") from exc
    if not match or "SPDX-License-Identifier: BSD-3-Clause" not in match[0]:
        raise BuildError(
            "CUTLASS redistribution notice missing from installed TileLang"
        )
    result["licenses/cutlass-notice.txt"] = match[0].encode()
    dist = distribution("tilelang")
    for index, item in enumerate(dist.files or []):
        if "licenses" in item.parts:
            path = Path(dist.locate_file(item))
            if path.is_file():
                result[f"licenses/tilelang-{index}-{path.name}"] = path.read_bytes()
    return result
