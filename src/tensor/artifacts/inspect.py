"""tensor.artifacts.inspect: migrated implementation with explicit ownership."""

from __future__ import annotations

import json
from pathlib import Path
from tensor.artifacts.format import read_artifact


def inspect_artifact(path: Path, stage: str) -> str:
    manifest, files = read_artifact(path)
    if stage == "manifest":
        return json.dumps(manifest, indent=2)
    if stage == "tirx":
        return files["kernel.tirx.json"].decode("utf-8")
    raise ValueError(
        "artifact inspection supports manifest or tirx; pass source.py for target or passes"
    )
