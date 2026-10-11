"""Cache identity for every WebGPU lowering pass and schedule implementation."""

import hashlib
from pathlib import Path


def lowering_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for directory in (root / "passes", root / "schedules"):
        for path in sorted(directory.glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
