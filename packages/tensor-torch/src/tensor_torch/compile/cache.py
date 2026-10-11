"""Content-addressed FX artifacts, validation, and atomic cache publication."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from tensor.artifacts.format import read_artifact

_CACHE_SCHEMA = 1


def artifact(source, target, cache_dir, nvrtc_home):
    identity = {
        "schema": _CACHE_SCHEMA,
        "source": source,
        "target": target,
        "frontend": "tilelang==0.1.14",
        "runtime_abi": "1.1",
        "compiler": "nvrtc",
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = cache_dir / (digest + ".tbin")
    hit = False
    if path.is_file():
        try:
            manifest, _ = read_artifact(path)
            hit = (
                manifest["target"] == target
                and manifest["source_sha256"]
                == hashlib.sha256(source.encode()).hexdigest()
            )
        except ValueError:
            pass
    if not hit:
        from tensor import build

        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir, prefix=".region-") as temp:
            source_path = Path(temp) / "kernel.py"
            source_path.write_bytes(source.encode())
            output = Path(temp) / "kernel.tbin"
            build(
                source_path,
                output,
                target=target,
                compiler="nvrtc",
                nvrtc_home=nvrtc_home,
                cache_dir=cache_dir / "compiler",
            )
            os.replace(output, path)
    return path, hit
