"""Experiment identity, including uncommitted source and the exact environment."""

from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def snapshot() -> dict:
    def git(*args):
        try:
            proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=10)
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    files = [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / ".python-version"]
    files += sorted(path for path in (ROOT / "experiments").rglob("*")
                    if path.suffix in (".py", ".cpp"))
    files += sorted((ROOT / "tools").glob("*.py"))
    files += sorted((ROOT / "tools").glob("*.ps1"))
    files += sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    digest = hashlib.sha256()
    for path in files:
        if path.is_file() and "out" not in path.relative_to(ROOT).parts:
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    packages = {}
    for name in ("tilelang", "apache-tvm-ffi", "torch", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    status = git("status", "--porcelain")
    lock = ROOT / "uv.lock"
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "source_sha256": digest.hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "packages": packages,
    }
