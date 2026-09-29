"""Audit successful compiler include/source reads from strace's openat output."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOTS = ("/nvrtc/include/", "/packages/tilelang/src/",
         "/packages/tilelang/3rdparty/cutlass/include/")
HEADER_SUFFIXES = {".h", ".hpp", ".cuh", ".inl", ".inc", ".hxx", ".hh"}


def audit(trace: Path, roots=ROOTS) -> dict:
    pending = {}
    headers, outside = set(), set()
    for line in trace.read_text().splitlines():
        if "<unfinished ...>" in line:
            if "openat(" in line:
                pending[line.split()[0]] = line.replace("<unfinished ...>", "")
            continue
        if "<... openat resumed>" in line:
            pid = line.split()[0]
            if pid not in pending:
                raise ValueError("resumed openat without its original call")
            line = pending.pop(pid) + line.split("<... openat resumed>", 1)[1]
        if "O_DIRECTORY" in line or not re.search(r"\)\s+= [0-9]+(?:\s|$)", line):
            continue
        match = re.search(r'openat\([^,]+, "([^"]+)"', line)
        if not match:
            continue
        path = match[1]
        if any(path.startswith(root) for root in roots):
            # Includes extensionless CCCL headers and source files read while
            # hashing compiler inputs, not just .h files used by NVRTC.
            headers.add(path)
        elif "/include/" in path or Path(path).suffix in HEADER_SUFFIXES:
            outside.add(path)
    if pending:
        raise ValueError("trace ends with unresolved openat calls")
    if not headers:
        raise ValueError("trace contains no successful compiler include reads")
    if outside:
        raise ValueError(f"compiler include reads outside explicit roots: {sorted(outside)}")
    return {"sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "successful_distinct_include_files": len(headers),
            "files_by_root": {root: sum(path.startswith(root) for path in headers) for root in roots},
            "suffix_counts": dict(sorted(Counter(Path(path).suffix for path in headers).items())),
            "allowed_roots": list(roots), "outside_roots": [], "files": sorted(headers)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--root", action="append", help="allowed include root; repeat to replace defaults")
    args = parser.parse_args()
    print(json.dumps(audit(args.trace, args.root or ROOTS), indent=2))
