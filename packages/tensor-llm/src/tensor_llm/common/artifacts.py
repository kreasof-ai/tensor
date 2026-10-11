"""Stable kernel identities shared by model producers and runtime executors."""

import hashlib
import json
from pathlib import Path


def identity(kind, parameters):
    return hashlib.sha256(
        json.dumps([kind, parameters], sort_keys=True).encode()
    ).hexdigest()[:24]


def artifact_path(directory, record):
    """Resolve and check a physical artifact for either manifest format."""
    directory = Path(directory).resolve()
    descriptor = record.get("logical", record)
    path = (
        directory / descriptor.get("path", descriptor.get("artifact", ""))
    ).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("artifact path escapes its bundle or is missing")
    with path.open("rb") as image:
        checksum = hashlib.file_digest(image, "sha256").hexdigest()
    if checksum != descriptor["sha256"]:
        raise ValueError("artifact checksum/integrity mismatch")
    return path


def load_kernels(device, directory, records, *, requirements=None):
    """Share integrity checks and cleanup across all executor/model adapters."""
    if requirements is not None and records.keys() != requirements.keys():
        raise ValueError("incomplete/incompatible kernel coverage")
    kernels = {}
    try:
        for key, row in records.items():
            if requirements is not None:
                kind, parameters = requirements[key]
                if row["kind"] != kind or row["parameters"] != parameters:
                    raise ValueError("artifact identity mismatch")
            kernels[key] = device.load(artifact_path(directory, row))
        return kernels
    except BaseException:
        for kernel in kernels.values():
            kernel.release()
        raise
