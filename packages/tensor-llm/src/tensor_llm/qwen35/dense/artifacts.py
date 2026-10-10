"""Implementation identity shared by the AOT producer and native consumer."""

import hashlib
import importlib.util
from pathlib import Path


def implementation():
    modules = [
        f"tensor_llm.qwen35.dense.{name}"
        for name in (
            "artifacts",
            "checkpoint",
            "engine",
            "head",
            "kernels",
            "mtp",
            "projections",
            "recurrent",
            "recurrent_kernels",
            "speculative",
            "spec_graph",
            "spec_kernels",
            "server",
        )
    ]
    modules.extend(
        (
            "tensor_llm.common.scheduler",
            "tensor_llm.common.artifacts",
            "tensor_llm.qwen35.kernels.decode",
            "tensor_llm.qwen35.kernels.mtp",
        )
    )
    return {
        name: hashlib.sha256(
            Path(importlib.util.find_spec(name).origin).read_bytes()
        ).hexdigest()
        for name in modules
    }


def validate_implementation(manifest, expected):
    if manifest.get("implementation") != expected:
        raise ValueError("dense implementation changed; regenerate the AOT bundle")
