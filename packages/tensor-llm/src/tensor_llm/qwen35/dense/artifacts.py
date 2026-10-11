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
            "mtp",
            "recurrent",
            "speculative",
            "speculative_graph",
            "server",
        )
    ]
    modules.extend(
        (
            "tensor_llm.common.scheduler",
            "tensor_llm.common.artifacts",
            "tensor_llm.common.checkpoint",
            "tensor_llm.common.state",
            "tensor_llm.common.plan",
            "tensor_llm.kernels.elementwise",
            "tensor_llm.kernels.hybrid_batch",
            "tensor_llm.kernels.split_linear",
            "tensor_llm.kernels.head",
            "tensor_llm.kernels.recurrent",
            "tensor_llm.kernels.speculation",
            "tensor_llm.kernels.hybrid",
            "tensor_llm.kernels.mtp",
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
