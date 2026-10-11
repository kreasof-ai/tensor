"""Specialized Tensor kernel factories shared by the module templates."""

import hashlib
import importlib
import json

FAMILIES = {
    "elementwise": ("zero", "cast", "add", "gelu", "gelu_backward"),
    "linear": ("gemm", "gemm_gelu", "gemm_residual"),
    "embedding": ("embedding", "embedding_backward"),
    "normalization": ("norm", "norm_backward", "column_sum"),
    "attention": (
        "pack_qkv",
        "unpack_qkv",
        "merge_heads",
        "split_heads",
        "softmax",
        "softmax_backward",
    ),
    "loss": ("ce_parts", "ce_loss", "ce_backward"),
    "optimizer": ("sumsq", "sum_parts", "clip", "adamw"),
}


def identity(kind, parameters):
    return hashlib.sha256(
        json.dumps([kind, parameters], sort_keys=True).encode()
    ).hexdigest()[:24]


def factory_module(kind):
    for family, kinds in FAMILIES.items():
        if kind in kinds:
            return __name__ + "." + family
    raise ValueError(f"unknown training kernel {kind}")


def source(kind, parameters, schedule=None):
    from tensor.compiler.frontend.source import export_source

    module = factory_module(kind)
    return export_source(
        module,
        "make_kernel",
        kind,
        parameters,
        schedule,
        dependencies=(module, __name__, 'tensor.compiler.frontend.source'),
    )


def make_kernel(kind, parameters, schedule=None):
    return importlib.import_module(factory_module(kind)).make_kernel(
        kind, parameters, schedule
    )
