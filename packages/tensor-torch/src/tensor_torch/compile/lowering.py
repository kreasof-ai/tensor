"""Validate FX regions and emit Tensor kernel factories."""

from __future__ import annotations
import math
import torch
from torch.fx import Node
from .partition import kind, GEMM


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def emit(nodes, external, inputs):
    """Validate an FX region and emit metadata for an actual DSL factory."""
    from tensor.compiler.entry import export_source

    values = dict(zip(external, inputs))
    root = nodes[-1]
    base = next((n for n in nodes if kind(n) in {"gemm", "attention"}), None)
    if any(not isinstance(v, torch.Tensor) for v in inputs):
        raise ValueError("symbolic scalar FX inputs currently remain in PyTorch")
    if any(
        v.device.type != "cuda"
        or not v.is_contiguous()
        or (v.numel() == 0 or v.ndim == 0)
        for v in inputs
    ):
        raise ValueError("region needs nonempty contiguous CUDA tensors")
    if len({v.device for v in inputs}) != 1 or len({v.dtype for v in inputs}) != 1:
        raise ValueError("region needs one device and one dtype")
    dtype = _dtype(inputs[0])
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("region supports float16, bfloat16 and float32")
    if base is not None and kind(base) == "attention":
        if root is not base or len(nodes) != 1:
            raise ValueError("attention epilogues remain separate regions")
        q, k, v = [values[n] for n in base.args[:3]]
        options = dict(
            zip(
                ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"),
                base.args[3:],
            )
        )
        options.update(base.kwargs)
        if (
            dtype not in ("float16", "bfloat16")
            or q.ndim != 4
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-1] not in (64, 128)
            or options.get("attn_mask") is not None
            or options.get("dropout_p", 0) != 0
            or options.get("enable_gqa", False)
            or options.get("scale") not in (None, q.shape[-1] ** -0.5)
        ):
            raise ValueError(
                "SDPA profile requires FP16/BF16 BHSD self-attention, D64/128, no mask/dropout/GQA"
            )
        module = "tensor_torch.compile.attention"
        source = export_source(
            module,
            "make_kernel",
            tuple(q.shape),
            bool(options.get("is_causal", False)),
            dtype,
            dependencies=(
                module,
                "tensor_torch.compile.lowering",
                "tensor.compiler.entry",
            ),
            outputs=["out"],
        )
        return source, tuple(external.index(n) for n in base.args[:3]), "attention"

    specification = {"dtype": dtype, "inputs": [tuple(v.shape) for v in inputs]}
    if base is None:
        shape = tuple(torch.broadcast_shapes(*(v.shape for v in inputs)))
        if any(
            tuple(v.shape) not in (shape, (shape[-1],)) and v.numel() != 1
            for v in inputs
        ):
            raise ValueError(
                "pointwise broadcast supports full tensors, trailing bias, or singleton"
            )
        available = set(external)
    else:
        if kind(base) != "gemm" or dtype not in ("float16", "bfloat16"):
            raise ValueError("matrix multiplication needs float16 or bfloat16")
        a, b = [values[n] for n in base.args[:2]]
        transpose = GEMM[base.target]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matrix multiplication currently needs rank 2")
        m, k = a.shape
        n, kb = (b.shape[0], b.shape[1]) if transpose else (b.shape[1], b.shape[0])
        if k != kb:
            raise ValueError("GEMM reduction dimension mismatch")
        shape = (m, n)
        if any(
            tuple(v.shape) not in ((m, k), tuple(b.shape), shape, (n,)) for v in inputs
        ):
            raise ValueError("unsupported GEMM epilogue broadcast")
        if any(base.kwargs):
            raise ValueError("GEMM keyword arguments unsupported")
        specification["gemm"] = {
            "m": m,
            "k": k,
            "n": n,
            "transpose": transpose,
            "a": external.index(base.args[0]),
            "b": external.index(base.args[1]),
        }
        if transpose and len(base.args) > 2 and base.args[2] is not None:
            specification["gemm"]["bias"] = external.index(base.args[2])
        available = (set(external) - set(base.args[:2])) | {base}
    specification["shape"] = shape
    names = {n: "arg" + str(i) for i, n in enumerate(external)}
    if base is not None:
        names[base] = base.name
        specification["gemm"]["name"] = base.name

    def operand(value):
        if isinstance(value, Node):
            if value not in available:
                raise ValueError("unsupported region dependency")
            return {"node": names[value]}
        if type(value) in (int, float) and math.isfinite(value):
            return {"scalar": value}
        raise ValueError("unsupported pointwise operand")

    operations = []
    for node in nodes:
        if node is base:
            continue
        category = kind(node)
        args = [operand(a) for a in node.args]
        kwargs = node.kwargs
        if set(kwargs) - {"alpha", "inplace"}:
            raise ValueError("unsupported pointwise keyword arguments")
        if category in {"add", "sub", "mul", "div"}:
            if len(args) != 2:
                raise ValueError("binary operation needs two arguments")
            alpha = kwargs.get("alpha", 1)
            if type(alpha) not in (float, int) or not math.isfinite(alpha):
                raise ValueError("alpha must be a finite scalar")
        elif category not in {"neg", "relu", "sigmoid", "tanh"}:
            raise ValueError("unsupported epilogue")
        operations.append(
            {
                "name": node.name,
                "kind": category,
                "args": args,
                "alpha": kwargs.get("alpha", 1),
            }
        )
        names[node] = node.name
        available.add(node)
    specification["operations"], specification["root"] = operations, names[root]
    source = export_source(
        "tensor_torch.compile.kernels",
        "make_kernel",
        specification,
        dependencies=("tensor_torch.compile.lowering", "tensor.compiler.entry"),
        outputs=["out"],
    )
    return source, tuple(range(len(inputs))), "gemm" if base else "pointwise"
