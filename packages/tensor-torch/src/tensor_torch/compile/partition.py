"""Recognize supported FX operations and select bounded compilation regions."""

from __future__ import annotations

import operator

import torch

POINTWISE = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "div",
    operator.neg: "neg",
    torch.add: "add",
    torch.sub: "sub",
    torch.mul: "mul",
    torch.div: "div",
    torch.relu: "relu",
    torch.sigmoid: "sigmoid",
    torch.tanh: "tanh",
    torch.nn.functional.relu: "relu",
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.mul.Scalar: "mul",
    torch.ops.aten.div.Tensor: "div",
    torch.ops.aten.neg.default: "neg",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.tanh.default: "tanh",
}
GEMM = {
    operator.matmul: False,
    torch.matmul: False,
    torch.mm: False,
    torch.nn.functional.linear: True,
    torch.ops.aten.mm.default: False,
}
SDPA = {torch.nn.functional.scaled_dot_product_attention}


def kind(node):
    if node.op == "call_method" and node.target in {"relu", "sigmoid", "tanh"}:
        return node.target
    if node.op == "call_function":
        return POINTWISE.get(node.target) or (
            "gemm"
            if node.target in GEMM
            else "attention" if node.target in SDPA else None
        )
    return None


def eligible(node):
    value = node.meta.get("example_value", node.meta.get("val"))
    return (
        kind(node) is not None
        and isinstance(value, torch.Tensor)
        and value.device.type == "cuda"
        and value.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and value.is_contiguous()
        and not node.kwargs.get("inplace", False)
        and not node.kwargs.get("out")
        and not node.kwargs.get("rounding_mode")
    )


def regions(graph):
    claimed = set()
    result = []
    for root in reversed(list(graph.nodes)):
        if root in claimed or not eligible(root):
            continue
        selected = set()
        bases = []

        def collect(node):
            if node in selected or node in claimed or not eligible(node):
                return
            category = kind(node)
            if category in {"gemm", "attention"}:
                if bases or (category == "attention" and node is not root):
                    return
                bases.append(node)
                selected.add(node)
                return
            selected.add(node)
            for predecessor in node.all_input_nodes:
                if len(predecessor.users) == 1:
                    collect(predecessor)

        collect(root)
        ordered = [n for n in graph.nodes if n in selected]
        external = list(
            dict.fromkeys(
                n for item in ordered for n in item.all_input_nodes if n not in selected
            )
        )
        result.append((root, ordered, external))
        claimed.update(selected)
    return list(reversed(result))
