"""Validate PyTorch arguments and infer concrete or symbolic output metadata."""

import torch
from tensor.runtime.signature import bind_shapes, buffer_argument, resolve_shape


class _Metadata:
    def __init__(self, value):
        self.shape = tuple(value.shape)
        self.dtype = str(value.dtype).removeprefix("torch.")


def metadata(kernel, args, *, symbolic=False):
    if len(args) != len(kernel.inputs):
        raise ValueError(f"expected {len(kernel.inputs)} inputs, received {len(args)}")
    values = {}
    device = None
    for descriptor, value in zip(kernel.inputs, args):
        if buffer_argument(descriptor):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{descriptor['name']} must be a PyTorch tensor")
            if value.device.type != "cuda" or not value.is_contiguous():
                raise ValueError("Tensor kernels require contiguous CUDA tensors")
            if device is not None and value.device != device:
                raise ValueError("all inputs must be on the same CUDA device")
            device = value.device
            values[descriptor["name"]] = _Metadata(value)
        else:
            values[descriptor["name"]] = value
    if device is None:
        raise ValueError("a PyTorch kernel needs at least one tensor input")
    if not symbolic:
        symbols, _ = bind_shapes(kernel.manifest, values, {})
        shapes = [resolve_shape(d["shape"], symbols) for d in kernel.outputs]
    else:
        symbols = {}
        for descriptor in kernel.inputs:
            value = values[descriptor["name"]]
            if buffer_argument(descriptor):
                if (
                    len(value.shape) != len(descriptor["shape"])
                    or value.dtype != descriptor["dtype"]
                ):
                    raise ValueError("input rank or dtype mismatch")
                for dim, actual in zip(descriptor["shape"], value.shape):
                    if isinstance(dim, dict) and "var" in dim:
                        name = dim["var"]
                        if name in symbols:
                            torch._check(symbols[name] == actual)
                        symbols[name] = actual
            elif descriptor["name"] in kernel.manifest.get("symbols", {}):
                symbols[descriptor["name"]] = value

        def evaluate(expr):
            if type(expr) is int:
                return expr
            if "var" in expr:
                return symbols[expr["var"]]
            if "cast" in expr:
                return evaluate(expr["value"])
            lhs, rhs = map(evaluate, expr["args"])
            import operator

            return {
                "add": operator.add,
                "sub": operator.sub,
                "mul": operator.mul,
                "floordiv": operator.floordiv,
                "floormod": operator.mod,
                "min": torch.sym_min,
                "max": torch.sym_max,
            }[expr["op"]](lhs, rhs)

        for descriptor in kernel.inputs:
            if buffer_argument(descriptor):
                for dim, actual in zip(
                    descriptor["shape"], values[descriptor["name"]].shape
                ):
                    torch._check(evaluate(dim) == actual)
        shapes = [tuple(map(evaluate, d["shape"])) for d in kernel.outputs]
        for shape in shapes:
            for extent in shape:
                torch._check(extent > 0)
    return [
        (shape, getattr(torch, d["dtype"])) for shape, d in zip(shapes, kernel.outputs)
    ], device
