"""Small export files for kernels defined by inspectable Python DSL factories."""

import hashlib
from importlib.util import find_spec
from pathlib import Path


def _canonical(value):
    """Keep specialization text stable across manifest serialization."""
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def export_source(module, factory, *arguments, dependencies, outputs=None):
    """Bind producer cache identity to every factory implementation it uses.

    The entry file contains specialization metadata only. Algorithm bodies live
    in importable Python functions; changes there must invalidate source caches.
    """
    digest = hashlib.sha256()
    for name in sorted(set((module, *dependencies))):
        digest.update(name.encode())
        digest.update(Path(find_spec(name).origin).read_bytes())
    arguments = tuple(_canonical(arg) for arg in arguments)
    extra = f", 'outputs': {outputs!r}" if outputs is not None else ""
    return (
        f"# DSL implementation SHA256: {digest.hexdigest()}\n"
        f"from {module} import {factory}\n\n"
        f"def tensor_export():\n"
        f'    return {{"kernel": {factory}({", ".join(repr(arg) for arg in arguments)}){extra}}}\n'
    )


def primitive(arguments, algorithm):
    """Wrap a DSL macro in a PrimFunc with its specialized buffer/scalar ABI.

    Fused operators select optional operands in Python. The public IR builder
    creates parameters in ABI order, without generating or executing Python text.
    """
    import tilelang.language as T
    from tvm.script.ir_builder.tirx import arg

    def parameters():
        return [
            arg(
                name,
                T.Tensor(count if isinstance(count, (tuple, list)) else (count,), dtype)
                if count is not None
                else T.Var(name, dtype),
            )
            for name, count, dtype in arguments
        ]

    @T.prim_func
    def kernel():
        algorithm(*parameters())

    return kernel
