"""Registers implementation for WebGPU kernels."""

from itertools import product


def registers(shape, dtype="float32", *, initialize=True):
    """Allocate separate scalar registers, rather than a shader private array."""
    import tilelang.language as T

    return {
        index: T.alloc_var(dtype, init=0) if initialize else T.alloc_var(dtype)
        for index in product(*(range(size) for size in shape))
    }


def shared_buffers(shape, dtype, count):
    import tilelang.language as T

    return [T.alloc_shared(shape, dtype) for _ in range(count)]


def vector(values):
    import tilelang.language as T

    if len(values) == 1:
        return values[0]
    return T.call_extern(f"float32x{len(values)}", f"vec{len(values)}<f32>", *values)
