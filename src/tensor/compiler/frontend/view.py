"""Checked dense buffer aliases without narrowing shape arithmetic to int32."""

from math import prod


def buffer_view(source, shape=None, dtype=None):
    """Return a dense, zero-offset alias with the same storage bit count.

    Static products are evaluated as Python integers. Symbolic products use
    int64 before multiplication, so large buffers cannot overflow an int32
    shape check. This does not allocate storage or change indexing types.
    """
    import tilelang.language as T
    import tvm
    from tvm import tirx

    shape = tuple(source.shape if shape is None else shape)
    dtype = source.dtype if dtype is None else dtype
    analyzer = tvm.arith.Analyzer()
    if not analyzer.can_prove_equal(source.elem_offset, 0):
        raise ValueError("buffer_view requires zero-offset source storage")
    stride = 1
    for extent, actual in zip(reversed(source.shape), reversed(source.strides)):
        if not analyzer.can_prove_equal(actual, stride):
            raise ValueError("buffer_view requires dense source storage")
        stride = stride * extent

    def bits(extents, element_type):
        element = tvm.DataType(element_type)
        if all(isinstance(v, (int, tirx.IntImm)) for v in extents):
            values = [int(v) for v in extents]
            if any(v < 0 for v in values):
                raise ValueError("buffer_view extents cannot be negative")
            return prod(values) * element.bits * element.lanes
        result = tirx.IntImm("int64", element.bits * element.lanes)
        for value in extents:
            result = result * tirx.Cast("int64", value)
        return result

    actual, expected = bits(shape, dtype), bits(source.shape, source.dtype)
    if isinstance(actual, int) and isinstance(expected, int):
        equal = actual == expected
    else:
        actual = tirx.IntImm("int64", actual) if isinstance(actual, int) else actual
        expected = tirx.IntImm("int64", expected) if isinstance(expected, int) else expected
        equal = tvm.arith.Analyzer().can_prove_equal(actual, expected)
    if not equal:
        raise ValueError("buffer_view must preserve the source storage bit count")
    return T.Tensor(shape, dtype, source.data, scope=source.scope())
