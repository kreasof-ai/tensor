"""Packed schedules construct macros with locally resolved reference types."""

import pytest


def test_packed_integer_factory_resolves_local_macro_annotations():
    pytest.importorskip("tilelang")
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.packed_integer import packed_integer_matmul

    @T.macro
    def word(weight, column, block, component):
        return weight[(column * 8 + block) * 8 + component]

    @T.macro
    def scale(weight, column, block):
        return T.float32(1)

    @T.macro
    def store(output, residual, index, gate, up):
        output[index] = gate

    algorithm = packed_integer_matmul(4, 256, 16, word, scale, store)
    assert callable(algorithm)
