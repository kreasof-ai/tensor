import pytest


@pytest.mark.parametrize("rows", [8192, 32768, 65536])
def test_large_static_reinterpret_view(rows):
    pytest.importorskip("tilelang")
    from tvm import tirx
    from tensor.compiler.frontend.view import buffer_view

    source = tirx.decl_buffer((rows, 11776), "uint8")
    alias = buffer_view(source, shape=(rows * 11776 // 4,), dtype="uint32")
    assert alias.data.same_as(source.data)
    assert int(alias.shape[0]) == rows * 11776 // 4


def test_symbolic_reinterpret_view():
    pytest.importorskip("tilelang")
    from tvm import tirx
    from tensor.compiler.frontend.view import buffer_view

    rows = tirx.Var("rows", "int32")
    source = tirx.decl_buffer((rows, 128), "uint8")
    alias = buffer_view(source, shape=(rows, 32), dtype="uint32")
    assert alias.data.same_as(source.data)


def test_rejects_mismatched_storage():
    pytest.importorskip("tilelang")
    from tvm import tirx
    from tensor.compiler.frontend.view import buffer_view

    source = tirx.decl_buffer((32768, 11776), "uint8")
    with pytest.raises(ValueError, match="storage bit count"):
        buffer_view(source, shape=(32768 * 11776,), dtype="uint32")
