"""Producer entry files bind imported DSL implementations to the build cache."""

import ast
from types import SimpleNamespace


def test_source_identity_tracks_every_factory_dependency(tmp_path, monkeypatch):
    from tensor.compiler import entry

    paths = {}
    for name in ("factory", "decoder"):
        path = tmp_path / (name + ".py")
        path.write_text(f"# {name} implementation\n")
        paths[name] = path
    monkeypatch.setattr(entry, "find_spec", lambda name: SimpleNamespace(origin=paths[name]))

    def source():
        return entry.export_source(
            "factory",
            "make_kernel",
            {"label": "quotes ' and \nlines", "n": 7},
            dependencies=("decoder",),
            outputs=["out"],
        )

    before = source()
    ast.parse(before)
    paths["decoder"].write_text("# updated decoder\n")
    after = source()
    assert before.splitlines()[0] != after.splitlines()[0]
    assert before.splitlines()[1:] == after.splitlines()[1:]
    assert "outputs" in after
    first = entry.export_source(
        "factory", "make_kernel", {"z": {"y": 2, "a": 1}, "a": (3, 4)}, dependencies=()
    )
    restored = entry.export_source(
        "factory", "make_kernel", {"a": [3, 4], "z": {"a": 1, "y": 2}}, dependencies=()
    )
    assert first == restored


def test_macro_primitive_preserves_named_buffers_shapes_and_scalar_abi():
    import pytest

    pytest.importorskip("tilelang")
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    from tensor.compiler.lowering import frontend_arguments

    @T.macro
    def affine(x, out, scale):
        with T.Kernel(1, threads=32):
            for i, j in T.Parallel(3, 7):
                out[i, j] = x[i, j] * scale

    kernel = primitive(
        [("x", (3, 7), "float32"), ("out", (3, 7), "float32"), ("scale", None, "float32")], affine
    )
    arguments = frontend_arguments(kernel, {})
    assert [a["name"] for a in arguments] == ["x", "out", "scale"]
    assert arguments[0]["shape"] == [3, 7]
    assert arguments[-1]["kind"] == "scalar"
