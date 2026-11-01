"""Shape assumptions enter real lowering and runtime dimensions share one binary."""

import hashlib
import os

import numpy as np
import pytest

import tensor as tx
from tensor.artifacts.format import read_artifact
from tensor.compiler.frontend.domain import constrain
from tensor.compiler.frontend.signature import frontend_arguments


def test_assumptions_keep_runtime_signature_and_are_visible_in_ir():
    from examples.dynamic_gemm import matmul

    symbols = {}
    arguments = frontend_arguments(matmul, symbols)
    domain = {"version": 1, "bounds": {"rows": {"min": 1, "max": 256}}, "relations": []}
    lowered = constrain(matmul, domain, symbols)
    assert frontend_arguments(lowered, {}) == arguments
    from tvm import tirx

    assumptions = []
    tirx.stmt_functor.post_order_visit(lowered.body, lambda node: assumptions.append(str(node.node))
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.assume" else None)
    assert any("rows <= 256" in fact for fact in assumptions)


def test_guarded_bounds_remove_an_impossible_runtime_branch():
    import tilelang.language as T

    rows = T.dynamic("rows")

    @T.prim_func
    def guarded(x: T.Tensor((rows,), "float32"), out: T.Tensor((rows,), "float32")):
        with T.Kernel(1, threads=32):
            for i in T.serial(rows):
                if rows > 256:
                    out[i] = T.float32(-100003)
                else:
                    out[i] = x[i] + 1

    domain = dict(version=1, bounds={"rows": dict(min=1, max=256)}, relations=[])
    assert "100003" in guarded.script()
    lowered = constrain(guarded, domain, {"rows": "int32"})
    assert "100003" not in lowered.script()
    assert frontend_arguments(guarded, {}) == frontend_arguments(lowered, {})


def test_explicit_extent_scalar_can_guard_a_capacity_buffer():
    import tilelang.language as T

    @T.prim_func
    def affine(x: T.Tensor((256,), "float32"), out: T.Tensor((256,), "float32"), rows: T.int32):
        with T.Kernel(T.ceildiv(rows, 32), threads=32) as block:
            for j in T.Parallel(32):
                i = block * 32 + j
                if i < rows:
                    out[i] = x[i] + 1

    domain = dict(version=1, bounds={"rows": dict(min=1, max=256)}, relations=[])
    symbols = {}
    arguments = frontend_arguments(affine, symbols, shape_domain=domain)
    assert symbols == {"rows": "int32"}
    lowered = constrain(affine, domain, symbols)
    assert frontend_arguments(lowered, {}) == arguments


@pytest.mark.parametrize("orientation", ["weight", "activation"])
def test_dynamic_matmul_uses_existing_semantic_prelowering(orientation):
    import tilelang.language as T
    from tensor.compiler.cuda.matmul import matmul
    from tensor.compiler.cuda.prelower import lower_cuda_operations

    rows = T.dynamic("rows")
    logical = matmul(rows, 97, 65, dtype="float16", orientation=orientation)
    lowered, metadata = lower_cuda_operations(logical, "sm_89")
    assert frontend_arguments(logical, {}) == frontend_arguments(lowered, {})
    assert metadata["parameters"]["m"] == {"var": "rows"}
    assert "T.gemm" in lowered.script()


@pytest.mark.skipif(os.getenv("TENSOR_CUDA") != "1", reason="native CUDA qualification")
def test_bounded_binary_rebinds_irregular_shapes_and_preserves_unused_capacity(tmp_path):
    source = tmp_path / "bounded.py"
    source.write_text('''import tilelang.language as T
from tensor.compiler.cuda.matmul import matmul
def tensor_export():
    return {"kernel": matmul(T.dynamic("rows"), 32, 32, dtype="float16"), "outputs": ["out"], "shape_domain": {
        "version": 1, "bounds": {"rows": {"min": 1, "max": 256}}, "relations": []}}
''')
    artifact = tmp_path / "bounded.tbin"
    tx.build(source, artifact, target="sm_89")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest, _ = read_artifact(artifact)
    assert manifest["format_version"] == 4
    with tx.Device() as device:
        rng = np.random.default_rng(19)
        a = (rng.normal(size=(256, 32)) * .1).astype("float16")
        b = (rng.normal(size=(32, 32)) * .1).astype("float16")
        left, right = device.from_numpy(a), device.from_numpy(b)
        out = device.full((256, 32), 31, "float16")
        sentinel = device.full((256, 32), 31, "float16")
        kernel = device.load(artifact)
        for rows in (1, 3, 7, 31, 32, 33, 127, 128, 129, 255, 256):
            # Unused logical capacity must never be overwritten by masked tiles.
            out.copy_from(sentinel)
            av, ov = left.prefix((rows, 32)), out.prefix((rows, 32))
            call = kernel.bind(av, right, ov)
            call.launch()
            expected = a[:rows].astype("float32") @ b.astype("float32").T
            np.testing.assert_allclose(ov.to_numpy(), expected, rtol=.02, atol=.002)
            np.testing.assert_array_equal(out.to_numpy()[rows:], 31)
            ov.release()
            av.release()
        with pytest.raises(ValueError, match="outside shape domain"):
            bad = device.empty((257, 32), "float16")
            kernel.bind(bad, right, bad)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == checksum


@pytest.mark.skipif(os.getenv("TENSOR_CUDA") != "1", reason="native CUDA qualification")
def test_capacity_buffers_use_a_guarded_explicit_live_extent(tmp_path):
    source = tmp_path / "capacity.py"
    source.write_text('''import tilelang.language as T
@T.prim_func
def affine(x: T.Tensor((256,), "float32"), out: T.Tensor((256,), "float32"), rows: T.int32):
    with T.Kernel(T.ceildiv(rows, 32), threads=32) as block:
        for j in T.Parallel(32):
            i = block * 32 + j
            if i < rows:
                out[i] = x[i] + 1
def tensor_export():
    return {"kernel": affine, "outputs": ["out"], "shape_domain": {
        "version": 1, "bounds": {"rows": {"min": 1, "max": 256}}, "relations": []}}
''')
    path = tmp_path / "capacity.tbin"
    tx.build(source, path, target="sm_89")
    with tx.Device() as device:
        values = np.arange(256, dtype="float32")
        x, out = device.from_numpy(values), device.full((256,), -31)
        sentinel = device.full((256,), -31)
        executable = device.load(path)
        for rows in (1, 3, 33, 127, 256):
            out.copy_from(sentinel)
            executable.bind(x, out, rows).launch()
            actual = out.to_numpy()
            np.testing.assert_array_equal(actual[:rows], values[:rows] + 1)
            np.testing.assert_array_equal(actual[rows:], -31)
        for rows in (0, 257):
            with pytest.raises(ValueError, match="outside shape domain"):
                executable.bind(x, out, rows)


@pytest.mark.skipif(os.getenv("TENSOR_CUDA") != "1", reason="native CUDA qualification")
def test_two_runtime_axes_share_one_guarded_image(tmp_path):
    source = tmp_path / "matrix.py"
    source.write_text('''import tilelang.language as T
batch, tokens = T.dynamic("batch"), T.dynamic("tokens")
@T.prim_func
def affine(x: T.Tensor((batch, tokens), "float32"), out: T.Tensor((batch, tokens), "float32")):
    with T.Kernel(T.ceildiv(batch * tokens, 256), threads=256) as block:
        for j in T.Parallel(256):
            i = block * 256 + j
            if i < batch * tokens:
                out[i // tokens, i % tokens] = x[i // tokens, i % tokens] + 7
def tensor_export():
    return {"kernel": affine, "outputs": ["out"], "shape_domain": {
        "version": 1, "bounds": {"batch": {"min": 1, "max": 256},
        "tokens": {"min": 8, "max": 4096, "multiple_of": 8}},
        "relations": [{"left": {"var": "batch"}, "op": "le", "right": {"var": "tokens"}}]}}
''')
    path = tmp_path / "matrix.tbin"
    tx.build(source, path, target="sm_89")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with tx.Device() as device:
        values = np.arange(256 * 4096, dtype="float32") / 128
        x, out = device.from_numpy(values), device.full(values.shape, -31)
        sentinel = device.full(values.shape, -31)
        executable = device.load(path)
        for batch, tokens in ((1, 8), (3, 24), (7, 136), (31, 248), (33, 264), (256, 4096)):
            out.copy_from(sentinel)
            left, result = x.prefix((batch, tokens)), out.prefix((batch, tokens))
            executable.bind(left, result).launch()
            np.testing.assert_array_equal(result.to_numpy().reshape(-1), values[:batch * tokens] + 7)
            np.testing.assert_array_equal(out.to_numpy()[batch * tokens:], -31)
            left.release()
            result.release()
        for shape in ((3, 25), (32, 24)):
            left, result = x.prefix(shape), out.prefix(shape)
            with pytest.raises(ValueError, match="divisible|relation"):
                executable.bind(left, result)
            left.release()
            result.release()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
