"""Semantic CUDA prelowering preserves explicit code and rejects false claims."""

import os

import pytest


def test_explicit_kernel_is_not_rescheduled():
    pytest.importorskip("tilelang")
    import tilelang.language as T

    from tensor.compiler.cuda.prelower import lower_cuda_operations

    @T.prim_func
    def plain(x: T.Tensor((32,), "float32"), out: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                out[i] = x[i] + 1

    result, metadata = lower_cuda_operations(plain, "sm_89")
    assert result.same_as(plain) and metadata is None


@pytest.mark.parametrize("orientation", ["activation", "weight"])
def test_operation_becomes_native_gemm_and_retains_signature(orientation):
    pytest.importorskip("tilelang")
    from tensor.compiler.cuda.matmul import matmul
    from tensor.compiler.cuda.prelower import lower_cuda_operations
    from tensor.compiler.frontend.signature import frontend_arguments

    logical = matmul(129, 97, 65, orientation=orientation)
    lowered, metadata = lower_cuda_operations(logical, "sm_89")
    assert frontend_arguments(logical, {}) == frontend_arguments(lowered, {})
    assert "T.gemm" in lowered.script()
    assert metadata["schedule"]["orientation"] == orientation
    again, unchanged = lower_cuda_operations(lowered, "sm_89")
    assert again.same_as(lowered) and unchanged is None


def test_annotation_cannot_replace_different_computation():
    pytest.importorskip("tilelang")
    import tvm

    from tensor.compiler.cuda.matmul import matmul
    from tensor.compiler.cuda.prelower import lower_cuda_operations

    logical = matmul(32, 32, 64)
    altered = logical.with_body(tvm.tirx.Evaluate(0))
    with pytest.raises(ValueError, match="does not match"):
        lower_cuda_operations(altered, "sm_89")
    with pytest.raises(ValueError, match="shared-memory"):
        lower_cuda_operations(logical.with_attr("tensor.cuda.block_k", 1024), "sm_89")
    with pytest.raises(ValueError, match="warps"):
        lower_cuda_operations(matmul(32, 32, 64, block_m=16, block_n=16), "sm_89")


def test_partition_boundaries_and_accumulation_are_explicit():
    from tensor.compiler.cuda.matmul import matmul_partials
    from tensor.compiler.cuda.prelower import lower_cuda_operations

    op = matmul_partials(128, 97, 128, 2, block_m=128, block_n=32, block_k=32, stages=2)
    result, metadata = lower_cuda_operations(op, "sm_89")
    assert metadata["parameters"]["splits"] == 2
    assert metadata["parameters"]["out_dtype"] == "float32"
    assert tuple(int(x) for x in list(result.buffer_map.values())[-1].shape) == (
        2,
        128,
        97,
    )
    with pytest.raises(ValueError, match="align"):
        lower_cuda_operations(matmul_partials(128, 97, 128, 4, block_k=64), "sm_89")


@pytest.mark.skipif(
    os.environ.get("TENSOR_P1_CUDA") != "1", reason="native CUDA execution"
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("splits", [2, 4])
def test_partitioned_matmul_and_reduction_graph(tmp_path, dtype, splits):
    import torch

    import tensor as tx
    from tensor.providers.cuda.graph import CudaGraph

    partial = tmp_path / "partial.py"
    partial.write_text(f"""
from tensor.compiler.cuda.matmul import matmul_partials
def tensor_export():
    return {{'kernel':matmul_partials(129,97,128,{splits},dtype={dtype!r},block_m=64,block_n=32,block_k=32,stages=2,threads=512),'outputs':['out']}}
""")
    reduce = tmp_path / "reduce.py"
    reduce.write_text(f"""
from tensor.compiler.cuda.matmul import matmul_reduce
def tensor_export():
    return {{'kernel':matmul_reduce(129,97,{splits},{dtype!r}),'outputs':['out']}}
""")
    major, minor = torch.cuda.get_device_capability()
    target = "sm_90a" if major == 9 else f"sm_{major}{minor}"
    pa, ra = tmp_path / "partial.tbin", tmp_path / "reduce.tbin"
    tx.build(partial, pa, target=target)
    tx.build(reduce, ra, target=target)
    dt = getattr(torch, dtype)
    x = (torch.randn((129, 128), device="cuda") * 0.2).to(dt)
    w = (torch.randn((97, 128), device="cuda") * 0.2).to(dt)
    parts = torch.empty((splits, 129, 97), device="cuda", dtype=torch.float32)
    out = torch.empty((129, 97), device="cuda", dtype=dt)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), tx.Device(stream=int(stream.cuda_stream)) as device:
        kp, kr = device.load(pa), device.load(ra)
        bx, bw, bp, bo = [device.from_dlpack(t) for t in (x, w, parts, out)]

        def launch():
            kp.launch(bx, bw, bp)
            kr.launch(bp, bo)

        with CudaGraph(device, launch, resources=(kp, kr, bx, bw, bp, bo)) as graph:
            for scale in (1.0, -0.75):
                x.mul_(scale)
                graph.launch()
                device.synchronize()
                ref = torch.mm(x.float(), w.float().T)
                torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.003)


@pytest.mark.skipif(
    os.environ.get("TENSOR_NVRTC") != "1", reason="GPU-free pinned CUDA compiler check"
)
@pytest.mark.parametrize("target", ["sm_89", "sm_90a"])
def test_operation_builds_artifact_with_selected_geometry(tmp_path, target):
    from tensor.artifacts.format import read_artifact
    from tensor.compiler.build import build_artifact

    source = tmp_path / "matmul.py"
    source.write_text("""
from tensor.compiler.cuda.matmul import matmul
def tensor_export():
    return {'kernel':matmul(128,96,128,block_m=128,block_n=32,block_k=32,stages=2,orientation='weight'),'outputs':['out']}
""")
    path = tmp_path / "matmul.tbin"
    build_artifact(source, path, target=target)
    manifest, _ = read_artifact(path)
    assert manifest["cuda_prelowering"]["schedule"]["orientation"] == "weight"
    assert manifest["cuda_prelowering"]["parameters"]["m"] == 128
    assert manifest["compiler"]["prelowering_sha256"]
    assert {x["name"] for x in manifest["abi"]} == {"x", "w", "out"}


@pytest.mark.skipif(
    os.environ.get("TENSOR_P1_CUDA") != "1", reason="native CUDA execution"
)
@pytest.mark.parametrize("orientation", ["activation", "weight"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("threads", [128, 512])
def test_tails_and_fresh_input_graph_replay(tmp_path, orientation, dtype, threads):
    import torch

    import tensor as tx
    from tensor.providers.cuda.graph import CudaGraph

    source = tmp_path / "product.py"
    source.write_text(f"""
from tensor.compiler.cuda.matmul import matmul
def tensor_export():
    return {{'kernel':matmul(129,97,65,dtype={dtype!r},block_m=64,block_n=32,block_k=32,stages=2,threads={threads},orientation={orientation!r}),'outputs':['out']}}
""")
    artifact = tmp_path / "product.tbin"
    major, minor = torch.cuda.get_device_capability()
    target = "sm_90a" if major == 9 else f"sm_{major}{minor}"
    tx.build(source, artifact, target=target)
    dt = getattr(torch, dtype)
    x = (torch.randn((129, 65), device="cuda") * 0.2).to(dt)
    w = (torch.randn((97, 65), device="cuda") * 0.2).to(dt)
    out = torch.empty((129, 97), device="cuda", dtype=dt)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), tx.Device(stream=int(stream.cuda_stream)) as device:
        kernel = device.load(artifact)
        buffers = [device.from_dlpack(t) for t in (x, w, out)]
        with CudaGraph(
            device, lambda: kernel.launch(*buffers), resources=(kernel, *buffers)
        ) as graph:
            for scale in (1.0, -0.75):
                x.mul_(scale)
                graph.launch()
                device.synchronize()
                ref = torch.mm(x.float(), w.float().T)
                torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.003)
