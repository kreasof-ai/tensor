"""Work conservation, bounded partials and fresh-value Stream-K execution."""

import os

import pytest

from tensor.compiler.cuda.streamk import (
    streamk_matmul,
    streamk_reduce,
    streamk_segments,
)


@pytest.mark.skipif(
    os.environ.get("TENSOR_NVRTC") != "1", reason="NVRTC instruction inspection"
)
def test_fixup_instruction_budget_does_not_expand_lookup_per_output(tmp_path):
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import tilelang
    from tvm.target import Target

    from tensor.compiler.cuda.lowering import lower_cuda_intrinsics
    from tensor.compiler.cuda.nvrtc import NvrtcCompiler

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump unavailable")
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with target, tilelang.transform.PassContext(opt_level=3):
        lowered = tilelang.lower(
            streamk_reduce(128, 2304, 8192, 132, block_m=64),
            target=target,
            enable_device_compile=False,
        )
    includes = (
        Path(tilelang.__file__).parent / "src",
        Path(tilelang.__file__).parent / "3rdparty/cutlass/include",
    )
    cubin = tmp_path / "fixup.cubin"
    cubin.write_bytes(
        NvrtcCompiler().compile(
            lower_cuda_intrinsics(str(lowered.kernel_source)), "sm_90a", includes
        )
    )
    sass = subprocess.check_output([cuobjdump, "--dump-sass", str(cubin)], text=True)
    instructions = re.findall(
        r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?[A-Z][A-Z0-9_.]+", sass
    )
    assert len(instructions) < 500


@pytest.mark.parametrize("tiles", [1, 3, 8, 9, 31, 256])
@pytest.mark.parametrize("k_tiles", [1, 7, 64])
def test_partition_covers_every_iteration_once_and_reduction_reads_written_slots(
    tiles, k_tiles
):
    for workers in (1, 2, 7, 17, 128, 132, 264, 528):
        if workers > tiles * k_tiles:
            continue
        coverage = [0] * (tiles * k_tiles)
        parts, complete = {}, {}
        for worker, tile, first, count, slot in streamk_segments(
            tiles, k_tiles, workers
        ):
            for pos in range(tile * k_tiles + first, tile * k_tiles + first + count):
                coverage[pos] += 1
            value = sum(range(first + 1, first + count + 1))
            if slot < 0:
                assert tile not in complete
                complete[tile] = value
            else:
                assert (worker, slot) not in parts
                parts[worker, slot] = value
        assert set(coverage) == {1}
        total = tiles * k_tiles
        for tile in range(tiles):
            first = ((tile * k_tiles + 1) * workers - 1) // total
            last = ((tile + 1) * k_tiles * workers - 1) // total
            if first == last:
                result = complete[tile]
            else:
                assert tile not in complete
                result = sum(
                    parts[w, 0 if tile == w * total // workers // k_tiles else 1]
                    for w in range(first, last + 1)
                )
            assert result == k_tiles * (k_tiles + 1) // 2


@pytest.mark.skipif(os.environ.get("TENSOR_P1_CUDA") != "1", reason="CUDA execution")
@pytest.mark.parametrize(
    "workers,out_dtype",
    [
        (1, "bfloat16"),
        (7, "bfloat16"),
        (18, "float32"),
        (33, "bfloat16"),
        (264, "bfloat16"),
        (528, "bfloat16"),
    ],
)
def test_full_and_boundary_tiles_ignore_stale_scratch_and_read_fresh_graph_values(
    workers, out_dtype
):
    import tilelang
    import torch

    m, n, k, bm, bn, bk = 35, 65, 4096 if workers >= 264 else 512, 16, 32, 64
    major, minor = torch.cuda.get_device_capability()
    target = {"kind": "cuda", "arch": f"sm_{major}{minor}"}
    opts = {"block_m": bm, "block_n": bn, "block_k": bk, "out_dtype": out_dtype}
    passes = {
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True,
        "tl.disable_wgmma": True,
    }
    producer = tilelang.compile(
        streamk_matmul(m, n, k, workers, **opts), target=target, pass_configs=passes
    )
    reducer = tilelang.compile(
        streamk_reduce(m, n, k, workers, **opts), target=target, pass_configs=passes
    )
    x = torch.randn((m, k), device="cuda").bfloat16()
    weight = torch.randn((n, k), device="cuda").bfloat16()
    x[::8] = 0
    x[::8, 71] = 8
    x[1].mul_(128)
    parts = torch.empty((workers, 2, bm, bn), device="cuda")
    out = torch.empty(
        (m, n),
        device="cuda",
        dtype=torch.bfloat16 if out_dtype == "bfloat16" else torch.float32,
    )
    producer(x, weight, parts, out)
    reducer(parts, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        producer(x, weight, parts, out)
        reducer(parts, out)
    for scale in (1.0, -0.5, 2.0):
        x.mul_(scale)
        weight.mul_(-0.25)
        parts.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        ref = x.double() @ weight.double().T
        error = torch.linalg.vector_norm(out.double() - ref, dim=1)
        norm = torch.linalg.vector_norm(ref, dim=1)
        assert torch.isfinite(out).all()
        assert float((error / norm.clamp_min(norm.mean() * 0.1)).max()) < (
            1e-2 if out_dtype == "bfloat16" else 2e-5
        )
