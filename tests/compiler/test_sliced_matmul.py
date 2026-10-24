"""Single-launch warp-sliced products preserve precision and fresh graph values."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("TENSOR_P1_CUDA") != "1", reason="native CUDA execution"
)
@pytest.mark.parametrize(
    "paired,bm,bn,warps,chunk",
    [
        (False, 8, 8, 4, 64),
        (False, 8, 16, 8, 64),
        (True, 8, 2, 4, 64),
        (True, 16, 4, 8, 64),
    ],
)
def test_fresh_values_tails_sparse_rows_and_outliers(paired, bm, bn, warps, chunk):
    import tilelang
    import torch

    from tensor.compiler.cuda.sliced import sliced_matmul

    m, n, k = bm + 3, 65, 1024
    x = torch.randn((m, k), device="cuda").bfloat16()
    weight = torch.randn((n, k), device="cuda")
    packed = torch.empty(
        (n * (2 if paired else 1), k), device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty(
        (m, n), device="cuda", dtype=torch.float32 if paired else torch.bfloat16
    )
    major, minor = torch.cuda.get_device_capability()
    kernel = tilelang.compile(
        sliced_matmul(
            m,
            n,
            k,
            block_m=bm,
            block_n=bn,
            warp_splits=warps,
            chunk=chunk,
            paired=paired,
            out_dtype="float32" if paired else "bfloat16",
        ),
        target={"kind": "cuda", "arch": f"sm_{major}{minor}"},
        pass_configs={
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
            "tl.disable_wgmma": True,
        },
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(x, packed, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            kernel(x, packed, out)
        for turn in range(3):
            weight.normal_()
            high = weight.bfloat16()
            low = (weight - high.float()).bfloat16()
            packed.copy_(
                torch.stack((high, low), 1).reshape(2 * n, k) if paired else high
            )
            x.normal_()
            x[:, 3] *= 30
            x[::8] = 0
            x[::8, k // 3] = 16
            graph.replay()
            reference = x.double() @ (weight if paired else high).double().T
            diff = (out.double() - reference).norm(dim=1)
            norm = reference.norm(dim=1)
            assert torch.isfinite(out).all()
            assert float(diff.norm() / norm.norm()) < (5e-5 if paired else 0.006)
            assert float((diff[::8] / norm[::8]).max()) < 0.001
