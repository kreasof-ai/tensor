"""Packed E2M1 grouped GEMM preserves routes, scales and fresh graph values."""

import os

import pytest

from tensor.compiler.cuda.nvfp4 import grouped_matmul, pipelined_grouped_matmul


def test_grouped_nvfp4_rejects_unaligned_tiles():
    for options in ({"block_m": 7}, {"block_n": 15}, {"block_k": 127}):
        with pytest.raises(ValueError):
            grouped_matmul(3, 96, 256, 3, 3, **options)


@pytest.mark.skipif(os.environ.get("TENSOR_P1_CUDA") != "1", reason="CUDA execution")
@pytest.mark.parametrize("m,bm,routed,k", [(3, 8, False, 512), (17, 16, True, 512), (35, 8, False, 512), (8, 8, False, 4096)])
@pytest.mark.parametrize("pipeline", [False, True])
def test_routed_fp4_tails_scales_and_fresh_graph_values(m, bm, routed, k, pipeline):
    import tilelang
    import torch

    n, experts, top, workers = 96, 3, 2, 7
    route_lists = [
        [r for r in range(m * top) if r % experts == e] for e in range(experts)
    ]
    routes = torch.full((experts, m), -1, device="cuda", dtype=torch.int32)
    for e, selected in enumerate(route_lists):
        routes[e, : len(selected)] = torch.tensor(selected, device="cuda")
    counts = torch.tensor(
        [len(r) for r in route_lists], device="cuda", dtype=torch.int32
    )
    tile_e, tile_r = [], []
    for e, selected in enumerate(route_lists):
        for start in range(0, len(selected), bm):
            tile_e.append(e)
            tile_r.append(start)
    maximum = experts * ((m + bm - 1) // bm)
    ebuf = torch.zeros(maximum, device="cuda", dtype=torch.int32)
    rbuf = torch.zeros_like(ebuf)
    ebuf[: len(tile_e)] = torch.tensor(tile_e, device="cuda")
    rbuf[: len(tile_r)] = torch.tensor(tile_r, device="cuda")
    total = torch.tensor([len(tile_e)], device="cuda", dtype=torch.int32)
    gen = torch.Generator(device="cuda").manual_seed(80193)
    x = torch.randn(
        (m * top if routed else m, k), generator=gen, device="cuda"
    ).bfloat16()
    x[0].mul_(32)
    packed = torch.randint(
        0, 256, (experts, n, k // 2), generator=gen, device="cuda", dtype=torch.uint8
    )
    # Every finite positive E4M3 bit pattern, including subnormal and zero scales.
    scale_bits = (torch.arange(experts * n * (k // 16), device="cuda") % 127).byte()
    scales = scale_bits.view(torch.float8_e4m3fn).reshape(experts, n, k // 16)
    gs = torch.tensor([0.003, 0.031, 0.25], device="cuda")
    out = torch.empty((m * top, n), device="cuda", dtype=torch.bfloat16)
    major, minor = torch.cuda.get_device_capability()
    kernel = tilelang.compile(
        (pipelined_grouped_matmul if pipeline else grouped_matmul)(
            m,
            n,
            k,
            experts,
            experts,
            top=top,
            block_m=bm,
            workers=workers,
            routed_input=routed,
            **({"native_async": False} if pipeline else {}),
        ),
        target={"kind": "cuda", "arch": f"sm_{major}{minor}"},
        pass_configs={
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
            "tl.disable_wgmma": True,
            **({"tl.disable_thread_storage_sync": True} if pipeline else {}),
        },
    )
    kernel(x, packed, scales, gs, routes, counts, ebuf, rbuf, total, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(x, packed, scales, gs, routes, counts, ebuf, rbuf, total, out)
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    for factor in (1.0, -0.75):
        x.mul_(factor)
        packed.bitwise_xor_(0x88)
        gs.mul_(1.25)
        out.fill_(float("nan"))
        graph.replay()
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        values = grid[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
        weight = (
            values.reshape(experts, n, k // 16, 16)
            * scales.float()[..., None]
            * gs[:, None, None, None]
        ).reshape(experts, n, k)
        ref = torch.empty_like(out, dtype=torch.float64)
        for e, selected in enumerate(route_lists):
            rows = torch.tensor(selected, device="cuda")
            ref[rows] = (
                x[rows if routed else rows // top].double() @ weight[e].double().T
            )
        error = torch.linalg.vector_norm(out.double() - ref) / torch.linalg.vector_norm(
            ref
        )
        assert torch.isfinite(out).all()
        assert float(error) < 0.006
