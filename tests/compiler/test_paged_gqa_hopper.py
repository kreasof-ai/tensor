"""Paged attention tails, precision, split merging and native overlap contract."""

import os

import pytest

from tensor.compiler.cuda.paged_gqa_hopper import make_hopper_partial


@pytest.mark.parametrize("bk", [64, 128])
def test_native_attention_queues_next_qk_before_pv_and_only_waits_for_qk(bk):
    import tilelang
    from tvm.target import Target

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with target, tilelang.transform.PassContext(opt_level=3, config={
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True,
        "tl.disable_thread_storage_sync": True,
    }):
        lowered = tilelang.lower(
            make_hopper_partial(3, 8, 16, 2, 2048, 6144, 1, 0, 4, bk=bk),
            target=target, enable_device_compile=False,
        )
    source = str(lowered.kernel_source)
    loop = source[source.index("for (int tile_1"):]
    assert "tensor_gqa_cp16" in source
    assert "kFloat8_e4m3" in source and "kBFloat16" in source
    assert "tl::wait_wgmma<1>" in loop
    assert loop.index("desc_a_1") < loop.index("desc_a_2") < loop.index("tl::wait_wgmma<1>")
    assert "__syncthreads()" not in loop
    assert "warpgroup_fence_operand" in loop and ".arrive()" in loop


@pytest.mark.skipif(os.environ.get("TENSOR_P1_CUDA") != "1", reason="CUDA execution")
@pytest.mark.parametrize("length,page,window,stages", [(1, 1, 0, 2), (8, 64, 129, 3)])
def test_mirror_matches_dense_attention_after_fresh_graph_values(length, page, window, stages):
    import tilelang
    import torch

    batch, hq, hkv, prefix, splits = 2, 16, 2, 1089, 2
    pages = (prefix + length + page - 1) // page
    rows = batch * pages * page
    gen = torch.Generator(device="cuda").manual_seed(81019)
    query = torch.randn((batch * length, hq, 64), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    kc = torch.randn((rows, hkv, 64), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    vc = torch.randn((rows, hkv, 64), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    table = torch.randperm(batch * pages, generator=gen, device="cuda").reshape(batch, pages).int()
    lengths = torch.tensor([prefix, 41], device="cuda", dtype=torch.int32)
    ks, vs = torch.tensor([1 / 64], device="cuda"), torch.tensor([0.25], device="cuda")
    out = torch.empty((batch, hkv, splits, 64, 64), device="cuda")
    stats = torch.empty((batch, hkv, splits, 64, 2), device="cuda")
    major, minor = torch.cuda.get_device_capability()
    kernel = tilelang.compile(make_hopper_partial(batch, length, hq, hkv, pages, rows, page,
        window, splits, bk=64, stages=stages, native_async=False),
        target={"kind": "cuda", "arch": f"sm_{major}{minor}"}, pass_configs={
            "tl.disable_tma_lower": True, "tl.disable_warp_specialized": True,
            "tl.disable_wgmma": True, "tl.disable_thread_storage_sync": True,
        })
    kernel(query, kc, vc, ks, vs, table, lengths, out, stats)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(query, kc, vc, ks, vs, table, lengths, out, stats)
    for factor in (1.0, -0.5):
        query.copy_((query.float() * factor).to(query.dtype))
        vc.copy_((-vc.float()).to(vc.dtype))
        ks.mul_(1.25)
        table.copy_(table.roll(1, 1))
        graph.replay()
        for b, seq in enumerate(lengths.tolist()):
            positions = torch.arange(seq + length, device="cuda")
            physical = table[b, positions // page].long() * page + positions % page
            keys, values = kc[physical].float() * ks, vc[physical].float() * vs
            q = query[b * length:(b + 1) * length].float().reshape(length, hkv, 8, 64)
            scores = torch.einsum("lhgd,shd->lhgs", q, keys) * (5.4730064863875 * .125)
            at = seq + torch.arange(length, device="cuda")
            mask = positions[None, :] <= at[:, None]
            if window:
                mask &= positions[None, :] > at[:, None] - window
            ref = torch.einsum("lhgs,shd->lhgd", torch.softmax(scores.masked_fill(~mask[:, None, None, :], -torch.inf), -1), values)
            maximum, denominator = stats[b, ..., 0], stats[b, ..., 1]
            weights = torch.exp2(maximum - maximum.amax(1, keepdim=True))
            actual = (out[b] * weights[..., None]).sum(1) / (denominator * weights).sum(1)[..., None]
            actual = actual[:, :length * 8].reshape(hkv, length, 8, 64).permute(1, 0, 2, 3)
            error = torch.linalg.vector_norm(actual - ref) / torch.linalg.vector_norm(ref)
            assert torch.isfinite(actual).all()
            assert float(error) < .004
