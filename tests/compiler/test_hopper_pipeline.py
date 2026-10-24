"""Ownership, generated instructions and fresh-value execution of Hopper schedules."""

import os

import pytest

from tensor.compiler.cuda.hopper import ring_events, staged_matmul


@pytest.mark.parametrize("bm,queue", [(8, 0), (32, 1), (64, 1), (128, 2)])
def test_manual_sync_keeps_entry_and_async_ownership_without_loop_cta_barriers(
    bm, queue
):
    import tilelang
    from tvm.target import Target

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with (
        target,
        tilelang.transform.PassContext(
            opt_level=3,
            config={
                "tl.disable_tma_lower": False,
                "tl.disable_warp_specialized": True,
                "tl.disable_thread_storage_sync": True,
            },
        ),
    ):
        lowered = tilelang.lower(
            staged_matmul(
                128,
                4096,
                8192,
                block_m=bm,
                block_n=64,
                block_k=128,
                stages=max(2, queue + 1),
                queue_depth=queue,
                splits=4,
                register_epilogue=True,
                manual_sync=True,
            ),
            target=target,
            enable_device_compile=False,
        )
    source = str(lowered.kernel_source)
    warm, steady, drain = (
        source.index("for (int warm"),
        source.index("for (int tile"),
        source.rindex("tl::wait_wgmma<0>"),
    )
    assert source.index(".init(128)") < source.index("__syncthreads()") < warm
    assert "__syncthreads()" not in source[steady:]
    assert ".wait(" in source[steady:drain] and ".arrive();" in source[steady:drain]
    assert f"tl::wait_wgmma<{queue}>" in source[steady:drain]
    assert "warpgroup_fence_operand" in source[drain:]


def test_manual_sync_rejects_kernels_without_the_async_ownership_contract():
    for opts in (
        {"native_async": False},
        {"register_epilogue": False},
        {"cluster_reduce": True},
    ):
        with pytest.raises(ValueError):
            staged_matmul(
                128,
                256,
                8192,
                block_n=64,
                splits=4,
                **({"register_epilogue": True} | opts),
                manual_sync=True,
            )


def test_search_exposes_pipeline_and_weight_layout_and_filters_unsafe_choices():
    from tensor.compiler.cuda.schedules import hopper_legal, hopper_space
    from tensor.compiler.search import ScheduleSearch

    spaces = hopper_space(128, paired=True)
    seed = {
        "family": "hopper_pipeline",
        "block_m": 32,
        "block_n": 32,
        "block_k": 256,
        "stages": 2,
        "queue_depth": 1,
        "splits": 4,
        "transposed_weights": True,
    }
    search = ScheduleSearch(
        [seed],
        spaces=spaces,
        legal=lambda c: hopper_legal(
            c, target="sm_90a", depth=8192, paired=True, outputs=256
        ),
    )
    assert search.next() == seed
    assert not hopper_legal(
        seed | {"stages": 1}, target="sm_90a", depth=8192, paired=True
    )
    assert not hopper_legal(seed, target="sm_89", depth=8192, paired=True)
    assert not hopper_legal(seed, target="sm_90a", depth=8192, paired=True, outputs=65)


@pytest.mark.parametrize(
    "stages,queue", [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (4, 2), (4, 3)]
)
def test_ring_never_overwrites_an_outstanding_mma(stages, queue):
    for tiles in (1, 2, 3, 8, 17, 64):
        slots, pending, completed = {}, set(), set()
        loaded = set()
        for event, tile, slot in ring_events(tiles, stages, queue):
            if event == "load":
                assert slots.get(slot) not in pending
                assert tile not in loaded
                slots[slot] = tile
                loaded.add(tile)
            elif event == "issue":
                assert slots[slot] == tile
                pending.add(tile)
                assert len(pending) <= queue + 1
            else:
                assert tile in pending
                pending.remove(tile)
                completed.add(tile)
        assert not pending and loaded == completed == set(range(tiles))


@pytest.mark.parametrize("splits", [2, 4, 8])
def test_cluster_reduction_orders_shared_memory_exchange_and_lifetime(splits):
    import tilelang
    from tvm.target import Target

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with (
        target,
        tilelang.transform.PassContext(
            opt_level=3,
            config={"tl.disable_tma_lower": False, "tl.disable_warp_specialized": True},
        ),
    ):
        lowered = tilelang.lower(
            staged_matmul(
                128,
                256,
                8192,
                paired=True,
                splits=splits,
                queue_depth=0,
                cluster_reduce=True,
            ),
            target=target,
            enable_device_compile=False,
        )
    source = str(lowered.kernel_source)
    fn = next(iter(lowered.device_mod.functions.values()))
    assert [int(x) for x in fn.attrs["cluster_dims"]] == [1, 1, splits]
    assert f"completed[0].init({splits})" in source
    assert source.count("cute::cluster_sync()") == 2
    drain = source.rindex("tl::wait_wgmma<0>")
    exchange = source.index("cute::cluster_sync()", drain)
    copy = source.index("tl::tma_store_cluster", exchange)
    wait = source.index("completed[0].wait(0)", copy)
    retire = source.index("cute::cluster_sync()", wait)
    assert drain < exchange < copy < wait < retire
    assert "tl::cluster_sync()" in source[:drain]


@pytest.mark.parametrize(
    "options",
    [
        {"stages": 1, "queue_depth": 1},
        {"queue_depth": 8},
        {"splits": 3},
        {"block_n": 16},
        {"stages": 8},
    ],
)
def test_rejects_unsafe_or_illegal_schedule(options):
    with pytest.raises(ValueError):
        staged_matmul(128, 256, 8192, paired=True, **options)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("queue", [0, 1])
def test_native_lowering_has_explicit_arrivals_and_bounded_waits(transposed, queue):
    import re

    import tilelang
    from tvm.target import Target

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    with (
        target,
        tilelang.transform.PassContext(
            opt_level=3,
            config={"tl.disable_tma_lower": False, "tl.disable_warp_specialized": True},
        ),
    ):
        lowered = tilelang.lower(
            staged_matmul(
                128,
                256,
                8192,
                paired=True,
                splits=4,
                transposed_weights=transposed,
                queue_depth=queue,
            ),
            target=target,
            enable_device_compile=False,
        )
    source = str(lowered.kernel_source)
    assert "CUtensorMap" in source and "wgmma_ss" in source
    assert f"tl::wait_wgmma<{queue}>" in source
    assert "tl::wait_wgmma<0>" in source
    assert ".arrive();" in source and ".init(128)" in source
    array_size = int(re.search(r"float acc\[(\d+)\]", source)[1])
    counts = [
        int(x) for x in re.findall(r"warpgroup_fence_operand\([^;]*, (\d+)\);", source)
    ]
    assert counts and all(x <= array_size for x in counts)
    if transposed:
        assert "for (int i_1 = 0; i_1 < 4;" in source


@pytest.mark.skipif(
    os.environ.get("TENSOR_P1_CUDA") != "1", reason="native CUDA execution"
)
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_synchronous_mirror_reads_fresh_inputs_and_weights(paired, transposed):
    import tilelang
    import torch

    m, n, k = 35, 65, 256
    x = torch.randn((m, k), device="cuda").bfloat16()
    original = torch.randn((n, k), device="cuda")
    high = original.bfloat16()
    low = (original - high.float()).bfloat16()
    packed = torch.stack((high, low), dim=1).reshape(2 * n, k) if paired else high
    weight = (packed.T if transposed else packed).contiguous()
    out = torch.empty((2, m, n), device="cuda", dtype=torch.float32)
    major, minor = torch.cuda.get_device_capability()
    kernel = tilelang.compile(
        staged_matmul(
            m,
            n,
            k,
            block_k=64,
            splits=2,
            paired=paired,
            transposed_weights=transposed,
            native_async=False,
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
        kernel(x, weight, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            kernel(x, weight, out)
        for _ in range(3):
            x.normal_()
            original.normal_()
            high = original.bfloat16()
            low = (original - high.float()).bfloat16()
            packed = (
                torch.stack((high, low), dim=1).reshape(2 * n, k) if paired else high
            )
            weight.copy_((packed.T if transposed else packed).contiguous())
            graph.replay()
            ref = x.double() @ (original if paired else high).double().T
            error = (out.double().sum(0) - ref).norm() / ref.norm()
            assert float(error) < (5e-5 if paired else 1e-5)
