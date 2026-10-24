"""Direct accumulator stores and PTX pair exchange preserve FP32 router math."""

import os

import pytest

from tensor.compiler.cuda.hopper import staged_matmul


def test_search_exposes_register_control_and_filters_incompatible_clusters():
    from tensor.compiler.cuda.schedules import hopper_legal, hopper_register_space

    axes = hopper_register_space(128, paired=True)["hopper_register"]
    assert axes["register_epilogue"] == (False, True)
    assert axes["block_n"] == (32,)
    config = {
        "block_m": 32,
        "block_n": 32,
        "block_k": 128,
        "stages": 2,
        "queue_depth": 0,
        "splits": 4,
        "transposed_weights": True,
        "register_epilogue": True,
    }
    assert hopper_legal(config, target="sm_90a", depth=8192, paired=True, outputs=256)
    assert not hopper_legal(
        config | {"cluster_reduce": True},
        target="sm_90a",
        depth=8192,
        paired=True,
        outputs=256,
    )


@pytest.mark.parametrize("paired", [False, True])
def test_register_epilogue_removes_shared_round_trip_after_drain(paired):
    import tilelang
    from tvm.target import Target

    sources = []
    for direct in (False, True):
        target = Target({"kind": "cuda", "arch": "sm_90a"})
        with (
            target,
            tilelang.transform.PassContext(
                opt_level=3,
                config={
                    "tl.disable_tma_lower": False,
                    "tl.disable_warp_specialized": True,
                },
            ),
        ):
            lowered = tilelang.lower(
                staged_matmul(
                    128,
                    256,
                    8192,
                    block_n=32 if paired else 64,
                    paired=paired,
                    splits=4,
                    stages=3,
                    register_epilogue=direct,
                ),
                target=target,
                enable_device_compile=False,
            )
        sources.append(str(lowered.kernel_source))
    old, new = sources
    epilogue = new[new.rindex("tl::wait_wgmma<0>") :]
    assert "void* epilogue" not in new
    assert "__syncthreads" not in epilogue
    assert "warpgroup_fence_operand" in epilogue
    assert old.count("__syncthreads") == new.count("__syncthreads") + 2
    if paired:
        assert "shfl.sync.bfly.b32 %0, %1, 4, 31, 0xffffffff" in new
        assert epilogue.index("tensor_pair_sum_xor4") < epilogue.index("if (")


def test_register_epilogue_rejects_incompatible_cluster_and_tiles():
    for opts in (
        {"cluster_reduce": True},
        {"block_n": 16},
        {"block_n": 64},
        {"block_m": 7},
    ):
        with pytest.raises(ValueError):
            staged_matmul(
                128, 256, 8192, paired=True, splits=4, register_epilogue=True, **opts
            )


@pytest.mark.skipif(os.environ.get("TENSOR_P1_CUDA") != "1", reason="CUDA execution")
@pytest.mark.parametrize(
    "paired,bm,bn,transposed,out_dtype,coalesced",
    [
        (True, 8, 32, True, "float32", False),
        (True, 16, 32, False, "float32", False),
        (True, 32, 32, True, "float32", False),
        (True, 64, 32, False, "float32", False),
        (False, 8, 64, False, "bfloat16", False),
        (False, 32, 64, True, "float32", False),
        (True, 8, 32, True, "float32", True),
        (True, 16, 32, False, "float32", True),
        (True, 32, 32, True, "float32", True),
        (True, 64, 32, True, "float32", True),
    ],
)
def test_register_mapping_with_tails_outliers_and_fresh_graph_values(
    paired, bm, bn, transposed, out_dtype, coalesced
):
    import tilelang
    import torch

    m, n, k, splits = 35, 65, 512, 2
    major, minor = torch.cuda.get_device_capability()
    kernel = tilelang.compile(
        staged_matmul(
            m,
            n,
            k,
            block_m=bm,
            block_n=bn,
            block_k=64,
            splits=splits,
            paired=paired,
            transposed_weights=transposed,
            native_async=False,
            register_epilogue=True,
            coalesced_epilogue=coalesced,
        ),
        target={"kind": "cuda", "arch": f"sm_{major}{minor}"},
        pass_configs={
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
            "tl.disable_wgmma": True,
        },
    )
    for seed in (51817, 51819):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((m, k), device="cuda", generator=gen).bfloat16()
        x[1].mul_(128)
        x[::8] = 0
        x[::8, 71] = -8
        original = torch.randn((n, k), device="cuda", generator=gen)
        original[:, 13].mul_(128)
        high = original.bfloat16()
        low = (original - high.float()).bfloat16()
        packed = torch.stack((high, low), dim=1).reshape(2 * n, k) if paired else high
        weight = (packed.T if transposed else packed).contiguous()
        output = torch.empty((splits, m, n), device="cuda")
        kernel(x, weight, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel(x, weight, output)
        for scale in (1.0, -0.75):
            x.mul_(scale)
            graph.replay()
            ref = x.double() @ (original if paired else high).double().T
            actual = output.sum(0)
            if out_dtype == "bfloat16":
                actual = actual.bfloat16()
            error = torch.linalg.vector_norm(actual.double() - ref, dim=1)
            norm = torch.linalg.vector_norm(ref, dim=1)
            bound = norm.clamp_min(norm.mean() * 0.1)
            assert float((error / bound).max()) < (1e-4 if paired else 1e-2)
            assert float((error[::8] / bound[::8]).max()) < 1e-3
