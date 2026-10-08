"""L40S workload contracts; opt in with TENSOR_LLT_CUDA=1."""

import gc
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tensor_torch.llt import Operators
import tensor

pytestmark = pytest.mark.skipif(
    os.environ.get("TENSOR_LLT_CUDA") != "1", reason="set TENSOR_LLT_CUDA=1"
)


@pytest.fixture(scope="module")
def ops():
    return Operators(os.environ.get("TENSOR_LLT_CACHE_DIR", "build/llt-tests"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_owned_borrowed_dlpack_export_lifetime(dtype):
    name = str(dtype).removeprefix("torch.")
    with tensor.Device() as device:
        buf = device.from_numpy(np.linspace(-3, 3, 129, dtype="float32"), dtype=name)
        value = torch.from_dlpack(buf)
        torch.testing.assert_close(
            value, torch.linspace(-3, 3, 129, device="cuda").to(dtype)
        )
        assert len(buf.to_bytes()) == 258
        with pytest.raises(RuntimeError, match="DLPack consumers"):
            buf.release()
        borrowed = device.from_dlpack(value)
        np.testing.assert_array_equal(borrowed.to_numpy(), value.float().cpu().numpy())
        borrowed.release()
        del value
        gc.collect()
        buf.release()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shared,causal,dv", [(True, True, 32), (False, False, 64)])
def test_attention_forward_backward_tails_and_score_value_dims(
    ops, dtype, shared, causal, dv
):
    torch.manual_seed(4)
    q = torch.randn(1, 4, 33, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(
        1, 1 if shared else 4, 67, 64, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        1, k.shape[1], 67, dv, device="cuda", dtype=dtype, requires_grad=True
    )
    out = ops.attention(q, k, v, causal=causal, query_offset=7, scale=0.125)
    mask = (
        torch.arange(67, device="cuda")[None, :]
        <= torch.arange(33, device="cuda")[:, None] + 7
        if causal
        else None
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(4 // k.shape[1], 1),
        v.float().repeat_interleave(4 // k.shape[1], 1),
        attn_mask=mask,
        scale=0.125,
    )
    g = torch.randn_like(out)
    expected = torch.autograd.grad(ref, (q, k, v), g.float())
    out.backward(g)
    tol = 0.035 if dtype == torch.bfloat16 else 0.005
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
    for x, y in zip((q, k, v), expected):
        torch.testing.assert_close(x.grad, y, atol=tol, rtol=tol)


def test_aliased_latent_gradient_and_checkpoint(ops):
    from torch.utils.checkpoint import checkpoint

    q = torch.randn(
        1, 4, 35, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    c = torch.randn(1, 1, 35, 32, device="cuda", dtype=q.dtype, requires_grad=True)

    def forward(q, c):
        return ops.attention(q, c, c, causal=True, scale=0.125)

    out = forward(q, c)
    direct = torch.autograd.grad(out.float().square().sum(), (q, c))
    replay = checkpoint(forward, q, c, use_reentrant=False)
    recomputed = torch.autograd.grad(replay.float().square().sum(), (q, c))
    for a, b in zip(direct, recomputed):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        c.float().expand(-1, 4, -1, -1),
        c.float().expand(-1, 4, -1, -1),
        is_causal=True,
        scale=0.125,
    )
    expected = torch.autograd.grad(ref.square().sum(), (q, c))
    for a, b in zip(direct, expected):
        torch.testing.assert_close(a, b, atol=0.05, rtol=0.05)


def test_model_operators_and_parameter_gradients(ops):
    torch.manual_seed(7)
    x = torch.randn(37, 48, device="cuda", requires_grad=True)
    w = torch.randn(65, 48, device="cuda", requires_grad=True)
    y = ops.linear(x, w)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = torch.nn.functional.linear(x, w)
    dy = torch.randn_like(y)
    expected = torch.autograd.grad(ref, (x, w), dy)
    actual = torch.autograd.grad(y, (x, w), dy)
    torch.testing.assert_close(y, ref, atol=0.0625, rtol=0.02)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0.0625, rtol=0.02)
    norm_weight = torch.randn(48, device="cuda", requires_grad=True)
    y = ops.rms_norm(x, norm_weight)
    ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * norm_weight
    dy = torch.randn_like(y)
    for a, b in zip(
        torch.autograd.grad(y, (x, norm_weight), dy),
        torch.autograd.grad(ref, (x, norm_weight), dy),
    ):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(y, ref, atol=2e-6, rtol=2e-6)
    y = ops.gelu(x)
    ref = torch.nn.functional.gelu(x)
    torch.testing.assert_close(y, ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(y, x, torch.ones_like(y))[0],
        torch.autograd.grad(ref, x, torch.ones_like(ref))[0],
        atol=1e-6,
        rtol=1e-6,
    )
    w = torch.randn(113, 48, device="cuda", requires_grad=True)
    index = torch.tensor([1, 1, 2, 8, 55, 112], device="cuda")
    y = ops.embedding(index, w)
    ref = torch.nn.functional.embedding(index, w)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(y, w, torch.ones_like(y))[0],
        torch.autograd.grad(ref, w, torch.ones_like(ref))[0],
        atol=0,
        rtol=0,
    )


def test_rotary_forward_backward_and_offsets(ops):
    x = torch.randn(1, 3, 19, 32, device="cuda", requires_grad=True)
    theta = (torch.arange(19, device="cuda") + 5)[:, None] * 10000.0 ** (
        -torch.arange(16, device="cuda") / 16
    )
    cs = torch.cat((theta.cos(), theta.cos()), -1)
    ss = torch.cat((theta.sin(), theta.sin()), -1)
    ref = x * cs + torch.cat((-x[..., 16:], x[..., :16]), -1) * ss
    y = ops.rotary(x, offset=5)
    torch.testing.assert_close(y, ref, atol=2e-5, rtol=2e-5)
    dy = torch.randn_like(y)
    torch.testing.assert_close(
        torch.autograd.grad(y, x, dy)[0],
        torch.autograd.grad(ref, x, dy)[0],
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("vocab", [259, 50257])
def test_full_vocabulary_cross_entropy_and_ignore(ops, vocab):
    x = torch.randn(7, vocab, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([0, 1, 8, -100, 5, 12, vocab - 1], device="cuda")
    y = ops.cross_entropy(x, target)
    ref = torch.nn.functional.cross_entropy(x.float(), target)
    torch.testing.assert_close(y, ref, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(y, x)[0],
        torch.autograd.grad(ref, x)[0],
        atol=2e-5,
        rtol=0.01,
    )


def test_adamw_clipping_state_and_nonfinite_skip(ops):
    from tensor_torch.llt import AdamW

    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(1025, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    optimizer = AdamW([p], ops, lr=1e-3, max_norm=0.7)
    reference = torch.optim.AdamW([q], lr=1e-3, foreach=False)
    for i in range(5):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        q.grad = grad.clone()
        torch.nn.utils.clip_grad_norm_([q], 0.7)
        optimizer.step()
        reference.step()
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)
    snapshot = p.detach().clone()
    state = optimizer.state[p]["step"].clone()
    p.grad.fill_(float("nan"))
    optimizer.step()
    torch.testing.assert_close(p, snapshot, atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state[p]["step"], state, atol=0, rtol=0)


def test_dynamic_cache_decode_graph_and_capacity(ops):
    from tensor_torch.llt import KVCache

    with torch.inference_mode():
        cache = KVCache(ops, 1, 1, 19, 48, 32)
        cache.keys.fill_(float("nan"))
        cache.values.fill_(float("nan"))
        k = torch.randn(1, 1, 11, 48, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 1, 11, 32, device="cuda", dtype=k.dtype)
        q = torch.randn(1, 4, 1, 48, device="cuda", dtype=k.dtype)
        cache.append(k, v)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(),
            k.float().expand(-1, 4, -1, -1),
            v.float().expand(-1, 4, -1, -1),
            scale=0.125,
        )
        torch.testing.assert_close(
            ops.decode(q, cache, scale=0.125).float(), ref, atol=0.01, rtol=0.02
        )
        # Warm append/decode specifications before capture, then reset the prefix.
        kk = k[:, :, :1].contiguous()
        vv = v[:, :, :1].contiguous()
        cache.append(kk, vv)
        ops.decode(q, cache, scale=0.125)
        cache.reset()
        cache.append(k, v)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cache.append(kk, vv)
            out = ops.decode(q, cache, scale=0.125)
        graph.replay()
        graph.replay()
        assert cache.check() == 13
        allk = torch.cat((k, kk.expand(-1, -1, 2, -1)), -2)
        allv = torch.cat((v, vv.expand(-1, -1, 2, -1)), -2)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(),
            allk.float().expand(-1, 4, -1, -1),
            allv.float().expand(-1, 4, -1, -1),
            scale=0.125,
        )
        torch.testing.assert_close(out.float(), expected, atol=0.01, rtol=0.02)
        for _ in range(7):
            graph.replay()
        with pytest.raises(ValueError, match="capacity"):
            cache.check()
        cache.reset()
        assert cache.length == 0 and not cache.overflow.any().item()


def test_variable_length_batched_decode(ops):
    from tensor_torch.llt import KVCache

    with torch.inference_mode():
        c = torch.randn(4, 1, 111, 64, device="cuda", dtype=torch.float16)
        cache = KVCache(ops, 4, 1, 129, 64, dtype=c.dtype, shared=True)
        cache.keys.fill_(float("nan"))
        cache.append(c)
        cache.lengths.copy_(
            torch.tensor([1, 7, 65, 111], device="cuda", dtype=torch.int32)
        )
        q = torch.randn(4, 8, 1, 64, device="cuda", dtype=c.dtype)
        out = ops.decode(q, cache, scale=0.125)
        for b, n in enumerate((1, 7, 65, 111)):
            ref = torch.nn.functional.scaled_dot_product_attention(
                q[b : b + 1].float(),
                c[b : b + 1, :, :n].float().expand(-1, 8, -1, -1),
                c[b : b + 1, :, :n].float().expand(-1, 8, -1, -1),
                scale=0.125,
            )
            torch.testing.assert_close(
                out[b : b + 1].float(), ref, atol=0.005, rtol=0.005
            )


def test_chunked_classifier_loss_gradients_and_ignored_chunk(ops):
    x = torch.randn(7, 48, device="cuda", requires_grad=True)
    w = torch.randn(259, 48, device="cuda", requires_grad=True)
    target = torch.tensor([-100, -100, 1, 7, 30, 20, 5], device="cuda")
    direct = ops.cross_entropy(ops.linear(x, w), target)
    chunked = ops.linear_cross_entropy(x, w, target, chunk_size=2)
    torch.testing.assert_close(chunked, direct, atol=3e-6, rtol=3e-6)
    a = torch.autograd.grad(chunked, (x, w))
    b = torch.autograd.grad(direct, (x, w))
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, atol=0.02, rtol=0.02)


def test_frontend_alignment_instead_of_abi_default(ops):
    x = torch.arange(145, device="cuda", dtype=torch.float32)[16:]
    assert x.data_ptr() % 64 == 0 and x.data_ptr() % 256 != 0
    before = ops.report["layout_copies"]
    torch.testing.assert_close(
        ops.cast(x, torch.bfloat16), x.to(torch.bfloat16), atol=0, rtol=0
    )
    assert ops.report["layout_copies"] == before


@pytest.mark.parametrize(
    "ta,tb", [(False, False), (False, True), (True, False), (True, True)]
)
def test_batched_transpose_projection_gradients(ops, ta, tb):
    x = torch.randn(
        (3, 48, 37) if ta else (3, 37, 48), device="cuda", requires_grad=True
    )
    y = torch.randn(
        (3, 65, 48) if tb else (3, 48, 65), device="cuda", requires_grad=True
    )
    out = ops.bmm(x, y, transpose_a=ta, transpose_b=tb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = torch.bmm(
            x.transpose(-1, -2) if ta else x, y.transpose(-1, -2) if tb else y
        )
    g = torch.randn_like(out)
    torch.testing.assert_close(out, ref, atol=0.0625, rtol=0.02)
    for a, b in zip(
        torch.autograd.grad(out, (x, y), g), torch.autograd.grad(ref, (x, y), g)
    ):
        torch.testing.assert_close(a, b, atol=0.0625, rtol=0.02)


def test_large_gradient_norm_hierarchical_reduction(ops):
    from tensor_torch.llt import AdamW

    p = torch.nn.Parameter(torch.randn(1200001, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    optimizer = AdamW([p], ops, lr=1e-3, max_norm=0.7)
    norm = torch.nn.utils.clip_grad_norm_([q], 0.7)
    reference = torch.optim.AdamW([q], lr=1e-3, foreach=False)
    optimizer.step()
    reference.step()
    torch.testing.assert_close(optimizer.last_norm, norm, atol=0.001, rtol=1e-5)
    torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)


def test_split_head_backward_stream_and_large_logits(ops):
    torch.manual_seed(29)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        q = (
            torch.randn(1, 8, 257, 64, device="cuda", dtype=torch.bfloat16) * 2
        ).requires_grad_()
        c = (
            torch.randn(1, 1, 257, 64, device="cuda", dtype=q.dtype) * 2
        ).requires_grad_()
        out = ops.attention(q, c, c, causal=True, scale=0.125)
        dy = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, c), dy)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(),
            c.float().expand(-1, 8, -1, -1),
            c.float().expand(-1, 8, -1, -1),
            is_causal=True,
            scale=0.125,
        )
        expected = torch.autograd.grad(ref, (q, c), dy.float())
        torch.testing.assert_close(out.float(), ref, atol=0.07, rtol=0.05)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0.1, rtol=0.07)
    torch.cuda.current_stream().wait_stream(side)


def test_folded_projection_matches_unfolded_high_precision(ops):
    # The exact algebra uses original-head scaling; rotating only latent values
    # would not preserve the additional positional score component.
    torch.manual_seed(13)
    z = torch.randn(17, 32, device="cuda") * 0.1
    latent = torch.randn(19, 16, device="cuda") * 0.1
    wq = torch.randn(16, 32, device="cuda") * 0.1
    wk = torch.randn(16, 16, device="cuda") * 0.1
    wv = torch.randn(16, 16, device="cuda") * 0.1
    wo = torch.randn(32, 16, device="cuda") * 0.1
    qp = torch.randn(1, 1, 17, 16, device="cuda") * 0.1
    kp = torch.randn(1, 1, 19, 16, device="cuda") * 0.1
    folded_q = ops.matmul(wk, wq, transpose_a=True)
    folded_o = ops.matmul(wo, wv)
    q = torch.cat(
        (ops.linear(z, folded_q).reshape(1, 1, 17, 16), ops.cast(qp, torch.bfloat16)),
        -1,
    )
    k = torch.cat(
        (
            ops.cast(latent, torch.bfloat16).reshape(1, 1, 19, 16),
            ops.cast(kp, torch.bfloat16),
        ),
        -1,
    )
    v = ops.cast(latent, torch.bfloat16).reshape(1, 1, 19, 16)
    actual = ops.linear(
        ops.attention(q, k, v, scale=0.25).reshape(17, 16), folded_o
    ).float()
    score = (
        (z @ wq.T) @ (latent @ wk.T).T + qp.reshape(17, 16) @ kp.reshape(19, 16).T
    ) * 0.25
    expected = (score.softmax(-1) @ (latent @ wv.T)) @ wo.T
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.04)


def test_optimizer_restores_clipping_policy(ops):
    from tensor_torch.llt import AdamW

    p = torch.nn.Parameter(torch.randn(17, device="cuda"))
    optimizer = AdamW([p], ops, max_norm=0.25)
    optimizer.step()
    restored = AdamW([p], ops, max_norm=1.0)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.max_norm == 0.25


def test_rotary_runtime_batch_offsets_and_graph_reuse(ops):
    x = torch.randn(2, 3, 19, 32, device="cuda")
    offset = torch.tensor([3, 9], device="cuda", dtype=torch.int64)

    def reference():
        theta = (
            torch.arange(19, device="cuda")[None, :, None] + offset[:, None, None]
        ) * 10000.0 ** (-torch.arange(16, device="cuda") / 16)
        cs = torch.cat((theta.cos(), theta.cos()), -1)[:, None]
        ss = torch.cat((theta.sin(), theta.sin()), -1)[:, None]
        return x * cs + torch.cat((-x[..., 16:], x[..., :16]), -1) * ss

    y = ops.rotary(x, offset=offset)
    count = len(ops.report["artifacts"])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ops.rotary(x, offset=offset)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        y = ops.rotary(x, offset=offset)
    for first in (17, 33, 1000):
        offset.copy_(torch.tensor([first, first + 9], device="cuda"))
        g.replay()
        torch.testing.assert_close(y, reference(), atol=0.0003, rtol=0.0003)
        assert len(ops.report["artifacts"]) == count


def test_streamed_full_vocabulary_classifier_and_empty_labels(ops):
    torch.manual_seed(71)
    x = torch.randn(65, 128, device="cuda", requires_grad=True)
    w = (torch.randn(50257, 128, device="cuda") * 0.02).requires_grad_()
    labels = torch.arange(65, device="cuda")
    direct = ops.cross_entropy(ops.linear(x, w), labels)
    streamed = ops.linear_cross_entropy(x, w, labels, chunk_size=32)
    torch.testing.assert_close(streamed, direct, atol=1e-5, rtol=1e-5)
    actual = torch.autograd.grad(streamed, (x, w))
    expected = torch.autograd.grad(direct, (x, w))
    for a, b in zip(actual, expected):
        assert ((a - b).norm() / b.norm()).item() < 0.02
    ignored = torch.full_like(labels, -100)
    value = ops.linear_cross_entropy(x, w, ignored, chunk_size=32)
    assert torch.isnan(value).item()
    for grad in torch.autograd.grad(value, (x, w)):
        assert grad.count_nonzero().item() == 0


def test_pinned_nvrtc_warp_reduction_trait_and_poisoned_tail(ops):
    from tensor_torch.llt import KVCache

    q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(1, 1, 19, 32, device="cuda", dtype=q.dtype)
    cache = KVCache(ops, 1, 1, 33, 32, shared=True)
    cache.keys.fill_(float("nan"))
    cache.append(c)
    partial, stats = ops.call(
        "decode_partial",
        (1, 4, 1, 33, 32, 32, "bfloat16", 0.125, 8),
        ["partial", "stats"],
        q,
        cache.keys,
        cache.values,
        cache.lengths,
        module="benchmarks.llt.tensor_warp_decode",
    )
    actual = ops.call(
        "decode_merge",
        (1, 4, 32, "bfloat16", 8),
        ["out"],
        partial,
        stats,
        module="tensor_torch.templates.llt_decode",
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, c, c, enable_gqa=True, scale=0.125
    )
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


def test_adamw_infinite_norm_disables_clipping(ops):
    from tensor_torch.llt import AdamW

    p = torch.nn.Parameter(torch.randn(17, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    a = AdamW([p], ops, max_norm=float("inf"))
    b = torch.optim.AdamW([q], foreach=False)
    for _ in range(3):
        grad = torch.randn_like(p) * 20
        p.grad = grad.clone()
        q.grad = grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_affine_layer_norm_tails_and_parameter_gradients(ops, dtype):
    torch.manual_seed(9511)
    x = torch.randn(37, 79, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(79, device="cuda", requires_grad=True)
    bias = torch.randn(79, device="cuda", requires_grad=True)
    result = ops.layer_norm(x, weight, bias)
    reference = torch.nn.functional.layer_norm(x.float(), (79,), weight, bias).to(dtype)
    upstream = torch.randn_like(result)
    actual = torch.autograd.grad(result, (x, weight, bias), upstream)
    expected = torch.autograd.grad(reference, (x, weight, bias), upstream)
    tolerance = 0.035 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)
    for a, b in zip(actual, expected):
        assert (a.float() - b.float()).norm() / b.float().norm() < 0.025


def test_linear_channel_bias_gradient(ops):
    torch.manual_seed(9512)
    x = torch.randn(37, 79, device="cuda", requires_grad=True)
    weight = torch.randn(53, 79, device="cuda", requires_grad=True)
    bias = torch.randn(53, device="cuda", requires_grad=True)
    result = ops.linear(x, weight, bias)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = torch.nn.functional.linear(x, weight, bias)
    upstream = torch.randn_like(result)
    actual = torch.autograd.grad(result, (x, weight, bias), upstream)
    expected = torch.autograd.grad(reference, (x, weight, bias), upstream)
    # Tensor's separate bias epilogue adds one BF16 rounding before addition.
    torch.testing.assert_close(result, reference, atol=0.125, rtol=0.04)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0.125, rtol=0.04)


def test_inference_weight_cache_refresh_lifetime_and_training():
    import weakref

    ops = Operators(os.environ.get("TENSOR_LLT_CACHE_DIR", "build/llt-tests"), cache_inference_weights=True)
    weight = torch.nn.Parameter(torch.randn(33, 48, device="cuda"))
    with torch.inference_mode():
        first = ops.cast(weight, torch.bfloat16)
        assert ops.cast(weight, torch.bfloat16) is first
        weight.add_(0.25)
        second = ops.cast(weight, torch.bfloat16)
        assert second is not first
        torch.testing.assert_close(second, weight.to(torch.bfloat16), atol=0, rtol=0)
    # autograd.Function.forward disables grad, but is not inference mode.
    ops.clear_inference_weight_cache()
    x = torch.randn(17, 48, device="cuda", requires_grad=True)
    ops.linear(x, weight).float().sum().backward()
    assert not ops.inference_weight_casts
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    with torch.inference_mode():
        ops.cast(weight, torch.bfloat16)
    reference = weakref.ref(weight)
    del weight
    gc.collect()
    assert reference() is None and not ops.inference_weight_casts
