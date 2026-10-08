"""Native kernels needed by recurrent/cache-sharing research models."""

import os
import pytest

torch = pytest.importorskip("torch")
from tensor_torch.recurrent import RecurrentOperators

pytestmark = pytest.mark.skipif(
    os.environ.get("TENSOR_LLT_CUDA") != "1", reason="set TENSOR_LLT_CUDA=1"
)


@pytest.fixture(scope="module")
def ops():
    return RecurrentOperators(
        os.environ.get("TENSOR_LLT_CACHE_DIR", "build/recurrent-tests")
    )


@pytest.mark.parametrize(
    "prefix,window,offset,m,n,heads,kvheads",
    [
        (0, 13, 0, 37, 37, 2, 2),
        (37, 9, 0, 37, 74, 2, 2),
        (0, 17, 71, 19, 96, 4, 1),
        (0, 1, 0, 33, 33, 2, 2),
        (64, 65, 0, 257, 321, 4, 1),
    ],
)
def test_window_forward_backward(ops, prefix, window, offset, m, n, heads, kvheads):
    torch.manual_seed(2)
    q = torch.randn(
        1, heads, m, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(1, kvheads, n, 32, device="cuda", dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qi = torch.arange(m, device="cuda")[:, None] + offset
    kj = torch.arange(n, device="cuda")[None, :]
    mask = ((kj < prefix) & (kj <= qi)) | (
        (kj >= prefix) & (kj - prefix <= qi) & (kj - prefix > qi - window)
    )
    out = ops.window_attention(
        q, k, v, window_size=window, shared_prefix=prefix, query_offset=offset
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(heads // kvheads, 1),
        v.float().repeat_interleave(heads // kvheads, 1),
        attn_mask=mask,
    )
    dy = torch.randn_like(out)
    grads = torch.autograd.grad(expected, (q, k, v), dy.float())
    out.backward(dy)
    torch.testing.assert_close(out.float(), expected, rtol=0.035, atol=0.035)
    for tensor, reference in zip((q, k, v), grads):
        torch.testing.assert_close(tensor.grad, reference, rtol=0.045, atol=0.045)
    assert not ops.report["fallbacks"]


@pytest.mark.parametrize("kind", ["silu", "sigmoid", "multiply", "blend"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_elementwise_forward_backward(ops, kind, dtype):
    torch.manual_seed(3)
    tensors = [
        torch.randn(513, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    x, y, z = tensors
    if kind == "silu":
        actual = ops.silu(x)
        expected = torch.nn.functional.silu(x)
        used = (x,)
    elif kind == "sigmoid":
        actual = ops.sigmoid(x)
        expected = torch.sigmoid(x)
        used = (x,)
    elif kind == "multiply":
        actual = ops.multiply(x, y)
        expected = x * y
        used = (x, y)
    else:
        actual = ops.blend(x, y, z)
        expected = x * y + (1 - x) * z
        used = (x, y, z)
    dy = torch.randn_like(actual)
    refs = torch.autograd.grad(expected, used, dy)
    actual.backward(dy)
    tol = 0.04 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    for t, ref in zip(used, refs):
        torch.testing.assert_close(t.grad, ref, rtol=tol, atol=tol)


def test_old_attention_entry_point_and_checkpoint_are_unchanged(ops):
    from torch.utils.checkpoint import checkpoint

    q = torch.randn(
        1, 2, 33, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    c = torch.randn_like(q, requires_grad=True)

    def fn(q, c):
        return ops.window_attention(q, c, c, window_size=12)

    ordinary = fn(q, c)
    g = torch.randn_like(ordinary)
    direct = torch.autograd.grad(ordinary, (q, c), g)
    replay = checkpoint(fn, q, c, use_reentrant=False)
    repeat = torch.autograd.grad(replay, (q, c), g)
    for a, b in zip(direct, repeat):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    old = ops.attention(q, c, c, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, c, c, is_causal=True)
    torch.testing.assert_close(old, ref, rtol=0.035, atol=0.035)


@pytest.mark.parametrize("glength,llength", [(17, 5), (65, 17)])
def test_two_bank_cached_decode_and_graph(ops, glength, llength):
    from tensor_torch.llt import KVCache

    torch.manual_seed(8)
    with torch.inference_mode():
        caches = []
        for capacity, length in [(65, glength), (17, llength)]:
            cache = KVCache(ops, 1, 2, capacity, 32, dtype=torch.bfloat16)
            k = torch.randn(1, 2, length, 32, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cache.append(k, v)
            caches.append(cache)
        q = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.bfloat16)
        g, l = caches
        actual = ops.shared_decode(q, g, l)
        k = torch.cat([g.keys[:, :, :glength], l.keys[:, :, :llength]], 2)
        v = torch.cat([g.values[:, :, :glength], l.values[:, :, :llength]], 2)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(actual, expected, rtol=0.035, atol=0.035)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                ops.shared_decode(q, g, l)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = ops.shared_decode(q, g, l)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)


def test_mixed_gate_storage_backward(ops):
    gate = torch.sigmoid(torch.randn(513, device="cuda")).detach().requires_grad_()
    state = torch.randn_like(gate, requires_grad=True)
    proposal = torch.randn(513, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = ops.blend(gate, state, proposal)
    expected = gate * state + (1 - gate) * proposal
    dy = torch.randn_like(actual)
    references = torch.autograd.grad(expected, (gate, state, proposal), dy)
    actual.backward(dy)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for value, reference in zip((gate, state, proposal), references):
        torch.testing.assert_close(value.grad, reference, atol=2e-6, rtol=2e-6)


def test_window_api_rejects_empty_heads_and_invalid_rank(ops):
    q = torch.empty(1, 2, 3, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 0, 3, 32, device="cuda", dtype=q.dtype)
    with pytest.raises(ValueError, match="geometry"):
        ops.window_attention(q, k, k, window_size=2)
    with pytest.raises(ValueError, match="shared_prefix"):
        ops.window_attention(q, q[0], q[0], window_size=2)
