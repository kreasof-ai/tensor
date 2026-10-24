"""Independent Triton implementations; tuning is performed by the harness."""

import triton as tr
import triton.language as tl


@tr.jit
def matmul(
    X,
    W,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        ks = base * BK + ki
        x = tl.load(
            X + mi[:, None] * K + ks[None, :], (mi[:, None] < M) & (ks[None, :] < K), 0
        )
        w = tl.load(
            W + ni[None, :] * K + ks[:, None], (ni[None, :] < N) & (ks[:, None] < K), 0
        )
        acc = tl.dot(x, w, acc)
    tl.store(
        O + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N)
    )


@tr.jit
def attention(
    Q,
    K,
    V,
    O,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    d = tl.arange(0, D)
    q = tl.load(Q + bh * S * D + rows[:, None] * D + d[None, :], rows[:, None] < S, 0)
    mx = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    stop = tl.minimum(S, (tl.program_id(0) + 1) * BM) if CAUSAL else S
    for start in range(0, stop, BN):
        cols = start + tl.arange(0, BN)
        k = tl.load(
            K + bh * S * D + cols[None, :] * D + d[:, None], cols[None, :] < S, 0
        )
        score = tl.dot(q, k) * (D**-0.5)
        valid = cols[None, :] < S
        if CAUSAL:
            valid = valid & (rows[:, None] >= cols[None, :])
        score = tl.where(valid, score, -float("inf"))
        new = tl.maximum(mx, tl.max(score, 1))
        p = tl.exp(score - new[:, None])
        alpha = tl.exp(mx - new)
        denom = denom * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + bh * S * D + cols[:, None] * D + d[None, :], cols[:, None] < S, 0
        )
        acc = tl.dot(p.to(tl.float16), v, acc)
        mx = new
    tl.store(
        O + bh * S * D + rows[:, None] * D + d[None, :],
        acc / denom[:, None],
        rows[:, None] < S,
    )


@tr.jit
def mla_partial(
    Q, C, P, L, H: tl.constexpr, S: tl.constexpr, SPLITS: tl.constexpr, BN: tl.constexpr
):
    hs = tl.program_id(0) * 16 + tl.arange(0, 16)
    split, b = tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    q = tl.load(Q + (b * H + hs[:, None]) * 576 + d[None, :], hs[:, None] < H, 0)
    qr = tl.load(Q + (b * H + hs[:, None]) * 576 + 512 + r[None, :], hs[:, None] < H, 0)
    mx = tl.full((16,), -float("inf"), tl.float32)
    den = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, 512), 0, tl.float32)
    chunk: tl.constexpr = tr.cdiv(S, SPLITS * BN)
    for step in range(chunk):
        ns = (split * chunk + step) * BN + tl.arange(0, BN)
        k = tl.load(C + (b * S + ns[None, :]) * 576 + d[:, None], ns[None, :] < S, 0)
        kr = tl.load(
            C + (b * S + ns[None, :]) * 576 + 512 + r[:, None], ns[None, :] < S, 0
        )
        scores = (tl.dot(q, k) + tl.dot(qr, kr)) * (576**-0.5)
        scores = tl.where(ns[None, :] < S, scores, -float("inf"))
        new = tl.maximum(mx, tl.max(scores, 1))
        p = tl.exp(scores - new[:, None])
        alpha = tl.exp(mx - new)
        den = den * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        acc = tl.dot(p.to(tl.float16), tl.trans(k), acc)
        mx = new
    tl.store(
        P + ((b * H + hs[:, None]) * SPLITS + split) * 512 + d[None, :],
        acc / den[:, None],
        hs[:, None] < H,
    )
    tl.store(L + (b * H + hs) * SPLITS + split, mx + tl.log(den), hs < H)


@tr.jit
def mla_merge(P, L, O, SPLITS: tl.constexpr):
    h = tl.program_id(0)
    s, d = tl.arange(0, SPLITS), tl.arange(0, 512)
    lse = tl.load(L + h * SPLITS + s)
    weights = tl.exp(lse - tl.max(lse, 0))
    values = tl.load(P + (h * SPLITS + s[:, None]) * 512 + d[None, :])
    out = tl.sum(values * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(O + h * 512 + d, out)


@tr.jit
def decode_ggml(W, rows, channels, N: tl.constexpr, K: tl.constexpr, Q: tl.constexpr):
    if Q == 2:
        base = rows * (K // 32 * 18) + channels // 32 * 18
        scale = tl.load(tl.cast(W + base, tl.pointer_type(tl.float16)), rows < N, 0).to(
            tl.float32
        )
        byte = tl.load(W + base + 2 + channels % 16, rows < N, 0).to(tl.int32)
        val = ((byte >> (channels % 32 // 16 * 4)) & 15) - 8
        decoded = val * scale
    elif Q == 12:
        base = rows * (K // 256 * 144) + channels // 256 * 144
        local = channels % 256
        group = local // 32
        d = tl.load(tl.cast(W + base, tl.pointer_type(tl.float16)), rows < N, 0).to(
            tl.float32
        )
        dm = tl.load(
            tl.cast(W + base + 2, tl.pointer_type(tl.float16)), rows < N, 0
        ).to(tl.float32)
        sc0 = tl.load(W + base + 4 + group % 4, rows < N, 0).to(tl.int32)
        sc1 = tl.load(W + base + 8 + group % 4, rows < N, 0).to(tl.int32)
        sc2 = tl.load(W + base + 12 + group % 4, rows < N, 0).to(tl.int32)
        sc = tl.where(group < 4, sc0 & 63, (sc2 & 15) | ((sc0 >> 6) << 4))
        mn = tl.where(group < 4, sc1 & 63, (sc2 >> 4) | ((sc1 >> 6) << 4))
        byte = tl.load(W + base + 16 + local // 64 * 32 + local % 32, rows < N, 0).to(
            tl.int32
        )
        val = (byte >> (local % 64 // 32 * 4)) & 15
        decoded = d * sc * val - dm * mn
    else:
        base = rows * (K // 256 * 210) + channels // 256 * 210
        local = channels % 256
        half = local // 128
        lane = local % 32
        quarter = local % 128 // 32
        lo = tl.load(W + base + half * 64 + lane + (quarter % 2) * 32, rows < N, 0).to(
            tl.int32
        )
        hi = tl.load(W + base + 128 + half * 32 + lane, rows < N, 0).to(tl.int32)
        val = ((lo >> (quarter // 2 * 4)) & 15) | (((hi >> (quarter * 2)) & 3) << 4)
        sc = tl.load(
            tl.cast(W + base + 192 + local // 16, tl.pointer_type(tl.int8)), rows < N, 0
        ).to(tl.float32)
        d = tl.load(
            tl.cast(W + base + 208, tl.pointer_type(tl.float16)), rows < N, 0
        ).to(tl.float32)
        decoded = (val - 32) * sc * d
    return decoded


@tr.jit
def gguf(
    X,
    W,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = step * BK + rr
        decoded = decode_ggml(W, ni[None, :], kk[:, None], N, K, Q)
        x = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, 0).to(
            tl.float16
        )
        acc = tl.dot(x, decoded.to(tl.float16), acc)
    tl.store(
        O + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N)
    )


@tr.jit
def gguf_gemv(
    X, W, O, N: tl.constexpr, K: tl.constexpr, Q: tl.constexpr, ROWS: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ks = tl.arange(0, tr.next_power_of_2(K))
    weights = decode_ggml(W, rows[:, None], ks[None, :], N, K, Q)
    activation = tl.load(X + ks, ks < K, 0)
    result = tl.sum(weights * activation[None, :], 1)
    tl.store(O + rows, result, rows < N)
