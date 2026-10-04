"""Packed CUDA transfer: independent arithmetic, masking, and plan lifetimes."""
import os
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
GPU = pytest.mark.skipif(os.environ.get('TENSOR_LFM2_CUDA') != '1',
                         reason='requires the NVRTC bundle and CUDA GPU')


def test_cuda_profile_rows_are_explicit():
    from tensor_llm.model import valid_rows, requirements
    from test_contracts import fixture
    assert valid_rows('cuda', (1, 128), 'default')
    assert valid_rows('cuda', (1, 32, 128), 'optimized')
    assert not valid_rows('cuda', (1, 32, 128), 'default')
    assert not valid_rows('cuda', (1, 128, 32), 'optimized')
    with pytest.raises(ValueError, match='CUDA kernel profile'):
        requirements(fixture(), 512, cuda_profile='typo')


def build(kind, parameters, directory, generator=None):
    import tensor
    from tensor_llm.cuda_kernels import source
    path = directory / (kind + '.py')
    path.write_text((generator or source)(kind, parameters))
    artifact = path.with_suffix('.tbin')
    tensor.build(path, artifact, compiler='nvrtc', target='sm_86')
    return artifact


@GPU
@pytest.mark.parametrize('encoding', [0, 1, 2, 12, 14])
def test_packed_projection_and_epilogues_against_numpy(tmp_path, encoding):
    import tensor
    from tensor_llm import GGUF
    g = GGUF(ROOT / 'build/lfm2-diagnostic.gguf')
    k = 256
    o = 7  # Exercise partial warp-row tiles, including Q4_K.
    rng = np.random.default_rng(1743 + encoding)
    if encoding in (0, 1):
        weights = rng.normal(size=(o, k)).astype(np.float32 if encoding == 0 else np.float16)
        raw = weights.ravel()
    else:
        name = next(n for n, t in g.tensors.items() if t.type == encoding and len(t.shape) == 2 and t.shape[1] == k)
        info = g.tensors[name]
        raw = np.array(g.packed(name)[:o * info.nbytes // info.shape[0]], copy=True)
        weights = np.array(g.array(name)[:o], copy=True)
    x = rng.normal(size=k).astype(np.float32)
    residual = rng.normal(size=o).astype(np.float32)
    dot = weights.astype(np.float64) @ x.astype(np.float64)
    p = dict(r=1, k=k, o=o, type=encoding)
    with tensor.Device() as device:
        dx, dw, dr = [device.from_numpy(a) for a in (x, raw, residual)]
        out = device.full(o, np.nan)
        for kind in ('linear', 'linear_add', 'ffn'):
            path = build(kind, p, tmp_path)
            kernel = device.load(path)
            args = (dx, dw, dw, out) if kind == 'ffn' else (dx, dw, dr, out) if kind == 'linear_add' else (dx, dw, out)
            kernel.launch(*args)
            expected = dot / (1 + np.exp(-dot)) * dot if kind == 'ffn' else dot + residual if kind == 'linear_add' else dot
            np.testing.assert_allclose(out.to_numpy(), expected, rtol=2e-5, atol=2e-5)


@GPU
@pytest.mark.parametrize('kind,heads', [('attention_partial',4),('attention_grouped',4),('attention_grouped',8)])
def test_split_attention_short_and_tail_partitions(tmp_path,kind,heads):
    import tensor
    rng = np.random.default_rng(1907)
    h, kh, d, cap = heads, 2, 64, 448
    q = rng.normal(size=(h, d)).astype(np.float32)
    k = rng.normal(size=(cap, kh, d)).astype(np.float16)
    v = rng.normal(size=(cap, kh, d)).astype(np.float16)
    p = dict(r=1, h=h, kh=kh, d=d, cap=cap, splits=16)
    partial = build(kind, p, tmp_path)
    merge = build('attention_merge', dict(h=h, d=d, splits=16), tmp_path)
    with tensor.Device() as device:
        dq, dk, dv = [device.from_numpy(a.ravel()) for a in (q, k, v)]
        scratch = device.empty(h * 16 * 66)
        output = device.full(h * d, np.nan)
        control = device.zeros(2, dtype='int32')
        first, second = device.load(partial), device.load(merge)
        for length in (1, 7, 16, 17, 64, 65, 129, 384):
            host = np.array([length - 1, 1], np.int32)
            device.driver.call('cuMemcpyHtoD_v2', control.pointer, host.ctypes.data, host.nbytes)
            first.launch(dq, dk, dv, scratch, control)
            second.launch(scratch, output)
            expected = []
            for head in range(h):
                group = head // (h // kh)
                scores = k[:length, group].astype(np.float64) @ q[head].astype(np.float64) / 8
                probs = np.exp(scores - scores.max()); probs /= probs.sum()
                expected.append(probs @ v[:length, group].astype(np.float64))
            np.testing.assert_allclose(output.to_numpy().reshape(h, d), expected, rtol=2e-5, atol=2e-6)


@GPU
@pytest.mark.parametrize('encoding,rows', [(1,8),(2,32),(12,128),(14,8)])
@pytest.mark.parametrize('kind', ['linear','ffn'])
def test_staged_prefill_pairs_and_padding(tmp_path, encoding, rows, kind):
    import torch
    import tensor
    from tensor_llm import GGUF
    from tensor_llm.cuda_kernels import prefill_source
    g=GGUF(ROOT/'build/lfm2-diagnostic.gguf');k,o=256,63
    name=next(n for n,t in g.tensors.items() if t.type==encoding and len(t.shape)==2 and t.shape[1]==k)
    info=g.tensors[name];raw=np.array(g.packed(name)[:o*info.nbytes//info.shape[0]],copy=True)
    if encoding==1:raw=raw.view(np.float16)
    x=np.random.default_rng(2137).normal(size=(rows,k)).astype(np.float32)*.3
    tx=torch.from_numpy(x).cuda().half();w=torch.from_numpy(np.array(g.array(name)[:o],copy=True)).cuda().half()
    expected=torch.mm(tx,w.T,out_dtype=torch.float32)
    if kind=='ffn':expected=torch.nn.functional.silu(expected)*expected
    expected=expected.cpu().numpy().ravel()
    p=dict(r=rows,k=k,o=o,type=encoding,block_m=32,block_n=64,block_k=128,stages=2,threads=128,packed_pairs=True)
    path=build(kind,p,tmp_path,generator=prefill_source)
    with tensor.Device() as device:
        inp,weights=[device.from_numpy(a) for a in (x.ravel(),raw)];output=device.full(rows*o,np.nan)
        arguments=(inp,weights,weights,output) if kind=='ffn' else (inp,weights,output)
        device.load(path).launch(*arguments);actual=output.to_numpy()
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)


@GPU
def test_greedy_lowest_tie_and_control(tmp_path):
    import tensor
    kernel = build('argmax', dict(n=513), tmp_path)
    values = np.full(513, -3, np.float32); values[[7, 510]] = 5
    with tensor.Device() as device:
        logits = device.from_numpy(values)
        token = device.zeros(1, dtype='int32')
        control = device.from_numpy(np.array([17, 128], np.int32))
        device.load(kernel).launch(logits, token, control)
        np.testing.assert_array_equal(token.to_numpy(), [7])
        np.testing.assert_array_equal(control.to_numpy(), [17, 1])


@GPU
def test_split_norm_rope_and_partial_cache_store(tmp_path):
    import tensor
    rng = np.random.default_rng(1973)
    r, h, kh, d, cap = 8, 4, 2, 64, 64
    position, count, eps, theta = 23, 5, 1e-5, 10000
    q, k, v = [rng.normal(size=shape).astype(np.float32)
               for shape in ((r,h,d), (r,kh,d), (r,kh,d))]
    qw, kw = [rng.uniform(.5, 1.5, size=d).astype(np.float32) for _ in range(2)]
    def norm_rope(x, weights):
        x = x.astype(np.float64)
        normalized = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * weights
        angles = np.arange(position, position+r)[:,None,None] * theta**(-np.arange(d//2)*2/d)
        left, right = np.split(normalized, 2, axis=-1)
        return np.concatenate((left*np.cos(angles)-right*np.sin(angles),
                               left*np.sin(angles)+right*np.cos(angles)), axis=-1)
    p = dict(r=r, h=h, kh=kh, d=d, cap=cap, eps=eps, theta=theta)
    query, keys = [build(kind, p, tmp_path) for kind in ('qnorm', 'kvnorm')]
    with tensor.Device() as device:
        dq, dk, dv, dqw, dkw = [device.from_numpy(x.ravel()) for x in (q,k,v,qw,kw)]
        out = device.full(r*h*d, np.nan)
        kc, vc = [device.full(cap*kh*d, -17, dtype='float16') for _ in range(2)]
        control = device.from_numpy(np.array([position,count], np.int32))
        device.load(query).launch(dq,dqw,out,control)
        device.load(keys).launch(dk,dv,dkw,kc,vc,control)
        np.testing.assert_allclose(out.to_numpy().reshape(r,h,d), norm_rope(q,qw), rtol=2e-5, atol=3e-6)
        expected_k = np.full((cap,kh,d), -17, np.float16)
        expected_v = expected_k.copy()
        expected_k[position:position+count] = norm_rope(k,kw)[:count]
        expected_v[position:position+count] = v[:count]
        np.testing.assert_allclose(kc.to_numpy().reshape(cap,kh,d), expected_k, rtol=1e-3, atol=2e-3)
        np.testing.assert_array_equal(vc.to_numpy().reshape(cap,kh,d), expected_v)


@GPU
@pytest.mark.parametrize('graphs', [False, True])
def test_adaptive_model_state_reset_and_greedy(graphs):
    import tensor
    from tensor_llm import LFM2
    from benchmarks.lfm2.torch_reference import Reference
    path = ROOT / 'build/lfm2-diagnostic.gguf'
    bundle = ROOT / 'build/lfm2-cuda-transfer/diagnostic-optimized'
    reference = Reference(path)
    with tensor.Device() as device, LFM2(path, bundle, device, context=384, graphs=graphs) as model:
        for tokens in ([256] + [j % 256 for j in range(161)], [65], [66], [67]):
            actual = model.forward(tokens)
            for start in range(0, len(tokens), 128):
                expected = reference.forward(tokens[start:start + 128])
            assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < .01
            assert int(np.argmax(actual)) == int(np.argmax(expected))
        model.reset(); first = model.forward([256, 65, 66])
        model.reset(); np.testing.assert_array_equal(model.forward([256, 65, 66]), first)
        device_tokens = model.generate('A short raw completion', max_tokens=5, chat=False)
        host_tokens = model.generate('A short raw completion', max_tokens=5, chat=False, gpu_greedy=False)
        assert device_tokens == host_tokens
