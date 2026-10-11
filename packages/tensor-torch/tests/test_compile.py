"""Client packaging, FX contracts, and opt-in actual CUDA execution."""

import importlib.abc
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
import tensor_torch as tt

GPU = pytest.mark.skipif(
    os.environ.get("TENSOR_P4_CUDA") != "1", reason="set TENSOR_P4_CUDA=1"
)


def test_cpu_and_unsupported_nodes_fall_back_without_compiler(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    a, b = torch.randn(129), torch.randn(129)
    with torch.inference_mode():
        function = torch.compile(
            lambda a, b: torch.relu(a * 2 + b).sin(), backend=backend, fullgraph=True
        )
        torch.testing.assert_close(function(a, b), torch.relu(a * 2 + b).sin())
    assert not backend.report["regions"]
    assert backend.report["fallback_nodes"]
    assert not list(tmp_path.glob("*.tbin"))


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_pointwise_shapes_nan_broadcast_and_cache(tmp_path, dtype):
    backend = tt.Backend(cache_dir=tmp_path)
    function = torch.compile(
        lambda a, b: torch.relu(a * 2 + b),
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    with torch.inference_mode():
        for length in (129, 257, 129):
            a = torch.randn(3, length, device="cuda", dtype=dtype)
            b = torch.randn(length, device="cuda", dtype=dtype)
            a[0, 0] = float("nan")
            torch.testing.assert_close(
                function(a, b), torch.relu(a * 2 + b), equal_nan=True
            )
    specs = [s for r in backend.report["regions"] for s in r["specializations"]]
    assert len(specs) == 2 and all(s["kind"] == "pointwise" for s in specs)
    assert not backend.report["fallback_nodes"]
    # A fresh process must execute the same cached graph with all frontend imports blocked.
    script = """
import builtins,sys,torch,tensor_torch as tt
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in {'tilelang','tvm','tvm_ffi'}: raise ImportError('compiler blocked: '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
b=tt.Backend(cache_dir=sys.argv[1]); dtype=getattr(torch,sys.argv[2])
f=torch.compile(lambda a,b:torch.relu(a*2+b),backend=b,fullgraph=True,dynamic=True)
with torch.inference_mode():
 a=torch.randn(3,129,device='cuda',dtype=dtype);c=torch.randn(129,device='cuda',dtype=dtype)
 torch.testing.assert_close(f(a,c),torch.relu(a*2+c))
assert all(s['cache_hit'] for r in b.report['regions'] for s in r['specializations'])
assert not {'tilelang','tvm','tvm_ffi'} & sys.modules.keys()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(dtype).removeprefix("torch."),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@GPU
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_tail_bias_relu_and_mlp(tmp_path, linear, dtype):
    backend = tt.Backend(cache_dir=tmp_path)
    a = torch.randn(33, 64, device="cuda", dtype=dtype)
    w = torch.randn((65, 64) if linear else (64, 65), device="cuda", dtype=dtype)
    bias = torch.randn(65, device="cuda", dtype=dtype)
    function = lambda a, w, b: torch.relu(
        torch.nn.functional.linear(a, w, b) if linear else a @ w + b
    )
    with torch.inference_mode():
        compiled = torch.compile(function, backend=backend, fullgraph=True)
        torch.testing.assert_close(
            compiled(a, w, bias), function(a, w, bias), atol=0.005, rtol=0.02
        )
        mlp = lambda a, w1, w2: torch.relu(
            torch.nn.functional.linear(
                torch.relu(torch.nn.functional.linear(a, w1)), w2
            )
        )
        w2 = torch.randn(32, 65, device="cuda", dtype=dtype)
        wt = w if linear else w.t().contiguous()
        torch.testing.assert_close(
            torch.compile(mlp, backend=backend, fullgraph=True)(a, wt, w2),
            mlp(a, wt, w2),
            atol=0.25 if dtype == torch.bfloat16 else 0.05,
            rtol=0.03,
        )
    assert (
        len([s for r in backend.report["regions"] for s in r["specializations"]]) == 3
    )


@GPU
@pytest.mark.parametrize(
    "shape,causal",
    [((1, 2, 129, 64), True), ((2, 2, 257, 64), False), ((1, 2, 128, 128), True)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fx_attention_online_softmax(tmp_path, shape, causal, dtype):
    backend = tt.Backend(cache_dir=tmp_path)
    args = tuple(torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    function = lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal
    )
    with torch.inference_mode():
        output = torch.compile(function, backend=backend, fullgraph=True)(*args)
        torch.testing.assert_close(
            output,
            function(*args),
            atol=0.02 if dtype == torch.bfloat16 else 0.002,
            rtol=0.02,
        )
    assert backend.report["regions"][0]["specializations"][0]["kind"] == "attention"


@GPU
def test_graph_break_partial_fallback_and_layout(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)

    def function(a, b):
        c = torch.relu(a * 2 + b)
        torch._dynamo.graph_break()
        return torch.relu(c.sin() + b)

    with torch.inference_mode():
        a, b = (torch.randn(257, device="cuda") for _ in range(2))
        compiled = torch.compile(function, backend=backend)
        torch.testing.assert_close(compiled(a, b), function(a, b))
        noncontiguous = a[::2]
        torch.testing.assert_close(
            compiled(noncontiguous, b[::2]), function(noncontiguous, b[::2])
        )
    assert backend.report["graphs"] >= 2
    assert any("sin" in n for n in backend.report["fallback_nodes"])
    assert any(r["specializations"] for r in backend.report["regions"])


@GPU
def test_unaligned_views_and_specialization_limit_fall_back(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path, max_specializations=1)
    function = torch.compile(
        lambda a, b: torch.relu(a * 2 + b), backend=backend, dynamic=True
    )
    with torch.inference_mode():
        a, b = (torch.randn(129, device="cuda") for _ in range(2))
        torch.testing.assert_close(function(a, b), torch.relu(a * 2 + b))
        torch.testing.assert_close(
            function(a[1:], b[1:]), torch.relu(a[1:] * 2 + b[1:])
        )
    assert any("limit" in f for r in backend.report["regions"] for f in r["fallbacks"])
    backend = tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        compiled = torch.compile(lambda a, b: torch.relu(a * 2 + b), backend=backend)
        torch.testing.assert_close(
            compiled(a[1:], b[1:]), torch.relu(a[1:] * 2 + b[1:])
        )
    assert any(
        "alignment" in f for r in backend.report["regions"] for f in r["fallbacks"]
    )


@GPU
def test_corrupt_fx_cache_is_rebuilt(tmp_path):
    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    function = lambda a, b: torch.relu(a * 2 + b)
    backend = tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        torch.compile(function, backend=backend)(a, b)
        artifact = Path(backend.report["regions"][0]["specializations"][0]["artifact"])
        artifact.write_bytes(b"corrupt")
        fresh = tt.Backend(cache_dir=tmp_path)
        torch.testing.assert_close(
            torch.compile(function, backend=fresh)(a, b), function(a, b)
        )
    assert not fresh.report["regions"][0]["specializations"][0]["cache_hit"]


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_pointwise_activation_precision_and_arithmetic(tmp_path, dtype):
    a = torch.tensor(
        [-30, -12, -1, -0.01, 0, 0.01, 1, 12, 30] + [0.1] * 120,
        device="cuda",
        dtype=dtype,
    )
    b = torch.full_like(a, 2)
    backend = tt.Backend(cache_dir=tmp_path)
    function = lambda a, b: torch.tanh(torch.sigmoid(-(a - b) / b))
    with torch.inference_mode():
        compiled = torch.compile(function, backend=backend, fullgraph=True)
        torch.testing.assert_close(
            compiled(a, b),
            function(a, b),
            atol=0.001 if dtype == torch.float16 else 1e-6,
            rtol=0.001 if dtype == torch.float16 else 1e-6,
        )
        sigmoid = torch.compile(lambda a: a.sigmoid(), backend=backend, fullgraph=True)
        torch.testing.assert_close(
            sigmoid(a),
            a.sigmoid(),
            atol=1e-7,
            rtol=0.001 if dtype == torch.float16 else 1e-6,
        )
    assert all(
        r["specializations"] and not r["fallbacks"] for r in backend.report["regions"]
    )


@GPU
def test_attention_shared_operand_and_unaligned_storage_fallback(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    function = lambda q: torch.nn.functional.scaled_dot_product_attention(q, q, q)
    shape = (1, 2, 128, 64)
    from torch.nn.attention import sdpa_kernel, SDPBackend

    # Flash SDPA itself assumes vector-aligned pointers; use its math provider
    # to validate framework fallback for a deliberately unaligned view.
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        compiled = torch.compile(
            function, backend=backend, fullgraph=True, dynamic=False
        )
        aligned = torch.randn(shape, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(
            compiled(aligned), function(aligned), atol=0.002, rtol=0.02
        )
        unaligned = torch.randn(
            aligned.numel() + 1, device="cuda", dtype=torch.float16
        )[1:].reshape(shape)
        torch.testing.assert_close(
            compiled(unaligned), function(unaligned), atol=0.002, rtol=0.02
        )
    assert any(
        "alignment" in f for r in backend.report["regions"] for f in r["fallbacks"]
    )
