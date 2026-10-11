"""Client packaging, FX contracts, and opt-in actual CUDA execution."""

import importlib.abc
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import tensor_torch as tt

GPU = pytest.mark.skipif(
    os.environ.get("TENSOR_P4_CUDA") != "1", reason="set TENSOR_P4_CUDA=1"
)


def test_inference_backend_preserves_eager_autograd(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    a = torch.randn(33, requires_grad=True)
    function = torch.compile(lambda a: (a * 2).relu(), backend=backend)
    function(a).sum().backward()
    torch.testing.assert_close(a.grad, 2 * (a > 0).float())
    assert backend.report["autograd"][0]["stage"] == "fallback"


@GPU
def test_aot_autograd_uses_compiled_forward_and_backward_region(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path, training=True)
    function = torch.compile(
        lambda a, b: torch.relu(a * 2 + b), backend=backend, fullgraph=True
    )
    a, b = (torch.randn(129, device="cuda", requires_grad=True) for _ in range(2))
    output = function(a, b)
    output.sum().backward()
    torch.testing.assert_close(a.grad, 2 * (a * 2 + b > 0).float())
    torch.testing.assert_close(b.grad, (a * 2 + b > 0).float())
    assert {g["stage"] for g in backend.report["autograd"]} == {"forward", "backward"}
    assert len([r for r in backend.report["regions"] if r["specializations"]]) >= 2
    assert any("threshold_backward" in n for n in backend.report["fallback_nodes"])
