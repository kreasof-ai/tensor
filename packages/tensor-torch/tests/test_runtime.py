"""Client packaging, FX contracts, and opt-in actual CUDA execution."""

import importlib.abc
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
import tensor_torch as tt

GPU = pytest.mark.skipif(
    os.environ.get("TENSOR_P4_CUDA") != "1", reason="set TENSOR_P4_CUDA=1"
)


def _require_native():
    from tensor_torch.runtime.launch import _executor

    if _executor is None:
        pytest.skip("build with TENSOR_TORCH_BUILD_NATIVE=1 for C++ executor checks")


@GPU
def test_current_stream_lifetime_and_capture(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    function = torch.compile(
        lambda a, b: torch.relu(a * 2 + b), backend=backend, fullgraph=True
    )
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(producer):
        a, b = (torch.randn(129, device="cuda") for _ in range(2))
        expected = torch.relu(a * 2 + b)
        torch.cuda._sleep(1_000_000)
    consumer.wait_stream(producer)
    with torch.inference_mode(), torch.cuda.stream(consumer):
        result = function(a, b)
        observed = result + 1
        del result
        # Exercise pointer rebinding and immediate input destruction after submission.
        for _ in range(20):
            x, y = torch.ones(129, device="cuda"), torch.ones(129, device="cuda")
            out = function(x, y)
            del x, y
        torch.testing.assert_close(observed, expected + 1)
        torch.testing.assert_close(out, torch.full_like(out, 3))
        # Prepared launches must be capturable without host synchronization.
        static_a, static_b = torch.ones(129, device="cuda"), torch.ones(
            129, device="cuda"
        )
        function(static_a, static_b)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=consumer):
            captured = function(static_a, static_b)
        static_a.fill_(2)
        graph.replay()
        torch.testing.assert_close(captured, torch.full_like(captured, 5))


@GPU
def test_prepared_binding_rejects_changed_storage_and_metadata(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    with torch.inference_mode():
        function = torch.compile(lambda a, b: torch.relu(a * 2 + b), backend=backend)
        function(a, b)
        kernel = tt.load(backend.report["regions"][0]["specializations"][0]["artifact"])
        prepared = kernel.prepare(a, b)
        a.fill_(2)
        prepared()
        torch.testing.assert_close(prepared.outputs[0], torch.relu(a * 2 + b))
        a.set_(torch.empty_like(a))
        with pytest.raises(ValueError, match="storage changed"):
            prepared()


@GPU
def test_runtime_scalars_reuse_preparation_without_changing_results(tmp_path):
    import tensor

    artifact = tmp_path / "affine.tbin"
    tensor.build(
        Path(__file__).parents[3] / "examples/dynamic_affine.py",
        artifact,
        compiler="nvrtc",
    )
    kernel = tt.load(artifact)
    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    with torch.inference_mode():
        for scale in (2.0, 0.5, -3.0):
            torch.testing.assert_close(kernel(a, b, scale), torch.relu(a * scale + b))
    assert len(kernel._prepared) == 1


@GPU
def test_native_prepared_scalars_metadata_streams_and_capture(tmp_path):
    _require_native()
    import tensor

    artifact = tmp_path / "affine.tbin"
    tensor.build(
        Path(__file__).parents[3] / "examples/dynamic_affine.py",
        artifact,
        compiler="nvrtc",
    )
    kernel = tt.load(artifact)
    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.inference_mode():
        for scale in (2.0, 0.5, -3.0):
            with torch.cuda.stream(producer):
                a.fill_(2)
                call = kernel.prepare(a, b, scale)
                assert call._native is not None
            consumer.wait_stream(producer)
            with torch.cuda.stream(consumer):
                call()
                torch.testing.assert_close(call.outputs[0], torch.relu(a * scale + b))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=consumer):
                    call()
                a.fill_(3)
                graph.replay()
                torch.testing.assert_close(call.outputs[0], torch.relu(a * scale + b))
                call.outputs[0].resize_(128)
                with pytest.raises(ValueError, match="metadata or storage changed"):
                    call()
                call.outputs[0].resize_(129)
                a.resize_(128)
                with pytest.raises(ValueError, match="metadata or storage changed"):
                    call()
                a.resize_(129)


@GPU
def test_native_fx_pointer_rebinding_concurrency_and_alignment(tmp_path):
    _require_native()
    from concurrent.futures import ThreadPoolExecutor
    from tensor_torch.kernel import Kernel
    from tensor_torch.runtime.prepared import LaunchPlan

    backend = tt.Backend(cache_dir=tmp_path)
    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    with torch.inference_mode():
        torch.compile(
            lambda a, b: torch.relu(a * 2 + b), backend=backend, fullgraph=True
        )(a, b)
        artifact = backend.report["regions"][0]["specializations"][0]["artifact"]
        kernel = Kernel(artifact)
        fallback = lambda a, b: torch.relu(a * 2 + b)
        plan = LaunchPlan(kernel, (a, b), fallback)
        assert plan._native is not None and plan.prepared._native is None
        assert not plan.prepared.tensors
        unaligned = torch.randn(130, device="cuda")[1:]
        torch.testing.assert_close(plan(unaligned, b), fallback(unaligned, b))
        with pytest.raises(ValueError, match="metadata changed"):
            plan(a[:128], b[:128])
        del a, b

    def worker(seed):
        stream = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            for i in range(30):
                x, y = torch.full((129,), float(seed + i), device="cuda"), torch.ones(
                    129, device="cuda"
                )
                output = plan(x, y)
                del x, y
                torch.testing.assert_close(
                    output, torch.full_like(output, 2 * (seed + i) + 1)
                )
        stream.synchronize()

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, range(3)))


@GPU
def test_native_fixed_and_fx_calls_reject_closed_session(tmp_path):
    _require_native()
    from tensor_torch.kernel import Kernel
    from tensor_torch.runtime.prepared import LaunchPlan

    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    backend = tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        torch.compile(
            lambda a, b: torch.relu(a * 2 + b), backend=backend, fullgraph=True
        )(a, b)
        kernel = Kernel(backend.report["regions"][0]["specializations"][0]["artifact"])
        call = kernel.prepare(a, b)
        plan = LaunchPlan(kernel, (a, b), lambda a, b: torch.relu(a * 2 + b))
        assert call._native is not None and plan._native is not None
        tt.close()
        with pytest.raises(RuntimeError, match="closed session"):
            call()
        with pytest.raises(RuntimeError, match="closed session"):
            plan(a, b)


@GPU
def test_native_cuda_failures_preserve_provider_exception(tmp_path):
    _require_native()
    from tensor.providers.cuda import CudaError
    from tensor_torch.kernel import Kernel
    from tensor_torch.runtime.launch import _native_plan

    a, b = (torch.randn(129, device="cuda") for _ in range(2))
    backend = tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        torch.compile(
            lambda a, b: torch.relu(a * 2 + b), backend=backend, fullgraph=True
        )(a, b)
        kernel = Kernel(backend.report["regions"][0]["specializations"][0]["artifact"])
        prepared = kernel.prepare(a, b)
        # Zero block width is rejected by the driver before any device execution.
        prepared.launch = {**prepared.launch, "block": [0, 1, 1]}
        invalid = _native_plan(prepared, (a, b), prepared.outputs, fixed=True)
        with pytest.raises(CudaError, match="cuLaunchKernel failed with CUDA error"):
            invalid()
        prepared()
        torch.testing.assert_close(prepared.outputs[0], torch.relu(a * 2 + b))
