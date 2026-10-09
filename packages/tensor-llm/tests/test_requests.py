"""Shared ownership and request isolation on a real mixed-encoding CUDA fixture."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensor_llm import LFM2, LFM2Request


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_request_limit_rejected_before_loading(limit):
    with pytest.raises(ValueError, match='max_requests'):
        LFM2('missing-model', 'missing-bundle', SimpleNamespace(), max_requests=limit)


GPU = pytest.mark.skipif(os.environ.get('TENSOR_LFM2_CUDA') != '1',
                         reason='set TENSOR_LFM2_CUDA=1 for CUDA/NVRTC qualification')


@pytest.fixture(scope='module', params=['default', 'optimized'])
def fixture_bundle(request, tmp_path_factory):
    from benchmarks.lfm2.diagnostic import fixture
    from benchmarks.lfm2.producer import produce
    cache = os.environ.get('TENSOR_LFM2_REQUEST_CACHE')
    directory = (Path(cache) / request.param if cache
                 else tmp_path_factory.mktemp('lfm2-requests-' + request.param))
    model = fixture(directory / 'model.gguf')
    bundle = directory / 'bundle'
    produce(model, bundle, context=384, target='sm_89', cuda_profile=request.param)
    return model, bundle


@GPU
@pytest.mark.parametrize('graphs', [False, True])
def test_eight_interleaved_requests_and_slot_lifecycle(fixture_bundle, graphs):
    import tensor
    model_path, bundle = fixture_bundle
    prefixes = [[256] + [(j + 13 * i) % 256 for j in range(n - 1)]
                for i, n in enumerate((1, 2, 31, 32, 127, 128, 129, 257))]
    with tensor.Device() as device, LFM2(model_path, bundle, device,
            context=384, graphs=graphs, max_requests=8) as model:
        initial_bytes = model.allocated_bytes
        requests = [model] + [model.new_request() for _ in range(7)]
        assert all(isinstance(r, LFM2Request) for r in requests)
        assert model.request_count == 8
        assert model.allocated_bytes == model.shared_bytes + sum(r.private_bytes for r in requests)
        assert model.allocated_bytes - initial_bytes == sum(r.private_bytes for r in requests[1:])
        assert all(r.weights is model.weights and r.kernels is model.kernels
                   and r.workspaces is model.workspaces for r in requests)
        assert len({r.control.pointer for r in requests}) == 8
        assert len({r.logits.pointer for r in requests}) == 8
        assert len({r.states[0].pointer for r in requests}) == 8
        assert len({r.caches[1][0].pointer for r in requests}) == 8
        with pytest.raises(RuntimeError, match='capacity exhausted'):model.new_request()

        # Independent model instances expose any accidental shared cache/history.
        with LFM2(model_path, bundle, device, context=384, graphs=graphs) as independent:
            expected = []
            for prefix in prefixes:
                independent.reset()
                expected.append([independent.forward(prefix), independent.forward([65]),
                                 independent.forward([66, 67])])
            for index, r in enumerate(requests):
                np.testing.assert_array_equal(r.forward(prefixes[index]), expected[index][0])
            for index in reversed(range(8)):
                np.testing.assert_array_equal(requests[index].forward([65]), expected[index][1])
            for index in (3, 0, 6, 1, 7, 2, 5, 4):
                # Reset another request while this one's continuation is live.
                if index == 3:requests[2].reset()
                if index == 2:
                    requests[2].forward(prefixes[2]);requests[2].forward([65])
                np.testing.assert_array_equal(requests[index].forward([66, 67]), expected[index][2])
            for index, r in enumerate(requests):
                assert r.position == len(prefixes[index]) + 3
            for index, r in enumerate(requests):
                # read=False outputs remain private even when scratch is reused.
                independent.reset();want = independent.forward(prefixes[index])
                r.reset();r.forward(prefixes[index], read=False)
                requests[(index + 1) % 8].forward([68])
                np.testing.assert_array_equal(r.logits.to_numpy(), want)

        stale = requests[1]
        remaining = model.allocated_bytes - stale.private_bytes
        stale.close();stale.close()
        assert model.request_count == 7 and model.allocated_bytes == remaining
        replacement = model.new_request(context=128, graphs=not graphs)
        assert replacement.position == 0 and replacement is not stale
        with pytest.raises(RuntimeError, match='closed'):stale.forward([65])
        with pytest.raises(RuntimeError, match='closed'):stale.reset()
        np.testing.assert_array_equal(replacement.forward(prefixes[1]), expected[1][0])
        with pytest.raises(ValueError, match='capacity'):replacement.forward([65] * 128)
        model.close()
        assert replacement.closed and all(r.closed for r in requests)
        assert model.allocated_bytes == 0 and model.request_count == 0
        with pytest.raises(RuntimeError, match='closed'):model.new_request()


@GPU
def test_failed_request_allocation_releases_private_resources(fixture_bundle, monkeypatch):
    import tensor
    model_path, bundle = fixture_bundle
    with tensor.Device() as device, LFM2(model_path, bundle, device,
            context=384, max_requests=2) as model:
        baseline = model.forward([256, 65, 66])
        before = model.allocated_bytes
        allocations = []
        original = device.from_numpy
        def fail(host):
            if len(allocations) == 3:raise MemoryError('injected request allocation failure')
            buffer = original(host);allocations.append(buffer);return buffer
        with monkeypatch.context() as patch:
            patch.setattr(device, 'from_numpy', fail)
            with pytest.raises(MemoryError, match='injected'):model.new_request()
        assert all(b._released for b in allocations)
        assert model.allocated_bytes == before and model.request_count == 1
        model.reset();np.testing.assert_array_equal(model.forward([256, 65, 66]), baseline)
        with model.new_request() as other:
            np.testing.assert_array_equal(other.forward([256, 65, 66]), baseline)


@GPU
@pytest.mark.parametrize('operation', ['from_numpy', 'load'])
def test_failed_shared_model_loading_releases_allocations(fixture_bundle, monkeypatch, operation):
    import tensor
    model_path, bundle = fixture_bundle
    with tensor.Device() as device:
        buffers = set(device._buffers)
        executables = dict(device._executables)
        original = getattr(device, operation)
        calls = []
        def fail(*args, **kwargs):
            if len(calls) == 2:raise MemoryError('injected shared model loading failure')
            result = original(*args, **kwargs);calls.append(result);return result
        with monkeypatch.context() as patch:
            patch.setattr(device, operation, fail)
            with pytest.raises(MemoryError, match='injected'):
                LFM2(model_path, bundle, device, context=384)
        assert set(device._buffers) == buffers
        assert device._executables == executables
        assert all(item._released for item in calls)


@GPU
def test_failed_graph_capture_does_not_release_shared_resources(fixture_bundle, monkeypatch):
    import tensor
    import tensor_llm.lfm2.model as module
    model_path, bundle = fixture_bundle
    with tensor.Device() as device, LFM2(model_path, bundle, device,
            context=384, max_requests=2) as model:
        before = model.allocated_bytes
        original = module.CudaGraph
        captured = []
        def fail(*args, **kwargs):
            if captured:raise RuntimeError('injected graph capture failure')
            graph = original(*args, **kwargs);captured.append(graph);return graph
        with monkeypatch.context() as patch:
            patch.setattr(module, 'CudaGraph', fail)
            with pytest.raises(RuntimeError, match='injected'):model.new_request()
        assert captured[0]._closed
        assert model.allocated_bytes == before and model.request_count == 1
        assert all(not w._released for w in model.weights.values())
        with model.new_request() as other:
            np.testing.assert_array_equal(other.forward([256, 65]), model.forward([256, 65]))


@GPU
def test_reject_wrong_thread_before_state_mutation(fixture_bundle):
    import tensor
    model_path, bundle = fixture_bundle
    with tensor.Device() as device, LFM2(model_path, bundle, device, context=384) as model:
        with ThreadPoolExecutor(max_workers=1) as worker:
            for action in (lambda: model.forward([65]), model.reset, model.close):
                with pytest.raises(RuntimeError, match='owner thread'):worker.submit(action).result()
        assert model.position == 0 and not model.closed
        assert np.isfinite(model.forward([256, 65])).all()


@GPU
def test_metadata_copy_completed_before_nonblocking_execution(fixture_bundle, monkeypatch):
    """Model pageable DMA as deferred until default-stream completion.

    A host-returning HtoD copy is allowed to finish staging before device DMA.
    The executor uses a nonblocking stream, so stream completion must order it.
    """
    import ctypes as ct
    import tensor
    model_path, bundle = fixture_bundle
    with tensor.Device() as device, LFM2(model_path, bundle, device,
            context=384, max_requests=2) as model, model.new_request() as other:
        expected = [model.forward([256, 65, 66]), model.forward([67])]
        model.reset()
        original = device.driver.call
        pending = []
        def defer(name, *args):
            if name == 'cuMemcpyHtoD_v2':
                pointer, host, size = args
                pending.append((pointer, ct.string_at(host, size)))
                return
            if name == 'cuStreamSynchronize' and args[0] is None:
                for pointer, data in pending:
                    host = ct.create_string_buffer(data)
                    original('cuMemcpyHtoD_v2', pointer, host, len(data))
                pending.clear()
            return original(name, *args)
        with monkeypatch.context() as patch:
            patch.setattr(device.driver, 'call', defer)
            np.testing.assert_array_equal(model.forward([256, 65, 66]), expected[0])
            other.forward([256, 71, 72, 73])
            np.testing.assert_array_equal(model.forward([67]), expected[1])
            assert not pending


@GPU
def test_bundle_provenance_and_request_context_rejections(fixture_bundle, tmp_path):
    import tensor
    model_path, bundle = fixture_bundle
    manifest = json.loads((bundle / 'inference.json').read_text())
    manifest['implementation']['tensor_llm.lfm2.model'] = 'old-implementation'
    (tmp_path / 'inference.json').write_text(json.dumps(manifest))
    with tensor.Device() as device:
        with pytest.raises(ValueError, match='implementation mismatch'):
            LFM2(model_path, tmp_path, device, context=384)
        with LFM2(model_path, bundle, device, context=384, max_requests=2) as model:
            for context in (0, 385, True, 1.5):
                with pytest.raises(ValueError, match='capacity'):model.new_request(context=context)
            assert model.request_count == 1


@GPU
def test_gpu_host_sampling_and_independent_numeric_reference(fixture_bundle):
    import tensor
    from benchmarks.lfm2.torch_reference import Reference
    model_path, bundle = fixture_bundle
    reference = Reference(model_path)
    with tensor.Device() as device, LFM2(model_path, bundle, device,
            context=384, max_requests=2) as model, model.new_request() as other:
        for index, r in enumerate((model, other)):
            reference.reset()
            for tokens in ([256] + [(i + index) % 256 for i in range(128)], [65], [66, 67]):
                actual = r.forward(tokens)
                for start in range(0, len(tokens), 128):want = reference.forward(tokens[start:start + 128])
                assert np.isfinite(actual).all()
                assert np.linalg.norm(actual - want) / np.linalg.norm(want) < .01
        for r in (model, other):
            gpu = r.generate('Hello', max_tokens=8, chat=False)
            assert gpu == r.generate('Hello', max_tokens=8, chat=False, gpu_greedy=False)
