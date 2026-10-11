"""Provider submission, borrowed resources, and legacy precision/layout contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from tensor.runtime.abi import StreamDescriptor
from tensor_llm.common.plan import ExecutionPlan, capture, submit
from tensor_llm.common.state import allocate_hybrid_states


class Device:
    capabilities = frozenset()

    def __init__(self, provider="cuda"):
        self.info = dict(provider=provider)
        self.launched = []

    def stream_descriptor(self):
        return StreamDescriptor(1, 0, 0)

    def _launch(self, kernel, call):
        self.launched.append(kernel)

    def prepare_plan(self, plan):
        self.prepared = tuple(plan)
        return self.prepared


class Kernel:
    manifest = dict(arguments=[])

    def _bind(self, args, kwargs, *, include_outputs):
        assert include_outputs
        return {}, {}, dict(grid=[1, 1, 1], block=[256, 1, 1], shared_memory_bytes=0)


def test_derived_plan_and_labels_preserve_submission_order():
    device = Device()
    plan = ExecutionPlan(device, labeled=True)
    first, second = Kernel(), Kernel()
    plan.bind(first, (), label="normalization")
    plan.bind(second, (), label="projection")
    submit(device, tuple(plan))
    assert device.launched == [first, second]
    assert [entry[2] for entry in plan] == ["normalization", "projection"]
    inspected = []

    def inspect(entry):
        assert device.launched[-1] is entry[0]
        inspected.append(entry[2])

    submit(device, plan, after_launch=inspect)
    assert inspected == ["normalization", "projection"]


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_capture_uses_shared_order_and_borrows_resources(monkeypatch, provider):
    import tensor.providers.cuda_graph

    class Graph:
        def __init__(self, device, callback, *, resources):
            self.callback, self.resources = callback, resources

    monkeypatch.setattr(tensor.providers.cuda_graph, "CudaGraph", Graph)
    device = Device(provider)
    kernel, resource = Kernel(), object()
    plan = ExecutionPlan(device)
    plan.bind(kernel, ())
    graph = capture(device, tuple(plan), resources=(resource,))
    if provider == "cuda":
        assert graph.resources == (resource,)
        graph.callback()
        assert device.launched == [kernel]
    else:
        assert graph == tuple(plan) == device.prepared


def test_lfm_state_layout_and_zero_initialization_are_preserved():
    config = SimpleNamespace(
        layers=("conv", "attention"), conv=3, width=128, kv_heads=2, head_dim=64
    )
    arrays = []

    class Allocator:
        def from_numpy(self, array):
            arrays.append(array)
            return array

    states = allocate_hybrid_states(
        Allocator(),
        config,
        1,
        384,
        kv_dtype="float16",
        zero_recurrent=True,
        zero_cache=True,
        flat=True,
    )
    assert len(states[0]) == 1 and len(states[1]) == 2
    assert arrays[0].shape == (256,) and arrays[0].dtype == np.float32
    assert all(
        array.shape == (384 * 2 * 64,) and array.dtype == np.float16
        for array in arrays[1:]
    )
    assert all(not np.any(array) for array in arrays)


@pytest.mark.parametrize(
    "module,kind,parameters,contract",
    [
        (
            "ggml",
            "rms",
            dict(r=2, c=128, eps=1e-6),
            [
                ("x", (256,), "float32"),
                ("w", (128,), "float32"),
                ("out", (256,), "float32"),
            ],
        ),
        (
            "cuda",
            "add_rms",
            dict(r=2, c=128, eps=1e-6),
            [
                ("x", (256,), "float32"),
                ("mixed", (256,), "float32"),
                ("residual", (256,), "float32"),
                ("w", (128,), "float32"),
                ("out", (256,), "float32"),
            ],
        ),
        (
            "hybrid",
            "rms",
            dict(r=2, c=128, eps=1e-6),
            [
                ("residual", (2, 128), "bfloat16"),
                ("w", (128,), "bfloat16"),
                ("out", (2, 128), "bfloat16"),
            ],
        ),
        (
            "hybrid",
            "add_rms",
            dict(r=2, c=128, eps=1e-6),
            [
                ("x", (2, 128), "float32"),
                ("residual", (2, 128), "bfloat16"),
                ("w", (128,), "bfloat16"),
                ("out", (2, 128), "bfloat16"),
            ],
        ),
        (
            "ggml",
            "swiglu",
            dict(r=2, c=128),
            [
                ("x", (256,), "float32"),
                ("y", (256,), "float32"),
                ("out", (256,), "float32"),
            ],
        ),
        (
            "hybrid",
            "swiglu",
            dict(r=2, c=128),
            [
                ("g", (2, 128), "float32"),
                ("u", (2, 128), "float32"),
                ("out", (2, 128), "bfloat16"),
            ],
        ),
        (
            "hybrid_batch",
            "swiglu",
            dict(slots=2, chunk=4, pool=2, c=128),
            [("x", (8, 256), "float32"), ("out", (8, 128), "bfloat16")],
        ),
    ],
)
def test_shared_kernel_preserves_each_existing_adapter_abi(
    module, kind, parameters, contract
):
    import importlib

    pytest.importorskip("tilelang")
    factory = importlib.import_module(f"tensor_llm.kernels.{module}")
    kernel = factory.make_kernel(kind, parameters)
    observed = [
        (
            kernel.buffer_map[param].name,
            tuple(int(size) for size in kernel.buffer_map[param].shape),
            kernel.buffer_map[param].dtype,
        )
        for param in kernel.params
    ]
    assert observed == contract


@pytest.mark.parametrize("module", ["ggml", "cuda", "hybrid", "webgpu"])
def test_shared_operator_change_invalidates_producer_source(
    monkeypatch, tmp_path, module
):
    import importlib
    from pathlib import Path
    from tensor.compiler import entry

    original = entry.find_spec
    helper = tmp_path / "elementwise.py"
    helper.write_bytes(
        Path(original("tensor_llm.kernels.elementwise").origin).read_bytes()
    )

    def find_spec(name):
        return (
            SimpleNamespace(origin=str(helper))
            if name == "tensor_llm.kernels.elementwise"
            else original(name)
        )

    monkeypatch.setattr(entry, "find_spec", find_spec)
    source = importlib.import_module(f"tensor_llm.kernels.{module}").source
    first = source("rms", dict(r=2, c=128, eps=1e-6))
    helper.write_bytes(helper.read_bytes() + b"\n# Updated shared implementation.\n")
    assert source("rms", dict(r=2, c=128, eps=1e-6)) != first
