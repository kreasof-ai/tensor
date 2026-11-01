"""A faster invalid schedule must never win the kernel search."""
import numpy as np
import pytest
import os


def test_region_selection_reuses_bank_and_gates_each_measured_shape():
    from tensor.compiler.tuning import select_regions

    report = dict(shapes=[1, 3, 7, 31, 64, 129, 256], weights=[1] * 7,
                  trials=[dict(status="passed", measurements=[dict(seconds=v) for v in values])
                          for values in ([1, 1, 1, 2, 3, 4, 5], [3, 2, 2, 1, 1, 1, 1])])
    result = select_regions(report, symbol="rows", bounds=(1, 256), max_regions=2)
    assert [r["trial"] for r in result["regions"]] == [0, 1]
    assert [r["bounds"] for r in result["regions"]] == [
        {"rows": dict(min=1, max=19)}, {"rows": dict(min=20, max=256)}]
    assert result["weighted_relative_latency"] == 1
    assert result["performance_coverage"] == "measured shapes only"
    with pytest.raises(ValueError, match="no portfolio"):
        select_regions(report, symbol="rows", bounds=(1, 256), max_regions=1)
    # A global mean must not hide an isolated measured regression.
    with pytest.raises(ValueError, match="no portfolio"):
        select_regions(report, symbol="rows", bounds=(1, 256), reference_seconds=[.1] + [1] * 6)

from tensor.compiler.tuning import tune


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_cuda_working_set_rejects_invalid_footprints(size):
    from types import SimpleNamespace
    from tensor.compiler.tuning import cuda_working_set

    with pytest.raises(ValueError, match="positive integer"):
        with cuda_working_set(SimpleNamespace(info={"provider": "cuda"}), size):
            pass


def test_cuda_working_set_rejects_other_providers():
    from types import SimpleNamespace
    from tensor.compiler.tuning import cuda_working_set

    with pytest.raises(ValueError, match="requires CUDA"):
        with cuda_working_set(SimpleNamespace(info={"provider": "cpu"}), 1024):
            pass


@pytest.mark.skipif(os.environ.get("TENSOR_P1_CUDA") != "1", reason="native CUDA measurement")
def test_cuda_working_set_can_condition_measurements_without_changing_their_inputs(tmp_path):
    import tensor
    from tensor.compiler.tuning import cuda_working_set

    with tensor.Device() as device:
        value = device.from_numpy(np.arange(1024, dtype="float32"))
        try:
            identity = None
            for _ in range(2):
                with cuda_working_set(device, 1 << 20, out=tmp_path) as (prepare, description):
                    assert description["weight_bytes"] >= 1 << 20
                    assert description["working_set_bytes"] > description["weight_bytes"]
                    if identity is not None:
                        assert description["artifact_sha256"] == identity
                    identity = description["artifact_sha256"]
                    prepare()
                    np.testing.assert_array_equal(value.to_numpy(), np.arange(1024, dtype="float32"))
                with pytest.raises(RuntimeError):
                    prepare()
        finally:
            value.release()


class Output:
    def __init__(self):self.value=None
    def to_numpy(self):return self.value


class Candidate:
    def __init__(self,result,latency,*,saved=None):self.result=result;self.device=latency;self.saved=saved
    def launch(self,*arguments):
        arguments[-1].value=np.asarray(self.result,dtype=np.float32)
        if self.saved is not None:arguments[-2].value=np.asarray(self.saved,dtype=np.float32)


def test_tuning_rejects_fast_wrong_results_and_wrong_saved_intermediates(monkeypatch):
    monkeypatch.setattr('tensor.compiler.tuning.measure',lambda device,callback,**kw:{'median_seconds':device})
    output,saved=Output(),Output()
    candidates={'wrong_output':Candidate([9,9],0.01,saved=[1,2]),
                'wrong_saved':Candidate([1,2],0.02,saved=[9,9]),
                'correct':Candidate([1,2],1.0,saved=[1,2])}
    result=tune(candidates,[saved],output,np.array([1,2]),checks=[(saved,np.array([1,2]))])
    assert result['selected']=='correct'
    assert result['candidates']['wrong_output']['status']=='rejected'
    assert result['candidates']['wrong_saved']['status']=='rejected'


def test_tuning_requires_finite_reference_and_correct_candidates(monkeypatch):
    monkeypatch.setattr('tensor.compiler.tuning.measure',lambda device,callback,**kw:{'median_seconds':device})
    with pytest.raises(ValueError,match='finite'):tune({'bad':Candidate([0],1)},[],Output(),[np.nan])
    with pytest.raises(ValueError,match='all tuning'):tune({'bad':Candidate([0],1)},[],Output(),[10])


def test_webgpu_measurement_resets_every_invocation_outside_timing(monkeypatch):
    from types import SimpleNamespace

    from tensor.compiler.tuning import measure
    events = []
    clock = [0.0]
    def reset():
        events.append("reset")
        clock[0] += 100.0
    def callback():
        events.append("launch")
        clock[0] += 2.0
    device = SimpleNamespace(info={"provider": "webgpu"}, synchronize=lambda: events.append("sync"))
    monkeypatch.setattr("tensor.compiler.tuning.time.perf_counter", lambda: clock[0])
    result = measure(device, callback, warmup=1, samples=2, repeats=3, reset=reset)
    assert result["median_seconds"] == 2.0
    assert result["metric"] == "completed_seconds"
    assert "gpu_seconds" not in result
    assert events.count("reset") == events.count("launch") == 7
    assert all(events[i-1] in ("reset", "sync") for i,x in enumerate(events) if x == "launch")


def test_measurement_rejects_unsupported_provider_and_invalid_budget():
    from types import SimpleNamespace

    from tensor.compiler.tuning import measure
    with pytest.raises(ValueError, match="CUDA or WebGPU"):
        measure(SimpleNamespace(info={"provider": "cpu"}), lambda: None)
    with pytest.raises(ValueError, match="measurement counts"):
        measure(SimpleNamespace(info={"provider": "webgpu"}), lambda: None, samples=0)


def test_multi_shape_search_reuses_builds_and_releases_after_partial_failure():
    from contextlib import contextmanager
    from tensor.compiler.search import ScheduleSearch
    from tensor.compiler.tuning import tune_shapes

    events = []
    search = ScheduleSearch([{"family": "projection", "tile": 16}],
                            spaces={"projection": {"tile": (16, 32, 64)}})
    @contextmanager
    def build(config):
        events.append(("build", config["tile"]))
        try:
            yield config["tile"]
        finally:
            events.append(("release", config["tile"]))
    def evaluate(tile, shape):
        events.append(("evaluate", tile, shape))
        return {"correct": not (tile == 64 and shape == 7),
                "seconds": (1.0 if shape == 1 else 2.0) if tile == 16 else .5}
    result = tune_shapes(search, [1, 7, 33], build, evaluate, budget=3)
    assert result["compiled_candidates"] == 3
    assert result["selected"] == [1, 1, 1]
    assert result["trials"][2]["status"] == "rejected"
    assert events.count(("build", 16)) == events.count(("release", 16)) == 1
    assert events[-1] == ("release", 64)
    assert ("evaluate", 64, 33) not in events
