"""Explicit correctness-checked tuning of precompiled Tensor kernel candidates."""

from __future__ import annotations

import ctypes as c
import math
import statistics
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def cuda_working_set(device, weight_bytes, *, out=None, target=None, seed=17):
    """Condition cache residency with an independent canonical matrix product.

    Yield (launch, description). Pass launch through measurement's reset callback
    so the extra work completes outside timing. This is a controlled workload,
    not a hardware cache-invalidation guarantee; choose a footprint appropriate
    to the device and retain the returned description with measurements.
    """
    from contextlib import ExitStack
    import hashlib
    from importlib.metadata import version
    from pathlib import Path
    import tempfile

    if type(weight_bytes) is not int or weight_bytes < 1:
        raise ValueError("working-set weight bytes must be a positive integer")
    if device.info["provider"] != "cuda":
        raise ValueError("working-set conditioning requires CUDA")
    if type(seed) is not int:
        raise ValueError("working-set seed must be an integer")
    import tensor

    rows, depth = 32, 2048
    columns = max(128, ((weight_bytes + depth * 2 * 128 - 1) // (depth * 2 * 128)) * 128)
    target = device.info["arch"] if target is None else target
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tensor-working-set-"))) if out is None else Path(out)
        root.mkdir(parents=True, exist_ok=True)
        generated = (
            "from tensor.compiler.cuda.matmul import matmul\n"
            "def tensor_export():\n"
            f" return dict(kernel=matmul({rows}, {columns}, {depth}, dtype='float16', out_dtype='float32', "
            "block_m=32, block_n=128, block_k=64, threads=128, stages=2))\n"
        )
        from tensor.compiler.cuda import lowering
        from tensor.compiler.cuda.prelower import lowering_identity

        compiler_identity = (lowering_identity(),
                             hashlib.sha256(Path(lowering.__file__).read_bytes()).hexdigest(),
                             version("tilelang"), version("apache-tvm-ffi"))
        key = hashlib.sha256(repr((generated, target, compiler_identity)).encode()).hexdigest()[:20]
        source, artifact = root / f"{key}.py", root / f"{key}.tbin"
        source.write_text(generated)
        if not artifact.exists():
            tensor.build(source, artifact, target=target)
        executable = stack.enter_context(device.load(artifact))
        if (executable.manifest["source_sha256"] != hashlib.sha256(generated.encode()).hexdigest()
                or executable.manifest["target"] != target):
            raise ValueError("working-set artifact does not match its source and target")
        rng = np.random.default_rng(seed)
        buffers = []
        for shape in ((rows, depth), (columns, depth)):
            value = device.from_numpy(rng.standard_normal(shape, dtype=np.float32).astype(np.float16))
            stack.callback(value.release)
            buffers.append(value)
        result = device.empty((rows, columns), "float32")
        stack.callback(result.release)
        buffers.append(result)
        bound = executable.bind(*buffers)
        bound.launch()
        device.synchronize()
        description = dict(method="independent-canonical-matmul", target=target, seed=seed,
                           requested_weight_bytes=weight_bytes, weight_bytes=buffers[1].nbytes,
                           working_set_bytes=sum(value.nbytes for value in buffers),
                           geometry=dict(m=rows, n=columns, k=depth), dtype="float16",
                           artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                           source_sha256=executable.manifest["source_sha256"])
        yield bound.launch, description


def tune_shapes(search, shapes, build, evaluate, *, budget, weights=None):
    """Share each compiled schedule across a bounded set of dynamic shapes.

    ``build(config)`` is a context manager owning one candidate's resources.
    ``evaluate(candidate, shape)`` qualifies independently before returning a
    positive finite latency. Shapes run sequentially so reference working sets
    need not accumulate. A failed shape disqualifies the candidate everywhere.
    This returns measured points, not an inferred guarantee over untested shapes.
    """
    shapes = tuple(shapes)
    if not shapes or type(budget) is not int or budget < 1:
        raise ValueError("multi-shape tuning needs shapes and a positive budget")
    weights = tuple(weights) if weights is not None else (1.0,) * len(shapes)
    if (len(weights) != len(shapes)
            or any(not math.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError("shape weights must be positive and finite")
    trials, baseline = [], None
    for _ in range(budget):
        try:
            config = search.next()
        except StopIteration:
            break
        record = {"config": dict(config), "measurements": []}
        try:
            with build(config) as candidate:
                for shape in shapes:
                    measurement = evaluate(candidate, shape)
                    if (measurement.get("correct") is not True
                            or not math.isfinite(measurement["seconds"])
                            or measurement["seconds"] <= 0):
                        raise ValueError("shape qualification or timing failed")
                    record["measurements"].append(dict(measurement))
            seconds = [m["seconds"] for m in record["measurements"]]
            if baseline is None:
                baseline = seconds
            # Workload weighting guides exploration while each point retains
            # its own measurement and winner for later dispatch selection.
            score = math.exp(sum(w * math.log(t / b) for w, t, b in
                                 zip(weights, seconds, baseline)) / sum(weights))
            search.record(config, score)
            record.update(status="passed", score=score)
        except (ValueError, AssertionError) as error:
            record.update(status="rejected", reason=str(error))
        trials.append(record)
    valid = [i for i, trial in enumerate(trials) if trial["status"] == "passed"]
    if not valid:
        raise ValueError("all multi-shape tuning candidates failed qualification")
    selected = [min(valid, key=lambda i: trials[i]["measurements"][j]["seconds"])
                for j in range(len(shapes))]
    return {"shapes": list(shapes), "weights": list(weights), "trials": trials,
            "selected": selected, "compiled_candidates": len(trials)}


def select_regions(report, *, symbol, bounds, max_regions=8, max_slowdown=1.10,
                   reference_seconds=None):
    """Choose bounded contiguous regions from one shared one-axis candidate bank.

    Minimize weighted relative latency under a per-measured-point slowdown cap
    and a region-count budget. Dynamic programming reuses existing measurements;
    it performs no builds. Midpoints between measured shapes define dispatch
    boundaries. Unmeasured integers have coverage, but no performance claim.
    """
    shapes = report["shapes"]
    if (not isinstance(symbol, str) or not symbol.isidentifier()
            or not shapes or any(type(v) is not int for v in shapes)
            or list(shapes) != sorted(set(shapes))
            or len(bounds) != 2 or tuple(bounds) != (shapes[0], shapes[-1])
            or shapes[0] < 1 or type(max_regions) is not int or not 1 <= max_regions <= 64
            or not math.isfinite(max_slowdown) or max_slowdown < 1):
        raise ValueError("region selection needs ordered integer shapes including both bounds")
    valid = [i for i, trial in enumerate(report["trials"]) if trial["status"] == "passed"]
    if not valid:
        raise ValueError("region selection needs qualified candidates")
    weights = report["weights"]
    times = {i: [m["seconds"] for m in report["trials"][i]["measurements"]] for i in valid}
    reference = (list(reference_seconds) if reference_seconds is not None else
                 [min(times[i][j] for i in valid) for j in range(len(shapes))])
    if (len(reference) != len(shapes) or len(weights) != len(shapes)
            or any(len(row) != len(shapes) for row in times.values())
            or any(not math.isfinite(v) or v <= 0
                   for v in [*reference, *weights, *(v for row in times.values() for v in row)])):
        raise ValueError("region selection requires complete positive finite measurements")
    # dp[(number of regions, measured prefix length)] = (cost, [(begin,end,trial)])
    count = len(shapes)
    dp = {(0, 0): (0.0, [])}
    for end in range(1, count + 1):
        costs = {i: 0.0 for i in valid}
        eligible = set(valid)
        for begin in range(end - 1, -1, -1):
            for i in tuple(eligible):
                ratio = times[i][begin] / reference[begin]
                if ratio > max_slowdown:
                    eligible.remove(i)
                else:
                    costs[i] += weights[begin] * ratio
            if not eligible:
                break
            trial = min(eligible, key=lambda i: (costs[i], i))
            for size in range(1, min(max_regions, end) + 1):
                previous = dp.get((size - 1, begin))
                if previous is None:
                    continue
                cost = previous[0] + costs[trial]
                key = (size, end)
                if key not in dp or cost < dp[key][0]:
                    dp[key] = (cost, [*previous[1], (begin, end, trial)])
    choices = [(cost, size, regions) for (size, end), (cost, regions) in dp.items() if end == count]
    if not choices:
        raise ValueError("no portfolio satisfies the measured slowdown and region budgets")
    cost, _, chosen = min(choices, key=lambda choice: (choice[0], choice[1]))
    regions = []
    for begin, end, trial in chosen:
        low = shapes[0] if begin == 0 else (shapes[begin - 1] + shapes[begin]) // 2 + 1
        high = shapes[-1] if end == count else (shapes[end - 1] + shapes[end]) // 2
        regions.append(dict(bounds={symbol: dict(min=low, max=high)}, trial=trial,
                            measured_shapes=shapes[begin:end]))
    return dict(regions=regions, weighted_relative_latency=cost / sum(weights),
                max_slowdown=max_slowdown, max_regions=max_regions,
                reference_seconds=reference, performance_coverage="measured shapes only",
                selected_candidates=sorted({row["trial"] for row in regions}))


def measure_cuda(device, callback, *, warmup=3, samples=7, repeats=10, reset=None):
    """Return CUDA-event and completed-call samples for an allocation-free callback."""
    if device.info["provider"] != "cuda":
        raise ValueError("CUDA event measurement requires the CUDA provider")
    if (
        any(type(v) is not int or v < 1 for v in (samples, repeats))
        or type(warmup) is not int
        or warmup < 0
    ):
        raise ValueError("invalid measurement counts")
    begin, end = c.c_void_p(), c.c_void_p()
    device.driver.call("cuEventCreate", c.byref(begin), 0)
    try:
        device.driver.call("cuEventCreate", c.byref(end), 0)
        for _ in range(warmup):
            if reset is not None:
                reset()
            callback()
        device.synchronize()
        gpu, completed = [], []
        for _ in range(samples):
            if reset is not None:
                gpu_total, completed_total = 0.0, 0.0
                for _ in range(repeats):
                    reset()
                    device.synchronize()
                    start = time.perf_counter()
                    device.driver.call("cuEventRecord", begin, device.stream)
                    callback()
                    device.driver.call("cuEventRecord", end, device.stream)
                    device.driver.call("cuEventSynchronize", end)
                    completed_total += time.perf_counter() - start
                    value = c.c_float()
                    device.driver.call("cuEventElapsedTime", c.byref(value), begin, end)
                    gpu_total += value.value / 1000
                gpu.append(gpu_total / repeats)
                completed.append(completed_total / repeats)
                continue
            start = time.perf_counter()
            device.driver.call("cuEventRecord", begin, device.stream)
            for _ in range(repeats):
                callback()
            device.driver.call("cuEventRecord", end, device.stream)
            device.driver.call("cuEventSynchronize", end)
            completed.append((time.perf_counter() - start) / repeats)
            value = c.c_float()
            device.driver.call("cuEventElapsedTime", c.byref(value), begin, end)
            gpu.append(value.value / 1000 / repeats)
        return {
            "gpu_seconds": gpu,
            "completed_seconds": completed,
            "median_gpu_seconds": statistics.median(gpu),
            "median_completed_seconds": statistics.median(completed),
            "warmup": warmup,
            "samples": samples,
            "repeats": repeats,
        }
    finally:
        device.synchronize()
        if end.value:
            device.driver.call("cuEventDestroy_v2", end)
        device.driver.call("cuEventDestroy_v2", begin)


def measure_cuda_sequence(device, callbacks, *, warmup=1, samples=3, reset=None):
    """Attribute an ordered uncaptured sequence without replaying isolated state.

    Each sample executes the complete dependency chain once. Reset runs before
    that chain, outside timing. Event intervals include gaps from uncaptured
    host submission; these are diagnostics, not additive captured-call timings.
    """
    callbacks = tuple(callbacks)
    if (device.info["provider"] != "cuda" or not callbacks
            or type(warmup) is not int or warmup < 0
            or type(samples) is not int or samples < 1):
        raise ValueError("sequence profiling needs CUDA callbacks and valid counts")
    events = []
    try:
        for _ in range(len(callbacks) + 1):
            event = c.c_void_p()
            device.driver.call("cuEventCreate", c.byref(event), 0)
            events.append(event)
        for _ in range(warmup):
            if reset is not None:
                reset()
            for callback in callbacks:
                callback()
        device.synchronize()
        timings, completed = [], []
        for _ in range(samples):
            if reset is not None:
                reset()
            device.synchronize()
            start = time.perf_counter()
            for event, callback in zip(events, callbacks):
                device.driver.call("cuEventRecord", event, device.stream)
                callback()
            device.driver.call("cuEventRecord", events[-1], device.stream)
            device.driver.call("cuEventSynchronize", events[-1])
            completed.append(time.perf_counter() - start)
            row = []
            for first, last in zip(events, events[1:]):
                value = c.c_float()
                device.driver.call("cuEventElapsedTime", c.byref(value), first, last)
                row.append(value.value / 1000)
            timings.append(row)
        return dict(scope="uncaptured sequence including host submission gaps",
                    gpu_seconds=timings, completed_seconds=completed,
                    median_gpu_seconds=[statistics.median(row[i] for row in timings)
                                        for i in range(len(callbacks))],
                    warmup=warmup, samples=samples)
    finally:
        device.synchronize()
        for event in reversed(events):
            device.driver.call("cuEventDestroy_v2", event)


def measure(device, callback, *, warmup=3, samples=7, repeats=10, reset=None):
    """Measure completed work, excluding trial-state reset.

    CUDA reports event latency; WebGPU reports synchronized completed-call
    latency. Mutable callbacks supply reset before every invocation.
    """
    provider = device.info["provider"]
    if provider == "cuda":
        result = measure_cuda(
            device, callback, warmup=warmup, samples=samples, repeats=repeats,
            reset=reset,
        )
        return {**result, "metric": "gpu_seconds",
                "median_seconds": result["median_gpu_seconds"]}
    if provider != "webgpu":
        raise ValueError("kernel tuning requires CUDA or WebGPU")
    if (any(type(v) is not int or v < 1 for v in (samples, repeats))
            or type(warmup) is not int or warmup < 0):
        raise ValueError("invalid measurement counts")
    for _ in range(warmup):
        if reset is not None:
            reset()
        callback()
    device.synchronize()
    completed = []
    for _ in range(samples):
        if reset is None:
            start = time.perf_counter()
            for _ in range(repeats):
                callback()
            device.synchronize()
            elapsed = time.perf_counter() - start
        else:
            elapsed = 0.0
            for _ in range(repeats):
                reset()
                device.synchronize()
                start = time.perf_counter()
                callback()
                device.synchronize()
                elapsed += time.perf_counter() - start
        completed.append(elapsed / repeats)
    median = statistics.median(completed)
    return {"metric": "completed_seconds", "completed_seconds": completed,
            "median_completed_seconds": median, "median_seconds": median,
            "warmup": warmup, "samples": samples, "repeats": repeats}


def tune(candidates, inputs, output, reference, *, rtol=0.02, atol=0.002, checks=(), reset=None):
    """Select a schedule by measured GPU latency, rejecting incorrect candidates.

    candidates maps configuration labels to loaded Executables. The caller owns
    compilation, the search space and an independently validated reference.
    Inputs/output are already allocated; transfers stay outside the timing.
    """
    if not candidates:
        raise ValueError("tuning needs candidates")
    reference = np.asarray(reference)
    if not np.isfinite(reference).all():
        raise ValueError("tuning reference must be finite")
    results = {}
    for name, kernel in candidates.items():
        try:
            if reset is not None:
                reset()
            kernel.launch(*inputs, output)
            actual = output.to_numpy()
            np.testing.assert_allclose(actual, reference, rtol=rtol, atol=atol)
            for buffer, expected in checks:
                np.testing.assert_allclose(
                    buffer.to_numpy(), expected, rtol=rtol, atol=atol
                )
            results[name] = {
                "status": "passed",
                "timing": measure(
                    kernel.device, lambda kernel=kernel: kernel.launch(*inputs, output), reset=reset,
                ),
                "maximum_absolute_error": float(
                    np.max(
                        np.abs(actual.astype(np.float32) - reference.astype(np.float32))
                    )
                ),
            }
        except (AssertionError, ValueError) as error:
            results[name] = {"status": "rejected", "reason": str(error)}
    valid = [name for name, result in results.items() if result["status"] == "passed"]
    if not valid:
        raise ValueError("all tuning candidates failed correctness")
    selected = min(
        valid, key=lambda name: results[name]["timing"]["median_seconds"]
    )
    return {"selected": selected, "candidates": results, "rtol": rtol, "atol": atol}
