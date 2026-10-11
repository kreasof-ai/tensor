"""Completed-call and host-enqueue measurements."""

from __future__ import annotations

import statistics
import time
from tensor.runtime.executable import Executable


def bench(
    executable: Executable,
    buffers: tuple,
    *,
    warmup: int = 10,
    iters: int = 100,
    **bindings,
) -> dict:
    """Measure host launch plus stream synchronization, with preallocated buffers."""
    if type(warmup) is not int or warmup < 0 or type(iters) is not int or iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    device = executable.device
    for _ in range(warmup):
        executable.launch(*buffers, **bindings)
        device.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        executable.launch(*buffers, **bindings)
        device.synchronize()
        samples.append(time.perf_counter() - start)
    enqueue = []
    for _ in range(iters):
        start = time.perf_counter()
        executable.launch(*buffers, **bindings)
        enqueue.append(time.perf_counter() - start)
    device.synchronize()
    return {
        "warmup": warmup,
        "iters": iters,
        "median_launch_and_sync_seconds": statistics.median(samples),
        "min_launch_and_sync_seconds": min(samples),
        "median_host_enqueue_seconds": statistics.median(enqueue),
    }
