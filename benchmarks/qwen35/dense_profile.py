"""Local captured kernel timings; diagnostic, never a serving speedup claim."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics

from tensor.providers.cuda import Device
from tensor.providers.cuda_graph import CudaGraph
from tensor_llm.qwen35.dense.engine import DenseEngine


def measure(batch, calls):
    device = batch.device
    recorder = device.driver.lib.cuEventRecordWithFlags
    recorder.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
    recorder.restype = ct.c_int
    handles = []

    def event():
        handle = ct.c_void_p()
        device.driver.call("cuEventCreate", ct.byref(handle), 0)
        handles.append(handle)
        return handle

    rows = []
    for kernel, call in calls:
        key = next(
            key for key, candidate in batch.kernels.items() if candidate is kernel
        )
        rows.append((key, event(), event(), kernel, call))

    def submit():
        for _, start, end, kernel, call in rows:
            device.driver.call("cuEventRecordWithFlags", start, device.stream, 1)
            device._launch(kernel, call)
            device.driver.call("cuEventRecordWithFlags", end, device.stream, 1)

    samples = {key: [] for key, *_ in rows}
    graph = None
    try:
        graph = CudaGraph(device, submit, resources=batch.graph.resources)
        for _ in range(5):
            graph.launch()
            device.synchronize()
            totals = {}
            for key, start, end, *_ in rows:
                elapsed = ct.c_float()
                device.driver.call("cuEventElapsedTime", ct.byref(elapsed), start, end)
                totals[key] = totals.get(key, 0.0) + elapsed.value
            for key, elapsed in totals.items():
                samples[key].append(elapsed)
        return {key: statistics.median(values) for key, values in samples.items()}
    finally:
        if graph:
            graph.close()
        for handle in handles:
            device.driver.call("cuEventDestroy_v2", handle)


def profile(checkpoint, bundle, references, out, counts):
    references = json.loads(Path(references).read_text())
    rows = []
    with Device() as device:
        engine = DenseEngine(checkpoint, bundle, device)
        try:
            for count in counts:
                slots = [engine.admit() for _ in range(count)]
                predictions = engine.prefill(
                    slots,
                    [
                        references[i % len(references)]["request"]["prompt_token_ids"]
                        for i in range(count)
                    ],
                )
                ar = engine.batch(count)
                ar.run(slots, [[t] for t in predictions])
                verifier = engine.batch(count, 4, verify=True)
                verifier.run(slots, [[t, 10, 20, 30] for t in predictions])
                manifest = json.loads(
                    (
                        engine.bundle / f"inference-s{verifier.slots}-t4-verify.json"
                    ).read_text()
                )
                kinds = {
                    key: (row["kind"], row["parameters"])
                    for key, row in manifest["kernels"].items()
                }
                manifest = json.loads(
                    (engine.bundle / f"inference-s{ar.slots}-t1.json").read_text()
                )
                kinds.update(
                    {
                        key: (row["kind"], row["parameters"])
                        for key, row in manifest["kernels"].items()
                    }
                )
                phases = {}
                for name, batch, calls in [
                    ("ar", ar, ar.calls),
                    ("verify", verifier, verifier.calls),
                    ("commit", verifier, verifier.commit_calls),
                ]:
                    measured = measure(batch, calls)
                    groups = {}
                    for key, milliseconds in measured.items():
                        kind, parameters = kinds[key]
                        if kind == "split_linear":
                            kind += f"-k{parameters['k']}-o{parameters['o']}"
                        groups[kind] = groups.get(kind, 0.0) + milliseconds
                    phases[name] = dict(
                        total_ms=sum(measured.values()), groups=groups, kernels=measured
                    )
                verifier.discard()
                for slot in slots:
                    engine.release(slot)
                rows.append(dict(concurrency=count, phases=phases, diagnostic=True))
                Path(out).write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(rows[-1]), flush=True)
        finally:
            engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "bundle", "references", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 32, 256])
    args = parser.parse_args()
    profile(args.checkpoint, args.bundle, args.references, args.out, args.counts)
