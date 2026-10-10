"""Cold-cache kernel search across batch shapes, separate from HTTP timing."""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np

from tensor.compiler.entry import export_source
from tensor_llm.common.artifacts import identity
from tensor_llm.qwen35.dense.checkpoint import DenseCheckpoint
from benchmarks.qwen35.build import needs_build, build_artifact


def flush_kernel(p):
    import tilelang.language as T

    size = p["size"]

    @T.prim_func
    def kernel(x: T.Tensor((size,), "int32")):
        with T.Kernel(T.ceildiv(size, 256), threads=256) as block:
            for j in T.Parallel(256):
                i = block * 256 + j
                if i < size:
                    x[i] += 1

    return kernel


def cases(*, extended=False):
    geometries = (
        (16, 64, 128),
        (32, 64, 128),
        (64, 64, 128),
        (16, 128, 64),
        (32, 128, 64),
        (64, 128, 64),
        (32, 64, 64),
        (64, 64, 64),
    )
    row_counts = (1, 8, 32, 128, 256, 512, 1024) if extended else (1, 8, 32, 128, 256)
    if extended:
        geometries = (
            (16, 64, 128),
            (16, 32, 128),
            (32, 32, 128),
            (64, 64, 128),
            (128, 64, 128),
            (64, 128, 64),
            (128, 128, 64),
            (256, 64, 64),
        )
    for rows in row_counts:
        for family, k, o in (
            ("head", 1024, 248320),
            ("input", 1024, 8224),
            ("ffn", 1024, 7168),
            ("down", 3584, 1024),
            ("out", 2048, 1024),
        ):
            for m, n, bk in geometries:
                parts = 8 if o == 1024 else 1
                # Preserve the control's fixed Split-K boundaries.
                if parts > 1 and bk != 128:
                    continue
                p = dict(r=rows, k=k, o=o, m=m, n=n, bk=bk, parts=parts)
                if n == 32:
                    p["threads"] = 64
                yield dict(
                    family=family,
                    kind="split_linear",
                    parameters=p,
                    module="tensor_llm.qwen35.dense.projections",
                    factory="make_kernel",
                )
        for tile in (() if extended else (8, 16, 32, 64)):
            yield dict(
                family="gdn",
                kind="gdn_scan",
                parameters=dict(
                    slots=rows, chunk=1, pool=256, heads=16, d=128, tile=tile
                ),
                module="tensor_llm.qwen35.dense.kernels",
                factory="make_kernel",
            )


def _compile(job):
    source, artifact = job
    build_artifact(source, artifact, target="sm_89")


def prepare(directory, *, extended=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(cases(extended=extended))
    rows.append(
        dict(
            family="flush",
            kind="flush",
            parameters=dict(size=24 * 1024**2),
            module=__name__,
            factory="flush_kernel",
        )
    )
    pending = []
    for row in rows:
        key = identity(row["kind"], row["parameters"])
        row["path"] = key + ".tbin"
        args = (
            (row["parameters"],)
            if row["kind"] == "flush"
            else (row["kind"], row["parameters"])
        )
        source = export_source(
            row["module"],
            row["factory"],
            *args,
            dependencies=("tensor.compiler.entry", "tensor.compiler.cuda_lowering"),
        )
        entry = directory / (key + ".py")
        artifact = directory / row["path"]
        if needs_build(entry, artifact, source, "sm_89"):
            artifact.unlink(missing_ok=True)
            entry.write_text(source)
            pending.append((entry, artifact))
    with ProcessPoolExecutor(max_workers=4) as workers:
        for i, _ in enumerate(workers.map(_compile, pending), 1):
            print("compiled", i, "/", len(pending), flush=True)
    for row in rows:
        row["sha256"] = hashlib.sha256(
            (directory / row["path"]).read_bytes()
        ).hexdigest()
    (directory / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")


def measure(checkpoint, directory):
    from tensor.providers.cuda import Device
    from tensor.providers.cuda_graph import CudaGraph
    from tensor.runtime.abi import BoundCall

    directory = Path(directory)
    rows = json.loads((directory / "cases.json").read_text())
    results = []
    with Device() as device:
        weights = DenseCheckpoint(checkpoint).upload(device)
        flush_row = rows.pop()
        flush = device.load(directory / flush_row["path"])
        scratch = device.from_numpy(
            np.zeros((flush_row["parameters"]["size"],), dtype="int32")
        )
        fv, fs, fl = flush._bind((scratch,), {}, include_outputs=True)
        flush_call = BoundCall(device, flush.manifest, fv, fs, fl, validated=True)
        recorder = device.driver.lib.cuEventRecordWithFlags
        recorder.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
        recorder.restype = ct.c_int
        device.driver.lib.cuMemsetD8_v2.argtypes = [
            ct.c_uint64,
            ct.c_ubyte,
            ct.c_size_t,
        ]
        device.driver.lib.cuMemsetD8_v2.restype = ct.c_int
        rng = np.random.default_rng(20261010)
        references = {}
        names = dict(
            head="model.language_model.embed_tokens.weight",
            input="model.language_model.layers.0.linear_attn.input.weight",
            ffn="model.language_model.layers.0.mlp.input.weight",
            down="model.language_model.layers.0.mlp.down_proj.weight",
            out="model.language_model.layers.0.linear_attn.out_proj.weight",
        )
        for row in rows:
            p = row["parameters"]
            r = p.get("r", p.get("slots"))
            family = row["family"]
            owned = []

            def upload(x, dtype=None):
                b = device.from_numpy(x, dtype)
                owned.append(b)
                return b

            def empty(shape, dtype):
                b = device.empty(shape, dtype)
                owned.append(b)
                return b

            if family != "gdn":
                # Every geometry sees identical activations for this shape.
                local = np.random.default_rng(20261010 + r + p["k"])
                x = upload(
                    local.standard_normal((r, p["k"])).astype("float32") * 0.1,
                    "bfloat16",
                )
                out = empty((r, p["parts"], p["o"]), "float32")
                args = (x, weights[names[family]], out)
            else:
                local = np.random.default_rng(20261010 + r)
                q = upload(local.standard_normal((r, 16, 128)).astype("float32") * 0.01)
                k = upload(local.standard_normal((r, 16, 128)).astype("float32") * 0.08)
                v = upload(local.standard_normal((r, 16, 128)).astype("float32"))
                g = upload(np.full((r, 16), -0.1, dtype="float32"))
                beta = upload(np.full((r, 16), 0.5, dtype="float32"))
                mapping = upload(np.arange(r, dtype="int32"))
                lengths = upload(np.ones(r, dtype="int32"))
                state = empty((256, 16, 128, 128), "float32")
                out = empty((r, 16, 128), "float32")
                device.driver.call("cuMemsetD8_v2", state.pointer, 0, state.nbytes)
                args = (q, k, v, g, beta, mapping, lengths, state, out)
            path = directory / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("beam artifact changed")
            kernel = device.load(path)
            values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
            call = BoundCall(
                device, kernel.manifest, values, symbols, launch, validated=True
            )
            device._launch(kernel, call)
            device.synchronize()
            result = out.to_numpy()
            if family != "gdn":
                result = result.sum(axis=1)
            ref = references.setdefault((family, r), result.copy())
            error = float(
                np.linalg.norm(result - ref) / max(np.linalg.norm(ref), 1e-12)
            )
            if not np.isfinite(result).all() or error > 1e-5:
                raise AssertionError(f"beam arithmetic changed: {family} C{r}: {error}")
            start, end = ct.c_void_p(), ct.c_void_p()
            device.driver.call("cuEventCreate", ct.byref(start), 0)
            device.driver.call("cuEventCreate", ct.byref(end), 0)

            def submit():
                device._launch(flush, flush_call)
                device.driver.call("cuEventRecordWithFlags", start, device.stream, 1)
                device._launch(kernel, call)
                device.driver.call("cuEventRecordWithFlags", end, device.stream, 1)

            with CudaGraph(device, submit) as graph:
                samples = []
                for _ in range(7):
                    graph.launch()
                    device.synchronize()
                    ms = ct.c_float()
                    device.driver.call("cuEventElapsedTime", ct.byref(ms), start, end)
                    samples.append(ms.value)
            measured = dict(
                row,
                relative_rms=error,
                median_milliseconds=float(np.median(samples)),
                samples_milliseconds=samples,
                timing="captured external events after a 96 MiB cache flush; not model throughput",
            )
            results.append(measured)
            print(family, r, p, measured["median_milliseconds"], flush=True)
            device.driver.call("cuEventDestroy_v2", start)
            device.driver.call("cuEventDestroy_v2", end)
            kernel.release()
            for b in owned:
                b.release()
            (directory / "measured.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
        flush.release()
        scratch.release()
        for w in weights.values():
            w.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "measure"))
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--extended", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.out, extended=a.extended)
    else:
        measure(a.checkpoint, a.out)
