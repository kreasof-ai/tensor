"""Check fused greedy projection against the full FP32-logit projection."""

import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np

from tensor.providers.cuda import Device
from tensor_llm.common.artifacts import identity
from tensor_llm.qwen35.dense.checkpoint import DenseCheckpoint
from tensor_llm.qwen35.dense.projections import schedule


def qualify(checkpoint, bundle, out, counts):
    checkpoint = DenseCheckpoint(checkpoint)
    c = checkpoint.config
    bundle = Path(bundle)
    rows = []
    with Device() as device:
        device.driver.lib.cuMemsetD8_v2.argtypes = [
            ct.c_uint64,
            ct.c_ubyte,
            ct.c_size_t,
        ]
        device.driver.lib.cuMemsetD8_v2.restype = ct.c_int
        raw = np.ascontiguousarray(
            checkpoint.read("model.language_model.embed_tokens.weight")
        )
        weight = device.empty(raw.shape, "bfloat16")
        device.driver.call(
            "cuMemcpyHtoD_v2", weight.pointer, ct.c_void_p(raw.ctypes.data), raw.nbytes
        )
        device.driver.call("cuStreamSynchronize", None)
        try:
            for count in counts:
                parameters = schedule(count, c.width, c.vocab, c.width)
                full = device.load(
                    bundle / (identity("split_linear", parameters) + ".tbin")
                )
                fused = device.load(
                    bundle / (identity("head_linear", parameters) + ".tbin")
                )
                tiles = (c.vocab + 63) // 64
                merge = device.load(
                    bundle
                    / (
                        identity(
                            "head_argmax", dict(r=count, tiles=tiles, vocab=c.vocab)
                        )
                        + ".tbin"
                    )
                )
                x = device.from_numpy(
                    np.random.default_rng(7842 + count)
                    .standard_normal((count, c.width))
                    .astype("float32"),
                    dtype="bfloat16",
                )
                logits = device.empty((count, 1, c.vocab), "float32")
                maximum = device.empty((count, tiles), "float32")
                indices = device.empty((count, tiles), "int32")
                active = device.from_numpy(
                    np.array([int(i % 3 != 1) for i in range(count)], dtype="int32")
                )
                predicted = device.empty((count,), "int32")
                try:
                    full.launch(x, weight, logits)
                    fused.launch(x, weight, maximum, indices)
                    merge.launch(maximum, indices, active, predicted)
                    reference = logits.to_numpy().reshape(count, c.vocab)
                    expected = np.argmax(reference, axis=1)
                    expected[np.arange(count) % 3 == 1] = -1
                    assert np.array_equal(
                        predicted.to_numpy(), expected
                    ), f"fused argmax changed at rows={count}"
                    assert np.array_equal(
                        np.max(maximum.to_numpy(), axis=1), np.max(reference, axis=1)
                    ), f"fused maximum changed at rows={count}"
                    if count == 1:
                        device.driver.call("cuMemsetD8_v2", x.pointer, 0, x.nbytes)
                        device.driver.call("cuStreamSynchronize", None)
                        fused.launch(x, weight, maximum, indices)
                        merge.launch(maximum, indices, active, predicted)
                        assert predicted.to_numpy().tolist() == [
                            0
                        ], "ties must choose lowest vocabulary ID"
                    rows.append(dict(rows=count, exact_scores=True, exact_argmax=True))
                    Path(out).write_text(json.dumps(rows, indent=2) + "\n")
                    print(rows[-1], flush=True)
                finally:
                    for resource in (
                        predicted,
                        active,
                        indices,
                        maximum,
                        logits,
                        x,
                        merge,
                        fused,
                        full,
                    ):
                        resource.release()
        finally:
            weight.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "bundle", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    )
    a = p.parse_args()
    qualify(a.checkpoint, a.bundle, a.out, a.counts)
