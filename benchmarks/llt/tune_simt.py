"""Compare the single-query vector schedule with declared split candidates."""

import torch
from tensor_torch.llt import Operators, KVCache
from .qualify import ROOT, setup, TOLERANCES
from .scaling import measure, finish


def main():
    setup()
    ops = Operators(ROOT / "build/llt-qualification/artifacts")
    rows = []
    with torch.inference_mode():
        for d in (32, 64, 96, 128):
            c = torch.randn(1, 1, 65537, d, device="cuda", dtype=torch.bfloat16)
            q = torch.randn(1, 8, 1, d, device="cuda", dtype=c.dtype)
            cache = KVCache(ops, 1, 1, 65537, d, dtype=c.dtype, shared=True)
            cache.append(c)
            expected = torch.nn.functional.scaled_dot_product_attention(
                q, c, c, enable_gqa=True, scale=0.125
            )
            for partitions in (32, 64, 128, 256):

                def call():
                    partial, stats = ops.call(
                        "decode_partial",
                        (1, 8, 1, 65537, d, d, "bfloat16", 0.125, partitions),
                        ["partial", "stats"],
                        q,
                        cache.keys,
                        cache.values,
                        cache.lengths,
                        module="benchmarks.llt.tensor_warp_decode",
                    )
                    return ops.call(
                        "decode_merge",
                        (1, 8, d, "bfloat16", partitions),
                        ["out"],
                        partial,
                        stats,
                        module="tensor_torch.templates.llt_decode",
                    )

                torch.testing.assert_close(
                    call(), expected, **TOLERANCES["attention_bf16"]
                )
                report = measure(call)
                rows.append({"rank": d, "partitions": partitions, **report})
                print(
                    "Vector candidate:",
                    d,
                    partitions,
                    report["graph_median_ms"],
                    flush=True,
                )
    finish(
        "decode-simt-tuning",
        {"status": "passed", "cases": rows, "coverage": ops.report},
    )


if __name__ == "__main__":
    main()
