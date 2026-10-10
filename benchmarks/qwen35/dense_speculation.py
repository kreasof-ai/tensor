"""Exact target equivalence and acceptance diagnostics for dense MTP."""

import argparse
import json
from pathlib import Path
import time

from tensor.providers.cuda import Device
from tensor_llm.qwen35.dense.engine import DenseEngine
from tensor_llm.qwen35.dense.scheduler import Request, Scheduler
from tensor_llm.qwen35.dense.speculative import SpeculativeEngine
from .dense_qualification import snapshot


def execute(engine, specifications):
    requests = [Request(prompt, limit) for prompt, limit in specifications]
    scheduler = Scheduler(engine, lambda events: None)
    for request in requests:
        scheduler.submit(request)
    states = {}
    release = engine.release

    def capture(slot):
        index = next(i for i, r in enumerate(requests) if r.slot == slot)
        states[index] = snapshot(getattr(engine, "target", engine), slot)
        release(slot)

    engine.release = capture
    start = time.perf_counter()
    try:
        while scheduler.turn():
            pass
        elapsed = time.perf_counter() - start
        return [r.output for r in requests], elapsed, scheduler.stats, states
    finally:
        engine.release = release


def qualify(
    checkpoint,
    bundle,
    references,
    out,
    *,
    counts,
    tokens=64,
    window=4,
    single_graph=False,
    deferred=False,
    fused_head=False,
):
    references = json.loads(Path(references).read_text())
    rows = []
    with Device() as device:
        target = DenseEngine(checkpoint, bundle, device, fused_head=fused_head)
        speculative = SpeculativeEngine(
            target, window=window, single_graph=single_graph, deferred=deferred
        )
        try:
            for count in counts:
                # Unequal lengths exercise accepted-prefix clipping and drain.
                specifications = [
                    (
                        references[i % len(references)]["request"]["prompt_token_ids"],
                        tokens - i % 7,
                    )
                    for i in range(count)
                ]
                control, ar_seconds, _, control_states = execute(target, specifications)
                actual, mtp_seconds, stats, actual_states = execute(
                    speculative, specifications
                )
                assert actual == control, f"MTP changed greedy output at C{count}"
                assert (
                    actual_states == control_states
                ), f"MTP changed terminal state at C{count}"
                row = dict(
                    concurrency=count,
                    tokens=sum(map(len, actual)),
                    ar_seconds=ar_seconds,
                    mtp_seconds=mtp_seconds,
                    greedy_identical=True,
                    bitwise_terminal_states=True,
                    scheduler=stats,
                    speculation=dict(speculative.stats),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                Path(out).write_text(json.dumps(rows, indent=2) + "\n")
        finally:
            speculative.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32, 256])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--single-graph", action="store_true")
    parser.add_argument("--deferred", action="store_true")
    parser.add_argument("--fused-head", action="store_true")
    args = parser.parse_args()
    qualify(
        args.checkpoint,
        args.bundle,
        args.references,
        args.out,
        counts=args.counts,
        tokens=args.tokens,
        window=args.window,
        single_graph=args.single_graph,
        deferred=args.deferred,
        fused_head=args.fused_head,
    )
