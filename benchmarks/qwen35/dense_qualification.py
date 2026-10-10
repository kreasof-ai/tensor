"""Physical request-state and batch-independence checks for dense Qwen."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np


def snapshot(engine, request):
    """Hash the full recurrent state and only the initialized KV prefix."""
    if hasattr(engine, "deferred"):
        engine.deferred.materialize([request])
    engine.device.synchronize()
    position = int(engine.positions[request])
    records = {}
    for layer, kind in enumerate(engine.config.layers):
        for index, buffer in enumerate(engine.states[layer]):
            parts = []
            if kind == "linear_attention":
                size = buffer.nbytes // engine.capacity
                ranges = [(buffer.pointer + request * size, size)]
            else:
                # KV layout is pool, heads, context, dimension. Padding beyond
                # position is not part of a request's state contract.
                head_bytes = engine.context * engine.config.head_dim * 2
                ranges = [
                    (
                        buffer.pointer
                        + (request * engine.config.kv_heads + head) * head_bytes,
                        position * engine.config.head_dim * 2,
                    )
                    for head in range(engine.config.kv_heads)
                ]
            for pointer, size in ranges:
                host = ct.create_string_buffer(size)
                engine.device.driver.call("cuMemcpyDtoH_v2", host, pointer, size)
                parts.append(host.raw)
            records[f"{layer}:{index}"] = hashlib.sha256(b"".join(parts)).hexdigest()
    return dict(position=position, states=records)


def qualify(
    checkpoint,
    bundle,
    references,
    out,
    *,
    capacities=(1, 2, 4, 8, 16, 32, 64, 128, 256),
):
    from tensor.providers.cuda import Device
    from tensor_llm.qwen35.dense.engine import DenseEngine

    references = json.loads(Path(references).read_text())
    controls = []
    results = []
    with Device() as device:
        engine = DenseEngine(checkpoint, bundle, device)
        try:
            for record in references:
                request = engine.admit()
                tokens = engine.prefill(
                    [request], [record["request"]["prompt_token_ids"]]
                )
                states = [snapshot(engine, request)]
                teacher = [
                    p[1]
                    for p in record["response"]["meta_info"]["output_token_logprobs"]
                ]
                for token in teacher[:4]:
                    tokens.extend(engine.decode([request], [token]))
                    states.append(snapshot(engine, request))
                controls.append(dict(tokens=tokens, states=states))
                engine.release(request)
            for capacity in capacities:
                requests = [engine.admit() for _ in range(capacity)]
                prompts = [
                    references[i % len(references)]["request"]["prompt_token_ids"]
                    for i in range(capacity)
                ]
                # Reorder logical execution rows. Ownership remains attached
                # to the same persistent cache slot across every permutation.
                order = list(reversed(range(capacity)))
                predicted = engine.prefill(
                    [requests[i] for i in order], [prompts[i] for i in order]
                )
                outputs = {i: [token] for i, token in zip(order, predicted)}
                valid = True
                checks = 0
                for step in range(5):
                    for i in range(min(capacity, len(references))):
                        same = (
                            snapshot(engine, requests[i]) == controls[i]["states"][step]
                        )
                        valid &= same
                        checks += 1
                        if not same:
                            raise AssertionError(
                                f"C{capacity} request {i} step {step}: state changed"
                            )
                    if step < 4:
                        order = order[1:] + order[:1]
                        teacher = [
                            references[i % len(references)]["response"]["meta_info"][
                                "output_token_logprobs"
                            ][step][1]
                            for i in order
                        ]
                        predicted = engine.decode([requests[i] for i in order], teacher)
                        for i, token in zip(order, predicted):
                            outputs[i].append(token)
                for i in range(min(capacity, len(references))):
                    if outputs[i] != controls[i]["tokens"]:
                        raise AssertionError(
                            f"C{capacity} request {i}: predictions changed"
                        )
                for request in requests:
                    engine.release(request)
                results.append(
                    dict(
                        capacity=capacity,
                        bitwise_states=valid,
                        predictions=True,
                        state_checks=checks,
                    )
                )
                print(results[-1], flush=True)
                Path(out).write_text(
                    json.dumps(dict(status="running", results=results), indent=2) + "\n"
                )
        finally:
            engine.close()
    Path(out).write_text(
        json.dumps(dict(status="passed", results=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--capacities", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 64, 128, 256)
    )
    a = p.parse_args()
    qualify(a.checkpoint, a.bundle, a.references, a.out, capacities=a.capacities)
