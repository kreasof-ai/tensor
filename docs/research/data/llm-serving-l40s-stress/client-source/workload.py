"""Prepare and validate immutable token-ID request traces."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

SCHEMA = "tensor.llm-serving-workload.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def positive_integer(value):
    return type(value) is int and value > 0


def validate(workload):
    if workload.get("schema") != SCHEMA:
        raise ValueError("unsupported workload schema")
    if not workload.get("tokenizer") or not workload.get("requests"):
        raise ValueError("workload needs tokenizer provenance and requests")
    if any(not isinstance(workload["tokenizer"].get(key), str) or not workload["tokenizer"][key]
           for key in ("name", "revision")):
        raise ValueError("workload needs tokenizer name and revision")
    if type(workload.get("seed")) is not int or workload["seed"] < 0 or not workload.get("kind"):
        raise ValueError("workload needs a nonnegative integer seed and kind")
    if workload.get("sampling") != {"temperature": 0.0, "ignore_eos": True}:
        raise ValueError("this harness supports greedy forced-length sampling only")
    ids = set()
    previous = -1.0
    for request in workload["requests"] + workload.get("warmup", []):
        tokens = request.get("prompt_token_ids")
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("prompts must be nonempty nonnegative integer token IDs")
        if not positive_integer(request.get("output_tokens")):
            raise ValueError("output_tokens must be a positive integer")
        if not isinstance(request.get("id"), str) or request["id"] in ids:
            raise ValueError("request IDs must be unique strings")
        ids.add(request["id"])
    mode = workload.get("arrival", {}).get("mode")
    if mode not in ("closed-loop", "poisson"):
        raise ValueError("arrival mode must be closed-loop or poisson")
    for request in workload["requests"]:
        arrival = request.get("arrival_seconds")
        if mode == "poisson":
            if (type(arrival) not in (int, float) or not math.isfinite(arrival)
                    or arrival < previous or arrival < 0):
                raise ValueError("arrival timestamps must be finite, nonnegative and ordered")
            previous = arrival
        elif arrival is not None:
            raise ValueError("closed-loop requests cannot specify fixed arrival times")
    if mode == "poisson":
        rate = workload["arrival"].get("requests_per_second")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("poisson workloads need a positive finite request rate")
    expected = workload.get("sha256")
    if expected != digest({key: value for key, value in workload.items() if key != "sha256"}):
        raise ValueError("workload checksum mismatch")
    return workload


def prepare(pool, tokenizer, *, lengths=(512,), output_tokens=128, requests=128,
            warmup=1, seed=20261009, request_rate=None):
    if (not lengths or any(not positive_integer(length) for length in lengths)
            or not positive_integer(output_tokens) or not positive_integer(requests)
            or type(warmup) is not int or warmup < 0):
        raise ValueError("lengths/counts must be positive; warmup may be zero")
    if (not pool or any(type(token) is not int or token < 0 for token in pool)
            or len(set(pool)) < 2):
        raise ValueError("token pool needs at least two distinct valid token IDs")
    if request_rate is not None and (not math.isfinite(request_rate) or request_rate <= 0):
        raise ValueError("request rate must be positive and finite")
    # Independent RNGs keep prompt IDs identical when only arrival policy changes.
    prompts = random.Random(seed)
    arrivals = random.Random(seed + 1)
    elapsed = 0.0
    measured, discarded = [], []
    for index in range(requests + warmup):
        item = {"id": f"request-{index:06d}", "output_tokens": output_tokens,
                "prompt_token_ids": prompts.choices(pool, k=lengths[index % len(lengths)])}
        if index < requests:
            if request_rate is not None:
                if index:
                    elapsed += arrivals.expovariate(request_rate)
                item["arrival_seconds"] = elapsed
            measured.append(item)
        else:
            discarded.append(item)
    value = {"schema": SCHEMA, "tokenizer": tokenizer, "seed": seed,
             "kind": "synthetic-random-token-stress", "token_pool_sha256": digest(pool),
             "arrival": {"mode": "poisson" if request_rate is not None else "closed-loop",
                         "requests_per_second": request_rate},
             "sampling": {"temperature": 0.0, "ignore_eos": True},
             "requests": measured, "warmup": discarded}
    value["sha256"] = digest(value)
    return validate(value)


def load(path):
    return validate(json.loads(Path(path).read_text()))


def token_pool(model, revision):
    from transformers import AutoTokenizer

    if not revision:
        raise ValueError("pin --revision when preparing a Hugging Face tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    excluded = set(tokenizer.all_special_ids)
    vocabulary = tokenizer.get_vocab()
    pool = sorted(set(vocabulary.values()) - excluded)
    probe = "A shared tokenizer makes inference measurements comparable.\n0123456789"
    return pool, {"name": model, "revision": revision,
                  "vocabulary_sha256": digest(vocabulary), "probe_text": probe,
                  "probe_token_ids": tokenizer.encode(probe, add_special_tokens=False)}
