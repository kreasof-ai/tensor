"""Local, matched dense-Qwen serving exercise; never launches a cloud GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.llm_serving.runner import atomic_json, run
from benchmarks.llm_serving.workload import digest, validate

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TASKS = (
    "Explain how a hash table resolves collisions, with a Python example.",
    "Write a careful comparison of renewable energy sources for a small town.",
    "Explain why matrix multiplication benefits from batching on a GPU.",
    "Describe a practical plan for restoring a neglected vegetable garden.",
    "Solve a probability problem involving two dice and explain every step.",
    "Write a short story about a researcher repairing an old radio telescope.",
    "Explain TCP congestion control to a software engineer learning networking.",
    "Review a database design for a library and propose its tables and indexes.",
    "Compare sorting algorithms, including stability and memory requirements.",
    "Explain photosynthesis and how drought affects the process.",
    "Write a travel diary describing a fictional journey through mountain villages.",
    "Design a unit test strategy for a concurrent work queue.",
)


def prepare(checkpoint, directory, *, seed=20261010, drain=False):
    from transformers import AutoTokenizer

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    vocabulary = tokenizer.get_vocab()
    probe = "A shared tokenizer makes inference measurements comparable.\n0123456789"
    provenance = dict(
        name=MODEL,
        revision=REVISION,
        vocabulary_sha256=digest(vocabulary),
        probe_text=probe,
        probe_token_ids=tokenizer.encode(probe, add_special_tokens=False),
    )
    # The task permutation and examples change on held-out seeds. No generated
    # output is inspected while constructing or choosing the request set.
    import random

    rng = random.Random(seed)
    tasks = list(TASKS)
    rng.shuffle(tasks)
    for concurrency in CONCURRENCIES:
        requests = []
        for index in range(max(16, 4 * concurrency) + 2):
            task = tasks[index % len(tasks)]
            text = (
                f"Independent exercise {seed}-{index}. {task} "
                f"Use the example values {rng.randrange(10, 999)} and "
                f"{rng.randrange(10, 999)} when useful. Explain your reasoning "
                "clearly and include concrete examples. "
                "Assume the reader has basic background knowledge.\n"
            )
            rendered = tokenizer.apply_chat_template(
                [dict(role="user", content=text)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            item = dict(
                id=f"request-{index:06d}",
                prompt_token_ids=tokens,
                output_tokens=(64, 128, 256, 512)[index % 4] if drain else 256,
            )
            requests.append(item)
        value = dict(
            schema="tensor.llm-serving-workload.v1",
            tokenizer=provenance,
            seed=seed,
            kind="varied-text-drain" if drain else "varied-text-scaling",
            arrival=dict(mode="closed-loop", requests_per_second=None),
            sampling=dict(temperature=0.0, ignore_eos=True),
            requests=requests[:-2],
            warmup=requests[-2:],
        )
        value["sha256"] = digest(value)
        validate(value)
        atomic_json(directory / f"c{concurrency}.json", value)


def server(engine, port, *, kv_dtype="bfloat16", speculative=False):
    import importlib.metadata

    return dict(
        name=engine,
        engine=engine,
        base_url=f"http://127.0.0.1:{port}",
        model="qwen08",
        engine_version=(
            importlib.metadata.version(engine) if engine != "tensor" else "0.1.0"
        ),
        model_revision=REVISION,
        weight_format="BF16",
        kv_dtype=kv_dtype,
        state_dtype="float32",
        hardware="NVIDIA L40S x1",
        cpu_offload="none",
        prefix_cache=False,
        speculative=speculative,
        tokenizer_name=MODEL,
        tokenizer_revision=REVISION,
        gpu_indices=["0"],
    )


async def replay(
    engine,
    port,
    workloads,
    out,
    *,
    repeats=2,
    kv_dtype="bfloat16",
    speculative=False,
    concurrencies=CONCURRENCIES,
):
    manifest = dict(
        schema="tensor.llm-serving-servers.v1",
        comparison_group="qwen35-08b-local-scaling",
        servers=[server(engine, port, kv_dtype=kv_dtype, speculative=speculative)],
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(out) / "servers.json", manifest)
    for concurrency in concurrencies:
        workload = json.loads((Path(workloads) / f"c{concurrency}.json").read_text())
        report = await run(
            manifest,
            workload,
            Path(out) / f"c{concurrency}",
            concurrencies=[concurrency],
            repeats=repeats,
            external=True,
            timeout=300,
            startup_timeout=600,
            interval=2,
        )
        print(
            json.dumps(
                dict(
                    concurrency=concurrency,
                    status=report["status"],
                    results=report.get("results"),
                ),
                default=str,
            ),
            flush=True,
        )
        if report["status"] != "completed":
            raise RuntimeError(
                f"failed C{concurrency}; retain evidence before retrying"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--checkpoint", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--seed", type=int, default=20261010)
    prep.add_argument("--drain", action="store_true")
    bench = commands.add_parser("run")
    bench.add_argument("--engine", choices=("vllm", "sglang", "tensor"), required=True)
    bench.add_argument("--port", type=int, required=True)
    bench.add_argument("--workloads", required=True)
    bench.add_argument("--out", required=True)
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--kv-dtype", default="bfloat16")
    bench.add_argument("--speculative", action="store_true")
    bench.add_argument("--concurrency", type=int, nargs="+", default=CONCURRENCIES)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.checkpoint, args.out, seed=args.seed, drain=args.drain)
    else:
        asyncio.run(
            replay(
                args.engine,
                args.port,
                args.workloads,
                args.out,
                repeats=args.repeats,
                kv_dtype=args.kv_dtype,
                speculative=args.speculative,
                concurrencies=args.concurrency,
            )
        )


if __name__ == "__main__":
    main()
