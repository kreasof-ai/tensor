"""Exercise all three wire protocols locally; these are NOT inference measurements."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path

from .plot import plot
from .runner import CONFIG_SCHEMA, run
from .workload import prepare


@asynccontextmanager
async def mock_server(*, delay=.002, short=False, truncated=False, terminate=True, failure=None):
    from aiohttp import web

    state = {"active": 0, "peak_active": 0, "prompts": [], "payloads": [], "connections": set()}

    async def generate(request):
        data = await request.json()
        engine = {"/v1/completions": "vllm", "/generate": "sglang", "/completion": "llama.cpp"}[request.path]
        prompt = data["input_ids"] if engine == "sglang" else data["prompt"]
        count = data["sampling_params"]["max_new_tokens"] if engine == "sglang" else data.get("max_tokens", data.get("n_predict"))
        if short:
            count -= 1
        state["prompts"].append(prompt)
        state["payloads"].append(data)
        state["connections"].add(request.transport)
        if failure:
            return web.json_response({"error": failure}, status=500)
        state["active"] += 1
        state["peak_active"] = max(state["peak_active"], state["active"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def emit(value):
            # Deliberately fragment frames across network writes.
            encoded = ("data: " + json.dumps(value) + "\n\n").encode()
            midpoint = len(encoded) // 2
            await response.write(encoded[:midpoint])
            await response.write(encoded[midpoint:])

        try:
            await asyncio.sleep(delay)
            for index in range(count):
                await asyncio.sleep(delay)
                if engine == "vllm":
                    event = {"choices": [{"index": 0, "text": "x", "token_ids": [index + 10],
                                           "finish_reason": None}]}
                elif engine == "sglang":
                    event = {"text": "x" * (index + 1),
                             "meta_info": {"prompt_tokens": len(prompt),
                                           "completion_tokens": index + 1, "finish_reason": None}}
                else:
                    event = {"content": "x", "tokens": [index + 10], "stop": False}
                await emit(event)
            if terminate:
                if engine == "vllm":
                    await emit({"choices": [{"index": 0, "text": "", "token_ids": [],
                                              "finish_reason": "length"}]})
                    await emit({"choices": [], "usage": {"prompt_tokens": len(prompt),
                                                          "completion_tokens": count}})
                elif engine == "sglang":
                    await emit({"text": "x" * count, "meta_info": {"prompt_tokens": len(prompt),
                                "completion_tokens": count, "finish_reason": {"type": "length"},
                                "num_retractions": 0}})
                else:
                    await emit({"content": "", "tokens": [], "stop": True,
                                "tokens_evaluated": len(prompt), "tokens_predicted": count,
                                "truncated": truncated})
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        finally:
            state["active"] -= 1

    async def tokenize(request):
        return web.json_response({"tokens": [1, 2]})

    async def metrics(request):
        return web.Response(text=f"fixture_running_requests {state['active']}\n")

    async def info(request):
        return web.json_response({"fixture": True, "version": "synthetic", "api_key": "fixture-secret"})

    app = web.Application()
    for path in ("/v1/completions", "/generate", "/completion"):
        app.router.add_post(path, generate)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/{path:.*}", info)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await runner.cleanup()


def fixture_config(engine, url):
    return {"schema": CONFIG_SCHEMA, "comparison_group": "synthetic-protocol-test",
            "servers": [{"name": engine, "engine": engine, "base_url": url,
                         "model": "fixture", "model_revision": "fixture-v1",
                         "engine_version": "synthetic", "weight_format": "fixture",
                         "kv_dtype": "fixture", "state_dtype": "fixture",
                         "hardware": "synthetic HTTP fixture", "cpu_offload": "none",
                         "prefix_cache": False, "speculative": False, "synthetic": True,
                         "tokenizer_name": "fixture", "tokenizer_revision": "fixture-v1"}]}


async def smoke(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    workload = prepare([3, 4, 5, 6], {"name": "fixture", "revision": "fixture-v1",
                                    "probe_text": "fixture probe", "probe_token_ids": [1, 2]},
                       lengths=(32,), output_tokens=8, requests=32, warmup=1)
    paths = []
    for engine in ("vllm", "sglang", "llama.cpp"):
        async with mock_server() as (url, _):
            report = await run(fixture_config(engine, url), workload, out / engine,
                               concurrencies=(1, 2, 4, 8), external=True, interval=.05)
            if report["status"] != "completed":
                raise RuntimeError(f"{engine} protocol smoke failed")
            paths.append(out / engine / "report.json")
    plot(paths, out / "plots")
    print(f"SYNTHETIC PROTOCOL TEST ONLY: {out / 'plots'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(smoke(parser.parse_args().out))
