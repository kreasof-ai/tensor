"""Local HTTP endpoint for the dense Tensor engine and matched harness."""

import argparse
import asyncio
import json
from threading import Thread

from .scheduler import Request, Scheduler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--port", type=int, default=8103)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--mtp", action="store_true")
    p.add_argument("--single-graph", action="store_true")
    p.add_argument("--deferred", action="store_true")
    p.add_argument("--fused-head", action="store_true")
    p.add_argument("--verification-window", type=int, choices=(2, 4), default=4)
    a = p.parse_args()
    from aiohttp import web
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    app = web.Application(client_max_size=8 * 1024**2)

    async def startup(app):
        loop = asyncio.get_running_loop()
        ready = loop.create_future()

        def publish(events):
            def deliver():
                for request, event in events:
                    if not request.cancelled.is_set():
                        request.channel.put_nowait(event)

            loop.call_soon_threadsafe(deliver)

        def worker():
            from tensor.providers.cuda import Device
            from .engine import DenseEngine

            try:
                with Device() as device:
                    engine = DenseEngine(
                        a.checkpoint,
                        a.bundle,
                        device,
                        capacity=a.capacity,
                        context=a.context,
                        fused_head=a.fused_head,
                    )
                    target = engine
                    if a.mtp:
                        from .speculative import SpeculativeEngine

                        engine = SpeculativeEngine(
                            target,
                            window=a.verification_window,
                            single_graph=a.single_graph,
                            deferred=a.deferred,
                        )
                    scheduler = Scheduler(engine, publish)
                    app["scheduler"] = scheduler
                    for count in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                        if count <= a.capacity:
                            target.batch(count)
                            if a.mtp:
                                target.batch(count, a.verification_window, verify=True)
                                engine.drafter.batch(count)
                                engine.drafter.batch(count, a.verification_window)
                                if a.single_graph:
                                    engine.round(count)
                        if count <= min(a.capacity, 32):
                            target.batch(count, 32)
                            if a.mtp:
                                engine.drafter.batch(count, 32)
                    loop.call_soon_threadsafe(ready.set_result, True)
                    try:
                        scheduler.run()
                    finally:
                        engine.close()
            except BaseException as error:
                import traceback

                traceback.print_exc()

                def failed(error=error):
                    app["failure"] = str(error)
                    if not ready.done():
                        ready.set_exception(error)
                    if "scheduler" in app:
                        for request in (
                            *app["scheduler"].decoding,
                            *app["scheduler"].prefilling,
                        ):
                            request.channel.put_nowait(dict(error=str(error)))

                loop.call_soon_threadsafe(failed)

        app["worker"] = Thread(target=worker, daemon=True)
        app["worker"].start()
        await ready

    async def cleanup(app):
        if "scheduler" in app:
            app["scheduler"].stopped.set()
        await asyncio.to_thread(app["worker"].join, 30)

    async def health(request):
        return web.json_response(
            dict(status="failed" if app.get("failure") else "ok"),
            status=503 if app.get("failure") else 200,
        )

    async def tokenize(request):
        payload = await request.json()
        return web.json_response(
            dict(tokens=tokenizer.encode(payload["prompt"], add_special_tokens=False))
        )

    async def info(request):
        return web.json_response(
            dict(
                engine="tensor",
                model="Qwen/Qwen3.5-0.8B",
                weights="BF16",
                kv_dtype="bfloat16",
                state_dtype="float32",
                prefix_cache=False,
                speculative=a.mtp,
                capacity=a.capacity,
                context=a.context,
                stats=app["scheduler"].stats,
                speculation_stats=getattr(app["scheduler"].engine, "stats", None),
                implementation=getattr(
                    app["scheduler"].engine, "target", app["scheduler"].engine
                ).implementation,
                quality_qualified=False,
            )
        )

    async def generate(request):
        data = await request.json()
        prompt = data.get("input_ids")
        sampling = data.get("sampling_params", {})
        limit = sampling.get("max_new_tokens", 256)
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(t) is not int or not 0 <= t < 248320 for t in prompt)
            or type(limit) is not int
            or limit < 1
            or len(prompt) + limit > a.context
            or sampling.get("temperature", 0) != 0
            or not sampling.get("ignore_eos", False)
        ):
            raise web.HTTPBadRequest(
                text="requires valid token IDs, greedy forced length, and sufficient context"
            )
        item = Request(prompt, limit, asyncio.Queue())
        app["scheduler"].submit(item)
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        try:
            while True:
                event = await item.channel.get()
                await response.write(
                    (
                        "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
                    ).encode()
                )
                if event.get("error") or event.get("meta_info", {}).get(
                    "finish_reason"
                ):
                    break
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        finally:
            item.cancelled.set()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_get("/health", health)
    app.router.add_get("/server_info", info)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/generate", generate)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None)


if __name__ == "__main__":
    main()
