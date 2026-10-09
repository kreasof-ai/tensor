"""Explicit server protocols; counts come from token IDs or server usage."""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Event:
    count: int | None = None
    prompt_tokens: int | None = None
    token_ids: list[int] | None = None
    text: str = ""
    finished: bool = False
    metadata: dict | None = None


class Adapter:
    def __init__(self, engine, model):
        if engine not in ("vllm", "sglang", "llama.cpp", "tensor"):
            raise ValueError(f"unsupported engine: {engine}")
        self.engine, self.model = engine, model
        self.path = {"vllm": "/v1/completions", "sglang": "/generate",
                     "llama.cpp": "/completion", "tensor": "/generate"}[engine]

    def payload(self, request, *, seed, prefix_cache):
        tokens, output = request["prompt_token_ids"], request["output_tokens"]
        if self.engine == "vllm":
            return {"model": self.model, "prompt": tokens, "max_tokens": output,
                    "temperature": 0.0, "ignore_eos": True, "seed": seed,
                    "stream": True, "stream_options": {"include_usage": True},
                    "add_special_tokens": False, "return_token_ids": True}
        if self.engine in ("sglang", "tensor"):
            return {"input_ids": tokens, "stream": True,
                    "sampling_params": {"max_new_tokens": output, "temperature": 0.0,
                                        "ignore_eos": True, "sampling_seed": seed}}
        return {"prompt": tokens, "n_predict": output, "temperature": 0.0,
                "ignore_eos": True, "seed": seed, "stream": True,
                "return_tokens": True, "cache_prompt": prefix_cache,
                "stop": []}

    def parse(self, data, current):
        if data.get("error"):
            raise ValueError(f"server stream error: {data['error']}")
        if self.engine == "vllm":
            choice = next(iter(data.get("choices", [])), {})
            token_ids = choice.get("token_ids")
            usage = data.get("usage") or {}
            count = current + len(token_ids) if token_ids is not None else None
            if usage.get("completion_tokens") is not None:
                count = usage["completion_tokens"]
            metadata = {}
            if usage:
                metadata["usage"] = usage
            if choice.get("finish_reason") is not None:
                metadata["finish_reason"] = choice["finish_reason"]
            return Event(count, usage.get("prompt_tokens"), token_ids,
                         choice.get("text", ""), choice.get("finish_reason") is not None,
                         metadata)
        if self.engine in ("sglang", "tensor"):
            metadata = data.get("meta_info", {})
            return Event(metadata.get("completion_tokens"), metadata.get("prompt_tokens"),
                         None, data.get("text", ""), metadata.get("finish_reason") is not None,
                         metadata)
        token_ids = data.get("tokens")
        count = current + len(token_ids) if token_ids is not None else None
        if data.get("tokens_predicted") is not None:
            count = data["tokens_predicted"]
        if count is None and data.get("stop"):
            count = data.get("timings", {}).get("predicted_n")
        return Event(count, data.get("tokens_evaluated"), token_ids,
                     data.get("content", ""), bool(data.get("stop")),
                     {key: data[key] for key in ("timings", "truncated", "tokens_cached",
                                                "stop_type", "stopped_eos", "stopped_limit")
                      if key in data})

    def tokenize_payload(self, text):
        if self.engine == "llama.cpp":
            return {"content": text, "add_special": False}
        return {"model": self.model, "prompt": text, "add_special_tokens": False}


async def sse_events(stream):
    """Parse SSE frames across arbitrary network fragments, including multiline data."""
    fields = []
    while True:
        raw = await stream.readline()
        if not raw:
            if fields:
                value = "\n".join(fields)
                if value == "[DONE]":
                    yield None
                else:
                    yield json.loads(value)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if fields:
                value = "\n".join(fields)
                fields = []
                if value == "[DONE]":
                    yield None
                    return
                yield json.loads(value)
        elif line.startswith("data:"):
            fields.append(line[5:].lstrip(" "))
