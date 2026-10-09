"""Replay a common workload, recording completed client calls and telemetry."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
from urllib.parse import urlsplit

from .adapters import Adapter, sse_events
from .metrics import summarize
from .workload import digest, positive_integer

REPORT_SCHEMA = "tensor.llm-serving-report.v1"
CONFIG_SCHEMA = "tensor.llm-serving-servers.v1"


def is_oom(message):
    return re.search(r"\b(?:out of memory|oom|cuda_error_out_of_memory)\b", message, re.IGNORECASE) is not None


def redact(value):
    secret_keys = {"api_key", "hf_token", "access_token", "authorization", "password",
                   "token", "environment", "env", "launch_command"}
    if isinstance(value, dict):
        return {key: "[redacted]" if key.lower().replace("-", "_") in secret_keys
                else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        result, hide_next = [], False
        for item in value:
            if hide_next:
                result.append("[redacted]")
                hide_next = False
            elif isinstance(item, str) and item.split("=", 1)[0] in (
                    "--api-key", "--hf-token", "--access-token", "--password"):
                result.append(item.split("=", 1)[0] + "=[redacted]" if "=" in item else item)
                hide_next = "=" not in item
            else:
                result.append(redact(item))
        return result
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_servers(path, engines=None):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != CONFIG_SCHEMA or not config.get("comparison_group"):
        raise ValueError("server manifest needs its schema and comparison_group")
    names = set()
    for server in config.get("servers", []):
        name = server.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError("server names must be unique safe directory names")
        names.add(name)
        Adapter(server.get("engine"), server.get("model"))
        if engines and name not in engines and server["engine"] not in engines:
            continue
        if server.get("disposition") in ("unsupported", "not-run", "oom"):
            if not server.get("reason"):
                raise ValueError("unmeasured server cells need a reason")
            continue
        for key in ("base_url", "model", "engine_version", "model_revision", "weight_format",
                    "kv_dtype", "state_dtype", "tokenizer_name", "tokenizer_revision",
                    "hardware", "cpu_offload"):
            if not isinstance(server.get(key), str) or not server[key] or "REPLACE_" in server[key]:
                raise ValueError(f"{name}: supply a pinned {key}")
        for key in ("prefix_cache", "speculative"):
            if type(server.get(key)) is not bool:
                raise ValueError(f"{name}: explicitly declare {key}")
        url = urlsplit(server["base_url"])
        if (url.scheme not in ("http", "https") or not url.netloc or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("base_url needs HTTP(S), without credentials/query/fragment")
        if "command" in server and (not isinstance(server["command"], list)
                or not server["command"] or any(not isinstance(arg, str) or "REPLACE_" in arg
                                               for arg in server["command"])):
            raise ValueError(f"{name}: command must be a complete argument array")
        indices = server.get("gpu_indices", [])
        if not isinstance(indices, list) or any(not isinstance(index, str) for index in indices):
            raise ValueError("gpu_indices must be a list of local GPU IDs/UUIDs")
        environment = server.get("environment", {})
        if not isinstance(environment, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                                    for key, value in environment.items()):
            raise ValueError("environment must map variable names to string values")
    if not names:
        raise ValueError("server manifest is empty")
    return config


async def fetch(session, base_url, path, *, payload=None):
    try:
        async with session.request("POST" if payload is not None else "GET",
                                   base_url.rstrip("/") + path, json=payload,
                                   timeout=3) as response:
            text = await response.text()
            value = {"http_status": response.status, "path": path}
            if "json" in response.headers.get("Content-Type", ""):
                value["data"] = redact(json.loads(text))
            else:
                value["text"] = text[:2_000_000]
            return value
    except Exception as error:
        return {"path": path, "error": f"{type(error).__name__}: {error}"}


def signal_process(process, sig):
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass  # The child can exit between the returncode check and signalling.


@asynccontextmanager
async def server_lifecycle(session, server, directory, *, external, startup_timeout):
    if external:
        health = await fetch(session, server["base_url"], "/health")
        if health.get("http_status") != 200:
            raise RuntimeError(f"server is not healthy: {health}")
        yield {"managed": False, "startup_seconds": None}
        return
    if not server.get("command"):
        raise ValueError("managed runs require command; use --external for an existing server")
    health = await fetch(session, server["base_url"], "/health")
    if "http_status" in health:
        raise RuntimeError("endpoint already responds; stop it or use --external explicitly")
    process = None
    log = (directory / "server.log").open("wb")
    start = time.perf_counter()
    try:
        environment = os.environ | server.get("environment", {})
        executable = Path(server["command"][0])
        if executable.parent != Path("."):
            directory_root = Path(server.get("cwd", Path.cwd()))
            bin_directory = (directory_root / executable.parent).resolve()
            environment["PATH"] = str(bin_directory) + os.pathsep + environment.get("PATH", "")
        process = await asyncio.create_subprocess_exec(
            *server["command"], cwd=server.get("cwd"), stdout=log, stderr=log,
            env=environment,
            start_new_session=os.name == "posix")
        while True:
            if process.returncode is not None:
                raise RuntimeError(f"server exited ({process.returncode}); see server.log")
            if time.perf_counter() - start > startup_timeout:
                raise TimeoutError("server startup exceeded budget; see server.log")
            health = await fetch(session, server["base_url"], "/health")
            if health.get("http_status") == 200:
                break
            await asyncio.sleep(.5)
        yield {"managed": True, "startup_seconds": time.perf_counter() - start}
    finally:
        if process is not None and (os.name == "posix" or process.returncode is None):
            signal_process(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 15)
            except asyncio.TimeoutError:
                signal_process(process, signal.SIGKILL)
                await process.wait()
            finally:
                if os.name == "posix":
                    signal_process(process, signal.SIGKILL)  # Clean up any surviving worker children.
        log.close()


async def request_once(session, adapter, server, request, *, origin, offered,
                       seed, phase, timeout):
    import aiohttp

    sent = time.perf_counter()
    first, last = None, None
    count, prompt_count, finished = 0, None, False
    milestones, chunk_gaps, token_gaps, metadata = [], [], [], {}
    stream_hash = hashlib.sha256()
    status, error, disposition = "failed", None, "incorrect"
    http_status = None
    try:
        async with session.post(
                server["base_url"].rstrip("/") + adapter.path,
                json=adapter.payload(request, seed=seed, prefix_cache=server["prefix_cache"]),
                timeout=timeout) as response:
            http_status = response.status
            if response.status != 200:
                message = (await response.text())[:4096]
                disposition = "oom" if is_oom(message) else "environment-failed"
                raise RuntimeError(f"HTTP {response.status}: {message}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("endpoint did not return an SSE stream")
            async for data in sse_events(response.content):
                now = time.perf_counter()
                if data is None:
                    finished = True
                    break
                stream_hash.update(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
                stream_hash.update(b"\n")
                event = adapter.parse(data, count)
                if event.prompt_tokens is not None:
                    prompt_count = event.prompt_tokens
                if event.metadata:
                    metadata.update(event.metadata)
                finished |= event.finished
                if event.count is not None:
                    if type(event.count) is not int or event.count < count:
                        raise ValueError("server token count is invalid or decreased")
                    if event.count > count:
                        if first is None:
                            first = now
                        if last is not None:
                            chunk_gaps.append(now - last)
                            if event.count == count + 1 and milestones[-1]["delta_tokens"] == 1:
                                token_gaps.append(now - last)
                        milestones.append({"seconds": now - origin, "output_tokens": event.count,
                                           "delta_tokens": event.count - count})
                        last = now
                    count = event.count
                elif event.text:
                    # A protocol can emit text before the cumulative usage count.
                    # Preserve first visible output; never count words or SSE chunks as tokens.
                    if first is None:
                        first = now
            if not finished:
                raise ValueError("stream ended without a terminal event")
            if first is None or last is None:
                raise ValueError("stream supplied no usable generated-token counts")
            if prompt_count != len(request["prompt_token_ids"]):
                raise ValueError(f"prompt count mismatch: expected {len(request['prompt_token_ids'])}, got {prompt_count}")
            if count != request["output_tokens"]:
                raise ValueError(f"output count mismatch: expected {request['output_tokens']}, got {count}")
            if metadata.get("truncated"):
                raise ValueError("server truncated or shifted the context")
            finish_reason = metadata.get("finish_reason")
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("type")
            if finish_reason in ("abort", "error"):
                raise ValueError("server reported an aborted generation")
            status, disposition = "completed", "measured"
    except asyncio.CancelledError:
        raise
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        if is_oom(str(exception)):
            disposition = "oom"
        elif isinstance(exception, (TimeoutError, OSError, aiohttp.ClientError)):
            disposition = "environment-failed"
    ended = time.perf_counter()
    return {"id": request["id"], "phase": phase, "status": status, "disposition": disposition,
            "error": error, "http_status": http_status,
            "offered_seconds": offered - origin, "sent_seconds": sent - origin,
            "ended_seconds": ended - origin, "client_queue_seconds": sent - offered,
            "prompt_tokens": len(request["prompt_token_ids"]), "server_prompt_tokens": prompt_count,
            "expected_output_tokens": request["output_tokens"], "output_tokens": count,
            "ttft_seconds": first - offered if first is not None else None,
            "service_ttft_seconds": first - sent if first is not None else None,
            "latency_seconds": ended - offered,
            "tpot_seconds": (last - first) / (count - 1) if count > 1 and last is not None else None,
            "token_events": milestones, "stream_chunk_gaps_seconds": chunk_gaps,
            "single_token_gaps_seconds": token_gaps, "stream_sha256": stream_hash.hexdigest(),
            "server_metadata": redact(metadata)}


async def gpu_snapshot(indices):
    if not indices:
        return {"status": "disabled", "reason": "no local server GPU IDs declared"}
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--id=" + ",".join(indices),
            "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            return {"status": "unavailable", "error": stderr.decode().strip()}
        rows = []
        for line in stdout.decode().splitlines():
            uuid, name, used, total, utilization, driver = [part.strip() for part in line.split(",")]
            rows.append({"uuid": uuid, "name": name, "used_mib": float(used),
                         "total_mib": float(total), "utilization_percent": float(utilization),
                         "driver": driver})
        return {"status": "sampled", "location": "client-host", "devices": rows}
    except Exception as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}


async def sample_telemetry(session, server, path, origin, stop, interval):
    with path.open("w") as output:
        while True:
            gpu, metrics = await asyncio.gather(
                gpu_snapshot(server.get("gpu_indices", [])),
                fetch(session, server["base_url"], server.get("metrics_path", "/metrics")))
            output.write(json.dumps({"seconds": time.perf_counter() - origin,
                                     "gpu": gpu, "metrics": metrics}) + "\n")
            output.flush()
            if stop.is_set():
                return
            try:
                await asyncio.wait_for(stop.wait(), interval)
            except asyncio.TimeoutError:
                pass


def point_disposition(rows):
    errors = {row["disposition"] for row in rows if row["status"] != "completed"}
    if not errors:
        return "measured"
    if errors == {"oom"}:
        return "oom"
    return "incorrect" if "incorrect" in errors else "environment-failed"


async def run_point(session, server, workload, directory, *, concurrency, repeat,
                    timeout, interval, ttft_slo, tpot_slo):
    adapter = Adapter(server["engine"], server["model"])
    prefix = f"c{concurrency}-r{repeat}"
    rows = []
    stop = asyncio.Event()
    origin = time.perf_counter()
    monitor = asyncio.create_task(sample_telemetry(session, server, directory / f"{prefix}-telemetry.jsonl",
                                                  origin, stop, interval))
    workers = []
    with (directory / f"{prefix}-requests.jsonl").open("w") as output:
        iterator = iter(enumerate(workload["requests"]))

        async def worker():
            for index, request in iterator:
                offered = origin + request["arrival_seconds"] if "arrival_seconds" in request else time.perf_counter()
                if "arrival_seconds" in request:
                    await asyncio.sleep(max(0, offered - time.perf_counter()))
                row = await request_once(session, adapter, server, request, origin=origin,
                                         offered=offered, seed=workload["seed"] + index,
                                         phase="measured", timeout=timeout)
                row["index"] = index
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                if len(rows) % 25 == 0 or len(rows) == len(workload["requests"]):
                    print(f"{server['name']} {prefix}: {len(rows)}/{len(workload['requests'])} finished calls", flush=True)

        try:
            workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(workload["requests"])))]
            await asyncio.gather(*workers)
            elapsed = time.perf_counter() - origin
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            stop.set()
            await monitor
    summary = summarize(rows, elapsed, ttft_slo=ttft_slo, tpot_slo=tpot_slo)
    return {"concurrency": concurrency, "repeat": repeat, "disposition": point_disposition(rows),
            "requests_file": f"{prefix}-requests.jsonl", "telemetry_file": f"{prefix}-telemetry.jsonl",
            "summary": summary, "warnings": (["fewer than two waves of requests; fill/drain effects can dominate"]
                                               if len(rows) < 2 * concurrency else [])}


async def tokenizer_probe(session, server, workload):
    provenance = workload["tokenizer"]
    if (server["tokenizer_name"] != provenance["name"]
            or server["tokenizer_revision"] != provenance["revision"]):
        raise ValueError("server tokenizer identity differs from the workload")
    if not provenance.get("probe_text"):
        if not server.get("synthetic", False):
            raise ValueError("real server runs require a tokenizer text/ID probe in the workload")
        return {"status": "synthetic-fixture"}
    adapter = Adapter(server["engine"], server["model"])
    probe = await fetch(session, server["base_url"], "/tokenize",
                        payload=adapter.tokenize_payload(provenance["probe_text"]))
    if probe.get("http_status") != 200:
        raise ValueError(f"tokenizer probe unsupported/failed: {probe}")
    tokens = probe.get("data", {}).get("tokens")
    if tokens != provenance["probe_token_ids"]:
        raise ValueError("server tokenizer probe differs from prepared token IDs")
    return {"status": "matched", "probe_sha256": digest(tokens)}


async def run(config, workload, out, *, engines=None, concurrencies=(1, 2, 4, 8, 16, 32),
              repeats=1, external=False, timeout=3600, startup_timeout=1800,
              interval=2, ttft_slo=None, tpot_slo=None, api_key_env="LLM_BENCH_API_KEY"):
    import aiohttp

    if (not concurrencies or any(not positive_integer(c) for c in concurrencies)
            or len(set(concurrencies)) != len(concurrencies) or not positive_integer(repeats)
            or any(not math.isfinite(value) or value <= 0 for value in
                   (timeout, startup_timeout, interval, *[v for v in (ttft_slo, tpot_slo) if v is not None]))):
        raise ValueError("concurrency must be unique; repeats, timeouts, interval and SLOs must be finite and positive")
    selected = [server for server in config["servers"]
                if not engines or server["name"] in engines or server["engine"] in engines]
    if not selected or engines and any(not any(item in (server["name"], server["engine"])
                                              for server in selected) for item in engines):
        raise ValueError("unknown or empty engine selection")
    if external and sum(not server.get("disposition") for server in selected) > 1:
        raise ValueError("external runs select one engine at a time to avoid GPU contention")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "workload.json", workload)
    headers = {}
    if os.environ.get(api_key_env):
        headers["Authorization"] = "Bearer " + os.environ[api_key_env]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {"schema": REPORT_SCHEMA, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "comparison_group": config["comparison_group"], "workload_sha256": workload["sha256"],
              "workload_kind": workload["kind"], "arrival": workload["arrival"],
              "synthetic": any(server.get("synthetic", False) for server in selected),
              "protocol": {"boundary": "client HTTP dispatch/arrival to completed stream",
                           "connections": "fresh TCP connection per call; no transparent retries",
                           "arrival_latency": "includes client admission delay in poisson mode",
                           "throughput_denominator": "entire measured point including fill/drain and failed calls",
                           "input_throughput": "prompt tokens / whole point; not isolated prefill speed",
                           "tpot": "(last token event - first token event) / (output tokens - 1)",
                           "chunking": "client-observed; coalesced tokens have a shared arrival timestamp",
                           "warmup": "separate requests excluded from measurement; run before every point",
                           "retries": 0},
              "settings": {"concurrencies": list(concurrencies), "repeats": repeats,
                           "request_timeout_seconds": timeout, "telemetry_interval_seconds": interval,
                           "ttft_slo_seconds": ttft_slo, "tpot_slo_seconds": tpot_slo},
              "provenance": {"platform": platform.platform(), "python": platform.python_version(),
                             "git_commit": commit, "aiohttp": aiohttp.__version__,
                             "harness_sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                 for path in Path(__file__).parent.glob("*.py")}},
              "servers": []}
    atomic_json(out / "report.json", report)
    # cpp-httplib can close its keep-alive stream while a pooled socket is reused.
    # Use the same fresh-connection policy for every engine, without retrying calls.
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    try:
        async with aiohttp.ClientSession(connector=connector, headers=headers, read_bufsize=1024 * 1024) as session:
            for server in selected:
                directory = out / server["name"]
                directory.mkdir()
                result = {"configuration": redact(server), "points": []}
                report["servers"].append(result)
                if server.get("disposition"):
                    result.update(disposition=server["disposition"], reason=server["reason"])
                    continue
                try:
                    async with server_lifecycle(session, server, directory, external=external,
                                                startup_timeout=startup_timeout) as lifecycle:
                        result["lifecycle"] = lifecycle
                        result["hardware_snapshot"] = await gpu_snapshot(server.get("gpu_indices", []))
                        result["tokenizer_probe"] = await tokenizer_probe(session, server, workload)
                        paths = {"vllm": ["/version", "/v1/models"],
                                 "sglang": ["/server_info", "/get_server_info"],
                                 "llama.cpp": ["/props", "/slots"],
                                 "tensor": ["/server_info"]}[server["engine"]]
                        result["server_info"] = await asyncio.gather(*(fetch(session, server["base_url"], path) for path in paths))
                        adapter = Adapter(server["engine"], server["model"])
                        failed_point = None
                        for concurrency in concurrencies:
                            for repeat in range(repeats):
                                if failed_point is not None:
                                    result["points"].append({"concurrency": concurrency, "repeat": repeat,
                                                             "disposition": "not-run",
                                                             "reason": f"stopped after failed point {failed_point}; server may have pending work"})
                                    continue
                                warmup = []
                                for index, request in enumerate(workload.get("warmup", [])):
                                    origin = time.perf_counter()
                                    row = await request_once(session, adapter, server, request,
                                                             origin=origin, offered=origin,
                                                             seed=workload["seed"] + index,
                                                             phase="warmup", timeout=timeout)
                                    warmup.append(row)
                                atomic_json(directory / f"c{concurrency}-r{repeat}-warmup.json", warmup)
                                if any(row["status"] != "completed" for row in warmup):
                                    failed_point = f"c{concurrency}-r{repeat}"
                                    result["points"].append({"concurrency": concurrency, "repeat": repeat,
                                                             "disposition": point_disposition(warmup),
                                                             "reason": "warmup/protocol qualification failed",
                                                             "warmup_file": f"{failed_point}-warmup.json"})
                                    atomic_json(out / "report.json", report)
                                    continue
                                point = await run_point(session, server, workload, directory,
                                                        concurrency=concurrency, repeat=repeat, timeout=timeout,
                                                        interval=interval, ttft_slo=ttft_slo, tpot_slo=tpot_slo)
                                result["points"].append(point)
                                if point["disposition"] != "measured":
                                    failed_point = f"c{concurrency}-r{repeat}"
                                atomic_json(out / "report.json", report)
                        result["disposition"] = "measured" if all(point["disposition"] == "measured" for point in result["points"]) else "failed"
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    result.update(disposition="environment-failed", reason=f"{type(error).__name__}: {error}")
                atomic_json(out / "report.json", report)
        report["status"] = "completed" if all(server.get("disposition") == "measured" for server in report["servers"]) else "incomplete"
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        atomic_json(out / "report.json", report)
    return report
