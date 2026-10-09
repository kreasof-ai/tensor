"""Benchmark contracts that would otherwise produce misleading comparisons."""
import asyncio
import json

import pytest

from benchmarks.llm_serving.adapters import Adapter, sse_events
from benchmarks.llm_serving.metrics import summarize
from benchmarks.llm_serving.workload import prepare, validate


def workload(**options):
    return prepare([3, 4, 5, 6], {"name": "fixture", "revision": "fixture-v1",
                                "probe_text": "fixture probe", "probe_token_ids": [1, 2]},
                   lengths=(8, 13), output_tokens=3, requests=6, warmup=0, **options)


def test_trace_identity_and_exact_length():
    first = workload()
    assert first == workload()
    assert [len(item["prompt_token_ids"]) for item in first["requests"]] == [8, 13] * 3
    offered = workload(request_rate=10)
    assert [r["prompt_token_ids"] for r in first["requests"]] == [r["prompt_token_ids"] for r in offered["requests"]]
    assert offered["requests"][0]["arrival_seconds"] == 0
    assert offered["requests"][-1]["arrival_seconds"] > 0
    first["requests"][0]["prompt_token_ids"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        validate(first)


@pytest.mark.parametrize("options", [{"request_rate": float("nan")}, {"request_rate": -1},
                                    {"lengths": (0,)}, {"output_tokens": 0}, {"requests": 0}])
def test_invalid_workload_options(options):
    with pytest.raises(ValueError):
        prepare([3, 4], {"name": "fixture"}, **options)


def test_workload_cannot_declare_sampling_that_is_not_replayed():
    from benchmarks.llm_serving.workload import digest
    value = workload()
    value["sampling"]["temperature"] = 1
    value["sha256"] = digest({key: item for key, item in value.items() if key != "sha256"})
    with pytest.raises(ValueError, match="sampling"):
        validate(value)


def test_selected_manifest_allows_unconfigured_other_engines(tmp_path):
    from benchmarks.llm_serving.runner import load_servers
    from benchmarks.llm_serving.smoke import fixture_config
    value = fixture_config("vllm", "http://localhost:8000")
    other = fixture_config("sglang", "http://localhost:8001")["servers"][0]
    other["engine_version"] = "REPLACE_VERSION"
    value["servers"].append(other)
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(value))
    assert load_servers(path, ["vllm"]) == value
    with pytest.raises(ValueError, match="engine_version"):
        load_servers(path)


@pytest.mark.parametrize("options", [{"timeout": float("nan")}, {"interval": float("inf")},
                                    {"ttft_slo": float("nan")}, {"concurrencies": (1, 1)}])
def test_invalid_run_options_are_rejected_before_creating_output(tmp_path, options):
    pytest.importorskip("aiohttp")
    from benchmarks.llm_serving.runner import run
    from benchmarks.llm_serving.smoke import fixture_config
    with pytest.raises(ValueError):
        asyncio.run(run(fixture_config("vllm", "http://localhost:8000"), workload(),
                        tmp_path / "run", **options))
    assert not (tmp_path / "run").exists()


def test_sse_multiline_comments_and_terminal_marker():
    async def check():
        reader = asyncio.StreamReader()
        reader.feed_data(b': heartbeat\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata: [DONE]\n\n')
        reader.feed_eof()
        return [value async for value in sse_events(reader)]
    assert asyncio.run(check()) == [{"a": 1}, None]


def test_usage_and_chunk_counts_are_not_word_counts():
    adapter = Adapter("vllm", "fixture")
    assert adapter.parse({"choices": [{"text": "many words but one token", "token_ids": [7]}]}, 0).count == 1
    assert adapter.parse({"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 8}}, 3).count == 3
    sglang = Adapter("sglang", "fixture")
    assert sglang.parse({"text": "cumulative output", "meta_info": {"completion_tokens": 3}}, 2).count == 3
    payload = sglang.payload(workload()["requests"][0], seed=9, prefix_cache=False)
    assert payload["sampling_params"]["sampling_seed"] == 9
    assert "seed" not in payload["sampling_params"]


def test_failed_outputs_and_slos_do_not_inflate_throughput():
    rows = [{"status": status, "output_tokens": count, "prompt_tokens": 10,
             "ttft_seconds": ttft, "service_ttft_seconds": ttft, "tpot_seconds": .1,
             "latency_seconds": 2, "client_queue_seconds": 0,
             "stream_chunk_gaps_seconds": [.1], "single_token_gaps_seconds": [.1]}
            for status, count, ttft in (("completed", 5, .5), ("completed", 5, 3), ("failed", 100, .1))]
    result = summarize(rows, 10, ttft_slo=1)
    assert result["output_tokens_per_second"] == 1
    assert result["goodput_output_tokens_per_second"] == .5
    assert result["failed"] == 1
    assert result["observed_output_tokens_including_failures"] == 110


@pytest.mark.parametrize("engine", ["vllm", "sglang", "llama.cpp"])
def test_real_http_replay_batches_and_preserves_prompt_ids(tmp_path, engine):
    pytest.importorskip("aiohttp")
    from benchmarks.llm_serving.runner import run
    from benchmarks.llm_serving.smoke import fixture_config, mock_server

    async def check():
        value = workload()
        async with mock_server() as (url, state):
            report = await run(fixture_config(engine, url), value, tmp_path / "run",
                               concurrencies=(2,), external=True, interval=.01)
            assert state["peak_active"] == 2
            assert len(state["connections"]) == len(value["requests"])
            assert state["prompts"] == [r["prompt_token_ids"] for r in value["requests"]]
            return report
    report = asyncio.run(check())
    assert report["status"] == "completed"
    assert report["synthetic"]
    point = report["servers"][0]["points"][0]
    assert point["summary"]["completed"] == 6
    assert point["summary"]["output_tokens"] == 18
    assert point["summary"]["ttft_seconds"]["p50"] > 0
    assert "fixture-secret" not in (tmp_path / "run" / "report.json").read_text()
    rows = [json.loads(line) for line in (tmp_path / "run" / engine / point["requests_file"]).read_text().splitlines()]
    assert all(row["server_prompt_tokens"] == row["prompt_tokens"] for row in rows)
    if engine == "vllm":
        assert all(row["server_metadata"]["finish_reason"] == "length" for row in rows)


@pytest.mark.parametrize("short,truncated,terminate", [(True, False, True),
                                                     (False, True, True), (False, False, False)])
def test_short_truncated_or_broken_stream_cannot_pass(tmp_path, short, truncated, terminate):
    pytest.importorskip("aiohttp")
    import aiohttp
    from benchmarks.llm_serving.runner import request_once
    from benchmarks.llm_serving.smoke import fixture_config, mock_server
    import time

    async def check():
        async with mock_server(short=short, truncated=truncated, terminate=terminate) as (url, _):
            server = fixture_config("llama.cpp", url)["servers"][0]
            async with aiohttp.ClientSession() as session:
                origin = time.perf_counter()
                return await request_once(session, Adapter("llama.cpp", "fixture"), server,
                                          workload()["requests"][0], origin=origin, offered=origin,
                                          seed=1, phase="measured", timeout=2)
    assert asyncio.run(check())["status"] == "failed"


def test_poisson_client_queue_is_included_in_latency(tmp_path):
    pytest.importorskip("aiohttp")
    from benchmarks.llm_serving.runner import run
    from benchmarks.llm_serving.smoke import fixture_config, mock_server

    async def check():
        async with mock_server(delay=.005) as (url, _):
            return await run(fixture_config("vllm", url), workload(request_rate=100000),
                             tmp_path / "run", concurrencies=(1,), external=True, interval=.01)
    report = asyncio.run(check())
    point = report["servers"][0]["points"][0]
    assert point["summary"]["client_queue_seconds"]["p95"] > .05
    assert point["summary"]["ttft_seconds"]["p95"] > point["summary"]["service_ttft_seconds"]["p95"]


def test_failed_point_stops_later_loads_without_retry(tmp_path):
    pytest.importorskip("aiohttp")
    from benchmarks.llm_serving.runner import run
    from benchmarks.llm_serving.smoke import fixture_config, mock_server

    async def check():
        async with mock_server(short=True) as (url, state):
            report = await run(fixture_config("vllm", url), workload(), tmp_path / "run",
                               concurrencies=(1, 2), repeats=2, external=True, interval=.01)
            assert len(state["prompts"]) == len(workload()["requests"])
            return report
    report = asyncio.run(check())
    assert report["status"] == "incomplete"
    assert [point["disposition"] for point in report["servers"][0]["points"]] == [
        "incorrect", "not-run", "not-run", "not-run"]


def test_oom_is_retained_without_retry_or_fallback(tmp_path):
    pytest.importorskip("aiohttp")
    from benchmarks.llm_serving.runner import is_oom, run
    from benchmarks.llm_serving.smoke import fixture_config, mock_server

    assert not is_oom("no room for an unknown model")

    async def check():
        async with mock_server(failure="CUDA out of memory") as (url, state):
            report = await run(fixture_config("sglang", url), workload(), tmp_path / "run",
                               concurrencies=(1, 2), external=True, interval=.01)
            assert len(state["payloads"]) == len(workload()["requests"])
            return report
    report = asyncio.run(check())
    points = report["servers"][0]["points"]
    assert [point["disposition"] for point in points] == ["oom", "not-run"]
    assert points[0]["summary"]["output_tokens_per_second"] == 0


def test_warmup_oom_keeps_the_reason_and_does_not_measure(tmp_path):
    pytest.importorskip("aiohttp")
    from benchmarks.llm_serving.runner import run
    from benchmarks.llm_serving.smoke import fixture_config, mock_server

    async def check():
        value = prepare([3, 4], {"name": "fixture", "revision": "fixture-v1"},
                        lengths=(8,), output_tokens=3, requests=2, warmup=1)
        async with mock_server(failure="CUDA out of memory") as (url, state):
            report = await run(fixture_config("sglang", url), value, tmp_path / "run",
                               concurrencies=(1, 2), external=True, interval=.01)
            assert len(state["payloads"]) == 1
            return report
    points = asyncio.run(check())["servers"][0]["points"]
    assert [point["disposition"] for point in points] == ["oom", "not-run"]
    assert "summary" not in points[0]
    assert points[0]["warmup_file"] == "c1-r0-warmup.json"


def test_managed_server_is_cleaned_up_when_benchmark_raises(tmp_path):
    pytest.importorskip("aiohttp")
    import aiohttp
    import os
    import socket
    import sys
    from benchmarks.llm_serving.runner import server_lifecycle

    if os.name != "posix":
        pytest.skip("process-group check is POSIX-specific")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = tmp_path / "server.py"
    pid_file = tmp_path / "pid"
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    (bin_directory / "python").symlink_to(sys.executable)
    helper = bin_directory / "tensor-benchmark-helper"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o755)
    script.write_text("""import http.server, os, pathlib, shutil, sys
pathlib.Path(sys.argv[2]).write_text(str(os.getpid()))
pathlib.Path(sys.argv[2] + '.env').write_text(os.environ['LLM_BENCH_FIXTURE_VALUE'] + '\\n' + shutil.which('tensor-benchmark-helper'))
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
""")
    server = {"base_url": f"http://127.0.0.1:{port}",
              "command": [str(bin_directory / "python"), str(script), str(port), str(pid_file)],
              "environment": {"LLM_BENCH_FIXTURE_VALUE": "checked"}}

    async def check():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(RuntimeError, match="benchmark failed"):
                async with server_lifecycle(session, server, tmp_path, external=False, startup_timeout=5):
                    os.kill(int(pid_file.read_text()), 0)
                    raise RuntimeError("benchmark failed")
    asyncio.run(check())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert (tmp_path / "server.log").exists()
    assert (tmp_path / "pid.env").read_text().splitlines() == ["checked", str(helper)]


def test_unmatched_plots_require_explicit_label_and_preserve_failed_cells(tmp_path):
    pytest.importorskip("aiohttp")
    pytest.importorskip("matplotlib")
    from benchmarks.llm_serving.runner import atomic_json, run
    from benchmarks.llm_serving.smoke import fixture_config, mock_server
    from benchmarks.llm_serving.plot import load_points, plot

    async def check():
        paths = []
        for engine in ("vllm", "sglang"):
            async with mock_server() as (url, _):
                await run(fixture_config(engine, url), workload(), tmp_path / engine,
                          concurrencies=(1, 2), external=True, interval=.01)
                paths.append(tmp_path / engine / "report.json")
        return paths
    paths = asyncio.run(check())
    altered = json.loads(paths[1].read_text())
    altered["servers"][0]["configuration"]["weight_format"] = "different"
    altered["servers"][0]["points"][0]["disposition"] = "oom"
    atomic_json(paths[1], altered)
    with pytest.raises(ValueError, match="differs"):
        load_points(paths)
    plot(paths, tmp_path / "plots", allow_unmatched=True, latency_stat="mean")
    svg = (tmp_path / "plots" / "throughput-latency.svg").read_text()
    assert "SYNTHETIC PROTOCOL TEST" in svg
    assert "Unmatched formats" in svg
    assert "TTFT mean" in svg
    coverage = json.loads((tmp_path / "plots" / "plot-coverage.json").read_text())
    assert coverage["servers"][1]["points"][0]["disposition"] == "oom"
