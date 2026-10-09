# Benchmark vLLM, SGLang and llama.cpp with the same load

[Documentation](../README.md) · [Harness](../../benchmarks/llm_serving/README.md) · [Unified engine plan](../plan/unified-llm-engine.md)

The producer-independent [serving harness](../../benchmarks/llm_serving) supplies
an immutable request workload, three explicit server adapters, client timing,
raw records, telemetry and plots. It implements the baseline harness portion
of BENCH-01/04 and LLM-01. It does not implement Tensor batching or certify model
quality. The [L40S pilot](../research/llm-serving-l40s-pilot.md) records a real
three-engine Qwen3-0.6B run. The [bounded 35B stress test](../research/llm-serving-l40s-stress.md)
retains real 32K/16K measurements at concurrency 1, 2 and 4, including a converted
GGUF arm. Larger sweeps, numerical quality and Tensor serving comparisons remain open.

The native protocols were audited against upstream sources on **2026-10-09**:
[vLLM completion fields](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py),
[SGLang native generation](https://docs.sglang.io/docs/basic_usage/native_api),
and [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
Their installed revisions must be pinned and tested. Older revisions may lack
token-ID streaming or tokenization endpoints; the harness reports that failure
rather than counting text words or silently changing the request protocol.

## Set up the independent client and engines

```sh
uv venv --python 3.12 build/llm-bench-client
uv pip install --python build/llm-bench-client/bin/python -r benchmarks/llm_serving/requirements.txt
```

Windows uses `build/llm-bench-client/Scripts/python.exe`. Run modules from the
repository root; these benchmark modules are not part of the core runtime wheel.
Keep the engine stacks in separate environments so their Torch/CUDA dependencies
do not replace Tensor's producer stack. For example, after choosing tested pins:

```sh
uv venv --python 3.12 build/vllm-env
uv pip install --python build/vllm-env/bin/python 'vllm==VLLM_VERSION'
uv venv --python 3.12 build/sglang-env
uv pip install --python build/sglang-env/bin/python 'sglang[all]==SGLANG_VERSION'
uv pip install --python build/sglang-env/bin/python ninja
git clone https://github.com/ggml-org/llama.cpp build/llama.cpp
git -C build/llama.cpp checkout LLAMA_CPP_COMMIT
cmake -S build/llama.cpp -B build/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/llama.cpp/build --config Release --target llama-server -j
```

The version/commit tokens above are placeholders, not recommended releases.
Retain installed dependency locks, container identities if used, llama.cpp build
flags and startup logs. Audit architecture, dtype and kernel support before
downloading large checkpoints. This runbook does not install or qualify an engine
merely by recording its name.

## Freeze requests and tokenizer identity

Start with a small pilot using a pinned tokenizer revision:

```sh
build/llm-bench-client/bin/python -m benchmarks.llm_serving prepare \
  --tokenizer Qwen/Qwen3.5-35B-A3B-FP8 --revision CHECKPOINT_COMMIT \
  --input-lengths 512 --output-tokens 128 --requests 128 --warmup 1 \
  --out build/llm-serving/pilot.json
```

The client downloads tokenizer files, not model weights. It samples ordinary
token IDs, excludes special IDs and records the tokenizer vocabulary hash,
revision, seed, lengths and full prompt IDs. No chat template or BOS insertion
is applied. A text/ID probe must agree with every server before measurement;
the server also must report the expected full prompt count for every request.
The probe checks a sample, not the entire vocabulary mapping. Pin the original
tokenizer and verify GGUF conversion independently for full tokenizer agreement.

`--input-lengths 128 512 2048` creates a deterministic heterogeneous workload.
The synthetic prompts and greedy forced-length output are systems stress cases;
they are not representative language-quality tests. Outputs may contain repeated
EOS-like tokens because EOS termination is disabled. Use a separately validated
real request corpus for quality and representative application claims. Existing
corpora can be converted to the documented workload schema with exact token IDs,
unique request IDs, output lengths and the same tokenizer provenance; calculate
the canonical hash with `benchmarks.llm_serving.workload.digest` and validate it.

For a local tokenizer-independent protocol fixture, `prepare --token-pool FILE`
accepts a JSON object with `token_ids` and `tokenizer` provenance. Real runs still
require `name`, `revision`, `probe_text` and `probe_token_ids` in that provenance.

The long-context workload is explicit:

```sh
build/llm-bench-client/bin/python -m benchmarks.llm_serving prepare \
  --tokenizer Qwen/Qwen3.5-35B-A3B-FP8 --revision CHECKPOINT_COMMIT \
  --input-lengths 32000 --output-tokens 16000 --requests 1024 --warmup 1 \
  --out build/llm-serving/32k-16k.json
```

1024 requests at 16K output means over 16 million generated tokens **per point**.
A three-engine, ten-concurrency sweep is substantial GPU work. Preflight the
pilot and memory first, and choose the request count/repeats deliberately. The
harness warns when fewer than two waves of requests make fill/drain effects
especially strong. Use more waves for a stable plateau and retain repetitions.

## Configure server capacity separately from client load

Copy [servers.example.json](../../benchmarks/llm_serving/servers.example.json)
to the run directory and replace every `REPLACE_*` field. Each record declares
the actual engine version, model revision, weight format, KV and recurrent-state
precision, tokenizer identity, hardware, CPU offloading, prefix-cache and speculative policies, exact
argument array, and any local GPU IDs to monitor. Credentials belong in an
environment variable selected by `--api-key-env`; their values are not retained.
Do not put credentials in startup arguments or logs.

The example initially caps resident execution at four requests. It is a starting
configuration, not a tuned baseline or a guarantee the checkpoint fits. Tune
vLLM's sequence/token/memory budgets and SGLang's running-request/prefill/memory
budgets against actual allocations. Record each variant under a unique `name`.
Client concurrency may greatly exceed those resident limits and produce queueing.

SGLang 0.5.9 rejects requests whose input plus output budget equals its context
limit. Reserve more than 48,000 tokens for the 32K/16K shape; the example uses
49,152. Keep the request lengths fixed. Its `--language-only` flag configures
encoder disaggregation and requires encoder URLs; it is not a text-only loading
shortcut. This version also checks Torch 2.9.1's multimodal initialization against
CuDNN 9.15 or newer; retain any dependency override with the tested environment.

llama.cpp's slot count, unified/non-unified KV policy, total context allocation
and effective per-slot context must be audited for the pinned revision. The
example reserves `4 * 48000 = 192000` context tokens; it must not silently divide
a 48K total allocation into four 12K slots or shorten requests. Context shifting
and automatic fitting are disabled. Confirm GPU layer placement and no unintended
CPU expert/weight offloading in the retained startup log. KV offload flag names
can refer to placement **on the GPU**; record actual residency rather than
inferring RAM offloading from a flag's name.

The manifest's settings are user declarations. Available server info endpoints
and startup logs are retained for auditing actual resolved settings; the harness
does not guarantee all declarations match the engine's effective configuration.
GPU sampling runs on the client host only when `gpu_indices` is supplied, so
remote servers need their own co-located telemetry collector. Raw `/metrics`
samples preserve running/waiting counts, cache occupancy and preemption/retraction
counters where the server exports them. Missing endpoints/metrics are explicit
telemetry gaps, not zero values or proof of no preemption.

A baseline record can instead declare `disposition: "unsupported"`, `"oom"` or
`"not-run"` with a `reason`; it remains in coverage without making requests.
The three engines need not accept the same weight container. Native FP8 versus
a converted GGUF requires separately retained conversion/quality evidence and
format labels. Do not report such an overlay as a controlled same-weight speedup.

## Run engines independently

```sh
build/llm-bench-client/bin/python -m benchmarks.llm_serving run \
  --servers build/llm-serving/servers.json --workload build/llm-serving/pilot.json \
  --concurrency 1 2 4 8 16 32 --repeats 3 --out build/llm-serving/pilot-run
```

Managed runs start one server, wait for health, check tokenization, retain its
info/logs, warm up and measure each point, then terminate only the process group
they started before launching the next engine. Commands are argv arrays, not
shell expressions. An already responding endpoint is rejected rather than
mistaken for the newly launched server. Stop unrelated GPU jobs yourself.
The executable's directory is prepended to the child `PATH`, making that
environment's JIT tools available without activating it in the client shell.
An optional `environment` object supplies child variable overrides, such as
`CUDA_HOME` and `MAX_JOBS`; values are redacted from public report configuration.
Record nonsecret build paths/toolchain settings separately for reproduction.

For a server you started independently, select exactly one engine:

```sh
build/llm-bench-client/bin/python -m benchmarks.llm_serving run \
  --servers build/llm-serving/servers.json --engines vllm --external \
  --workload build/llm-serving/32k-16k.json \
  --concurrency 1 2 4 8 16 32 64 128 256 512 \
  --ttft-slo 120 --tpot-slo 0.1 --out build/llm-serving/vllm-stress
```

Repeat independently for SGLang and llama.cpp with the identical workload file.
These example SLOs are illustrative; freeze your own latency targets before
ranking goodput. A request is successful only with a terminal stream and exact
input/output token counts, without reported truncation. Short output, protocol
errors, OOM and timeouts remain failures. There are no automatic retries or
precision/context fallbacks. An incomplete run exits nonzero and retains its
records; interrupted runs retain partial request files and an interrupted report.
After a failed measured point, later points for that server are marked `not-run`
to avoid measuring against requests that might still be executing after a timeout.
Managed runs clean up that engine and proceed to the next baseline. `--engines`
validates only selected records' launch details, so other records can retain
unfilled placeholders during the pilot.

## Arrival and timing semantics

The default is a finite **closed-loop** concurrency sweep: each client issues
its next request after its prior request completes. All engines see the same
ordered prompts, output budgets and maximum outstanding request count. Their
actual arrival timestamps differ because completion rates differ. This measures
the whole finite run, including initial fill and final drain; it does not claim
an automatically detected steady-state window.

Adding `--request-rate RATE` during preparation stores a deterministic Poisson
arrival trace. Every engine replays identical offered timestamps. The concurrency
cap can queue requests **in the client**; both that delay and HTTP service latency
are retained. TTFT and end-to-end latency include client admission delay in this
mode; `service_ttft_seconds` begins at HTTP dispatch. Choose a rate and duration
appropriate to the load rather than silently changing it per engine.

All clocks are client monotonic clocks. Counts come from server usage or generated
token IDs, never words, characters or event counts. TTFT ends at first observed
generated output; TPOT is `(last generated-token event - first) / (output tokens - 1)`.
It is undefined for one output token. Stream events may contain multiple tokens,
especially with speculation, so their arrival time is shared. Chunk gaps and
gaps between individually streamed tokens are retained separately; neither is
misrepresented as an exact per-token timing for coalesced output.
Every call opens a fresh TCP connection under the same policy for all engines.
This avoids a pooled-socket race observed with llama.cpp's keep-alive limit
without hiding failed calls behind retries. Connection establishment is inside
the client timing boundary, including TLS for remote HTTPS endpoints.

Output throughput counts successful generated tokens divided by the entire
measured point duration, including failed-call time. Partial failed output is
retained separately. Goodput counts only successful requests meeting the declared
TTFT/TPOT limits. Input token throughput and total input+output throughput use
that same whole-run denominator: **input throughput is not isolated prefill
speed**. Loading/startup/warmup are outside the measured interval and separately
retained. Network, scheduling, sampling and streaming are inside it.

## Retained records and plots

Each run retains the exact workload, a versioned `report.json`, declared server
configuration, source hashes, server information and startup time/log. Each
point retains request JSONL with timing, cumulative token milestones, finish
metadata/errors and stream hashes, separate warmup records, and telemetry JSONL.
Plots use completed length/protocol-qualified points; all failed/unsupported
points remain in `plot-coverage.json`. Numerical model quality is a separate gate.

```sh
build/llm-bench-client/bin/python -m benchmarks.llm_serving plot \
  build/llm-serving/vllm-stress/report.json \
  build/llm-serving/sglang-stress/report.json \
  build/llm-serving/llama-stress/report.json \
  --latency-stat mean --allow-unmatched --out build/llm-serving/stress-plots
```

The plots include output throughput, TTFT, TPOT, request latency, input and total
token throughput, SLO goodput, sampled peak GPU memory and a latency/throughput
scatter. Latencies default to p95; `--latency-stat mean` gives mean curves like
[Netra's article](https://netraruntime.com/blog/netra-runtime-is-up-to-4x-faster-than-vllm),
which reports means. The article describes steady-state measurement; this
harness currently measures the complete finite replay, so that boundary differs.
Goodput equals successful output throughput when no SLO limits are supplied.
Repeated-point error bars show standard deviation across repetitions. An observed
frontier is drawn only for matched formats/configurations. Matching workload
hashes and comparison groups are mandatory. Different weight/cache/state formats,
checkpoint revisions, cache/speculative policies, hardware, CPU offloading or
SLO limits require `--allow-unmatched` and a
prominent exploratory label; there is no automatic speedup claim. A shared label
does not replace independently verified numerical comparability.
Declared hardware and available local GPU model/count/VRAM/driver snapshots
participate in matching. Remote hardware declarations need independent auditing.

```sh
build/llm-bench-client/bin/python -m benchmarks.llm_serving.smoke --out build/llm-serving-smoke
```

The local smoke command exercises all three native wire formats, real concurrent
HTTP streams, result retention and SVG/PNG plotting against mock servers. Its
plots carry **SYNTHETIC PROTOCOL TEST — NOT GPU PERFORMANCE**. Passing this check
does not establish that a real engine version runs Qwen or matches Netra's plot.
Their article does not publish the exact launch commands/request corpus, so this
harness reproduces workload shapes and comparison plot types, not an exact
replication of their undisclosed setup.
It names **Qwen3.6-35B-A3B on a single H200**; the repository's Qwen3.5/L40S
targets are a separate comparison. Use matching pinned model/hardware identities
when testing any claim about that article's absolute results.
