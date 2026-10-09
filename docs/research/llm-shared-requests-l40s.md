# Shared LFM2 requests on L40S

[Research](README.md) · [Unified engine plan](../plan/unified-llm-engine.md) · [Raw report](data/llm-shared-requests-l40s/real-report.json)

Measured **2026-10-09** on one NVIDIA L40S, `sm_89`. This qualifies the first
LLM-02 increment: independent request state sharing one model's packed weights,
loaded kernels and executor scratch. Calls execute serially on the model owner
thread and stream. These results do not establish batched throughput, Qwen
support or the planned 600 tok/s concurrency-8 target.

## Implementation and workload

`LFM2` retains its default `forward`, `reset` and `generate` entry points.
`max_requests=8` permits seven additional `new_request()` handles. Each owns
position/control, convolution history, KV cache, logits and bound plans/graphs.
Closing a sibling releases its private resources and allows replacement;
closing the owner closes every handle. Capacity, owner-thread and stale-handle
checks precede execution. Failed shared loading, private allocation or graph
capture releases successfully acquired resources without damaging live requests.

The real checkpoint is `LiquidAI/LFM2.5-230M-GGUF`, revision
`03502067c64ce32ac4fe87b0cec0310a1a13d3e9`, F16 file SHA-256
`4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`.
The CUDA optimized bundle compiles for context 768, capacity 896, rows 1/128,
and `sm_89`. Attention caches are F16 and convolution state is FP32.

Eight synthetic token prefixes have lengths 1, 2, 31, 32, 127, 128, 129 and 257.
Each continues with `[65]` then `[66, 67]`; execution order changes between
stages. All **24 logit arrays match independent model instances bitwise**.
Independent eager Torch operators check the full model's declared mixed
precision contract: maximum normalized error was **0.000973**, below the 0.01
gate. Reset/close/replacement checks preserve a live neighbor's continuation.
These are systems traces using real weights, not a semantic quality evaluation.

## Buffer sharing and singleton timing

| Allocation | Bytes |
|---|---:|
| Packed weights, once | 459,496,448 |
| Weights plus shared executor buffers | 469,902,852 |
| Private buffers per request | 11,337,736 |
| One request, total buffers | 481,240,588 |
| Eight requests, total buffers | 560,604,740 |

These counters exclude driver-owned CUDA graph storage and are not peak NVML
memory readings. The request limit bounds handle allocation; it does not promise
that a chosen checkpoint/context/graph configuration fits a device.

The frozen pre-refactor consumer and current consumer use **identical compiled
kernel images** and the same real checkpoint. Their manifests bind separately
to their own implementation fingerprints; fingerprint checks stay enabled.
Singleton timing uses alternating engine order, three warmups and nine measured
repetitions, with 64 forced decode calls per repetition. It includes completed
host logits and excludes loading, reset, tokenization and sampling.

| Prompt length | Before prefill (ms) | Shared prefill (ms) | Before decode (ms/token) | Shared decode (ms/token) |
|---:|---:|---:|---:|---:|
| 128 | 2.435 | 2.428 | 1.017 | 1.025 |
| 512 | 8.762 | 8.809 | 1.039 | 1.054 |

The observed decode cost is **0.8–1.4%** higher, including the copy-ordering fix
below. This is a short-context 230M comparison, not a general regression limit
for every supported checkpoint or a speedup claim. Raw paired samples remain
in the report. Full supported-checkpoint and physical WebGPU qualification remain
open before LLM-02 is fully accepted.

## Copy ordering and compiler-free replay

An initial installed replay intermittently disagreed with independent execution.
The ensuing audit found that `_write` did not order its default-stream copies
before nonblocking-stream graph launches. NVIDIA documents that pageable HtoD
copies may return after staging while device DMA remains incomplete.
[CUDA 12.9.1 synchronization behavior](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/api-sync-behavior.html)
defines this requirement. The runner now completes the default-stream transfer
before execution, including state writes during reset.

A deterministic regression defers copies until default-stream completion. The
frozen original write routine fails with wrong logits; the corrected routine
passes under both default and optimized profiles. Its
[before-failure log](data/llm-shared-requests-l40s/copy-ordering-before-failure.log)
is a controlled reproduction of the missing guarantee, not the initial replay's
raw log. The timing table above includes the corrected implementation.

The final wheels were installed into an isolated consumer containing exactly
Tensor, Tensor LLM, NumPy 2.5.3 and regex 2026.9.29. A fresh process ran from
`/tmp`, with compiler/framework imports blocked, and matched all 24 retained
GPU logit arrays bitwise. It also repeated reset/close/replacement checks.
The [consumer report](data/llm-shared-requests-l40s/consumer-report.json) and
[environment](data/llm-shared-requests-l40s/consumer-environment.json) are retained.

Validation passed 24 request tests (20 physical CUDA cases), eight legacy checks
(six physical CUDA cases), and 74 ordinary package/harness checks. The ordinary
suite reports 114 opt-in skips; they establish no additional hardware support.
Four wheels and four source archives passed metadata, source rebuild and
compiler-free distribution checks. Existing inference bundles require a producer
rebuild after the model implementation fingerprint changes.

## Reproduction and retained evidence

Prepare the pinned compiler environment and NVRTC bundle, then run:

```sh
python -m benchmarks.lfm2.download --model-size 230M --formats F16
python -m benchmarks.lfm2.producer \
  --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf \
  --out build/llm-shared-baseline/lfm2-230m \
  --context 768 --target sm_89 --cuda-profile optimized
python -m benchmarks.lfm2.shared_requests \
  --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf \
  --bundle build/llm-shared-baseline/lfm2-230m \
  --out build/llm-shared-baseline/real-validation
TENSOR_LFM2_CUDA=1 python -m pytest packages/tensor-llm/tests/test_requests.py
```

Use `--before-package` and `--before-bundle` to repeat the paired timing; both
source snapshots and source-bound manifests are retained. Extract the compiled
artifacts inside each bundle directory; the checkpoint is downloaded separately.
For installed replay, pass `--consumer --expected <qualified-report-directory>`
to the retained harness with absolute paths and run outside the checkout.

The [data directory](data/llm-shared-requests-l40s/) retains before/after consumer
sources, harness/tests, checkpoint provenance, manifests, compiled artifacts,
consumer wheels, all logit arrays, raw timing samples, test XML/logs and distribution
validation. [Checksums](data/llm-shared-requests-l40s/sha256.json) cover every file.
The next implementation stage is true batched LFM2 execution; Qwen FP8/MoE/GDN
support, serving, scheduler search and matched C8 stress baselines remain open.
