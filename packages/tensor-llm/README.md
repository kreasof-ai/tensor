# Tensor LLM

Standalone CUDA and WebGPU inference for LiquidAI LFM2.5 GGUF files, with
independent single-sequence requests sharing model resources.
The optional `tensor-llm` wheel adds GGUF parsing, byte-level BPE tokenization,
packed-weight projections, the hybrid convolution/attention forward plan, and
text generation. The installed consumer needs Tensor, NumPy, regex and the GPU
driver; WebGPU additionally needs Tensor's `webgpu` extra. It does not import
Torch, TileLang, TVM, Triton, GGML or llama.cpp.

Supported checkpoint profiles are F16, Q4_0 and Q4_K_M. Q4_K_M mixes Q4_K and
Q6_K matrices; the Q4_0 checkpoint also uses Q6_K for its tied embedding/output.
Weights stay packed on the GPU and are decoded in the projection kernels.
The reader additionally implements F32 and Q8_0; other quantizations and model
architectures are rejected. The kernels currently require 64-dimensional heads
and a three-tap short convolution. This is a bounded demonstration, rather than
a general GGUF inference engine.

The [unified engine roadmap](../../docs/plan/unified-llm-engine.md) proposes
extending this package with true batching, continuous scheduling and Qwen
FP8/MoE adapters. Shared model resources and independent request handles are
implemented for LFM2; its request handles execute one sequence per call. The
experimental Qwen adapter below implements native batched execution within the
same package.

## Source layout

The public classes stay available through `from tensor_llm import ...`. Internal
modules are grouped by model and responsibility:

```text
tensor_llm/
├── __init__.py, __main__.py, cli.py
├── common/
│   ├── artifacts.py              # Shared logical kernel identities
│   ├── gguf.py                   # GGUF parsing and packed weight utilities
│   └── tokenizer.py              # GGUF byte-level BPE
├── speculative/
│   ├── acceptance.py             # Model-independent greedy prefix acceptance
│   └── lookup.py                 # Bounded output-history proposals
├── lfm2/
│   ├── config.py, model.py, provenance.py
│   └── kernels/
│       └── baseline.py, cuda.py, webgpu.py
└── qwen35/
    ├── checkpoint.py             # Safetensors metadata and weight loading
    ├── artifacts.py              # Shared logical kernel requirements
    ├── decode.py, prefill.py, pipeline.py
    ├── kernels/                  # Producer factories shared by execution phases
    │   ├── decode.py, prefill.py, fp8_kv.py, matmul.py, mtp.py
    │   └── speculative.py, speculative_attention.py, speculative_linear.py
    ├── mtp/
    │   └── decode.py, prefill.py
    └── speculative/
        └── verifier.py, engine.py, attention.py, linear.py
```

Kernel factories import the compiler only when invoked by a producer. Executors,
checkpoint utilities and speculative helpers remain usable in the consumer
environment. `qwen35/speculative/attention.py` and `linear.py` install selected
kernel graphs; their producer algorithms live under `qwen35/kernels/`.

Internal module paths changed with this organization. Producers and source-hash
tracking use the new paths; regenerate inference bundles because their source
identities changed. Retained benchmark reports and frozen executed sources keep
their original provenance.

## Experimental native Qwen CUDA execution

`Qwen35Batch` and `Qwen35Prefill` execute the pinned
Qwen3.5-35B-A3B-FP8 text checkpoint using Tensor CUDA artifacts. One resident
weight allocation serves eight slots with independent FP32 GDN state and
BF16 or explicitly selected FP8 KV. The checkpoint loader streams safetensors
shards directly to the device. The forward path uses native FP8 tensor-core
projections, chunked recurrent scans, causal attention, greedy GPU argmax and
captured decode graphs. Runtime imports require neither Torch nor a producer
compiler. The local HTTP benchmark adapter additionally uses `aiohttp` and the
checkpoint's Rust `tokenizers` tokenizer.

This implementation is experimental. Whole-model numerical qualification
remains open; the FP8 KV calibration failed the retained 3% logit-error gate.
Kernel tests and short decode timings do not establish the 600 tok/s C8 target.
The serving adapter reports these qualification flags explicitly.

Produce the measured L40S profile from the repository root with the producer
environment on `PYTHONPATH=.:src:packages/tensor-llm/src`:

```bash
python -m benchmarks.qwen35.profile_producer --checkpoint build/models/qwen3.5-35b-a3b-fp8 --out build/qwen35-profile --kv-dtype fp8
python -m benchmarks.qwen35.prefill_producer --checkpoint build/models/qwen3.5-35b-a3b-fp8 --out build/qwen35-prefill --chunk 512 --block-m 64 --kv-dtype fp8 --packed-kv
python -m benchmarks.qwen35.server --checkpoint build/models/qwen3.5-35b-a3b-fp8 --bundle build/qwen35-profile/decoder --prefill-bundle build/qwen35-prefill --port 8013
```

Use `bfloat16` and omit `--packed-kv` for the separate BF16 cache reference.
The supplied producers default to `sm_89` and the pinned local NVRTC bootstrap.
The decoder/profile and prefill producers accept `--target sm_90` for Hopper;
MTP and verification producers inherit the parent bundle's target. Set
`TENSOR_NVRTC_HOME` to use the same pinned NVRTC bundle at another location.
The fixed slot scheduler accepts greedy token-ID requests and performs chunked
prefill and batched decode. Paged cache allocation, batch-1 optimization and
general continuous scheduling remain roadmap work.

The bounded Modal H200 runner prepares the official checkpoint and CUDA
artifacts on CPU, then requests one H200 for primitive checks, a two-row
verification-versus-serial check at 32K context, and the common streaming-client
replay at C8. It measures pure AR and MTP plus output lookup separately. Use the
retained workload from the L40S replay at
`docs/research/data/qwen35-native-h200/workload.json`, then run:

```bash
modal run benchmarks/qwen35/modal_h200.py --out build/qwen35-h200
```

Set `TENSOR_H200_WORKLOAD` to choose another local workload JSON. To repeat a
measurement with already prepared artifacts, pass `--prepared-file` pointing
to the prior local `prepared.json`; consumer implementation checks still apply.

Weights and artifacts persist in the `tensor-qwen35-h200` Modal volume. The
local output records the remote run directory for downloading raw reports with
`modal volume get`. CPU preparation has a one-hour timeout; GPU qualification
and both finite replays share a 30-minute timeout. The initial Hopper profile
uses ordinary pointer arguments and warp MMA: automatic TMA, WGMMA and warp
specialization are disabled for exact `sm_90` builds to fit the current runtime
ABI. These compatibility settings are retained in each artifact's compiler
identity. Whole-model qualification remains a separate gate.
The [first H200 report](../../docs/research/qwen35-native-h200.md) records
588.2 tok/s AR and 911.5 tok/s MTP plus output lookup over complete C8 32K/16K
client replays. Numerical qualification failed; these are experimental rates.

`Qwen35Checkpoint(path, branch='mtp')` also validates the official embedded
MTP weights. The experimental `Qwen35MTP(target, bundle)` drafter borrows the
target's embedding and output head, loads its own original FP8/BF16 weights,
and retains private KV. Produce its bundle with:

```bash
python -m benchmarks.qwen35.mtp_producer --checkpoint build/models/qwen3.5-35b-a3b-fp8 --decoder build/qwen35-profile/decoder --out build/qwen35-mtp
```

`draft(next_token_ids, target_final_normalized_hidden)` advances consecutive
draft positions from zero. Initialize the shifted prompt prefix before using
long-context drafts. `benchmarks.qwen35.mtp_calibrate.calibrate` resets the
target/drafter and measures short teacher-forced and autoregressive one-token
agreement against serial target evaluation. This is a drafting diagnostic.
MTP measurements retain separate labels from pure AR.

`Qwen35MTPPrefill` in `tensor_llm.qwen35.mtp.prefill` supports native chunked
initialization using `forward(shifted_token_ids, flat_target_hidden, lengths)`.
Build its artifacts with `benchmarks.qwen35.mtp_prefill_producer --checkpoint
... --prefill ... --out ...`. The `benchmarks.qwen35.mtp_benchmark` command
reproduces the long-prefix and three-proposal chain agreement diagnostics with
explicit target/draft/prefill bundles and a frozen workload. See the retained
[native Qwen report](../../docs/research/qwen35-native-l40s.md) for measured
agreement, cost and quality scope.

`Qwen35Verifier` now verifies several tokens together and retains FP32 GDN and
convolution snapshots for every possible accepted boundary. `commit(counts)`
restores each request independently; future KV rows are masked by the committed
position and overwritten on the next pass. `Qwen35Speculative` shares this target
with its MTP drafter, repairs draft KV from verified target hidden states and
returns the accepted tokens plus a corrected/bonus token. Optional bounded
output-history lookup proposes repeating continuations; target verification
still checks every output. Its results are labeled separately from pure MTP.

The measured L40S candidate verifies eight input tokens per request. Produce
its small-chunk verification/repair artifacts after the target and MTP bundles:

```bash
python -m benchmarks.qwen35.prefill_producer --checkpoint build/models/qwen3.5-35b-a3b-fp8 --out build/qwen35-prefill8 --chunk 8 --block-m 64 --kv-dtype fp8 --packed-kv
python -m benchmarks.qwen35.spec_producer --prefill build/qwen35-prefill8 --out build/qwen35-verify8
python -m benchmarks.qwen35.spec_attention_producer --prefill build/qwen35-verify8 --out build/qwen35-verify8-attention --key-rows 32
python -m benchmarks.qwen35.spec_linear_producer --prefill build/qwen35-verify8-attention --out build/qwen35-verify8-selected --expert-block-m 32
python -m benchmarks.qwen35.mtp_prefill_producer --checkpoint build/models/qwen3.5-35b-a3b-fp8 --prefill build/qwen35-prefill8 --out build/qwen35-repair8
python -m benchmarks.qwen35.spec_attention_producer --prefill build/qwen35-repair8 --out build/qwen35-repair8-selected --key-rows 32
python -m benchmarks.qwen35.mtp_prefill_producer --checkpoint build/models/qwen3.5-35b-a3b-fp8 --prefill build/qwen35-prefill --out build/qwen35-mtp-prefill
```

Run `benchmarks.qwen35.spec_run` with explicit `--decoder`, `--draft`, `--prefill`,
`--draft-prefill`, `--verify`, `--repair`, `--workload`, `--checkpoint` and `--out`
paths. `--output-lookup` enables the measured hybrid profile; omitting it selects
pure MTP. The native cohort includes prefix processing, executor setup and all
draft/verification/rollback/repair work, and records NVML telemetry and exact
token IDs. It excludes HTTP serialization.

For the common streaming client harness, `benchmarks.qwen35.server` accepts
`--draft-bundle`, `--draft-prefill-bundle`, `--verify-bundle`, `--repair-bundle`
and `--output-lookup` alongside its ordinary arguments. This mode admits requests
during prefill, then retains the cohort until completion. It closes large prefix
graphs before allocating verification snapshots, keeping the measured C8 profile
resident on the L40S. Accepted token chunks stream with exact cumulative counts.
General continuous admission during speculative decode remains roadmap work.

## Independent requests with shared weights

`LFM2` owns one weight/kernel allocation, shared executor scratch buffers and a
default request. Opt into additional handles with an explicit capacity:

```python
with LFM2(model_path, bundle_path, device, context=512, max_requests=8) as model:
    with model.new_request() as other:
        logits_a = model.forward(prompt_a)
        logits_b = other.forward(prompt_b)
        next_a = model.forward([token_a])
        next_b = other.forward([token_b])
```

Each handle has its own position, convolution history, KV cache, logits and
bound plans/graphs. Interleave calls on the model's creating thread and device
stream. Scratch buffers are shared and calls execute serially; this API does
not perform batched launches or continuous scheduling. Concurrent use from
another thread is rejected before request state changes. Avoid manipulating
the exposed execution plans directly while other requests are in use.

The capacity includes the default request and defaults to one. `new_request`
can select a smaller `context` or choose `graphs=False`; its cache allocation
still uses the bundle's compiled capacity. Resetting or closing a sibling does
not reset the others. Closing a sibling frees its buffers and graphs and makes
that capacity available to a fresh handle; stale handles remain invalid.
Closing the owner closes all handles. Close the owner before its device session.

`model.weight_bytes` counts packed weights; `shared_bytes` includes weights and
executor buffers once; `allocated_bytes` includes all live request buffers.
`request.private_bytes` counts only that handle's buffers. These counters exclude
driver-owned graph storage: use device telemetry for the complete GPU footprint.
Additional handles retain additional graph captures, so the request count is an
explicit allocation limit, not a guarantee that every capacity fits a device.

**Rebuild existing inference bundles after this refactor.** Implementation
fingerprints continue to reject bundles produced for the previous runner.
Re-run the matching producer command; unchanged kernel artifacts can be reused
while the producer binds the manifest to the installed implementation.

## Smaller local iteration workload

Use LiquidAI LFM2.5-230M for the Windows RX 6700 XT Vulkan work. The downloader
pins revision `03502067c64ce32ac4fe87b0cec0310a1a13d3e9` and verifies file sizes
and SHA256 checksums, independently of the retained 2.6B CUDA demonstration:

```powershell
uv sync --locked --extra webgpu
.venv/Scripts/python.exe benchmarks/lfm2/download.py --model-size 230M --formats F16 Q4_0
.venv/Scripts/python.exe -m tensor_llm inspect --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf
.venv/Scripts/python.exe -m tensor_llm inspect --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf
```

The F16 checkpoint is 461,884,256 bytes; Q4_0 is 149,080,928 bytes. Start local
iterations at 512 total prompt/decode tokens, increasing context after correctness
and reset checks. The model has width 1,024, FFN width 2,560, 14 layers, 16 query
heads, eight KV heads, head dimension 64 and vocabulary size 65,536. It fits the
existing architecture contract. F16 and Q4_0 generation, cache/reset correctness
and matched llama.cpp Vulkan timings were validated on this Radeon. The retained
2.6B CUDA timings do not establish 230M performance.
Single-user chat follows the GGUF's released generation prefix: 230M starts at
`assistant\n`, while the retained 2.6B template adds `<think>`. Other generation
prefix templates are rejected for chat; raw completion remains available.
The 230M Q4_0 file contains 82 Q4_0 matrices and one Q6_K tied embedding/output
matrix; the Vulkan path runs both decoders on packed u32 buffers. The F16
embedding is exactly 128 MiB and fits the current default WebGPU binding limit.

The [packed decode repeat](../../docs/research/lfm2-webgpu-decode-push.md) measures
packed floating dots and residual fusion on this GPU: 2.6B Q4_0 reaches
140 tok/s against llama.cpp's 169 at prefix 128, and 230M Q4_0 improves by 9.5%
in the same fresh comparison. Prefill is unchanged. Rebuild existing bundles
after updating the package because implementation fingerprints are enforced.

The later [2.6B QAD Q4_0 revisit](../../docs/research/lfm2-2.6b-q4_0-revisit.md)
reaches 146 tok/s decode versus native 169–170 at prefix 128. Ordered readback
adds 5% in an identical-shader ablation; the opt-in `prefill_chunked` profile
with rows 1/32/128 improves long-prompt prefill by 48–49%, to 346/339 tok/s.
Packed Q4 prefill still trails native substantially. Its Q6_K output matrix
requires `max_buffer_size=268435456` on this device.

The subsequent [packed parity search](../../docs/research/lfm2-2.6b-q4_0-parity.md)
reaches 164 tok/s decode versus native 170, and 522/517 tok/s prefill for
128/384 tokens. Build with `--webgpu-profile quant_searched --prefill-chunks 32 128`
to select replayed Q4/Q6 decode, fused attention and packed prefill outer products
for the measured 2.6B shapes. Other shapes retain existing schedules. The
256 MiB buffer limit is still required. Decode parity remains unproven;
the same-chunk prefill gain is 51%, with all 19 independent logit fixtures passing.

The [1K prefill experiment](../../docs/research/lfm2-2.6b-prefill-1k.md) adds the
opt-in `--webgpu-profile prefill_mixed --prefill-chunks 32 128` profile for this
2.6B QAD Q4_0 checkpoint, reaching 1,058/1,047 tok/s at 128/384 prompt tokens,
about twice the packed control. It combines searched short F16 accumulation tiles,
two-component Q8 activation dots, parallel RMS and a guarded final-layer suffix
plan. Its approximate prefill arithmetic passes the same independent logit
gates; decode projection WGSL stays identical to `quant_searched`. Weights
remain packed, and the 256 MiB buffer limit is required. See the report for
completed-forward throughput, native comparisons and the validated scope.

The 230M Q4_0 download uses the ordinary post-training-quantized checkpoint.
The 2.6B reports above use the distinct QAD Q4_0 checkpoint; do not interchange
these variants in matched numerical or performance comparisons. Q4_K_M can additionally
be downloaded with `--formats Q4_K_M`. Omitting `--model-size` preserves the
existing 2.6B download workflow and its default output directory.

Build and run the smaller Vulkan demonstration from the repository root:

```powershell
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --provider webgpu --webgpu-profile subgroup --context 512 --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --out build/lfm2-230m-q4_0-webgpu
.venv/Scripts/python.exe -m tensor_llm generate --provider webgpu --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --prompt 'What is 2 + 2?' --max-tokens 96
```

Use `F16` and `lfm2-230m-f16-webgpu` for the F16 bundle. The standard WebGPU profile uses
32-token prefill chunks, FP16 projection operands with FP32 accumulation, FP32
decode projections/attention, FP16 K/V caches and FP32 convolution history.
One prepared compute pass submits the per-layer dispatches per chunk, caching
bindings while encoding fresh command buffers. The default producer profile
`portable` requires `shader-f16`; the command above selects `subgroup` for this
Radeon and additionally requires the adapter's subgroup feature. Bundles record
and validate these requirements. Both retain the 32 KiB workgroup memory envelope.
Subgroup decode uses vectorized F16 and word-packed Q4 loads, segmented reductions,
coalesced attention scores, and shared-memory fallbacks for smaller subgroup sizes.
Attention skips cache products for inactive positions;
decode uses parallel RMS reductions; quantized decode additionally pairs
gate/up/SwiGLU projections. F16 decode keeps separate projections because the
paired variant was slower on this adapter.
Register tiles keep prefill accumulators private across the reduction. Queue
ordering replaces per-chunk host fences. WebGPU generation uses GPU greedy
argmax, feeds the selected token directly into decode, and reads one int32 per
step; `forward()` still returns full FP32 logits. Use `gpu_greedy=False` in
`generate()` to compare with host greedy sampling.
Prefill uses compiler-owned four-wide dot products for F16 and the large-K Q4
FFN-down projection. Installed wheels can include an optional native prepared-plan
encoder; it batches the dispatch loop into one C call with native validation
error capture. Ordinary Python wheels retain the Python encoder fallback.
Context 512 is the measured profile. See [the native submission report](../../docs/research/lfm2-230m-native-submission.md)
for numerical checks, remaining performance gaps and benchmark commands.

The experimental `prefill_outer` profile adds selected native-F16 projection
schedules and supports multiple prefill row sizes. Build the measured 230M
adaptive bundle with `--webgpu-profile prefill_outer --prefill-chunks 32 128`
and `--context 512`; the ordinary runner chooses a compiled row size from the
remaining prompt length. The profile retains the searched fused FFN and decode
projections. On RX 6700 XT it improves completed prefill by about 16% at 32
tokens and 2.19× at 128/384 tokens; larger native llama.cpp chunks still win.
See the [adaptive prefill comparison](../../docs/research/lfm2-prefill-chase.md)
for accuracy, chunk ablations and reproduction. Other checkpoints, quantized
weights and larger contexts have not been remeasured for this profile.

For a compiler-free Windows consumer, build the optional encoder on a producer
with MSVC C build tools. It creates a CPython 3.12 Windows x64 wheel; the
consumer needs no C or shader producer compiler. Omit `TENSOR_BUILD_WEBGPU_NATIVE`
to build an ordinary Python wheel with the fallback encoder.

```powershell
$env:TENSOR_BUILD_WEBGPU_NATIVE='1'
uv build --sdist --out-dir build/lfm2-230m-native-wheels
uv build --wheel build/lfm2-230m-native-wheels/tensor_workspace-0.1.0.tar.gz --out-dir build/lfm2-230m-native-wheels
uv build --wheel packages/tensor-llm --out-dir build/lfm2-230m-native-wheels
Remove-Item Env:TENSOR_BUILD_WEBGPU_NATIVE
uv venv --python 3.12 build/lfm2-230m-native-consumer
uv pip install --python build/lfm2-230m-native-consumer/Scripts/python.exe 'build/lfm2-230m-native-wheels/tensor_workspace-0.1.0-cp312-cp312-win_amd64.whl[webgpu]' build/lfm2-230m-native-wheels/tensor_llm-0.1.0-py3-none-any.whl
build/lfm2-230m-native-consumer/Scripts/python.exe -m tensor_llm generate --provider webgpu --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --prompt 'What is 2 + 2?' --max-tokens 96
```

## CUDA 2.6B: build once

Run these commands from the Tensor repository in the producer environment:

```sh
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
uv run --locked python benchmarks/lfm2/download.py --formats Q4_0
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" uv run --no-sync python benchmarks/lfm2/producer.py \
  --model build/lfm2-models/LFM2.5-2.6B-Q4_0.gguf \
  --out build/lfm2-q4_0 --context 8448 --target sm_86
uv build --wheel --out-dir build/lfm2-wheels
uv build --wheel packages/tensor-llm --out-dir build/lfm2-wheels
```

Choose the target architecture for your GPU. The shipped demonstration measures
SM86 on A10G; it does not establish execution on other GPU architectures.
NVRTC/TileLang are needed only by the producer. The consumer requires the GGUF,
`inference.json`, its `artifacts/*.tbin` files and matching installed wheels.
Generated source and compiler caches can be omitted. The manifest checks kernel
checksums, architecture, coverage and implementation fingerprints before loading.
Rebuild the plan after implementation changes.

### Optimized CUDA profile

Build with `--cuda-profile optimized --prefill-chunks 32 128` for the measured
LFM2.5-2.6B F16, ordinary Q4_0 and Q4_K_M schedules. This profile selects tuned
tensor-core prefill tiles, packed pair loaders, packed-word Q4/Q6 decode,
paired gate/up/SwiGLU, residual/RMS fusion, adaptive prefill and GPU greedy
sampling. Decode shares K/V across query heads from 4K context onward.
The guarded final attention/conv/conv suffix stores every K/V row before
cropping queries and the final two convolutions to eight rows. Intermediate
prefill chunks stop after their persistent state writes.

Download all three formats, then build a bundle for the format you will run:

```sh
uv run --no-sync python benchmarks/lfm2/download.py --formats F16 Q4_0 Q4_K_M
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" uv run --no-sync python benchmarks/lfm2/producer.py \
  --model build/lfm2-models/LFM2.5-2.6B-Q4_K_M.gguf \
  --out build/lfm2-q4km-cuda --context 8448 --target sm_86 \
  --cuda-profile optimized --prefill-chunks 32 128
```

Use this model and bundle with the ordinary consumer commands below. Packed
decode retains FP32 activations and accumulation. Prefill retains FP16
tensor-core operands with FP32 accumulation; the final FFN is evaluated only
for the last row using FP32 decode arithmetic. The Vulkan approximate short-half
and Q8 activation prefill arithmetic is not used by this CUDA profile.
See the [three-format CUDA report](../../docs/research/lfm2-cuda-formats.md)
for matched before/after and llama.cpp timings, accuracy and reproduction on
A10G. The [initial Vulkan transfer](../../docs/research/lfm2-cuda-vulkan-transfer.md)
measured the separate QAD Q4_0 checkpoint; that format retains a correctness
regression check. Other GPUs and model shapes need their own measurements.
The producer default remains `--cuda-profile default`.

CUDA kernels are expressed in TileLang/TIRx, including packed dequantization,
warp reductions and grouped attention. They use the ordinary `tensor.build`
interface. The shared discovery engine is `tensor.compiler.search`; CUDA and
WebGPU provide their own schedule spaces. Measured SM86 settings live in
[`benchmarks/lfm2/profiles`](../../benchmarks/lfm2/profiles), outside this package.
Use `--schedule-profile PATH` to supply a target-bound profile when building.
The bundle records selected schedules and the profile SHA256; the consumer
executes compiled artifacts without importing discovery or compiler modules.
See the [compiler cleanup report](../../docs/research/lfm2-compiler-cleanup.md)
for the fresh comparison with the frozen native CUDA implementation.

## Run without a compiler

```sh
uv venv --python 3.12 build/lfm2-consumer
uv pip install --python build/lfm2-consumer/bin/python build/lfm2-wheels/*.whl
build/lfm2-consumer/bin/tensor-llm generate \
  --model build/lfm2-models/LFM2.5-2.6B-Q4_0.gguf --bundle build/lfm2-q4_0 \
  --prompt 'What is the capital of France?' --max-tokens 256
```

On Windows use `build/lfm2-consumer/Scripts/python.exe -m tensor_llm` and set
`TENSOR_NVRTC_HOME` using the shell's environment syntax when producing.
`tensor-llm inspect --model FILE.gguf` reports the model contract without a GPU.
The Python API is `from tensor_llm import LFM2` with an explicit `tensor.Device()`
and a context manager. `forward(token_ids)` returns last-token FP32 logits;
`reset()` clears convolution history and starts a new attention cache prefix.

The CUDA profile uses chunks up to 128 tokens, followed by single-token cached
decode. The initial profile has a total prompt-plus-decode limit of 8,448 tokens.
It uses FP16 attention caches, FP32 convolution history, FP16 tensor-core prefill
projections and FP32 decode projections/accumulation. Quantized activation math
therefore differs from llama.cpp's MMQ/MMVQ paths; logits are not bitwise equal.
The default CUDA profile uses CPU greedy argmax; the optimized profile samples
on the GPU and reads one int32 per step. Use `gpu_greedy=False` to compare with
host sampling. Generation uses a single-user LFM2 chat template, or a plain
completion with `--raw`. Multi-turn chat, tools, sampling distributions, batching,
CPU inference and a persistent megakernel are outside the supported profiles.
CUDA Graph replay submits the existing per-layer kernels; it is not one fused
GPU kernel. See the repository's `docs/research/lfm2-inference.md` for correctness,
latency and limitations.
