# LFM2.5-230M on Windows Radeon Vulkan

This records the initial implementation. The
[optimization report](lfm2-230m-vulkan-optimization.md) describes the current
runner and fresh before/after measurements; the evidence below retains the
original implementation fingerprints and timings.

The installed `tensor-llm` runner generates text on this machine through wgpu's
Vulkan backend, using both F16 and packed Q4_0/Q6_K weights. This is a working
small-model iteration baseline. It remains substantially slower than native
llama.cpp Vulkan: about 5.3 times slower in F16 decode and 9.1 times slower in Q4
decode. Prefill needs a separate matrix-multiply optimization effort.

## Recorded environment and scope

Measured October 1, 2026: Windows, Ryzen 5 5600, AMD Radeon RX 6700 XT 12 GiB,
AMD proprietary driver 26.6.2, Python 3.12.13, NumPy 2.5.3, wgpu 0.29.0.
The adapter reports `DiscreteGPU` and `Vulkan`; no software adapter was used.
The [F16 report](data/lfm2-230m-f16-vulkan.json) and
[Q4 report](data/lfm2-230m-q4_0-vulkan.json) retain adapter capabilities and limits,
all timing samples, forced token fixtures, validation metrics and generation.

The checkpoint is [LiquidAI LFM2.5-230M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF),
revision `03502067c64ce32ac4fe87b0cec0310a1a13d3e9`. The downloader pins exact
sizes and SHA256 checksums. F16 is 461,884,256 bytes; Q4_0 is 149,080,928 bytes.
The latter is the ordinary quantized checkpoint, not the separate QAD release.
Its tied embedding/output matrix is Q6_K. This report does not validate Q4_K_M
execution, 2.6B on Radeon, other GPUs, or the model's full 128K context.

This profile uses a 512-token sequence limit and compiled cache capacity 576,
14 hybrid layers, 1,024 hidden channels, 2,560 FFN channels, head dimension 64
and 65,536 vocabulary entries. Model buffers total 447.8 MiB for F16 and
149.5 MiB for Q4. These numbers exclude pipeline objects, uniforms, allocator
overhead, host copies and the native baseline's allocations.

## Implementation and correctness

The WebGPU producer compiles 31 shape/encoding specializations for one-token
decode and 32-token prefill. Quantized weights remain packed in u32 storage;
shaders extract their bytes and reconstruct half scales, including subnormals.
Query normalization and KV normalization are separate kernels to respect the
eight-storage-binding limit. Residual buffers alternate to avoid writable
binding aliases. Convolution history and softmax accumulation use FP32;
attention K/V caches use FP16. Prefill projections round operands to FP16 and
accumulate FP32 products; decode projections use FP32 products/accumulation.
Explicit ties-to-even half rounding preserves that prefill contract across this
driver's native casts. Attention scans a fixed compiled capacity to keep
workgroup barriers uniform.

Decode GEMV assigns 32 lanes per output row, accumulates privately, then reduces
through a shared-memory tree. A prepared plan caches bindings and per-dispatch
scalar buffers, recording one ordered compute pass and submitting once per
chunk. Command buffers are encoded afresh. This is not captured CUDA Graph
replay or a fused megakernel. The profile requires `shader-f16`, uses no subgroup
or matrix extensions and stays within the portable 32 KiB shared-memory limit.

An independent NumPy forward implementation checks the entire model without
Tensor kernels or the Tensor runtime. Each format passes 19 full-vocabulary
logit fixtures: the chat prompt and three cached continuations; prefixes of
31, 32, 33, 127, 128, 129 and 511 tokens and their cached next tokens; then a
reset to the original prompt. All greedy argmax values agree with NumPy.
Worst relative RMS error is 0.240% for F16 and 0.236% for Q4; minimum cosine is
0.999997 for both. The gates were RMS below 1%, cosine above 0.9999 and equal
argmax. Reset is bitwise identical; empty IDs, out-of-vocabulary IDs, overflowing
context and a closed model are rejected.

Native llama.cpp argmax agrees on all 19 fixtures for each format, but logits
are not identical: maximum relative RMS is 4.79% for F16 and 10.53% for Q4.
The independent reference validates Tensor's stated precision. This fixture
set does not establish numerical equivalence or identical long generations to
llama.cpp, whose Vulkan projection/attention paths use different arithmetic.

Both Tensor formats complete this prompt at EOS in 39 generated tokens:

> To find the sum of 2 and 2, you simply add the numbers together:
>
> 2 + 2 = 4
>
> So, 2 + 2 equals 4.

The [62-test run](data/lfm2-230m-tests.xml) also checks prepared-plan replay,
scalar isolation, resource lifetime, actual adapter execution, half ties and
subnormals, exact Q4 byte/word boundaries and Q6 embedding gathers. CUDA GPU
regression was not rerun on this AMD machine. The
[isolated wheel consumer](data/lfm2-230m-clean-consumer.json), launched with
Python `-I`, reproduces both saved logits and generated token sequences without
Torch, TileLang, TVM, Triton, GGML or llama.cpp installed or imported. Tensor's
lightweight compiler wrapper modules are imported; compiler libraries are absent.

## Matched performance

The native comparator is the official Windows Vulkan release b11310,
commit `f872b591121761ac7b2af18283bd99bdc092a63a`. The checksum-pinned archive
is loaded through the public llama C API. Native logs confirm Vulkan0, all
15/15 layers offloaded (including output), F16 caches and flash attention.
See [F16 native log](data/lfm2-230m-f16-native.log) and
[Q4 native log](data/lfm2-230m-q4_0-native.log).

Both engines use the same GGUF, token IDs, total context 512, prefill chunks up
to 32, a fresh reset for each sample, and 64 forced single-token decode steps.
Host FP32 last-token logits and device completion are included in every call.
Loading, reset, tokenization and sampling are excluded. Timings are medians of
five repetitions after one warmup; engine order alternates and GPU workloads
run sequentially. Native CPU threads are six. This measures bounded public
forward calls, not default `llama-bench` throughput or end-to-end chat latency.

| Format | Prompt tokens | Tensor prefill tok/s | llama.cpp prefill tok/s | Tensor decode tok/s | llama.cpp decode tok/s |
|---|---:|---:|---:|---:|---:|
| F16 | 32 | 207.9 | 3,210.0 | 91.2 | 485.0 |
| F16 | 128 | 208.5 | 4,173.1 | 89.9 | 484.6 |
| F16 | 384 | 170.9 | 4,543.2 | 91.2 | 481.7 |
| Q4_0 | 32 | 198.1 | 3,740.2 | 90.4 | 817.0 |
| Q4_0 | 128 | 193.5 | 6,070.6 | 89.1 | 818.2 |
| Q4_0 | 384 | 175.8 | 7,054.3 | 87.3 | 790.4 |

Q4 cuts buffer memory by about three times but currently gives no decode speed
benefit. The shader uses scalar bit extraction for each weight, without a
specialized quantized dot-product path. Prefill uses the portable SIMT GEMM
lowering rather than hardware matrix instructions. These are clear candidate
optimization targets from the implementation; this run does not include GPU
timestamp attribution. Next work should profile projections versus attention
and submission, optimize tiled prefill and packed decode separately, then
consider feature-gated subgroups/fusion while retaining this correctness suite.

## Reproduction

From the repository root, in PowerShell:

```powershell
uv sync --locked --extra webgpu
.venv/Scripts/python.exe benchmarks/lfm2/download.py --model-size 230M --formats F16 Q4_0
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --provider webgpu --context 512 --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-230m-f16-webgpu
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --provider webgpu --context 512 --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --out build/lfm2-230m-q4_0-webgpu
.venv/Scripts/python.exe -m tensor_llm generate --provider webgpu --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --prompt 'What is 2 + 2?' --max-tokens 96
.venv/Scripts/python.exe benchmarks/lfm2/fetch_vulkan.py
$env:OPENBLAS_NUM_THREADS='6'
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-230m-f16-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-230m-f16-webgpu-validation
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-230m-q4_0-webgpu-validation
$env:TENSOR_WEBGPU='1'
$env:TENSOR_LFM2_WEBGPU='1'
.venv/Scripts/python.exe -m pytest tests/providers/test_webgpu.py tests/providers/test_webgpu_audit.py packages/tensor-llm/tests/test_gguf.py packages/tensor-llm/tests/test_contracts.py packages/tensor-llm/tests/test_webgpu.py
```

Wheel installation commands are in [the package README](../../packages/tensor-llm/README.md).
When building in a checkout with stale setuptools build output, build the core
sdist first and then its wheel to use a fresh staging directory:

```powershell
uv build --sdist --out-dir build/lfm2-230m-wheels
uv build --wheel build/lfm2-230m-wheels/tensor_workspace-0.1.0.tar.gz --out-dir build/lfm2-230m-wheels
build/lfm2-230m-consumer/Scripts/python.exe -I benchmarks/lfm2/webgpu_consumer.py --root (Get-Location).Path --out build/lfm2-230m-clean-consumer.json
```

The GGUFs, bundles, wheels, isolated environment and full logits are available
under `build/` on the measured machine and are intentionally not checked in.
The [verification record](data/lfm2-230m-verification.json) retains wheel and
benchmark source hashes; bundle manifests retain implementation fingerprints
and artifact checksums. Rebuild bundles after changing kernels or runtime code.
