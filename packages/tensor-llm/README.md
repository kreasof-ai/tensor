# Tensor LLM

Standalone, single-sequence CUDA and WebGPU inference for LiquidAI LFM2.5 GGUF files.
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
