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

The latest [decode repeat](../../docs/research/lfm2-webgpu-decode-push.md) measures
packed floating dots and residual fusion on this GPU: 2.6B Q4_0 reaches
140 tok/s against llama.cpp's 169 at prefix 128, and 230M Q4_0 improves by 9.5%
in the same fresh comparison. Prefill is unchanged. Rebuild existing bundles
after updating the package because implementation fingerprints are enforced.

This profile uses the ordinary post-training-quantized Q4_0 checkpoint. The
repository also publishes a distinct QAD Q4_0 checkpoint; do not interchange
them in matched numerical or performance comparisons. Q4_K_M can additionally
be downloaded with `--formats Q4_K_M`. Omitting `--model-size` preserves the
existing 2.6B download workflow and its default output directory.

Build and run the smaller Vulkan demonstration from the repository root:

```powershell
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --provider webgpu --webgpu-profile subgroup --context 512 --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --out build/lfm2-230m-q4_0-webgpu
.venv/Scripts/python.exe -m tensor_llm generate --provider webgpu --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --prompt 'What is 2 + 2?' --max-tokens 96
```

Use `F16` and `lfm2-230m-f16-webgpu` for the F16 bundle. The WebGPU profile uses
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
Generation uses CPU greedy argmax and a single-user LFM2 chat template, or a plain
completion with `--raw`. Multi-turn chat, tools, sampling distributions, batching,
CPU inference and a persistent megakernel are outside the supported profiles.
CUDA Graph replay submits the existing per-layer kernels; it is not one fused
GPU kernel. See the repository's `docs/research/lfm2-inference.md` for correctness,
latency and limitations.
