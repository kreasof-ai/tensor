# Tensor LLM

Standalone, single-sequence CUDA inference for LiquidAI LFM2.5-2.6B GGUF files.
The optional `tensor-llm` wheel adds GGUF parsing, byte-level BPE tokenization,
packed-weight projections, the hybrid convolution/attention forward plan, and
text generation. The installed consumer needs Tensor, NumPy, regex and an NVIDIA
driver. It does not import Torch, TileLang, TVM, Triton, GGML or llama.cpp.

Supported checkpoint profiles are F16, Q4_0 and Q4_K_M. Q4_K_M mixes Q4_K and
Q6_K matrices; the Q4_0 checkpoint also uses Q6_K for its tied embedding/output.
Weights stay packed on the GPU and are decoded in the projection kernels.
The reader additionally implements F32 and Q8_0; other quantizations and model
architectures are rejected. The kernels currently require 64-dimensional heads
and a three-tap short convolution. This is a bounded demonstration, rather than
a general GGUF inference engine.

## Build once

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

Prompt prefill uses chunks up to 128 tokens, followed by single-token cached
decode. The initial profile has a total prompt-plus-decode limit of 8,448 tokens.
It uses FP16 attention caches, FP32 convolution history, FP16 tensor-core prefill
projections and FP32 decode projections/accumulation. Quantized activation math
therefore differs from llama.cpp's MMQ/MMVQ paths; logits are not bitwise equal.
Generation uses CPU greedy argmax and a single-user LFM2 chat template, or a plain
completion with `--raw`. Multi-turn chat, tools, sampling distributions, batching,
CPU/WebGPU inference and a persistent megakernel are outside this profile.
CUDA Graph replay submits the existing per-layer kernels; it is not one fused
GPU kernel. See the repository's `docs/research/lfm2-inference.md` for correctness,
latency and limitations.
