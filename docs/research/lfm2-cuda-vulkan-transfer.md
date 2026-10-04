# Transferring the Vulkan LFM2 inference work to CUDA

[Research index](README.md) · [Vulkan history](rx6700xt-vulkan-history.md) ·
[Package instructions](../../packages/tensor-llm/README.md)

This report measures revision `4c6872674756454982af8c87c6a1d77b3fc5ef02`.
The subsequent [F16/Q4_0/Q4_K_M optimization pass](lfm2-cuda-formats.md)
extends this profile with tuned prefill and long-context grouped attention.
Use the recorded revision to reproduce the historical timings below.

The useful Vulkan optimizations transfer to CUDA as an opt-in
`--cuda-profile optimized` plan in `tensor_llm.LFM2`. The measured workload is
LFM2.5-2.6B **QAD Q4_0**, one sequence, on NVIDIA A10G. The weights stay packed;
the consumer still needs only Tensor, Tensor LLM, NumPy, regex and the driver.

## Matched completed-forward results

Measured on 2026-10-04, A10G / SM86, driver 595.91.07. Every runner uses the same
GGUF and explicit token IDs. Each measurement prefills the given prefix, then
executes 64 forced single-token cached decode calls. Times include submission,
completion and last-token FP32 host logits. Model loading, reset, tokenization
and sampling are outside the timer. These are `forward()` measurements;
GPU greedy generation is checked separately for correctness.

The native reference is llama.cpp b11310,
`f872b591121761ac7b2af18283bd99bdc092a63a`, matching the Vulkan campaign.
It uses CUDA, all layers offloaded, Flash Attention, F16 K/V, four CPU threads,
one sequence and batch/ubatch 128. Tensor's default has rows 1/128; the new
profile has rows 1/32/128 and CUDA Graph replay. The previous control is the
existing `fp16_half2` plus 16-way split-KV experimental runner, freshly rebuilt
and validated against this QAD checkpoint. Its packed decode products round
in half precision; the new packed decoder uses FP32 arithmetic.

All runners have three warmups and five retained samples per prefix. The
native helper runs first in a separate process on the same GPU; the three
Tensor runners rotate sequentially afterward. No GPU timing jobs overlap.
Rates below use the median total prefill/decode duration, rather than a median
of individual token rates. Raw samples and manifests are retained in
[the benchmark JSON](data/lfm2-cuda-vulkan-transfer-benchmark.json).

### Cached decode (tok/s)

| Prefix | Default CUDA | Previous CUDA experiment | Optimized CUDA | llama.cpp CUDA | Optimized / native |
|---:|---:|---:|---:|---:|---:|
| 128 | 136.1 | 177.5 | 247.2 | 235.5 | 105.0% |
| 512 | 122.8 | 177.4 | 245.4 | 233.2 | 105.2% |
| 2048 | 88.6 | 173.5 | 237.8 | 229.7 | 103.5% |
| 8192 | 42.4 | 159.4 | 212.7 | 217.9 | 97.6% |

### Prefill (tok/s)

| Prompt | Default CUDA | Previous CUDA experiment | Optimized CUDA | llama.cpp CUDA | Gain over default |
|---:|---:|---:|---:|---:|---:|
| 128 | 4,208 | 4,258 | 4,593 | 6,730 | 9.2% |
| 512 | 4,241 | 4,266 | 4,684 | 6,817 | 10.4% |
| 2048 | 4,217 | 4,245 | 4,673 | 6,884 | 10.8% |
| 8192 | 4,113 | 4,138 | 4,551 | 6,746 | 10.7% |

The optimized decoder gains **1.33–1.39× over the previous CUDA experiment**
and **1.82–5.02× over the default**. It runs 3.5–5.2% ahead of this native
reference at prefixes 128–2048; at 8192 it reaches 97.6% of native. Prefill
improves by 9.2–10.8%, while native remains 1.46–1.48× faster.

Tensor-owned buffers total 1.634 GiB for the default and 1.643 GiB for the
optimized profile; the new profile adds about 9.2 MiB of workspaces, rather
than an expanded weight copy. These counts exclude driver allocations and
CUDA Graph metadata; exact byte counts are in the benchmark profiles.

These results apply to the recorded QAD checkpoint and hardware. They do not
establish throughput parity for ordinary Q4_0, F16, Q4_K_M, other models or
other GPU architectures. Prefill remains behind llama.cpp.

## What transferred

- **Packed decode and scale reuse.** Each lane loads one Q4 word and consumes
  eight values, or reuses Q6 low/high words and signed scales. Aligned float4
  activation loads and four independent accumulator chains feed one warp per
  output row. GGML rows are only two-byte aligned, so packed words are assembled
  from two 16-bit loads. This avoids unsafe unaligned uint32 reads.
- **Fusion.** Quantized gate/up projections share activation work and write
  SwiGLU directly. FFN-down writes the residual sum; mixer residual addition
  and the following RMS normalization share one kernel. Prefill pairs gate/up
  around the existing tensor-core projections.
- **Attention and state.** The earlier CUDA 16-partition, four-warp attention
  experiment becomes part of the profile. It uses online FP32 softmax and a
  reusable 132 KiB partial buffer for 32 query heads. Intermediate prefill
  chunks stop after the final persistent state write and advance the control.
- **Suffix liveness.** For the guarded 2048-wide, 10752-FFN Q4 final
  attention/conv/conv suffix, all K/V are stored before cropping to eight query
  rows. Both width-three convolution histories and the last output are
  preserved. The final FFN runs only on the last row. Other shapes retain full
  suffix execution.
- **GPU greedy generation.** Argmax writes the next token on the device,
  breaking ties toward the lowest ID. Decode replays without a token upload
  and reads one int32 per step. The non-graph path follows the same plan.

Decode keeps FP32 activations and accumulation. Attention caches are F16;
convolution history is FP32. Prefill retains FP16 tensor-core projection and
attention operands with FP32 accumulation. Its final-row FFN uses FP32 decode
arithmetic. There is no expanded global weight cache or speculative unpacking.
Vulkan's short-half accumulator chains and two-component Q8 activation prefill
are not used here: CUDA already has tensor-core prefill.

The new decode plan has 246 launches, against 359 in the default. It is still
a graph of per-layer kernels, rather than a persistent inference megakernel.
The producer uses TileLang/TIR and NVRTC; there is no direct PTX emitter in
this transfer. The existing default profile remains available.

## Accuracy and compiler-free consumption

[Independent validation](data/lfm2-cuda-vulkan-transfer-validation.json)
retains 18 prefill/decode logit comparisons at prefix lengths
1, 32, 127, 128, 129, 384, 512, 2048 and 8192. An independent Torch reference
uses separate FP16 prefill and FP32 decode operators, with no shared kernel
templates. Every output is finite and every argmax matches. The maximum
relative RMS error is **0.1133%**, below the 1% gate; every cosine exceeds
0.9999. These gates establish bounded numerical agreement, not bitwise
agreement with Torch or llama.cpp or an evaluation of model quality.

Reset replay and graph/eager execution are bitwise equal. GPU and host greedy
generation agree for the retained 16-token example. Tests independently check
F32/F16/Q4_0/Q4_K/Q6_K projections and epilogues, output-row tails, short and
partial attention partitions, RMS/RoPE and partial K/V stores, absolute cropped
query positions, partial tail extraction, adaptive prefill, and lowest-ID
argmax ties. The shared causal-window proof checks both convolution histories
and the final output. The unchanged default runner's retained model checks also
pass after rebuilding its implementation fingerprints.

[The installed consumer audit](data/lfm2-cuda-vulkan-transfer-consumer.json)
replays all 18 saved arrays bit for bit and the same generation result. The
fresh environment contains exactly `tensor-workspace`, `tensor-llm`, NumPy
and regex. Imports of TileLang, TVM, TVM FFI, Torch, Triton and wgpu are blocked
during execution. CI compiles the diagnostic optimized plan and the guarded
suffix templates without a GPU on Linux and Windows.

## Reproduction

The checkpoint is pinned to LiquidAI/LFM2.5-2.6B-GGUF revision
`e7caca5d835a3901a8e0d63e94009429bafafdfc`, file
`LFM2.5-2.6B-QAD-Q4_0.gguf`, 1,593,894,944 bytes, SHA256
`a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03`.
It is distinct from the ordinary Q4_0 checkpoint used in earlier CUDA reports.

Run from the repository root in a prepared producer environment:

```sh
uv run --no-sync python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
uv run --no-sync python benchmarks/lfm2/download.py \
  --formats QAD-Q4_0 --out build/lfm2-cuda-transfer/models
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
model=build/lfm2-cuda-transfer/models/LFM2.5-2.6B-QAD-Q4_0.gguf
root=build/lfm2-cuda-transfer
uv run --no-sync python benchmarks/lfm2/producer.py --model "$model" \
  --out "$root/qad-default" --context 8448 --target sm_86
uv run --no-sync python benchmarks/lfm2/producer.py --model "$model" \
  --out "$root/qad-tail" --context 8448 --target sm_86 \
  --cuda-profile optimized --prefill-chunks 32 128
uv run --no-sync python benchmarks/lfm2/fp16_decode.py bundle \
  --base "$root/qad-default" --out "$root/qad-half2" --mode fp16_half2
uv run --no-sync python benchmarks/lfm2/decode_optimization.py \
  --base "$root/qad-half2" --out "$root/qad-previous" --splits 16
uv run --no-sync python benchmarks/lfm2/cuda_transfer.py validate \
  --model "$model" --bundle "$root/qad-tail" --out "$root/qad-tail-validation"
```

The native benchmark producer separately needs the CUDA toolkit, CMake, Ninja
and a C++ compiler. Its linker uses the toolkit's CUBLAS libraries; this is a
requirement of the llama.cpp reference build. Set `LD_LIBRARY_PATH` to the
toolkit's `lib64` when running a reference linked against a local toolkit.

```sh
uv run --no-sync python benchmarks/lfm2/build_reference.py \
  --out "$root/native-b11310" --cuda-root build/llama-cuda-toolkit \
  --arch 86 --jobs 2 --commit f872b591121761ac7b2af18283bd99bdc092a63a
uv run --no-sync python benchmarks/lfm2/cuda_transfer.py compare \
  --model "$model" --base "$root/qad-default" --bundle "$root/qad-tail" \
  --previous "$root/qad-previous" --reference "$root/native-b11310/lfm2-reference" \
  --out "$root/qad-final-comparison" --depths 128 512 2048 8192 \
  --generated 64 --repeats 5
```

Build current wheels and install them in a fresh environment to repeat the
compiler-free audit. Compiler source and caches are unnecessary for the consumer.

```sh
uv build --wheel --out-dir "$root/wheels"
uv build --wheel packages/tensor-llm --out-dir "$root/wheels"
uv venv --python 3.12 "$root/consumer"
uv pip install --python "$root/consumer/bin/python" "$root"/wheels/*.whl
"$root/consumer/bin/python" benchmarks/lfm2/cuda_transfer_consumer.py \
  --model "$model" --bundle "$root/qad-tail" --reference "$root/qad-tail-validation" \
  --out "$root/qad-tail-consumer.json"
```

The retained reports include the adapter, source fingerprints, model and
bundle hashes, raw timing samples and exact validation tokens. Rebuild bundles
after implementation changes; the runner rejects mismatched fingerprints.
