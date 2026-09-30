# FlashAttention through Tensor's NVRTC compiler and artifact runtime

Date: 2026-09-30. Hardware: NVIDIA A10G (`sm_86`).

The [attention example](../../examples/flash_attention.py) implements FP16
forward self-attention with online softmax, FP32 accumulators and tile-local
scores. It produces a Tensor `.tbin` through NVRTC and executes through the
existing CUDA driver runtime with no global attention-score workspace. The
[demo](../../tools/flash_attention_demo.py) specializes that one source and
compares against PyTorch 2.14.0's **forced FLASH_ATTENTION** SDPA backend.
This is an executable-kernel demonstration, not an implemented Phase 4 FX backend.

## Results

Twelve configurations pass three input families each: random normal Q/K/V,
Q/K multiplied by three for larger logits, and zero Q/K for uniform softmax.
Both Tensor and the forced PyTorch backend agree with an explicit FP32
matmul/softmax reference rounded to FP16, at `rtol=0.02`, `atol=0.002`. Tensor's
largest absolute difference across the 36 cases is `0.001953125`.

CUDA-event timings below measure seven median samples of graph replay with
100 operations per sample, after 20 warmups. Compilation, reference calculation,
input creation and host binding during capture are outside these GPU timings.

| B × H × S × D | Causal | Tensor GPU µs | PyTorch Flash GPU µs | PyTorch / Tensor |
|---|---|---:|---:|---:|
| 1 × 8 × 128 × 64 | No | 5.192 | 8.602 | 1.66× |
| 1 × 8 × 128 × 64 | Yes | 5.325 | 8.878 | 1.67× |
| 1 × 8 × 129 × 64 | No | 6.861 | 15.657 | 2.28× |
| 1 × 8 × 129 × 64 | Yes | 6.687 | 16.292 | 2.44× |
| 1 × 8 × 512 × 64 | No | 19.743 | 23.050 | 1.17× |
| 1 × 8 × 512 × 64 | Yes | 19.825 | 22.395 | 1.13× |
| 1 × 8 × 1024 × 64 | No | 68.946 | 49.039 | 0.71× |
| 1 × 8 × 1024 × 64 | Yes | 45.005 | 49.592 | 1.10× |
| 2 × 4 × 257 × 64 | No | 10.281 | 14.141 | 1.38× |
| 2 × 4 × 257 × 64 | Yes | 10.250 | 14.254 | 1.39× |
| 1 × 8 × 512 × 128 | No | 32.379 | 41.738 | 1.29× |
| 1 × 8 × 512 × 128 | Yes | 31.089 | 36.260 | 1.17× |

Tensor's kernel is faster on eleven of these twelve shapes, but the 1024-token
non-causal kernel is about 41% slower. These are fixed, untuned configurations
on one GPU and do not establish general FlashAttention performance.

Direct submission is a different result. Median batch wall time per completed
operation is approximately **67–71 µs for Tensor versus 22–51 µs for PyTorch**.
Tensor reuses pre-borrowed input/output descriptors and a preallocated output;
PyTorch SDPA allocates its returned outputs. Those favorable Tensor preparations
still do not offset the current Python submission cost. A finished adapter would
also need per-call ownership and output management. Kernel-level speedups must
not be reported as end-to-end PyTorch integration speedups.

All artifacts load in a fresh consumer process that rejects TileLang/TVM/FFI
imports. The default non-causal artifact additionally runs through `tensor run`
in the existing clean environment containing exactly Tensor and NumPy: no
PyTorch, frontend or compiler packages. Its output is byte-identical to the
demo's DLPack result. This demonstrates that PyTorch integration at the call
boundary does not make the compiled kernel depend on PyTorch.

## Reproduce

With the pinned producer and NVRTC bundle:

```bash
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
  uv run --locked python tools/flash_attention_demo.py --out-dir build/attention-demo
```

The output directory must be new. `--quick` runs four causal/non-causal cases
at sequence lengths 128/129. The producer builds `.tbin` files, then starts a
separate compiler-import-blocked process for correctness and both timing paths.
JSON samples and compile times are written to `results.json`; specialized
sources and reusable artifacts remain alongside it.

Build the default export directly or use `tensor-examples::flash_attention`:

```bash
tensor build examples/flash_attention.py --target sm_86 --out build/attention.tbin
TENSOR_ATTENTION_CUDA=1 TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
  uv run --locked python -m pytest tests/test_flash_attention.py -o addopts='' -q
```

Four GPU regressions passed in 21.10 seconds. They cover causal short blocks,
tail lengths and distinct batch/head values with zero-logit closed-form
references. Another 27 module/codegen/signature regressions passed in 15.63
seconds. Linux/Windows CI compile the default export with NVRTC; GPU execution
is measured locally on A10G.

CI [36654018983](https://github.com/kreasof-ai/tensor/actions/runs/36654018983)
passed for implementation `1769be3dafde8f8edccd20c3f7aa0d51a54adc4a`.
Linux passed 97 tests with 23 expected skips; Windows passed 87 with 33 skips.
Both compile the new default attention export with NVRTC. The four new
attention checks require GPU opt-in and skip on those GPU-free runners.
[CI metadata](data/flash-attention-ci.json) records the successful steps.

## Implementation boundary and evidence

This supports specialized contiguous `[batch, heads, sequence, head_dim]`
FP16 self-attention, default `1/sqrt(head_dim)` scale and optional causal masking.
It does not implement arbitrary masks, dropout, GQA, KV-cache decoding,
backward/autograd or dynamic sequence lengths within one artifact.

The first causal schedule used a query-dependent pipelined loop. It failed on
short query blocks; inspected generated code processed an incorrect epilogue
tile. The validated example uses a serial tile loop. The GPU regression checks
preserve that correctness boundary. The current runtime/compiler did not need
an ABI or provider change.

[Raw measurements](data/flash-attention-demo.json) retain build timings,
artifact/source hashes, every correctness case and every timing sample.
Measurements were made from the working tree based on `6f6646a`; the producer's
dirty flag is retained. The acceptance fields bind the measured generated
specializations to the committed example template and demo hashes. Reproduce
before making claims on other devices or software versions.
