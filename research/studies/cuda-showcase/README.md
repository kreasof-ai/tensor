# CUDA compiler showcase: measured NVIDIA L4 results

Measured on **24 October 2026**, on one NVIDIA L4 (SM89, 58 SMs).
All **17 cases** passed final numerical checks. The search measured **260 Tensor
candidates and 142 Triton candidates**. Final comparisons contain **112 provider
measurements**, each with 21 event samples over 200 operations per graph replay.
The artifact audit verified 308 compiled candidate artifacts and their members.

![Kernel latency comparison](evidence/l4-20261024/showcase.png)

| Workload | Tensor speedup over fixed TileLang | Finding |
|---|---:|---|
| FP16 matmul, four cases | 0.99–1.34× | Larger cases benefit from semantic prelowering and search; tuned Triton and PyTorch CUDA BLAS remain faster. |
| Flash Attention, four cases | 0.97–1.12× | Limited search gains. The causal S=129 case beats upstream FA2 by 2.20×; larger cases favor FA2 or Triton. |
| Dense MLA decode, three cases | 1.13–1.76× | Search improves split geometry and value tiles; Triton remains 10–17% faster than Tensor. |
| GGUF prefill, three encodings | 1.59–1.96× | Reused AKAS fused kernels beat the measured Triton implementations. GGML wins Q4_K/Q6_K prefill. |
| GGUF single-row decode, three encodings | About parity | AKAS's existing fixed GEMV is already strong: Tensor beats GGML by 1.18–1.31×. |

The matched native TileLang control usually tracks Tensor closely. The exception
is Q6_K prefill: **13.332 µs for Tensor/NVRTC versus 16.630 µs for native
TileLang/NVCC with the same schedule**, a repeatable 1.25× difference in this
run. This isolates the schedule but not every compiler/toolchain difference;
it does not establish which code-generation decision caused the gain.

These results support a benefit from producer schedule selection. They do not
support a general claim that Tensor generates faster device kernels than an
equally scheduled TileLang kernel. Both winning and losing shapes are retained.
The identical-cubin controls differ by at most 0.31% after interleaved timing.

## Evidence and reproduction

- [Complete per-case timing table](evidence/l4-20261024/REPORT.md)
- [Raw results, correctness, versions, search trials and artifact hashes](evidence/l4-20261024/results.json)
- [Timing CSV](evidence/l4-20261024/measurements.csv)
- [Search candidates](evidence/l4-20261024/candidates.json)
- [Selected target-bound schedule profile](evidence/l4-20261024/schedule-profile.json)
- [Artifact and timing audit](evidence/l4-20261024/audit.json)
- [Exportable SVG](evidence/l4-20261024/showcase.svg)
- [Harness, commands and methodology](../../../benchmarks/kernels/showcase/README.md)

Compiled artifacts, generated specialization sources, isolated baseline inputs,
and compiler caches remain under `build/showcase-l4-*` in the measurement
workspace. The raw JSON points to their paths and SHA256 identities. Compiler
and AKAS source revisions/hashes are recorded. The raw results retain
superseded qualification attempts and the reasons they were replaced; those
attempts do not enter the published timing table.

The main producer uses TileLang 0.1.14, TVM-FFI 0.1.12, Torch 2.14.1 with CUDA
13, Triton 3.8.0, and Tensor's pinned NVRTC 12.9 bundle. Native TileLang uses
NVCC 12.9 and TVM-FFI. The official FA2 2.8.3.post2 wheel runs in an isolated
Torch 2.8/CUDA 12 environment with identical inputs; the versions are retained
separately. GGML uses the existing unmodified CUDA build in `../akas` and a
benchmark-only bridge to time its full operation on its own stream.

## What is measured

Matmul follows `x[M,K] @ w[N,K].T`, with FP16 inputs/output and FP32
accumulation. Tensor performs its existing checked semantic CUDA prelowering,
then searches geometry, orientation and pipeline settings. The other workloads
are explicit TileLang kernels with producer schedule search; no semantic
attention, MLA or GGUF prelowering pass was added.

Attention covers causal/noncausal FP16 forward attention, including a sequence
tail and head dimensions 64/128. MLA is dense V3-style absorbed MQA with QK=576,
V=512, a contiguous shared latent cache, one query token, and both split and
merge launches included. It excludes page-table lookup, ragged batching and
model projections. **Upstream dense FlashMLA cannot run on this L4**: the
[pinned V3 upstream support matrix requires SM90 for dense decode](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7).

GGUF uses actual Q4_0/Q4_K/Q6_K raw block layouts and AKAS's packed GEMV/fused
dequantization kernels. Prefill uses FP16 operands and FP32 accumulation;
qualification independently rounds decoded weights and activations to FP16,
then multiplies in FP32. Errors against original FP32 operands are also saved.
For Q6_K prefill that additional relative L2 error is approximately **0.03%**.
GGML may quantize activations to Q8_1; its approximation is qualified separately.
PyTorch's resident-dequantized GGUF comparison excludes weight decoding and
serves as a lower-bound diagnostic, not a fused competitor.

Final timings use CUDA graphs and interleaved provider order. They exclude
compilation, tuning, allocation, transfers and host submission. FA2 and GGML
run in isolated processes and therefore are not interleaved with the main
providers. Clocks were not locked; raw min/max samples are retained and shown
as error bars. These small resident workloads exercise hot caches and must not
be presented as streaming-memory, full-model or serving performance.
