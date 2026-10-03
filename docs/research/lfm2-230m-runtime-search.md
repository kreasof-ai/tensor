# 230M decoding: runtime overhead before wider discovery

Runtime changes improve LFM2.5-230M F16 completed decode throughput by
**10.5–12.0%** on the RX 6700 XT. A subsequent wider projection search and
attention fusion sweep do not improve the full-model result. The retained
`decode_searched` profile reaches **442 tokens/s** at prefix 128 versus
**490 tokens/s** for llama.cpp Vulkan, a **1.11×** remaining throughput gap.

[Raw evidence](data/lfm2-230m-runtime-search.json) archives the paired runtime
ablation, final four-runner comparison, host profiles, all projection search
records, finalist replay, attention candidates, numerical checks, tests,
generated bundle shaders and implementation snapshots.
The [previous decode search](lfm2-230m-decode-search.md) provides the starting
kernel schedules.

## Completed-token measurements

Rates below are tokens/s. Every runner returns host FP32 logits for each
forward. Decode forces the same 64 token IDs after each prefix. Context is
512, prefill chunks are 32, and three warmups precede seven measured samples.
Runners rotate sequentially on the GPU; loading, compilation, initial calls,
reset and sampling are excluded.

| Prefix tokens | Original runtime | Improved runtime | Runtime + fused attention | llama.cpp Vulkan | Retained runtime gain |
|---:|---:|---:|---:|---:|---:|
| 32 | 398.5 | **446.4** | 445.9 | 487.3 | **12.0%** |
| 128 | 396.7 | **441.9** | 440.9 | 490.0 | **11.4%** |
| 384 | 392.6 | **434.0** | 425.7 | 488.7 | **10.5%** |

The original-runtime runner uses the same shaders as the improved-runtime
runner, with the previous queue fence, separate readback copy submission and
public `map_sync` flush restored by a benchmark-only wrapper. This isolates
runtime changes from kernel tuning. Runtime-only logits are **bitwise equal
on all 19 independent fixtures**.

At prefix 128, completed latency falls from **2.521 ms to 2.263 ms**.
llama.cpp takes **2.041 ms**. Parity still requires approximately **9.8% less
Tensor latency** at that prefix; this is not a claim of parity or a hardware
performance ceiling.

Prefill changes much less because its GPU compute dominates:

| Prefix tokens | Original runtime | Improved runtime | Runtime + fused attention | llama.cpp Vulkan |
|---:|---:|---:|---:|---:|
| 32 | 2488.9 | **2571.7** | 2575.2 | 3139.0 |
| 128 | 2614.2 | **2637.1** | 2642.5 | 4188.9 |
| 384 | 2550.7 | **2564.0** | 2567.5 | 4506.6 |

GPU-greedy generation of the same 39-token response takes median **106.64 ms**
with the original runtime and **97.40 ms** with the improved runtime, an
**8.7% latency reduction**. The fused variant takes **96.21 ms**, but its small
gain on this short response does not outweigh its completed-logit regression
at longer contexts. These generation timings include reset, tokenization,
prefill, greedy sampling and host completion. Host and GPU sampling agree.

## Runtime changes

Previously each completed forward performed:

1. Submit the compute command buffer.
2. Wait for queue-wide completion through `Buffer.to_numpy()`.
3. Submit a separate storage-to-staging copy.
4. Let wgpu 0.29 `map_sync(READ)` submit another empty command buffer.
5. Wait for mapping, copy the host array and unmap.

`PreparedPlan.launch(readback=buffer)` now appends the storage-to-staging copy
after the compute pass in the same encoder, submits once, then waits for map
completion. The explicit submission orders and flushes the copy after its
producers. Mapping its destination supplies the completion dependency, so the
earlier queue-wide fence is redundant.

The WebGPU buffer's ordinary `to_numpy()` path also relies on its ordered copy
and map completion. Other providers retain their existing synchronization.
Staging buffers are reused; returned NumPy arrays own their data and remain
valid after unmapping and subsequent launches. Resource, device, session and
released-plan validation remains active.

For **wgpu-native 0.29.0**, mapping uses its `READ_NOSYNC` path after an explicit
copy submission, following the installed implementation of
`GPUQueue.read_buffer`. Other versions use public `map_sync`. This version
guard avoids relying on the private mapping mode on an unverified release.
It does not bypass the map-completion wait.

The ordinary WebGPU LFM2 forward and GPU-greedy generation both use combined
submission/readback. Kernel arithmetic, precision, weight format and sampling
are unchanged by the runtime work.

Diagnostic cProfile runs over 64 warmed decode calls show:

| Host API event | Before | After |
|---|---:|---:|
| Command encoder finishes/submissions | 192 | 64 |
| Completion waits | 128 | 64 |
| Queue-wide completion waits | 64 | 0 |
| Python function calls | 74,769 | 53,393 |

Those instrumented runs took 206 ms and 168 ms respectively, but are not the
throughput acceptance result. Profiling includes GPU waits and perturbs host
timing; its cumulative times cannot be interpreted as independent CPU costs.
The earlier runtime-only three-runner experiment independently measured
**440.4 tokens/s** versus **398.5 tokens/s** at prefix 128.

## Wider discovery and what failed to help

The projection search runs for **357.45 seconds**, with beam width 16 and
deterministic restarts. It evaluates **601 candidates**: **478 pass** the
float64 oracle and GPU timing; **123 are rejected**. The grammar adds unroll
factors **5, 10, 16 and 20**, and supports **eight independent accumulator
chains**, subject to complete K coverage and existing register/shared-memory
limits. Seeds include the current selected schedules.

Every timing pass streams all model matrices, preserving the previous
full-weight-traffic protocol. Finalists are independently replayed with
separate outputs per matrix, three held-out activation scales on every affected
layer, warmup, and seven rotated GPU samples. The current schedule is included
explicitly as a control. Changes in thread count or accumulator count produced
only sub-percent improvements on fresh replay; no projection settings are
changed on that evidence. More unrolling or wider beams did not yield a
meaningful additional gain within this search budget.

Attention discovery adds a structural family: **QK scores, softmax and weighted
V in one dispatch**. Twelve fused candidates explore channels 16/32/64 and
value partitions 2/4/8/16. All candidates and the complete split baseline pass
five active-prefix tests with inactive K/V poisoned by NaNs. Timings measure
the complete one- or two-dispatch plan; GPU timestamps use the verified 10 ns
Vulkan timestamp period. The selected candidate uses **32 channels and 16
partitions**, eliminating six model dispatches: **132 to 126**.

Its per-layer GPU times illustrate the context tradeoff:

| Active position | Split scores + attention, µs | Fused candidate, µs |
|---:|---:|---:|
| 0 | 25.50 | 13.62 |
| 31 | 19.99 | 11.92 |
| 128 | 16.11 | 13.55 |
| 384 | 20.11 | 21.42 |
| 510 | 22.25 | 25.65 |

The fused candidate wins the short/medium microbenchmark score, but not the
completed-token full-model comparison. Fewer dispatches alone are insufficient:
the fused workgroups limit parallelism and duplicate QK work across channel
blocks. The experimental `decode_fused` profile is retained for reproducibility;
**use `decode_searched` for the measured best full-model throughput**.

## Accuracy and scope

Tensor uses native F16 weights and KV storage with FP32 activations and
accumulation. Its maximum independent-logit relative RMS error is **0.1937%**,
minimum cosine exceeds **0.999998**, and all **19 argmax checks** match. Reset
is bitwise deterministic and greedy generation is unchanged.

The same checkpoint is used by llama.cpp b11310, commit
`f872b591121761ac7b2af18283bd99bdc092a63a`, with all layers offloaded, Flash
Attention enabled and ordinary Vulkan precision. Its maximum relative RMS error
is **4.738%**, and all 19 argmax checks match. It does **not** pass Tensor's 1%
oracle threshold. These are matched-checkpoint throughput measurements with
different arithmetic precision, not precision-equivalent performance parity.

Hardware: RX 6700 XT / RDNA2, Ryzen 5600, 32 GB host memory, Windows, AMD
Vulkan 26.6.2 and wgpu 0.29.0. Model SHA256:
`4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`.
Runtime changes also apply to other WebGPU workloads that read buffers back,
but other models and latency-scaling workloads have not been remeasured here.

Validation includes native prepared encoding and Python fallback, retained
resource/error checks, owned cached snapshots, pending FP16 uploads without
an extra queue fence, eight-chain/non-power-of-two GEMV with tail outputs and
small-subgroup fallback, and fused attention with its QK fallback. Test logs
and all 19 model fixtures are recorded in the raw evidence.

## Reproduction

From the workspace root, with the prepared F16 model, native llama.cpp release
and independent NumPy fixtures:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
.venv/Scripts/python.exe benchmarks/lfm2/decode_kernel_search.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-runtime-search/gemv-extended --minutes 6 --extended
.venv/Scripts/python.exe benchmarks/lfm2/decode_search_recheck.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --root build/lfm2-runtime-search/gemv-extended --out build/lfm2-runtime-search/gemv-recheck
.venv/Scripts/python.exe benchmarks/lfm2/decode_fusion_search.py --out build/lfm2-runtime-search/fusion-v3
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-runtime-search/runtime --provider webgpu --context 512 --webgpu-profile decode_searched
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-runtime-search/extended --provider webgpu --context 512 --webgpu-profile decode_fused
.venv/Scripts/python.exe benchmarks/lfm2/runtime_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-runtime-search/runtime --searched build/lfm2-runtime-search/extended --reference build/llama-vulkan-b11310 --fixtures build/lfm2-230m-f16-webgpu-validation --out build/lfm2-runtime-search/comparison
```

The original host profile and first runtime-only run require the archived
implementation snapshots. Changes to implementation files invalidate earlier
LFM2 bundles: rebuild them before use. The final measurement bundles correspond
to the source hashes embedded in this report's raw evidence.
