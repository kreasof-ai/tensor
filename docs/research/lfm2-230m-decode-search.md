# 230M F16 decode search: narrowing the llama.cpp gap

Follow-up: [runtime overhead and wider discovery](lfm2-230m-runtime-search.md)
adds a further 10.5–12.0% completed decode gain. The measurements below retain
their original runtime and kernel scope.

Searching Tensor's decode schedules improves full-model LFM2.5-230M F16 decode
by **39–63%** on this RX 6700 XT. The remaining gap to llama.cpp Vulkan is
**1.23–1.24×**. This closes much of the gap, but does not establish parity.
The [earlier prefill gains](lfm2-prefill-search-throughput.md) are retained.

[Raw evidence](data/lfm2-230m-decode-search.json) contains all 1,081 projection
candidates, independent finalist replay, the attention sweep, full-model
samples and numerical metrics, profiles, bundle shaders and source snapshots.

## Full-model results

Same native-F16 checkpoint and completed host FP32 logits. Decode forces 64
identical tokens after each prompt. Values are tokens per second, from seven
repetitions after three warmups, rotating all three runners sequentially.
Loading, AOT compilation, initial pipeline compilation, reset and sampling are
excluded. Context is 512 and every engine uses 32-token prefill chunks.

| Prompt tokens | Tensor before decode search | Tensor searched decode | Change | llama.cpp Vulkan | Native / Tensor gap |
|---:|---:|---:|---:|---:|---:|
| 32 | 285.1 | **397.1** | **+39.3%** | 487.4 | **1.23×** |
| 128 | 271.2 | **396.3** | **+46.1%** | 485.7 | **1.23×** |
| 384 | 239.5 | **391.2** | **+63.3%** | 483.8 | **1.24×** |

The before bundle already contains the searched prefill schedules. At prefix
128 the native gap falls from **1.79× to 1.23×**; at prefix 384 it falls from
**2.02× to 1.24×**. Parity would still require about **18–19% less completed
time per token**, not merely 18–19% more throughput.

| Prompt tokens | Tensor prefill before | Tensor prefill after | llama.cpp prefill |
|---:|---:|---:|---:|
| 32 | 2,473 | 2,501 | 3,138 |
| 128 | 2,607 | 2,618 | 4,192 |
| 384 | 2,555 | 2,557 | 4,502 |

Prefill is essentially unchanged. Its row-32 kernels retain the earlier
schedules; the final vocabulary projection uses the new row-1 kernel.

The same 39-token GPU-greedy answer takes **0.1438 → 0.1071 seconds** including
reset, tokenization, prefill, generation and completion: **25.5% less time**.
The generated tokens and text are identical, and both profiles' host/GPU
greedy paths agree. These short-answer timings are separate from forced decode.

## Search and integration

The producer option is `--webgpu-profile decode_searched`. It retains searched
prefill and adds the measured F16 decode choices. Existing profiles keep their
defaults. No activation quantization or half-precision accumulator is added:
decode inputs, accumulation, nonlinear operations and output remain FP32;
matrix weights retain native GGUF F16 storage.

`streamed_gemv_schedule` in the compiler lowering module exposes subgroup
width, workgroup size, output microtiles, vector width, K unrolling, independent
accumulators, K access layout and optional shared activation storage. Output
tails are guarded, and reductions fall back to shared memory when the physical
subgroup is smaller than the requested reduction width. The discovery helper
now accepts caller-supplied spaces so the same beam/restart mechanism can search
decode schedules without adding them to the prefill grammar.

The projection search runs for **717.5 seconds** within its 12-minute candidate
budget. It explores **1,081** candidates across six matrix groups. **927** pass
and are timed; **147** violate arithmetic/tile/distribution constraints and
**7** exceed the vocabulary projection's dispatch limit. No numerical failure
is accepted for timing. Every candidate is checked on all affected layers with
actual F16 weights, an independent float64 F32-input dot-product oracle and
NaN output sentinels. Paired FFNs use a propagated SwiGLU error bound.

Every timing sample streams all model matrices, rather than repeatedly reading
one hot matrix. Search uses outputs shared by matrix shape. Finalist replay
uses independent output buffers to avoid introducing write-after-write hazards
between unrelated projections, and rotates the saved finalists. The replay
checks three held-out activation seeds/scales on every affected layer, including
very small inputs. The full-model measurement is the acceptance criterion;
traffic-plan microseconds are not substituted for end-to-end throughput.

The independently replayed choices are:

| Projection | Lanes / threads | Rows per thread | K unroll / accumulators |
|---|---:|---:|---:|
| Paired gate/up `[K,N]=[1024,2560]` | 32 / 256 | 1 | 8 / 4 |
| Vocabulary `[1024,65536]` | 32 / 256 | 1 | 8 / 4 |
| FFN down `[2560,1024]` | 32 / 64 | 2 | 4 / 4 |
| Conv input `[1024,3072]` | 32 / 256 | 1 | 8 / 4 |
| Width projections `[1024,1024]` | 32 / 64 | 1 | 8 / 4 |
| KV projections `[1024,512]` | 32 / 64 | 1 | 8 / 4 |

All use vec4 floating dots and striped K access. Shared activation staging
does not win. Several finalists have nearly equal timings; those small
differences do not establish a hardware optimum.

The paired gate/up kernel also computes SwiGLU. Relative to separate F16 gate,
up and activation kernels, this removes **28 dispatches per token** across 14
layers. The earlier fused residual addition in FFN down remains in place.
The decode plan falls from **160 to 132 dispatches**.

The attention sweep checks a baseline plus **15** channel/token-partition
combinations at active positions 0, 31, 128, 384 and 510. All pass the float64
softmax/value oracle. Inactive score entries retain the producer's `-infinity`
contract; inactive KV entries and output buffers are poisoned with NaNs.
The selected kernel partitions the value reduction across **16** token groups,
with 64 channels and 1,024 threads per head, then combines shared partial sums.
Its isolated GPU times are **13.82 µs at position 128** and **17.52 µs at 384**,
versus **25.05/56.52 µs** for baseline. Position 0 regresses from 10.15 to
18.53 µs; selection targets the measured 32/128/384 prompt workloads and is not
claimed to improve every context length.

## Profile and remaining gap

Seven-repeat profiles use the adapter's measured Vulkan timestamp period of
10 ns. Instrumentation can perturb scheduling, so ordinary forward throughput
above remains the accepted result.

| Decode at position 128 | Before | After |
|---|---:|---:|
| Whole GPU pass | 2.601 ms | **1.581 ms** |
| Instrumented projection sum | 2.314 ms | **1.429 ms** |
| Instrumented attention value sum | 0.242 ms | **0.057 ms** |
| Encode/submit host median | 0.444 ms | 0.447 ms |

GPU execution falls **39.2%**. End-to-end searched decode at prefix 128 is
about **2.524 ms/token**, compared with about **2.059 ms/token** for native
llama.cpp. The remaining difference is large enough that GPU kernel time alone
cannot explain the completed-call gap: encoding, submission, uploads, waits and
host-logit downloads remain material. The large vocabulary matrix is also a
streaming-bandwidth workload and gains only slightly from the search. Another
tile sweep does not by itself promise parity; reducing runtime cost and further
dispatch fusion are concrete remaining directions.

## Accuracy and environment

Both Tensor profiles pass all **19** independent NumPy fixtures: finite logits,
relative RMS below 1%, cosine above 0.9999, matching argmax and bitwise reset.
These include partial prefill chunks, recurrent convolution and attention state,
cached decode, and prefixes through 511 tokens. Maximum relative RMS is
**0.1937%** for both. Generation tokens and host/GPU greedy results also match.

llama.cpp uses its unchanged default inference arithmetic, all layers on
Vulkan, flash attention, F16 KV and six CPU threads. As in earlier comparisons,
its numerical metrics are recorded separately: maximum relative RMS **4.738%**,
minimum cosine **0.9989566**, matching argmax 19/19. It does not pass Tensor's
stricter oracle gate. These are same-checkpoint results with differing internal
arithmetic, rather than a precision-identical native comparison.

Windows, Ryzen 5 5600, RX 6700 XT, AMD Vulkan driver 26.6.2, Python 3.12 and
wgpu 0.29.0 with the native prepared encoder. llama.cpp is pinned to b11310 /
`f872b591121761ac7b2af18283bd99bdc092a63a`. Model SHA256:
`4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`.
The source base is `f9258441d599463f3802d958ec35ef1473fcb57d` plus archived
source snapshots. Tests pass: nine existing discovery checks, eight new decode
grammar/legality checks, and four GPU cases covering output tails, FP32
activation precision, fused FFN, shared input and forced subgroup fallback.

## Reproduction

From the repository root, with the existing model, native release and independent
NumPy fixtures installed:

```powershell
.venv/Scripts/python.exe benchmarks/lfm2/decode_kernel_search.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-decode-search/gemv --minutes 12
.venv/Scripts/python.exe benchmarks/lfm2/decode_search_recheck.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --root build/lfm2-decode-search/gemv --out build/lfm2-decode-search/recheck
.venv/Scripts/python.exe benchmarks/lfm2/decode_attention_search.py --out build/lfm2-decode-search/attention
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-decode-search/baseline --context 512 --provider webgpu --webgpu-profile searched
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-decode-search/searched --context 512 --provider webgpu --webgpu-profile decode_searched
.venv/Scripts/python.exe benchmarks/lfm2/decode_search_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --baseline build/lfm2-decode-search/baseline --searched build/lfm2-decode-search/searched --reference build/llama-vulkan-b11310 --fixtures build/lfm2-230m-f16-webgpu-validation --out build/lfm2-decode-search/comparison --repeats 7
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_profile.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-decode-search/searched --out build/lfm2-decode-search/profile-after.json --repeats 7 --timestamp-period-ns 10
```

Re-running search may choose different near-tied finalists. The production
profile contains the independently replayed choices listed above, rather than
loading a mutable search cache at inference time.
