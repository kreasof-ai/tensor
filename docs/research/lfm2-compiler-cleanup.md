# CUDA frontend and shared schedule discovery

[Research index](README.md) · [Original CUDA measurements](lfm2-cuda-formats.md) ·
[Package instructions](../../packages/tensor-llm/README.md)

The production CUDA path now expresses packed GEMV, fused epilogues,
tensor-core prefill loaders and grouped decode attention in TileLang/TIRx.
Complete algorithm bodies are no longer hidden in CUDA `prelude` strings.
They compile through the ordinary `tensor.build` interface and retain frontend
IR in each `.tbin`.

## Ownership

| Layer | Responsibility |
|---|---|
| `packages/tensor-llm/src/tensor_llm/cuda_kernels.py` | LFM2 frontend algorithms and explicit schedule parameters |
| `src/tensor/compiler/cuda_lowering.py` | Two-byte load and FP16 pair conversion helpers |
| `src/tensor/compiler/build.py` | TileLang lowering, hardware helper lowering, NVRTC, cache and artifact production |
| `src/tensor/compiler/search.py` | Generic beam exploration, family retention, deterministic restarts, legality filtering and producer profile selection |
| `src/tensor/compiler/cuda_schedules.py` | CUDA schedule spaces and resource constraints |
| `src/tensor/compiler/webgpu_schedules.py` | WebGPU spaces and coupled ownership moves |
| `benchmarks/lfm2/*search.py` | Workload construction, independent math, correctness gates, timing and candidate budgets |
| `benchmarks/lfm2/profiles/` | Measured target/model settings, outside runtime packages |

Warp shuffles, vectorized loads and IEEE FP32 fused multiply-add use TileLang
operations. The two Tensor hardware helpers contain no quantization, reduction,
GEMV or attention algorithm. Their implementation hash participates in the
compiler cache identity and is recorded in CUDA artifact metadata.

`ScheduleSearch` requires explicit backend spaces. Compilation failures and
incorrect candidates do not enter the beam; callers record positive finite
timings only after their independent checks pass. CUDA projection and attention
searches and the existing WebGPU searches use this same engine. The old
`tensor.compiler.webgpu_search` module has been removed without an alias.

## Producer profiles and standalone execution

The default SM86 optimized profile starts with the measured settings from
`0cffba8` and reselects five decode schedules for the frontend compiler in
[`cuda-sm86-lfm2.5-2.6b.json`](../../benchmarks/lfm2/profiles/cuda-sm86-lfm2.5-2.6b.json).
Selectors describe operations and semantic dimensions; schedules contain compiler
choices. The most specific selector wins. Duplicate selectors, conflicting
matches and provider/target mismatches are rejected.

The producer applies selected settings before compilation. Kernel records retain
semantic parameters separately from schedules, and the bundle records the
profile and its canonical SHA256. The grouped-attention transition is runtime
policy recorded in the bundle. No producer profile file, discovery engine,
TileLang or CUDA compiler is needed by the consumer.

An explicit profile replaces the default producer profile. Unmatched operations
use frontend defaults; a partial profile does not implicitly inherit all the
measured settings. Other GPU targets require their own measurements.

```sh
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" uv run --no-sync python benchmarks/lfm2/producer.py \
  --model build/lfm2-models/LFM2.5-2.6B-Q4_K_M.gguf \
  --out build/lfm2-q4km-frontend --target sm_86 \
  --cuda-profile optimized --prefill-chunks 32 128 \
  --schedule-profile benchmarks/lfm2/profiles/cuda-sm86-lfm2.5-2.6b.json

# Bounded discovery; compilation and validation are excluded from GPU timing.
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" uv run --no-sync python benchmarks/lfm2/cuda_format_search.py \
  --phase decode --candidates 3 --out build/lfm2-cuda-discovery
```

Projection search writes both `search.json` and a consumable `profile.json`.
Attention search records candidates at nine cache lengths, including empty/tail
partitions and the 4K transition. Search output is evidence for a candidate;
full-model validation and remeasurement remain necessary before selecting a
production profile.

## Validation and measurements

The frozen control is Tensor's optimized native CUDA implementation at
`0cffba8584b1a4da051202591c5a04b57d7233f1`. It is distinct from the llama.cpp
control. Both Tensor runners use packed weights, rows 1/32/128, identical
precision and CUDA Graph replay. Each workload uses one sequence, a shared
fixed prompt and 64 forced decode tokens. Rates divide tokens by the median
completed-forward duration after three warmups and five retained samples.
The Tensor runners rotate sequentially. llama.cpp b11310 is measured separately
before them with full CUDA offload, F16 K/V, Flash Attention and batch 128.
Loading, reset, tokenization and sampling are excluded. GPU jobs do not overlap.

Measured on 2026-10-04, NVIDIA A10G / SM86, driver 595.91.07:

### Prefill (tok/s)

| Format | Prefix | Frozen Tensor CUDA | Tensor frontend | llama.cpp CUDA | Frontend / frozen |
|---|---:|---:|---:|---:|---:|
| F16 | 32 | 2,094 | 2,084 | 1,882 | 0.995× |
| F16 | 512 | 6,486 | 6,545 | 5,375 | 1.009× |
| F16 | 8192 | 6,302 | 6,347 | 5,259 | 1.007× |
| Q4_0 | 32 | 2,211 | 2,288 | 3,126 | 1.035× |
| Q4_0 | 512 | 6,246 | 6,359 | 6,785 | 1.018× |
| Q4_0 | 8192 | 5,944 | 6,062 | 6,629 | 1.020× |
| Q4_K_M | 32 | 1,677 | 1,730 | 3,071 | 1.031× |
| Q4_K_M | 512 | 4,667 | 4,767 | 6,197 | 1.021× |
| Q4_K_M | 8192 | 4,563 | 4,655 | 6,030 | 1.020× |

### Decode (tok/s)

| Format | Prefix | Frozen Tensor CUDA | Tensor frontend | llama.cpp CUDA | Frontend / frozen |
|---|---:|---:|---:|---:|---:|
| F16 | 32 | 90.4 | 89.5 | 87.5 | 0.990× |
| F16 | 512 | 90.1 | 89.1 | 87.1 | 0.989× |
| F16 | 8192 | 86.3 | 85.6 | 84.9 | 0.992× |
| Q4_0 | 32 | 248.2 | 249.7 | 235.0 | 1.006× |
| Q4_0 | 512 | 247.7 | 248.5 | 232.1 | 1.003× |
| Q4_0 | 8192 | 220.1 | 220.3 | 214.8 | 1.001× |
| Q4_K_M | 32 | 225.7 | 223.8 | 222.8 | 0.992× |
| Q4_K_M | 512 | 223.7 | 220.9 | 220.1 | 0.988× |
| Q4_K_M | 8192 | 203.8 | 203.0 | 207.1 | 0.996× |

F16 prefill stays within 1% of the frozen implementation. Quantized prefill
improves 1.8–3.5%. Decode differs by −1.24% to +0.61% across these cases.
This preserves the demonstrated performance closely, with small remaining
F16/Q4_K_M decode regressions. These are observed ratios, not a proof of exact
performance equivalence or a claim for other hardware.

Reusing the native schedules initially cost Q4_K_M 3.3–4.4% decode throughput.
Two shared-discovery runs checked eight real projection shapes, each with a
baseline and eight candidates. Five schedule changes improved their GPU time
by more than 2% in both runs. The selected settings were revalidated and then
remeasured above; the initial measurements remain in the retained data.
A bounded prefill run also passed all 51 candidates across 17 real
projection/encoding combinations at 32 rows, using the same discovery engine.
Its exported settings remain experimental pending full-model remeasurement.
Grouped attention passed all seven configurations at nine cache lengths.
The current 32-token/two-warp production attention schedule is retained.

Each format passed 22 independent Torch comparisons: prefill and cached decode
at lengths 1, 32, 127, 128, 129, 384, 512, 2048, 4095, 4096 and 8192. Maximum
relative RMS was 0.1838% for F16, 0.1527% for Q4_0 and 0.1772% for Q4_K_M.
All logits were finite, every argmax matched, and cosine similarities exceeded
0.9999. Reset, graph/eager replay and GPU/host greedy generation also passed.

Artifact audits verified 196 kernel records against the frontend generator,
selected profile, artifact checksum and compiler lowering hash. Frontend IR
contains the algorithms; its external calls are limited to the two hardware
helpers. Regression tests inspect loops and external operations directly.

Fresh installed-wheel consumers, containing only Tensor, Tensor LLM, NumPy
and regex, reproduced all 66 saved logits bitwise and passed short/long greedy
generation. Their import guard rejects the entire `tensor.compiler` namespace
as well as framework/compiler packages. The audit exposed an eager compiler
import in `tensor.__init__`; `tensor.build()` now imports its implementation
only when called.

All 20 CUDA operator checks passed, covering five packed encodings, fused
linear/residual/FFN, staged prefill tails, split/grouped attention, normalization,
cache writes, greedy ties and adaptive model state. The complete CPU suite
passed 293 tests, with 351 opt-in tests skipped. Linux and Windows CI also
compile the selected schedules without a GPU.

Retained evidence:

- Full-model comparisons: [F16](data/lfm2-compiler-cleanup-F16-comparison.json),
  [Q4_0](data/lfm2-compiler-cleanup-Q4_0-comparison.json),
  [Q4_K_M](data/lfm2-compiler-cleanup-Q4_K_M-comparison.json).
- Independent validation: [F16](data/lfm2-compiler-cleanup-F16-validation.json),
  [Q4_0](data/lfm2-compiler-cleanup-Q4_0-validation.json),
  [Q4_K_M](data/lfm2-compiler-cleanup-Q4_K_M-validation.json).
- Compiled profile/IR audits: [F16](data/lfm2-compiler-cleanup-F16-audit.json),
  [Q4_0](data/lfm2-compiler-cleanup-Q4_0-audit.json),
  [Q4_K_M](data/lfm2-compiler-cleanup-Q4_K_M-audit.json).
- Installed consumers: [F16](data/lfm2-compiler-cleanup-F16-consumer.json),
  [Q4_0](data/lfm2-compiler-cleanup-Q4_0-consumer.json),
  [Q4_K_M](data/lfm2-compiler-cleanup-Q4_K_M-consumer.json).
- Shared discovery: [first projection run](data/lfm2-compiler-cleanup-decode-search-first.json),
  [projection recheck with source snapshots](data/lfm2-compiler-cleanup-decode-search.json),
  [attention](data/lfm2-compiler-cleanup-attention-search.json),
  [bounded prefill](data/lfm2-compiler-cleanup-prefill-search.json).
- Initial frontend results before reselection: [Q4_0](data/lfm2-compiler-cleanup-Q4_0-initial-comparison.json),
  [Q4_K_M](data/lfm2-compiler-cleanup-Q4_K_M-initial-comparison.json).

## Reproduction

Build optimized bundles with the normal producer into `F16-frontend`,
`Q4_0-frontend` and `Q4_K_M-frontend` beneath `build/lfm2-compiler-cleanup`.
Validate before benchmarking, in separate processes so Torch's reference
allocator does not retain GPU memory during timing:

```sh
uv run --no-sync python benchmarks/lfm2/cuda_frontend_compare.py audit
uv run --no-sync python benchmarks/lfm2/cuda_frontend_compare.py validate
uv run --no-sync python benchmarks/lfm2/cuda_frontend_compare.py compare
```

The comparison requires original optimized bundles named `F16-native`,
`Q4_0-native`, `Q4_K_M-native`, and a frozen `tensor_llm_native` source directory
from `0cffba8584b1a4da051202591c5a04b57d7233f1`. Build these in a checkout of that
revision and preserve their original manifests. The harness verifies every
frozen Python source against Git before loading it. The native implementation
is a benchmark reference outside the production package. Use the same pinned
llama.cpp b11310 helper as the original CUDA campaign.
