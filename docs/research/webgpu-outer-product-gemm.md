# Chasing CLBlast with staged outer products

[Research index](README.md) · [CLBlast baseline](clblast-rx6700xt.md) ·
[Whole-loop accumulation](webgpu-gemm-accumulation.md) · [WebGPU guide](../guides/webgpu.md)

On 2026-10-03, a new compiler schedule reduced 4096³ native-FP32 GEMM from
**112.956 ms to 33.515 ms** on the RX 6700 XT Vulkan backend. Stock CLBlast
OpenCL completed the identical operation in **16.373 ms**. The new Tensor
schedule is **3.37× faster**, reducing the freshly measured gap from **6.90×
to 2.05×**. This is useful progress toward CLBlast, with a substantial compute
gap still remaining.

This is an opt-in producer schedule, not a replacement of every `T.gemm` and
not a new LFM2 throughput measurement. Tensor's existing generic lowering and
LFM2 schedules retain their defaults. The 4096³ schedule does not generalize
well to 32-row matrices; those require independent schedule selection.

## Fresh matched comparison

All operands, accumulation and outputs are FP32: resident A[M,K] multiplied by
resident B[N,K].T, alpha=1, beta=0. Linear includes FP32 bias and ReLU. Every
implementation receives identical seeded arrays, recorded with SHA-256 hashes.
CLBlast is the unmodified 1.6.3 library and its device-selected parameters,
including any internal packing, padding, transformations and epilogue kernels.
No operand transpose or other prerequisite is charged only to one side outside
the timed call.

Completed-call medians below include submission and queue completion, exclude
output download, and use preallocated output. The generic column calls ordinary
`kernel.launch`; the new schedule uses a prebound prepared plan. GPU timestamps
independently establish that the large gain is in device computation. A separate
repeat also prepares the generic control to isolate this runtime distinction.
There are 20 warm completed calls and 45 timed calls per measurement.

| Workload | Generic Tensor (ms) | New Tensor (ms) | CLBlast (ms) | Tensor gain | Remaining gap |
|---|---:|---:|---:|---:|---:|
| Pure GEMM 512³ | 1.140 | 0.417 | 0.193 | 2.73× | 2.16× |
| Pure GEMM 1024³ | 3.721 | 0.990 | 0.462 | 3.76× | 2.14× |
| Pure GEMM 2048³ | 14.910 | 3.958 | 2.001 | 3.77× | 1.98× |
| Pure GEMM 4096³ | 112.956 | 33.515 | 16.373 | 3.37× | 2.05× |
| Linear 512³ | 1.161 | 0.423 | 0.275 | 2.74× | 1.54× |
| Linear 1024³ | 3.722 | 0.993 | 0.651 | 3.75× | 1.53× |
| Linear 2048³ | 14.745 | 4.091 | 2.178 | 3.60× | 1.88× |
| Linear 4096³ | 113.279 | 35.436 | 16.760 | 3.20× | 2.11× |

At 4096³, useful completed-call GEMM throughput is **1.22 → 4.10 TFLOP/s**,
versus **8.39 TFLOP/s** for CLBlast. This counts 2*M*N*K useful operations,
not actual executed instruction counts or measured hardware utilization.
Tensor's corresponding compute-pass timestamp is 33.462 ms; CLBlast's complete
routine queue interval is 16.419 ms. Those GPU timing boundaries differ, so
completed calls are the main cross-backend comparison.

[Full twelve-profile comparison](data/webgpu-outer-product-comparison.json).

Two independent 4096³ repeats measured **33.768 / 33.575 ms** for the winner
and **16.406 / 16.408 ms** for stock CLBlast. A generic prepared-plan control
measured **113.051 ms**, so eliminating ordinary-call binding does not explain
the large gain. On the same tile/layout/arithmetic, disabling explicit unroll
measured **37.219 / 37.065 ms**: explicit expansion contributes **1.10×** in
this controlled ablation. Most of the overall gain comes from the new schedule.

## Shape-specific 32-row schedules

The large winner is unsuitable as a universal prefill schedule. Two separate
one-minute searches, followed by one-minute batched refinements, select smaller
tiles for each projection shape. The first searches use single dispatches;
the refinements use three warm batches of 100 and seven measured batches of
20, normalizing the score per dispatch. This reduces unstable small-kernel
ranking, but a batched score is not an ordinary-call latency. Each finalist
is validated again on held-out inputs and timed as individual completed calls.

Both winners use 32×32 tiles, 256 threads, 2×2 private microtiles, K-major A/B,
column ownership and FMA expressions. The 1024-depth winner stages K=16 and
unrolls by 8; the 2560-depth winner stages K=8 and unrolls by 4. Their search
scores are 135.3 and 225.6 µs. Individual-dispatch timestamps differ, especially
for the short workloads, so these search scores are not the headline results.

The following fresh comparison prepares both Tensor schedules and includes
queue completion for all three implementations:

| M×N×K and mode | Generic prepared (ms) | Tuned prepared (ms) | CLBlast (ms) | Tensor gain | Remaining gap |
|---|---:|---:|---:|---:|---:|
| 32×2560×1024 GEMM | 0.972 | 0.525 | 0.183 | 1.85× | 2.87× |
| 32×2560×1024 linear | 0.678 | 0.378 | 0.257 | 1.80× | 1.47× |
| 32×1024×2560 GEMM | 0.913 | 0.520 | 0.254 | 1.76× | 2.05× |
| 32×1024×2560 linear | 0.721 | 0.562 | 0.303 | 1.28× | 1.86× |

Submission/completion matters here: tuned pure-GEMM compute-pass timestamps
are 0.213 and 0.344 ms, versus completed calls of 0.525 and 0.520 ms. Linear
fuses the epilogue in Tensor; CLBlast uses another kernel. Different GPU timing
boundaries and noticeable short-call variability make it inappropriate to
claim universal GPU parity from a favorable linear timestamp. CLBlast retains
the completed-call advantage in every measured case.

These are FP32 projection-shape experiments. The actual LFM2 runner uses F16
weights, its own activation contract and fused model operations. Neither
these timings nor the large-GEMM ratios establish new prefill or decode tokens/s.

[Independent repeats, controlled unroll ablation and skinny comparisons](data/webgpu-outer-product-refinement.json).
[Initial gate search](data/webgpu-outer-product-gate32.json),
[down search](data/webgpu-outer-product-down32.json),
[batched gate refinement](data/webgpu-outer-product-gate32-batched.json),
[batched down refinement](data/webgpu-outer-product-down32-batched.json).

## Compiler schedule and lowering

`outer_product_matmul_schedule` in `src/tensor/compiler/webgpu_lowering.py`
generates a reusable TileLang tile program. It cooperatively loads contiguous
pieces of each row-major operand into shared storage, keeps all output
accumulators private across K, and updates register microtiles with outer
products. Bias/ReLU can be fused into the final private-to-global store.
Zero-filled M/N/K tails and guarded output stores support incomplete tiles.
Both FP32 and FP16 operand storage accumulate into FP32 outputs.

The large winner has a **64×128 output tile, K=16, 256 threads and a 4×8
register microtile**: 32 FP32 accumulators per thread, 12 KiB actual shared
storage, M-major shared A, K-major shared B, column ownership, and explicit
unrolling by 16. It uses scalar multiply-add expressions. The driver owns their
instruction selection; this experiment does not assert identical arithmetic
or bitwise output across Tensor and CLBlast.

The WebGPU producer also accepts `tensor.webgpu.loop_unroll="explicit"`.
It expands explicitly marked `T.unroll` loops before WGSL generation because
the existing WGSL path can retain those as ordinary shader loops. This opt-in
pass requires static marked extents of at most 16, leaves serial K loops intact,
and retains the later uniform-barrier and resource verification. The helper's
`explicit_unroll=True` emits this attribute. Existing programs are unchanged
unless they opt in.

Discovery now couples tile or register-microtile moves with the thread count
needed to preserve exact output ownership. A single-coordinate mutation cannot
cross that equality constraint. Padding, A layout, row/column ownership,
K staging, FMA expressions and unroll factors remain independent search axes.

## Discovery and correctness

The first eight-minute search considered **858** candidates: **305 passed**
the oracle and **553 were rejected**. Its best 4096³ timestamp was 34.703 ms.
The four-minute follow-up enabled explicit unrolling and coupled discovery,
considered **190** candidates, and accepted **127**, reaching 33.477 ms.
Compilation, pipeline creation, download and validation consume search budget
but are excluded from scores. The square search takes seven single-dispatch
timestamp samples after three warm dispatches. The bounded search is not proof
of a globally optimal schedule, and these two phases alone are not an unroll
ablation because their winning layouts and arithmetic differ.

Every scored candidate overwrites a NaN-filled output and passes the same
FP32 comparison gate as CLBlast: abs(error) ≤ .002 + .02*abs(reference).
The selected schedule additionally passes independent float64 dot-product
checks on 35×131×1027 tail shapes, GEMM and linear, three held-out seeds and
scales 1, .01 and 1e-5, with the much tighter forward-error bound
3e-6*sum(abs(products)) + 1e-10 and an analogous bias rounding term.
The new GPU tests also cover FP16 storage, both shared layouts, row/column
ownership, padding and unexpanded/expanded unroll. All 21 tests in the new
outer-product module pass, including eight physical Vulkan cases. The related
CPU regression subset passes 42 tests, including coupled discovery and the
existing generic-lowering guards.

[Initial square search](data/webgpu-outer-product-search4096.json).
[Explicit-unroll follow-up](data/webgpu-outer-product-explicit4096.json).

## Reproduction

Use the producer environment with TileLang, current Tensor, NumPy, wgpu and the
native prepared-plan extension. The comparison also needs PyOpenCL and the
existing stock CLBlast DLL and parameter probe built as described in the
[CLBlast report](clblast-rx6700xt.md). Run GPU jobs sequentially on an otherwise
idle RX 6700 XT, with `WGPU_BACKEND_TYPE=Vulkan` and
`OPENBLAS_NUM_THREADS=6`. CPU references finish before timing.

```powershell
.venv/Scripts/python.exe benchmarks/inference/webgpu_outer_product_search.py `
  --out build/clblast-chase/search4096 --minutes 8 --size 4096
.venv/Scripts/python.exe benchmarks/inference/webgpu_outer_product_search.py `
  --out build/clblast-chase/explicit4096 --minutes 4 --size 4096 `
  --explicit-unroll --seeds-report build/clblast-chase/search4096/report.json
.venv/Scripts/python.exe benchmarks/inference/webgpu_outer_product_compare.py `
  --out build/clblast-chase/comparison `
  --search-report build/clblast-chase/explicit4096/report.json `
  --consumer-site build/clblast-consumer/Lib/site-packages
```

`source(M,N,K,config,mode)` in the search harness emits a complete buildable
TileLang program using the compiler helper. The search reports preserve source
snapshots and hashes; the comparison preserves generated source, WGSL and
artifact hashes, validation and all timing samples. The retained refinement
includes source snapshots and the pinned CLBlast commit and DLL hash.

For the shape-specific refinements:

```powershell
.venv/Scripts/python.exe benchmarks/inference/webgpu_outer_product_search.py `
  --out build/clblast-chase/gate32-batched --minutes 1 --shape 32 2560 1024 `
  --explicit-unroll
.venv/Scripts/python.exe benchmarks/inference/webgpu_outer_product_search.py `
  --out build/clblast-chase/down32-batched --minutes 1 --shape 32 1024 2560 `
  --explicit-unroll
.venv/Scripts/python.exe benchmarks/inference/webgpu_outer_product_compare.py `
  --out build/clblast-chase/refinement `
  --refine-comparison build/clblast-chase/comparison/comparison.json `
  --skinny-reports build/clblast-chase/gate32-batched/report.json build/clblast-chase/down32-batched/report.json `
  --consumer-site build/clblast-consumer/Lib/site-packages
```

The recorded batched phases resume their initial single-dispatch finalists with
`--seeds-report build/clblast-chase/gate32/report.json` and the corresponding
down report. Current code batches small shapes automatically; source snapshots
in the older reports preserve the earlier single-dispatch protocol. Search
results are hardware/timing-dependent, not deterministic performance promises.

## Remaining ceiling

The large FP32 compute gap is now about twofold. The next candidates are
vectorized global/shared loads, alternative contiguous register ownership,
packing/layout transformations charged to the complete operation, and software
prefetch or double-buffered staging. Those alter instruction count, register
pressure, occupancy and shared-memory traffic; more coordinate search of this
same scalar schedule does not establish their potential. Small workloads also
retain meaningful WebGPU submission/completion cost. The producer schedule and
unroll option remain explicit until broader workload and adapter evidence
supports a default selection policy.
