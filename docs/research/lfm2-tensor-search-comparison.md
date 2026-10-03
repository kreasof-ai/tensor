# Tensor schedule search versus tinygrad and llama.cpp

On this RX 6700 XT, searching **Tensor-generated WebGPU/Vulkan kernels** reduced
the GPU time of the two LFM2.5-230M F16 projections by **2.85× and 5.93×**.
Fresh measurements put Tensor near llama.cpp's **FP32-accumulation** gate kernel
and ahead of its down kernel. Tinygrad's searched OpenCL kernels remain faster
on GPU execution. Tensor's prepared calls have lower overhead than tinygrad's
TinyJit calls in this experiment.

These are hot, isolated 32-row projections. They do not establish full-model
prefill or decode parity. The ordinary inference bundles retain their defaults.
The subsequent [full-model throughput comparison](lfm2-prefill-search-throughput.md)
integrates the winners through an opt-in profile and measures their effect on
prefill and decoding for all three frameworks.

## Independent comparison

All numbers are microseconds; lower is better. The model weights are native
GGUF F16, and all providers receive identical FP32 inputs rounded to nearest-even
FP16 values before upload. Accumulation and output use FP32 for the accepted
comparison. Host preparation and output downloads are excluded from timing.

| Projection `[M,K,N]` | Tensor baseline GPU | Tensor searched GPU | tinygrad searched GPU | llama.cpp FP32 GPU |
|---|---:|---:|---:|---:|
| Gate `[32,1024,2560]` | 163.59 | **57.30** | **35.02** | 56.47 |
| Down `[32,2560,1024]` | 360.28 | **60.72** | **52.85** | 70.14 |

GPU batch timing includes command sequencing. For Tensor, timestamps bracket
20 dispatches in one compute pass. For tinygrad, OpenCL events bracket 20 queued
launches of the saved kernel; the corresponding means of individual kernel
durations are **34.68/52.56 µs**. For llama.cpp, its Vulkan performance logger
times a graph of 20 independent `MUL_MAT` nodes, and the total is divided by 20.
That graph uses distinct outputs and writes a timestamp after each node, whereas
Tensor overwrites one output and uses two timestamps for the batch. The small
gate difference is within this experiment's methodological limits.

| Projection | Tensor baseline completed | Tensor searched completed | tinygrad completed | llama.cpp FP32 completed |
|---|---:|---:|---:|---:|
| Gate | 235.45 | **141.61** | 187.75 | **133.33** |
| Down | 443.15 | **145.95** | 206.28 | 158.75 |

Completed calls use the normal prepared Tensor plan, cached TinyJit, or a
one-node GGML Vulkan graph. Providers and Tensor finalists rotate order. Each
is warmed for one second; the first batch is discarded, then seven batches of
20 calls plus synchronization are measured. Native completed-call timings have
the performance logger disabled. The Tensor winner is selected by fresh
**batched GPU time**, rather than by whichever completed-call sample happens to
be smallest.

Tensor's completed-call improvements are **1.66×/3.04×**. Relative to llama.cpp,
Tensor's gate call takes about **6.2% longer**, while its down call takes about
**8.1% less time**. Relative to tinygrad, Tensor's completed calls take about
**24.6%/29.2% less time** despite slower GPU kernels.

## Search and generated programs

The combined active controller budget was **1,797.87 seconds (29m58s)**. It
considered **2,368 configurations**: **1,769** compiled, passed the independent
oracle, and were timed; **599** were rejected. Almost all rejections were
schedule legality/resource checks. No timed candidate failed the numerical
gate. One incomplete artifact left during a restart caused an output-exists
rejection; that seed was retried successfully and the harness now handles it.

The search uses an eight-wide beam, widening to sixteen, adjacent coordinate
moves, coupled thread/partition moves, and deterministic restarts. It retains
two candidates per schedule family so a slower initial seed can still lead to
a useful neighbor. This is parameterized schedule discovery, not arbitrary TIR
rewriting or exhaustive optimization.

Two families are available:

- Existing staged register GEMM: M/N/K tile dimensions, threads, scalar or
  four-wide dot, inner unrolling, and shared LHS padding/transposition.
- New direct-load GEMM: workgroup-local K partitioning, row or column ownership,
  blocked or striped K distribution, scalar/two-wide/four-wide dot, unrolling,
  M/N tiles including non-power-of-two N choices, and thread count.

The latter is implemented in the reusable compiler schedule helper
`partitioned_matmul_schedule`. It retains private FP32 partial accumulators,
stores them into workgroup memory, synchronizes once, then reduces K partitions
locally. It requires neither atomic output updates nor a second dispatch.
The GPU-independent discovery layer is `tensor.compiler.webgpu_search`; the
benchmark supplies compilation, the oracle, Vulkan timing, and checkpoints.

The search resumed twice while retaining prior candidates and subtracting
elapsed controller time from the original 30-minute budget: first to add row
ownership, striped K and two-wide dots, then to preserve exploration of each
family. Restarts were sequential; no GPU search or measurement overlapped.

| Fresh Tensor winner | Gate | Down |
|---|---|---|
| Candidate | 1024 | 476 |
| Output tile | 16×8 | 4×8 |
| Threads | 128 | 64 |
| K partitions | 16 | 16 |
| Ownership | row | row |
| K distribution | blocked | striped |
| K unroll / dot width | 4 / 4 | 4 / 4 |
| Workgroup partial storage | 8 KiB | 2 KiB |

Gate candidate 856 won the search's original-input timing; candidate 1024 won
the independent rerun among four distinct finalists. Their fresh GPU times were
58.33 and 57.30 µs. This is why the final comparison uses remeasurement rather
than the lowest search observation. Down candidate 476 won both.

The tinygrad programs replay the prior 30-minute search's BEAM=8 winners,
without a new search. Its pinned revision is
`91b8cb5fa6c031c5a7440159d955f66952c5e2e9`. Llama.cpp uses the existing b11310
Vulkan binaries at revision `f872b591121761ac7b2af18283bd99bdc092a63a`.
Tensor and llama.cpp use Vulkan; tinygrad uses AMD OpenCL. The fresh rerun's
adapter and limits are recorded in the raw evidence.

## Correctness and measurement limits

During Tensor search, each candidate receives the original seeded FP32 input,
native F16 weights, and an output filled with NaNs before launch. Its result must
be finite and satisfy a float64 NumPy dot of nearest-even FP16 operands:

`abs(actual-reference) <= abs(lhs) @ abs(rhs).T * 3e-6 + 1e-10`.

The fresh comparison checks every finalist and both other frameworks on seeds
29, 101, 202 and 303, with scales 0.01, 0.01, 1 and 0.00001. The held-out scale
includes FP16 subnormals. The direct tinygrad timestamp runner is checked as
well as TinyJit. All 20 outputs of the batched native graph are validated.
Twelve native GPU tests additionally cover odd M/N dimensions, row/column
ownership, both K distributions and dot widths 1/2/4. Nine discovery/legality
tests pass; the existing CPU lowering checks also pass.

The comparison requests FP32 accumulation through GGML's
[precision API](https://github.com/ggml-org/llama.cpp/blob/f872b591121761ac7b2af18283bd99bdc092a63a/ggml/include/ggml.h#L1442).
The default native accumulator path was separately measured: **61.84/99.12 µs**
for gate/down, and it failed the shared oracle on all four fixtures for both
matrices. Its maximum seed-29 errors were approximately **0.000461/0.000667**,
versus the FP32 path's **4.67e-8/5.68e-8**. It is retained as a different-precision
diagnostic, not counted as an accepted winner.

WebGPU timestamp units use the previously verified **10 ns Vulkan
timestampPeriod**. A 200-dispatch calibration gave a raw-tick interval of
4.2393 ms if interpreted as nanoseconds versus 44.726 ms completed wall time,
confirming the required factor of ten. Candidate ranking warms three batches
of 100 dispatches, then takes the median of seven 20-dispatch timestamp batches.
The rerun retains both individual and batched timings. Per-call timestamp
readback and logger synchronization influence GPU duty cycle, so single-call
timings are not substituted for the primary batched comparison.

These two hot matrices fit the GPU's cache hierarchy. No occupancy, register
allocation or hardware instruction counters were collected. The remaining
tinygrad GPU advantage points to useful additional experiments, particularly
packed/vector loads and hoisting operand-rounding work; this run does not prove
which instruction or resource accounts for that gap.

## Evidence and reproduction

[Raw archive](data/lfm2-tensor-search-comparison.json) contains every Tensor
candidate, search phases, checks, timing samples, fresh finalists and programs,
compiler/runtime manifests, tinygrad source, native profiler logs, rejected
default-precision results, and harness sources/hashes. Local `.tbin` winners
remain under `build/tensor-search/search-30m/`.

Use the existing producer environment and pinned tinygrad clone. Searches are
sequential GPU jobs. Set `WGPU_BACKEND_TYPE=Vulkan` for the Tensor runs.

```powershell
.venv/Scripts/python.exe benchmarks/lfm2/tensor_projection_search.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/tensor-search/new-search --minutes 30

$env:PYTHONPATH=Join-Path (Get-Location) 'build/tinygrad-comparison/upstream'
$env:DEV='CL'; $env:BEAM='0'; $env:JITBEAM='2'; $env:BEAM_ESTIMATE='0'; $env:PARALLEL='0'
$env:CACHEDB=Join-Path (Get-Location) 'build/tensor-search/new-recheck/cache.db'
.venv/Scripts/python.exe benchmarks/lfm2/projection_search_recheck.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --search-root build/tensor-search/new-search --tiny-root build/tinygrad-comparison/long-search-30m --llama-directory build/llama-vulkan-b11310 --out build/tensor-search/new-recheck

.venv/Scripts/python.exe benchmarks/lfm2/llama_projection_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --directory build/llama-vulkan-b11310 --out build/tensor-search/new-native-profile --profile --graph-repeats 20
```

For archival profiling, redirect child stdout and stderr to the **same OS file
handle** using `subprocess.run(..., stdout=log, stderr=subprocess.STDOUT)`.
PowerShell's separately captured native streams can reorder the Python batch
markers and C++ profiler lines. The archived logs use the shared handle.

Rerunning the current harness starts with the complete action space; its search
trajectory will differ from this recorded run's two incremental broadening
phases. `--resume` preserves candidate history and subtracts elapsed time from
the combined budget. Search timing is device-local evidence, not a promise that
a wider beam always wins or that 30 minutes finds the global optimum.
