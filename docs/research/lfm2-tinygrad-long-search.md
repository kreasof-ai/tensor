# Wider LFM2 projection search with a 30-minute cap

This follows the [initial tinygrad comparison](lfm2-tinygrad-comparison.md)
on the same RX 6700 XT, Windows/OpenCL driver and LFM2.5-230M F16 checkpoint.
The experiment widens the existing schedule action space, rather than adding
new algebraic transformations or changing Tensor's shipping compiler.

**Result:** the run finishes in **29 minutes 49 seconds**. Independent
remeasurement confirms **1.12x/1.18x GPU speedups** for the gate/down
projections over the previous BEAM=2 winners. Completed-call speedups are
**1.03x/1.06x** because submission overhead remains included. Both winners
come from beam width 8; width 16 reaches its time cap and retains the old
schedules. [Raw evidence](data/lfm2-tinygrad-long-search.json) includes all
stage reports, timing batches, generated sources and validation metrics.

The combined search cap is **30 minutes across both prefill FFN projections**.
Previously validated BEAM=2 schedules remain fallbacks. The controller tries
beam widths 4, 8 and, if time remains, 16; upcast limits 64/128/256 and local
products 512/1024/1024. GPU execution is sequential. Each stage stops on
convergence or its deadline; it is not forced to consume its allocation.

| Stage | Wall seconds | Compilation requests | Passed candidate checks | Stop |
|---|---:|---:|---:|---|
| Gate, beam 4, upcast 64, local 512 | 163.30 | 352 | 282 | Converged |
| Down, beam 4, upcast 64, local 512 | 178.28 | 395 | 337 | Converged |
| Gate, beam 8, upcast 128, local 1024 | 435.24 | 881 | 625 | Converged |
| Down, beam 8, upcast 128, local 1024 | 600.62 | 1,172 | 880 | Converged |
| Gate, beam 16, upcast 256, local 1024 | 210.41 | 568 | 457 | Time cap |
| Down, beam 16, upcast 256, local 1024 | 201.07 | 549 | 457 | Time cap |

There are 3,917 compilation requests and 3,038 successful candidate checks,
including the fallbacks. These are evaluations across stages, not globally
unique programs. One compilation request fails; no executed candidate fails
the numerical gate. Separate private caches prevent a narrower search result
from short-circuiting a wider stage. The independent final remeasurement and
storage-correction validation run occur after the combined search cap.

## Correctness and deadlines

`tinygrad_search_budget.py` adds benchmark-local hooks to the pinned upstream
BEAM loop, without editing upstream. Search uses the actual block-0 F16
weight and seeded 32-row activation matrix. Before timing every candidate,
its output is filled with NaNs so a missing/partial write cannot reuse a
previous valid output. After timing, the output must be finite and satisfy
the independent float64 rounded-operand dot oracle:

`abs(actual - reference) <= abs(lhs) @ abs(rhs).T * 3e-6 + 1e-10`.

Incorrect candidates receive infinite latency and cannot become incumbents.
Full shapes are timed (`BEAM_ESTIMATE=0`); search is single-process
(`PARALLEL=0`). The best valid schedule and generated source are checkpointed
after improvements; progress counts are saved periodically. The original
passing schedule is timed and checked first.

The cooperative deadline is checked between compilation/timing operations.
A compilation already in progress may slightly overrun that stage's soft
deadline. The controller also enforces a combined process deadline. On
Windows, the harness terminates the child process tree on a hard timeout,
including a venv launcher and its Python child. Checkpointed schedules can
then be replayed without resuming search.

The short deadline smoke run passes, as do three CPU tests: rejection of
partial outputs despite a previously valid buffer, retention of the valid
checkpoint at a deadline, and restoration of compiler hooks after interruption.

## Independent remeasurement

Search ranking uses the minimum of the upstream GPU timing samples; those
numbers are candidate-selection observations. Final comparison replays the
previous BEAM=2 fallback and the wider-stage schedules on the **same OpenCL
device, weight and input buffers**, with search disabled.

Each variant is warmed for one second. Variants rotate order, with one
discarded and seven measured batches of 20 calls. The report separates:

- **GPU time:** OpenCL events, median of per-batch medians.
- **Completed-call time:** TinyJit launches plus queue completion, with host
  output copy excluded.

The direct GPU timing call's output is separately checked against the oracle.
Every finalist also passes three held-out activation fixtures: seeds 101, 202
and 303, scales .01, 1 and .00001. These stress different value ranges,
including half subnormals. The original seed 29 remains the search fixture.

| Projection / schedule | GPU us | Completed-call us |
|---|---:|---:|
| Gate, previous beam 2 | 42.46 | 183.25 |
| Gate, beam 4 | 41.84 | 190.11 |
| **Gate, beam 8** | **37.86** | **178.01** |
| Gate, beam 16 | 42.48 | 186.78 |
| Down, previous beam 2 | 46.48 | 187.95 |
| Down, beam 4 | 46.44 | 187.63 |
| **Down, beam 8** | **39.50** | **176.57** |
| Down, beam 16 | 46.50 | 183.10 |

The wider gate winner keeps 10 partial accumulators and the same 5 KiB local
reduction buffer, but reduces its serial loop from 64 to 16 iterations through
greater K unrolling and changes the thread mapping. The down winner increases
partial accumulators from 8 to 16 per thread, halves the workgroup from 256 to
128 threads, and reduces the serial loop from 160 to 20 iterations. Its local
reduction buffer remains 8 KiB. Both still use direct operand loads and a
single barrier before combining partial sums. These are generated-source
observations; register allocation and occupancy counters were not collected.

Width-16 finalists and the width-4 down finalist have source hashes identical
to the previous schedules. Their small timing differences are repeated
measurement variation, not new schedule discoveries. The width-8 sources are
different and win in the independent timing cohort. Longer/wider search finds
additional kernel gains here, with diminishing returns and a smaller benefit
at the completed-call boundary. Tensor must first express this direct-load,
workgroup-partitioned K family before its own search can explore the same area.

This remains a hot-matrix scheduling experiment. It does not establish
whole-model throughput, a compiler hardware ceiling, or an equally sized
benefit after transferring a schedule into Tensor's WebGPU lowering.

## Earlier full-model storage correction

Inspection during this run finds that the historical full-model adapter used
`GGUF.array`'s default FP32 dtype, expanding F16 GGUF weights to FP32. Its
earlier full-model timings were not a matched native-F16 storage comparison.
Those raw samples remain preserved and are explicitly labeled in the original
report. The projection harness has always loaded packed native F16 views;
**none of this search or remeasurement is affected**.

The full-model adapter now preserves GGUF F16 storage and asserts that every
F16 tensor remains half-typed. A fresh OpenCL run passes all 19 unchanged
NumPy fixtures and bitwise reset. That is a correctness-only run, with no new
full-model throughput claim. The evidence includes its report and adapter
source hash.

## Reproduction

Use the pinned tinygrad consumer environment from the initial report. The
existing BEAM=2 cache supplies the validated fallback. The controller sets
private cache paths and search limits for each stage:

```powershell
python benchmarks/lfm2/tinygrad_long_search.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-230m-f16-webgpu --out build/tinygrad-comparison/long-search-30m --fallback-cache build/tinygrad-comparison/projection-cl-beam2/cache.db --minutes 30
```

After all GPU search processes have stopped, remeasure the cached schedules:

```powershell
$env:DEV='CL'
$env:BEAM='0'
$env:JITBEAM='2'
$env:PARALLEL='0'
$env:CACHEDB=Join-Path (Get-Location) 'build\tinygrad-comparison\long-search-recheck\cache.db'
python benchmarks/lfm2/tinygrad_search_recheck.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --root build/tinygrad-comparison/long-search-30m --fallback-cache build/tinygrad-comparison/projection-cl-beam2/cache.db --out build/tinygrad-comparison/long-search-recheck
```

CPU boundary tests run in that optional environment:

```powershell
python -m unittest tests.test_tinygrad_search_budget -v
```
