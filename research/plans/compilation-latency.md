# Compilation latency and benchmark turnaround

Planning baseline: **2026-10-18**. Status: proposed measurement and improvement
plan. Compilation latency is a product acceptance criterion alongside execution
performance and validation memory. No latency measurements or speedups are
claimed by this document.

The [compiler design literature review](../studies/compiler-design/reports/literature-review.md)
connects these requirements to incremental-build design and shared dynamic-shape
search. Its LIT-04 experiment evaluates the dependency/cache architecture; LIT-01
checks whether shared exploration reduces repeated compile/qualification work.

Inspected revisions: Tensor `274d429ed9945b45ea5b8e492248355e1611a047` and
AKAS `5259580d29f1d0182f1346f43cd8ced15913a4e2`. The reported user experience is
slow compilation during AKAS benchmarking; the exact invocation, model, elapsed
time and cache state have not yet been captured.

## Current path and hypotheses

The user-visible wait includes graph construction, source/dependency processing,
kernel lowering, device compilation, reference generation, correctness checks,
tuning measurements, artifact publication and model initialization. A Python
entry point does not establish which stage dominates. Profile stage durations
and native work before considering a language change.

| Inspected behavior | Consequence to measure |
|---|---|
| AKAS `compilation/compile.py` enables search by default; `SearchSettings` defaults to eight search trials, in addition to baseline/final verification work | A compile command can include an optimization campaign across operation requirements and geometries |
| AKAS builds phase geometries, operations and trial candidates through sequential loops | Repeated setup and many small compilations can accumulate; concurrency is a later, memory-budgeted decision |
| [Tensor CUDA build](../../src/tensor/compiler/cuda/build.py) performs export, IR serialization and TileLang lowering before its binary cache lookup | A cubin cache hit skips device compilation, but does not skip the preceding frontend work |
| [Header hashing](../../src/tensor/compiler/cache.py) reads compiler include trees, and [NVRTC identity](../../src/tensor/compiler/cuda/nvrtc.py) hashes library files on each identity call | Stable toolchain fingerprinting may repeat for every kernel entering Tensor's build path |
| Tensor's build report separates `seconds`, `compile_seconds` and `cache_hit`; its main timer starts after package/provider/toolchain checks | Existing fields are partial timings, not command-to-ready latency or a complete stage breakdown |
| AKAS `compilation/emit.py` invokes Tensor but retains artifact/checksum records rather than its full timing report | Per-kernel compiler timing is not carried through this path into an end-to-end build breakdown |

These are observed execution paths and optimization candidates, not measured
attribution of the user's delay. Existing AKAS artifact pools and search caches
can bypass some work; measure each cache layer separately before changing it.

## Benchmarking workflow

In the inspected AKAS CLI, `bench --compile` invokes `compile_checkpoint()` before
measurement. For a program definition it requests decode/prefill geometries for
the selected batches in one program build; other profiles build per batch.
Lower-level caches can reuse work, but the compile entry point is still called.
The research `prepare()` helper also calls `compile_program()` directly.

Benchmark inference metrics intentionally exclude compilation, loading, reset and
warm-up. Preserve that timing protocol, while separately reporting:

- command-to-first-successful benchmark sample;
- preparation/build/qualification and model load/capture time;
- benchmark measurement time and total matrix/campaign wall time;
- artifact/profile reuse, cache misses and their invalidation reasons.

The current CLI can measure existing compatible bundles by omitting `--compile`.
Make preparation and measurement explicit in the product workflow, with a checked
reuse path for unchanged programs. Repeated measurements should reuse qualified
artifacts without repeating search. Changed source, target, shape domain,
precision or numerical contracts must trigger the required invalidation.

AKAS references: [benchmark compile dispatch](https://github.com/kreasof-ai/akas/blob/5259580d29f1d0182f1346f43cd8ced15913a4e2/src/akas/cli.py),
[compilation coordinator](https://github.com/kreasof-ai/akas/blob/5259580d29f1d0182f1346f43cd8ced15913a4e2/src/akas/compilation/compile.py),
[qualification](https://github.com/kreasof-ai/akas/blob/5259580d29f1d0182f1346f43cd8ced15913a4e2/src/akas/compilation/tuning.py),
and [benchmark protocol](https://github.com/kreasof-ai/akas/blob/5259580d29f1d0182f1346f43cd8ced15913a4e2/docs/benchmarking.md).

## Measurement contract

Add correlated stage events from AKAS into Tensor, using the
[tuning workbench's shared event/report contract](tuning-workbench.md). Capture
CLI startup/imports, checkpoint inspection, graph construction/rewrites,
dependency hashing, frontend export, IR lowering, toolchain identity, cache
lookup, NVRTC/NVCC execution, artifact I/O, reference generation, device upload,
correctness comparison, tuning timing and final load/capture/publication.

For each event record wall time, parent/correlation ID, operation and phase,
kernel/region/candidate identity, cache level and hit/miss reason, bytes processed,
peak/retained host/device memory and candidate status. Distinguish unique
requirements, requested trials, actual lowerings, actual device compilations and
cache copies. Use CPU sampling/native profiling when the stage trace identifies
an expensive region. With concurrent workers report critical-path wall time;
adding inclusive child durations can overcount the user's actual wait.

Start with these reproducible cases on a named model/target and fixed settings:

| Scenario | Acceptance intent |
|---|---|
| First build with cold producer caches | Establish command-to-usable-artifact and peak-memory baseline |
| Repeat in the same output directory | Reuse matching qualification and artifacts; explain any repeated expensive work |
| New output directory with shared populated caches | Reuse verified content independently of publication paths |
| One kernel or schedule change | Rebuild/requalify the affected dependency closure |
| Another benchmark repetition or live shape inside the compiled domain | Reuse the program without new code compilation |
| Deliberate deeper tuning | Report extra preparation cost and resulting execution improvement separately |

Record process-cold versus process-warm state, source/toolchain identities,
checkpoint/storage state, measurement repetitions and dispersion. Set numerical
latency/memory regression budgets from these baselines; do not invent a universal
seconds-per-model target before collecting them.

## Optimization order

1. **Expose the wait.** Retain stage timings and show the active phase/candidate,
   elapsed time, completed/remaining known work and cache activity. Long compiler
   calls should remain visible; counts must not imply a fabricated overall ETA.
2. **Reuse build-session invariants.** Snapshot toolchain/header identities and
   declared source/dependency fingerprints once per build session where valid.
   Detect mutations or restart the snapshot. Preserve content-based invalidation;
   timestamps or package version strings alone cannot replace it.
3. **Cache frontend work as well as images.** Add reusable lowered IR/source and
   complete artifact layers ahead of expensive lowering when identities permit.
   Early reuse requires the declared source/dependency, option and environment
   closure. Arbitrary Python exports can be non-hermetic: preserve execution/IR
   verification when that closure is unavailable. Keep corruption checks and
   target/toolchain/numerical-policy separation.
4. **Deduplicate qualification and compilation.** Share compatible operation
   requirements across layers, phases and geometries, carrying the exact storage,
   mapping and numerical contracts. Reuse qualified schedules and references with
   bounded caches. Keep deliberate fresh-finalist verification visible and scoped.
5. **Offer an explicit preparation budget.** Distinguish a usable qualified build
   from additional optimization, preserving a known-correct measured baseline.
   An initial build must include all phases and shape-domain coverage required by
   the requested workload. Report search depth, performance qualification and
   unmeasured regions; never silently trade correctness for a faster build.
6. **Add bounded concurrency where measured.** Schedule independent compilation
   tasks with CPU/RAM limits and appropriate worker isolation. GPU correctness
   and performance measurements need controlled scheduling; parallel compiler,
   BLAS and validation workers must not oversubscribe memory or contaminate timings.

Cancellation and resumption must retain usable artifacts and completed trial
evidence. Define deadline behavior explicitly: checking a budget between tasks
does not interrupt an in-progress compiler call. Publication remains atomic and
failed builds preserve the previous usable program.

## Relation to dynamic shapes and release gates

The [dynamic-shape plan](dynamic-shapes-and-batch-invariance.md) adds region
portfolios and multi-shape qualification, which can increase producer work.
Budget region count, trials, reference memory and total build time together.
Prefer measured useful partitions and shared compilation to enumerating every
integer shape. Keep runtime performance acceptance alongside compilation UX;
measure both costs for each producer mode.

Treat tracing, warm/incremental reuse and validation memory as early compiler
work, useful independently of dynamic-shape delivery. Tensor owns compiler-stage
instrumentation and safe reuse; AKAS owns phase/benchmark orchestration, workload
selection, qualification scheduling and command-to-ready reporting. Each layer
must clean up the resources it creates.

Exit evidence includes a representative benchmark invocation with complete stage
attribution, cold/warm/incremental measurements, verified invalidation behavior,
bounded memory, useful progress/cancellation and agreed latency regression gates.
This plan proposes no Python rewrite or speedup claim without that evidence.
