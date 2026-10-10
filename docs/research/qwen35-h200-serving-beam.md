# H200 serving beam search

An actual beam search and its subsequent candidates have measured **2,126.824 output tok/s** on one H200,
at C8 with eight distinct 32,000-token prompts and 16,000 output tokens per
request. This is an experimental timing: verification still fails its
comparison with serial native decoding, and the canonical model is unqualified.
The 3,000 output tok/s target has not been reached.

## Fixed measurement

The checkpoint remains `Qwen/Qwen3.5-35B-A3B-FP8`, revision
`9d1823d2dee688a6b25e77009dc727688c44936e`. Weights retain the official
block-128 FP8 values and scales. KV uses FP8 and recurrent state uses FP32.
There is one resident target, its embedded MTP head, no CPU offload, and no
prefix cache. All requests arrive together, with no warmup.

The workload semantic SHA is
`6e012fe014f8fc86d58d0065862c62e77e1374fccba612a7ab5d54c1679db44a`.
Every accepted replay completes eight requests, processes 256,000 prompt
tokens and produces 128,000 output tokens, with zero failures. The objective
is **128,000 divided by complete HTTP client elapsed time**, including prompt
processing, cohort initialization, and fill/drain effects. Adding input and
output token rates does not satisfy the output throughput target.

MTP and output-history lookup propose tokens; all proposals go through target
verification and independent per-request prefix commits. The synthetic prompts
produce substantial repetition, which helps history lookup. These measurements
do not establish equivalent performance on natural language requests or an
advantage over Netra's different model and insufficiently specified protocol.

## Completed observations

These are individual observations without repeated-run dispersion estimates.
The paired and asynchronous profiles use the same official FP8 weights with
temporary **FP16 shared-memory operands**; their historical CLI names start
with `bf16` but do not describe those operands accurately.

| Prefill / verification profile | Window | Whole-client output tok/s | Elapsed s |
|---|---:|---:|---:|
| Initial H200 port | 8 | 911.544 | 140.421 |
| Packed FP8 gathers, ordinary warp MMA | 8 | 1,069.126 | 119.724 |
| Packed FP8 gathers, ordinary warp MMA | 32 | 1,718.994 | 74.462 |
| Paired FP16 WGMMA prefill | 32 | 1,908.490 | 67.069 |
| Asynchronous paired prefill | 64 | 1,919.737 | 66.676 |
| **Paired prefill, beam winner** | **64** | **2,003.071** | **63.902** |
| Asynchronous paired prefill, later beam replay | 32 | 1,903.485 | 67.245 |
| Asynchronous paired prefill, later beam replay | 128 | 1,890.806 | 67.696 |
| Asynchronous prefill, independent 128-window replay | 128 | 1,868.863 | 68.491 |
| Asynchronous prefill, 16-query attention tile | 64 | 1,860.152 | 68.812 |
| Corrected two-stage prefill pipeline | 64 | 1,611.762 | 79.416 |
| Small warp-MMA verification expert tiles | 64 | 1,422.458 | 89.985 |
| Paired prefill, MTP fallback depth one | 64 | 1,945.031 | 65.809 |
| Packed prefill, later beam replay | 64 | 1,806.621 | 70.850 |
| Paired prefill, 128-window beam replay | 128 | 1,994.057 | 64.191 |
| Persistent 64-tile verification grid | 64 | 1,820.286 | 70.319 |
| Repaired packed two-stage prefill pipeline | 64 | 1,943.377 | 65.865 |
| **Paired prefill with padded Hopper attention** | **64** | **2,053.872** | **62.321** |
| **Padded prefill attention and paired verification dense projections** | **64** | **2,126.824** | **60.184** |

The first beam winner's target-only prompt processing takes 30.067 s and its complete client
run samples a peak of 73,461 MiB GPU memory. The newer padded-attention candidate
reduces target-only prompt processing to 27.580 s. The final dense-verification
candidate retains that prefill and measures 27.546 s for target-only prompt
processing. Larger verification windows alone
have not shown another improvement. The GPU has enough memory for the tested
128-token verification profile; the window is not request concurrency.

## What the beam actually searches

`benchmarks/qwen35/modal_beam.py` uses the compiler's existing `ScheduleSearch`
with width two. Its 24 configurations combine three prefill kernel variants,
verification windows 8/32/64/128, and MTP fallback depths 1/3. Four earlier full
replays seed the ranking. The first bounded search measures four additional
configurations, chooses neighboring configurations from the current beam, and
retains a trace of observations, seen configurations and pending candidates.
Two subsequent beam expansions measure fallback depth one, packed prefill with
window 64, and paired prefill with window 128. The frontier refreshes after each
observation so a small budget follows the current winners. The latest trace
adds the measured padded-attention profile as a fourth prefill choice, expanding
the space to 32 configurations. The retained frontier records unmeasured
configurations; it does not imply that more jobs are running or scheduled.

A candidate can enter the experimental beam only after physical GPU primitive
checks, initial and 32K same-state prefill gates, exact workload identity, and
the complete C8 replay. Hardware and token counts are checked, and the reported
rate must equal the fixed output count divided by elapsed time. The model
qualified beam additionally requires serial and model qualification. It is
currently empty. A kernel microbenchmark never substitutes for this objective.

Reproduction starts with prepared, source-bound control and verification
bundles on the persistent `tensor-qwen35-h200` Modal volume. The retained
prepared manifests identify those paths and source hashes. The search entrypoint
accepts a JSON inventory containing `prefixes`, `windows`, `observations` and
optional `seeds`. The retained inventory uses repository paths; run from the
repository root. Zero candidates reproduces the ranking without allocating a
GPU; a positive candidate count starts fresh measurements:

```bash
PYTHONPATH=src:packages/tensor-llm/src:. modal run \
  benchmarks/qwen35/modal_beam.py::beam_main \
  --inventory-file docs/research/data/qwen35-native-h200/serving-beam/inventory.json \
  --candidates 0 --width 2 --out build/qwen35-h200-beam-reproduction
```

Artifact producers regenerate controls when compiler/runtime identities change;
they do not overwrite recorded source hashes to load an incompatible bundle.
Each GPU replay has a 20-minute bound. CPU preparation uses separate functions.
Successful primitive checks may be reused only with matching source hashes,
package versions, driver, device and exact test command, and an intact retained
log. Full-model prefill comparisons and complete HTTP replays still run for
every measured candidate.

## Hopper implementation and numerical evidence

The original `sm_90` FP8 warp MMA is expanded into FP16 matrix operations in
the retained H200 disassembly. Moving to wide WGMMA changes reduction
association. Real dense projections showed RMS differences below `8e-8`, but
those differences can grow through quantization and recurrent state. Several
wide BF16 and native FP8 WGMMA prefill candidates failed the unchanged 1e-5
whole-prefill gate and were rejected before throughput measurement.

The selected prefill converts exact FP8 values into FP16 operands, reorders
each 32-element reduction to match the old instruction's two 16-element
groups, and explicitly preserves FP32 additions and scaled accumulation.
This restores bitwise agreement in focused projection comparisons, including
all finite FP8 encodings, signs and cancellation. Both initial and 32K full
prefill comparisons report zero logit and recurrent-state RMS differences and
8/8 greedy matches, including inactive slots and uneven request lengths.

The compiler now supports explicit `sm_90a` artifacts with ordinary pointer
arguments. The runtime accepts them only on compute capability 9.0. TMA and
warp specialization stay disabled in this path. Old compatible `sm_90`
artifacts continue to disable WGMMA. The target/ABI changes passed CPU target
validation and NVRTC ABI checks; the retained full replays include physical
H200 projection, causality, and recurrent rollback checks.

This kernel agreement does not resolve model accuracy. The winner's two-step,
teacher-forced serial check at the initialized 32K prefix measures 8.41076%
relative logit RMS and 13/16 matching greedy tokens; its threshold is 3% with
all greedy tokens matching. Both qualification flags remain false. Speculative
verification must agree with the intended target before a qualified throughput
or lossless speculative-decoding claim is made.

A separate per-layer diagnosis finds the first substantial difference in full
attention. Matching the expert reduction order and using 64-key attention
blocks passes a narrow two-token diagnostic (2.960% logit RMS, 16/16 greedy
matches). Extending the same comparison to the full 64-token window fails:
15.937% RMS and 504/512 greedy matches. Single-token query tiles compiled for
ordinary `sm_90` also fail that wider check (15.847%, 506/512). These are quality
diagnostics, not throughput results, and do not qualify speculative decoding.

## Captured phase diagnosis

CUDA event nodes were recorded inside the captured graph, using external event
record flags so they produce timestamps. A separate diagnostic profiles the
asynchronous paired profile at the initialized 32K prefix:

| Sum of captured kernel intervals, across layers | Prefill chunk 512 | Verification window 32 |
|---|---:|---:|
| Attention | 207.015 ms | 9.888 ms |
| Expert projections | 118.250 ms | 16.483 ms |
| Dense FP8 projections | 92.231 ms | 9.805 ms |
| GDN scan | 31.912 ms | 5.565 ms |

These intervals exclude host dispatch gaps between launches but include event
instrumentation overhead. They are diagnostics, not complete-client throughput
or achieved-bandwidth measurements. They show why the next candidates target
prefill attention, projection pipelines, and verification expert tile sizes.
Increasing the speculative window does not remove these costs.

The first two-stage pipeline and a 64-column prefill candidate produced
non-finite values in the full-model gate. Recompiling the old attention body
directly for `sm_90a` also failed that gate. No throughput result is credited to
these candidates. Removing the pipeline's packed-buffer alias passed focused GPU tests and both
full-prefill comparisons, but its complete replay was slower at 1,611.762 tok/s.
The smaller warp-MMA verification tile also slowed the complete replay. The
search keeps the faster measured configuration. New candidates require fresh
numerical checks before entering the search.

The corrected padded-attention candidate retains the original 16-key softmax
updates, explicitly zeros padding, and synchronizes encoded KV loads before
reusing shared storage. Its primitive check poisons unused cache bytes and
scales to expose invalid reads. Both initial and 32K full-model comparisons
then pass with zero logit and state RMS differences. The complete replay
passes 53 physical primitive checks, with eight unrelated cases skipped.
Its server phase totals are 26.531 s verification and 1.604 s MTP repair over
316 rounds; 127,992 of 128,370 proposed input positions are accepted. This
acceptance rate reflects the experimental verifier and is not evidence of
serial equivalence. Verification remains the largest decode cost.

The last candidate replaces only the selected verifier's dense projections
with paired FP16 WGMMA. It preserves the existing Split-K partitions and scale
arithmetic. Both paired and asynchronous factories pass separate eight-part
bitwise checks with all finite FP8 encodings and tail rows. The paired variant
completes the unchanged replay at 2,126.824 output tok/s with eight finished
requests, zero failures, and zero error in both full-prefill gates. Its serial
comparison still fails at 8.41076% RMS and 13/16 greedy matches. This follow-up
was measured separately from the 32-configuration serving beam; it does not
expand that beam's recorded search space or qualify the target model.
The final server records 25.012 s verification and 1.592 s MTP repair over
the same 316 rounds, reducing verification from the padded-prefill replay's
26.531 s. Reaching 3,000 output tok/s on this fixed load would require reducing
complete client elapsed time from 60.184 s to below 42.667 s; that was not achieved.

## Retained evidence

- [Beam trace and observations](data/qwen35-native-h200/serving-beam/beam.json)
- [Latest beam trace](data/qwen35-native-h200/serving-beam/beam-v4.json)
- [First beam winner: client streams, telemetry and numerical checks](data/qwen35-native-h200/serving-beam/paired64/summary.json)
- [Padded attention replay](data/qwen35-native-h200/serving-beam/padded-attention64/summary.json)
- [Final experimental best: paired verification dense projections](data/qwen35-native-h200/serving-beam/paired-dense64/summary.json)
- [Captured GPU phase records](data/qwen35-native-h200/captured-hopper-profile/summary.json)
- [Packed expert replay](data/qwen35-native-h200/packed-experts-selected/summary.json)
- [Real projection comparison and original disassembly](data/qwen35-native-h200/hopper-diagnosis/)
- [Rejected pipeline](data/qwen35-native-h200/pipeline-rejected/)
- [Rejected direct WGMMA attention](data/qwen35-native-h200/attention-rejected/)

The earlier [phase diagnosis](qwen35-h200-expert-scheduling.md) and
[initial H200 report](qwen35-native-h200.md) remain unchanged historical records.

## Closing the experiment

The experiment concluded at the user's request after the final dense
verification candidate. All Modal experiment apps and the retained local L40S
model process are stopped. No further candidate runs are scheduled. The results
establish an experimental speed improvement from the initial H200 port; they
do not establish correct canonical model inference, lossless speculation, or
an advantage over another engine on a matched protocol.

CPU serving, beam and target checks pass (50 tests), and explicit Hopper
target/ABI compilation passes with NVRTC (14 tests). The final dense candidate's
physical H200 checks pass (55 tests, eight unrelated cases skipped). The
separate serial comparisons fail as recorded above. Throughput and quality
qualification remain separate outcomes.
