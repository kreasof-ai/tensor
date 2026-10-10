# H200 bandwidth optimization

Optimization resumed on 2026-10-10 with the unchanged goal of more than 3,000
output tok/s at C8, 32,000 input tokens and 16,000 output tokens per request.
All new compute runs use the Modal **akbar2habibullah** profile and workspace.
The compiled control bundles were copied byte for byte from the former
`kreasof-ai` volume, preserving their artifact checksums and source identities.
The pinned model was downloaded into the new workspace.

The latest completed replay measures **2,414.471 output tok/s**, with mean
client TTFT **25.645 s**. The target remains open. These are experimental
timings: the unchanged serial-verification gate still fails at 8.411% relative
logit RMS and 13/16 matching greedy tokens. Model throughput qualification is
false. Synthetic output repetition benefits the history-lookup proposer;
these observations do not establish natural-language throughput or a Netra win.

## Unchanged load and precision

The model is `Qwen/Qwen3.5-35B-A3B-FP8`, revision
`9d1823d2dee688a6b25e77009dc727688c44936e`. Official FP8 weights and scales,
FP8 KV storage and FP32 recurrent state are retained. There is no CPU offload
or prefix cache. MTP and output-history lookup use a 64-position target
verification window in the earlier profiles and 128 positions in the latest
profile. Eight requests arrive together, without warmup; each
replay completes 256,000 prompt tokens and 128,000 output tokens. The workload
semantic SHA remains
`6e012fe014f8fc86d58d0065862c62e77e1374fccba612a7ab5d54c1679db44a`.

The objective divides output tokens by the complete HTTP client elapsed time,
including prefill, setup, streaming and fill/drain. A 3,000 tok/s observation
requires elapsed time below 42.667 s. Input throughput is reported separately.

## Completed client replays

| Configuration | Output tok/s | Client elapsed s | Mean TTFT s | Target-only prefill s |
|---|---:|---:|---:|---:|
| Previous paired projections and padded attention | 2,126.824 | 60.184 | 30.370 | 27.546 |
| Transient BF16 prefill attention workspace | 2,270.302 | 56.380 | 26.408 | 23.474 |
| Smaller projection CTAs plus workspace (rejected) | 2,240.394 | 57.133 | 27.152 | 24.321 |
| Prefill chunk 1024 plus verification workspace | 2,272.734 | 56.320 | 26.757 | 22.832 |
| Prefill chunk 2048 plus verification workspace | 2,281.844 | 56.095 | 27.406 | 22.422 |
| Accepted-prefix recurrent recomputation, window64 | 2,407.521 | 53.167 | 25.747 | 22.411 |
| Accepted-prefix recurrent recomputation, window128 | 2,414.471 | 53.014 | 25.645 | 22.735 |

The workspace decodes authoritative FP8 KV and its per-block scales once per
attention call. It reuses two BF16 scratch buffers across attention layers.
Their combined allocation is **786,432,000 bytes**, separately labeled from
the FP8 KV cache. The buffers are released when prefill closes. The candidate
retains the existing 16-key online softmax updates and reduction order.

The workspace replay passes 60 physical GPU checks, with eight unrelated cases
skipped. Both initial and 32K same-state prefill comparisons have zero logit
and recurrent-state RMS error and 8/8 matching greedy tokens. All eight client
requests finish with zero failures. Target verification still consumes
25.041 s over 316 rounds, essentially unchanged from the previous 25.012 s;
the prefill-only change cannot remove that remaining decode cost.

These are single observations; repeated-run dispersion has not been measured.

## Hardware counters and isolated schedules

Nsight Compute 2025.2.1 collected real H200 counters in a separate diagnostic.
The inputs are isolated random KV data at the 32K position, with eight slots,
512 prefill tokens and capacity 48K. They do not measure the whole model.

| Isolated operation | Kernel time ms | Sustained DRAM throughput % | Active warp occupancy % | Registers/thread |
|---|---:|---:|---:|---:|
| 1 GiB read plus 1 GiB write streaming copy | 0.526 | 82.08 | 77.99 | 26 |
| Original padded prefill attention | 28.990 | 0.86 | 12.50 | 191 |
| Joint encoded K/V loads | 27.269 | 0.91 | 12.50 | 185 |

CUDA event copy samples separately measure about **4.07 TB/s of useful
read-plus-write traffic**. This is a streaming-copy result, not achieved model
bandwidth. The attention counters show that this kernel spends its time before
saturating HBM; high register use and low occupancy justify changing its load
and conversion schedule.

The BF16 workspace reduces isolated attention **including its decode pass**
from roughly 28.7 ms to 15.6–15.8 ms, with bitwise identical output. Joint FP8
loads alone reduce it to about 27.0 ms. Splitting the value dimension into two
CTAs remains bitwise identical but takes about 26.2–26.7 ms. A 512-thread square
warp layout takes 34.3 ms and changes output bits, so it is rejected. A 64-query,
256-thread decoded layout preserves bits but takes 27.7 ms and is also slower.

## Reproduction and evidence

The retained [workspace replay](data/qwen35-native-h200/bandwidth/prefill-workspace/summary.json)
contains the exact prepared paths and source hashes. The new workspace owns
the `tensor-qwen35-h200` volume; run commands with the `akbar2habibullah` Modal
profile active. Existing control bundles must be prepared or migrated with
matching checksums before reuse.

```bash
PYTHONPATH=src:packages/tensor-llm/src:. modal run \
  benchmarks/qwen35/modal_bandwidth.py::workspace_main \
  --prepared-file build/qwen35-akbar-artifact-migration/prepared.json \
  --out build/qwen35-h200-workspace-replay
```

Each full candidate runs the required physical primitives, initial and 32K
same-state comparisons, unchanged serial diagnostic and complete C8 HTTP load.
Numerical gates and qualification flags are retained.

- [Physical attention primitive checks](data/qwen35-native-h200/bandwidth/attention-primitives.json)
- [Initial copy and load schedule microbenchmarks](data/qwen35-native-h200/bandwidth/prefill-micro-v1/measured.json)
- [BF16 workspace microbenchmarks](data/qwen35-native-h200/bandwidth/decoded-micro-v1/measured.json)
- [Value-split comparison](data/qwen35-native-h200/bandwidth/decoded-micro-v2/measured.json)
- [Additional warp layouts](data/qwen35-native-h200/bandwidth/decoded-micro-v3/measured.json)
- [Nsight hardware counters and commands](data/qwen35-native-h200/bandwidth/counters.json)

The 1024- and 2048-token chunks pass a complete 32K prefix comparison against
512-token control chunks: zero logit and recurrent-state RMS, 8/8 greedy
matches, identical positions and bitwise identical valid KV bytes and scales.
Their physical primitive suite passes 78 tests with eight skips. The 2048
replay reuses the source-bound proof from the 1024 replay; its retained log is
not a second fresh 820-second test run. Gains are small single observations.

The verification workspace additionally allocates 786,432,000 transient BF16
bytes during decode. Five dedicated GPU primitive checks pass. Isolated
attention plus decoding takes about 1.03 ms at 32K versus 1.23 ms for the
frozen FP8 control; at 47,936 tokens it takes 1.515 versus 1.816 ms. Parts and
statistics match bitwise in this diagnostic. The full-model serial diagnostic
still fails, so these timings do not qualify model throughput.

Packed FP8 operand conversion passes ten GPU checks but is slower than the
original 128-column expert projections in both balanced and hot routing. It
is rejected. Smaller 64-column projections also slow the full client replay
and are rejected. A 128-query, 512-thread full-row decoded attention layout
matches bits but takes about 28.3 ms, versus 15.7 ms for the selected layout,
and is rejected.

An accepted-prefix recurrent recomputation profile is now being evaluated.
The first inline recomputation failed exact checkpoint agreement and was
rejected. The corrected profile replays the unchanged scan kernel; it passes
all 64 acceptance depths with shortened and inactive requests. The 64-token profile additionally passes 79 physical GPU tests (eight skips)
and a full-model exact comparison against its frozen verifier at the real 32K
prefix. Forward results and commits at seven acceptance depths match bitwise.
The cache occupies 1,537,475,520 bytes instead of roughly 33 GB of snapshots.
Its completed client replay reaches 2,407.521 output tok/s with all eight
requests finished and zero failures. Across the same 316 rounds, verification
takes 22.530 s, commit 0.041 s and draft repair 1.696 s. The 3,000 tok/s goal
remains open. The 128-token window passes 80 GPU tests (eight skips), plus
full-model bitwise forward and rollback comparisons, and measures 2,414.471
output tok/s. Its recurrent cache uses 2,548,040,640 bytes. The window increase
gives little end-to-end benefit in these single observations. A pool of
captured graph widths sharing the same model is being implemented to avoid
large padded passes during short MTP fallback batches. A lossless resident FP16 expert operand cache passes the isolated bitwise
checks but is slower at 512 and 4096 rows in balanced and hot routing. It is
rejected; no model-sized cache was allocated. The original FP8 weights and
scales remain. [Measurements](data/qwen35-native-h200/bandwidth/resident-operands-micro.json).

Additional evidence:

- [1024-token prefill replay](data/qwen35-native-h200/bandwidth/chunk1024-verify-workspace/summary.json)
- [2048-token prefill replay](data/qwen35-native-h200/bandwidth/chunk2048-verify-workspace/summary.json)
- [Rejected smaller projection replay](data/qwen35-native-h200/bandwidth/projection64-workspace/summary.json)
- [Packed projection checks and timings](data/qwen35-native-h200/bandwidth/packed-projection-micro.json)
- [Verification attention workspace](data/qwen35-native-h200/bandwidth/verification-workspace-micro.json)
- [Exact recurrent replay primitive checks](data/qwen35-native-h200/bandwidth/recompute-primitives-v2.json)


A separate [expert tile microbenchmark](data/qwen35-native-h200/bandwidth/expert-shapes/measured.json)
compares the real 512-row, four-part verification geometry with finite FP8
encodings of both signs. The 16-row, 128-column, 128-thread candidate matches
control bits. Its up projection improves balanced routing from about 0.307 to
0.177 ms but slows hot routing from 0.103 to 0.158 ms. Its down projection
improves balanced routing from 0.437 to 0.196 ms and takes 0.171 versus 0.176 ms
for hot routing. These tradeoffs require a full-model replay before selection.
The [driver](data/qwen35-native-h200/bandwidth/expert-shapes/qwen35-expert-shape-micro.py)
is retained. The compiler emits synchronization warnings for smaller tiles;
no smaller tile has been promoted into the serving engine.

- [Recurrent replay full-client evidence](data/qwen35-native-h200/bandwidth/recompute64/summary.json)

- [128-token recurrent replay full-client evidence](data/qwen35-native-h200/bandwidth/recompute128/summary.json)

## Native warp-group attention experiment

Explicit native warp-group QK, PV, and combined QK/PV operations preserve the
control output bits in the isolated 0-token and 32K-prefix cases. They provide
no measurable improvement over the decoded workspace schedule: all four
32K cases take approximately 15.63–15.65 ms including FP8 KV decode. None is
selected. These are five event samples per candidate, with positive encoded
KV and signed queries; they are not a full model qualification.

- [Raw timing and numerical observations](data/qwen35-native-h200/bandwidth/warpgroup-attention/measured.json)

## Migration and current experiment

`akbar2habibullah` is the active Modal CLI profile and the owner of the new
model cache, compiler cache, compiled candidates, and run evidence. The old
workspace has no running apps. The [initial artifact transfer](data/qwen35-native-h200/bandwidth/migration/initial.json)
and [128-position transfer](data/qwen35-native-h200/bandwidth/migration/window128.json)
retain SHA-256 checks of the frozen controls; accessing the old workspace was
limited to reading those artifacts.

The adaptive verification candidate shares one resident model across 8- and
128-position captured graphs and uses the matching MTP repair graph and
hidden buffer. The first GPU job passed 86 tests (eight skipped), then timed
out during the full model checks before serving. Its 20-minute allocation
was insufficient for the 18-minute primitive suite plus model checks. The
retry has a one-hour limit. There is no completed adaptive throughput result
yet; it is not selected as a winner.

A second isolated candidate combines two QK blocks while retaining two
successive 16-key softmax updates. The compiler requires shared score staging
and emits synchronization warnings. All three geometries are slower (about
24.5–44.3 ms versus 15.7 ms at 32K) and change output bits with relative RMS
around 1.88e-5. The candidate is rejected, removed from the runtime package,
and retained only as [experiment evidence](data/qwen35-native-h200/bandwidth/batched-attention/measured.json).
