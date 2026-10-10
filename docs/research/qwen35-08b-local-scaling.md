# Qwen3.5-0.8B local scaling exercise

This is an experimental implementation checkpoint on one NVIDIA L40S, not a
completed speedup claim. The best measured native profile reaches 426 output
tok/s at C1 and 11,763 at C256. The requested 1.5× low-concurrency and 2×
middle/high-concurrency advantages over both vLLM and SGLang remain unmet.

The model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. These measurements use BF16
weights, BF16 KV and FP32 recurrent state, with no CPU offload or prefix cache.
Native MTP uses the checkpoint's official head with a two-position verifier.
The baselines in this table use autoregressive decoding; tuned speculative
baseline measurements remain outstanding.

| Concurrency | Native MTP output tok/s | vLLM AR output tok/s | SGLang AR output tok/s | Native mean TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 426.3 | 316.1 | 325.2 | 14.7 |
| 2 | pending | 526.5 | 518.2 | pending |
| 4 | pending | 1,015.0 | 1,003.6 | pending |
| 8 | 2,579.2 | 1,918.7 | 1,870.7 | 34.0 |
| 16 | pending | 3,316.7 | 3,240.1 | pending |
| 32 | 6,929.0 | 5,548.7 | 5,131.5 | 56.3 |
| 64 | pending | 7,730.8 | 7,297.7 | pending |
| 128 | 11,147.5 | 9,437.7 | 8,943.1 | 148.1 |
| 256 | 11,762.7 | 10,446.7 | 10,118.4 | 287.7 |

Values are arithmetic means of two complete HTTP replays. All displayed runs
completed with zero request failures. Throughput includes prefill, admission,
streaming and drain; it is not a selected decode interval. TTFT is measured at
the client. SGLang C32 repeats varied from 4,770 to 5,493 tok/s.

The workload uses varied natural-language prompts of 66–81 tokens and exactly
256 greedy output tokens, ignoring EOS. Each point has `max(16, 4*C)` requests,
closed-loop refill, and two separate warmup requests before each repeat. Context
capacity is 4,096. These are **not 32K-input/16K-output measurements**. Separate
long-context tests through C64 are still pending. Train, held-out and unequal
output-length drain workloads are preserved, but the displayed throughput
comes from the train workload. Held-out throughput qualification is pending.

The baseline versions are vLLM 0.17.1 and SGLang 0.5.9. The retained SGLang
comparison was run without overlapping kernel compilation, with streaming
interval four. vLLM used streaming interval one; a matching interval-four replay
is still needed. An earlier SGLang run overlapped CPU compilation and is excluded
from this table. Native streaming publishes the first token immediately and
subsequent groups at approximately four tokens.

## Implementation and validation status

The new `tensor_llm.qwen35.dense` modules separate checkpoint loading, projections,
attention and recurrent kernels, model execution, exact greedy head reduction,
MTP, speculative graph control, artifact validation and HTTP serving. The common
scheduler continuously refills freed slots and compacts execution batches while
keeping request state in stable pool slots. Compilation belongs to the producer;
the server loads AOT artifacts and captures CUDA graphs.

The measured V2 implementation combines draft, verification, acceptance, commit
and repair in one CUDA graph. It delays accepted recurrent updates through a
journal and reduces the vocabulary projection directly to exact greedy choices.
Native AR/MTP token equality and final physical state equality passed through
C256 for the retained four-position deferred profile. Batch independence passed
all powers of two through C256. The fused head matched full FP32 projection
scores and greedy IDs through 1,024 rows, including ties and inactive rows.
These checks establish internal consistency; they do not complete independent
model quality qualification.

The independent FP32-state Hugging Face oracle comparison remains open:
the initial native profile had 1.146% relative logit RMS and 135/136 greedy
matches. The mismatch occurs around a BF16-rounded tie. Server metadata therefore
continues to report `quality_qualified: false`; no model-quality win is claimed.

The source in this commit is newer than the measured V2 profiles. It includes an
unqualified optimization that commits the first verified recurrent candidate
inline, broader implementation fingerprints that reject stale AOT artifacts,
and an **unused, uncompiled FP8 KV attention prototype**. FP8 KV is not wired into
the engine or server yet. Rebuild the AOT bank and repeat token/state and model
quality checks before measuring this source. Formatting also changes source
fingerprints. The old reports' Git commit identifies the previously committed
workspace base, not an exact committed copy of these then-untracked sources;
retained server implementation fingerprints provide the available source hashes.

Five CPU contract tests cover continuous refill, cancellation, unequal-length
drain, speculative output limits and preservation of checkpoint storage bits.
They passed for this checkpoint. No new GPU validation of the latest source is
claimed.

## Evidence and reproduction

[Raw reports and compressed request streams](data/qwen35-08b-local-scaling/results/)
retain the baseline, native AR and native MTP iterations, with warmups, telemetry
and workloads. [The file index](data/qwen35-08b-local-scaling/files.json) records
sizes and SHA256 digests. Qualification JSON, server metadata, initial measured
kernel search and profiling evidence are included alongside them. Checkpoint
weights, compiled GPU binaries and virtual environments are excluded.

The entry points are `benchmarks.qwen35.dense_scaling` for workload preparation
and replay, `benchmarks.qwen35.dense_producer` for AOT compilation, and
`tensor_llm.qwen35.dense.server` for local serving. Their `--help` commands expose
the current options. Run from the repository with
`PYTHONPATH=src:packages/tensor-llm/src`; the producer requires the compiler
environment, while the server requires its runtime dependencies and tokenizer.
Use a fresh output directory for each benchmark and keep compilation and GPU
diagnostics separate from timed serving runs.
