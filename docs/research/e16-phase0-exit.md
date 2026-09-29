# E16 — Phase 0 exit validation

**Run:** 2026-09-29, Linux, NVIDIA A10G (`sm_86`), CUDA 12.9.86,
TileLang 0.1.14, TVM FFI 0.1.12, PyTorch 2.14.0, Python 3.12 and
Rust 1.98.1. **Experiment commit:**
`157ae365605eda3c4c9ca4aa739fb7f033b406e8`.

The final workers ran sequentially in fresh processes from a clean detached
checkout. The complete report is retained in
[the raw exit data](data/e16-phase0-exit.json). Together with E15 and its
two-host transfer result, every scoped Phase 0 exit gate passes. Cross-GPU
throughput remains deferred by explicit scope decision, so all performance
claims remain limited to the A10G.

## Independent non-CUDA provider (E6)

A new `p0_cpu` manifest is registered through TileLang's public
`BackendModule` API on TVM's unclaimed CPU `test` target kind. It does not
replace or mutate the built-in CPU provider. The manifest owns its execution
policy and native g++ compilation callback while delegating lowering and C
source generation to the existing CPU compiler components.

The resolved backend context routes five lowering calls, five code-generation
calls and five native compilation calls through the new manifest. Native
libraries execute extents 1/127/128/129/1025 over three random seeds each with
zero maximum absolute error. This passes the architectural requirement that a
second, independently registered non-CUDA provider can own compilation and
execution routing.

The measured boundary is narrower than a general plugin ABI: TileLang's
top-level `compile` function has a fixed JIT-adapter set and does not accept the
experimental `p0_native` adapter. The probe therefore uses the public backend
context directly. Phase 1 should add its product adapter above that context,
without patching TileLang's private registry.

## Serialized composition boundary (E4)

Two separately serialized frontend TIRx stages are reloaded and composed by
inlining an affine pointwise producer into a ReLU consumer. The consumer is
then compiled with either 64 or 128 threads while the producer artifact stays
unchanged. Across five boundary extents, all ten cases emit one CUDA kernel and
execute with zero error. A mismatched producer/consumer shape is explicitly
rejected.

This establishes that bounded composition and rescheduling can reuse frontend
TIRx serialization; Tensor does not need a new IR to support this first case.
It does not establish arbitrary fusion. The prototype accepts one-dimensional,
equal-type, identity-indexed pointwise stages with one output store. The prior
negative result also stands: post-`LowerTileOp` IR is not a freely re-lowerable
artifact through the standard pipeline.

## Rust and foreign-resource ABI checks (E7)

A standalone Rust executable uses the pinned stable TVM FFI C ABI directly.
It loads the actual CPU module, calls it with borrowed `DLTensor` views, checks
129 results, and verifies that a wrong dtype propagates an error. It asserts
the measured ABI layouts (`Any` 16 bytes and `DLTensor` 48 bytes) and TVM FFI
version 0.1.12. The same executable loads the compiler libraries and
round-trips 23,953 bytes of symbolic TIRx JSON byte-identically. Its dynamic
dependencies include neither Python nor PyTorch.

This proves native Rust hosting of executable calls and IR loading. It does
not prove a Python-free native compilation pipeline; that is a later packaging
choice rather than a Phase 0 exit requirement.

The CUDA ownership probe borrows PyTorch tensor addresses and two PyTorch CUDA
streams in the existing primary context. An owned CUDA event transfers
readiness from the producer stream, an opaque cubin runs on the consumer
stream, and a second event transfers completion to the default stream. Three
random cases pass with zero error. The probe unloads only its owned module and
events; the context, tensor addresses and borrowed streams remain valid.

This is a measured same-device CUDA contract. It does not define a
provider-neutral event type, distributed signal semantics or cross-device
memory ordering.

## Static versus symbolic throughput (E11)

Static and runtime-symbolic binaries use identical inputs, dtypes, outputs and
static scheduling tiles. Each result is the median of five CUDA-event samples,
with 20 warmups and 100 operations per captured CUDA graph.

| Workload and extent | Static (µs) | Symbolic (µs) | Symbolic / static |
|---|---:|---:|---:|
| elementwise 1 | 1.096 | 1.106 | 1.009× |
| elementwise 127 | 1.147 | 1.116 | 0.973× |
| elementwise 128 | 1.085 | 1.126 | 1.038× |
| elementwise 129 | 1.096 | 1.116 | 1.018× |
| elementwise 1025 | 1.126 | 1.126 | 1.000× |
| elementwise 1,048,576 | 24.535 | 24.504 | 0.999× |
| GEMM rows 1 | 1.321 | 1.341 | 1.016× |
| GEMM rows 31 | 1.290 | 1.434 | 1.111× |
| GEMM rows 32 | 1.280 | 1.423 | 1.112× |
| GEMM rows 33 | 1.341 | 1.423 | 1.061× |
| GEMM rows 65 | 1.372 | 1.444 | 1.052× |
| GEMM rows 1024 | 1.597 | 1.700 | 1.064× |

The symbolic elementwise kernel is within 3.8% of static in these samples. The
symbolic GEMM costs 1.6–11.2% with the same static tile. These are small-kernel
A10G observations, not general performance guarantees.

## Exit decision

Phase 0 is complete for the scoped MVP architecture. E4 has a reusable source
representation and a bounded composition path; E6 has an independently
registered CPU provider; E9 is measured; the load-bearing decisions are
recorded; the compiler-free artifact crosses hosts; and the remaining local
ABI and A10G performance questions now have measured answers.

Known restrictions carry forward as explicit product work: exact toolchain and
artifact identity, a Tensor-owned JIT adapter, general graph composition,
provider-neutral asynchronous primitives, native compiler packaging, and
performance validation on other GPU architectures. Compiler rejections in the
six-target E15 matrix remain capability data rather than hidden failures.

## Reproduction

```bash
uv sync --locked
uv run --locked python tools/bootstrap_cuda.py --out experiments/p0/out/cuda-12.9
uv run --locked python tools/bootstrap_rust.py --out experiments/p0/out/rust-1.98.1
export CUDA_HOME="$PWD/experiments/p0/out/cuda-12.9"
export TENSOR_P0_RUSTC="$PWD/experiments/p0/out/rust-1.98.1/bin/rustc"
uv run --locked python -m experiments.p0.phase0_exit --out experiments/p0/out/new-phase0-exit
```

Use a new output directory. The runner consumes the retained E15 and two-host
transfer evidence, then creates one report per final worker and an aggregate
exit report.
