# E15 — Remaining Phase 0 validation

**Run:** 2026-09-29, Linux, NVIDIA A10G (`sm_86`), CUDA 12.9.86,
TileLang 0.1.14, TVM FFI 0.1.12, PyTorch 2.14.0, Python 3.12.
**Experiment commit:** `37fd520ab7d03cc2476aec73a658783d18ee0335`.
Workers use a clean detached checkout, fresh processes, isolated caches, and
sequential execution. Raw results are retained in
[validation data](data/e15-phase0-validation.json) and
[transfer data](data/e15-artifact-transfer.json).

The bounded probes are complete, with both successful and negative results.
**This does not establish every Phase 0 claim.** Fusion, a newly registered
provider replacing a built-in target, Rust hosting, and performance on other
GPU architectures remain unverified. No product package has been scaffolded.
**Scope update — 2026-09-29:** the user deferred cross-GPU benchmarking, so
it no longer blocks Phase 0 completion. The measurements below are unchanged.

## Cache behavior (E5)

The elementwise cache experiment checks three paths: cold compilation,
same-process memory reuse, and fresh-process disk reuse. Both hits prohibit
lowering and CUDA compilation through observational hooks, rather than
inferring a hit from elapsed time. All paths execute and check the result.
Changing the static extent from 129 to 130 recompiles and changes the key.
Seven independent key changes cover IR, target, host target, execution
adapter, output indices, pass configuration, and compiler flags.

The isolated disk entry is deliberately corrupted. TileLang rejects the
manifest hash mismatch and re-lowers it, then produces a correct result.
Its separate device-code cache may still satisfy CUDA compilation.

The measured base key contains TileLang's version. It does not explicitly
contain TVM FFI or nvcc versions in this configuration. Tensor still needs an
outer identity including the exact toolchain/lock, ABI, and source hash.
The full workload matrix also measures repeated compilation of the original
PrimFunc and a freshly reconstructed frontend function; those paths must not
be assumed equivalent. IR hashes and compiler call counts identify mutation
and repeated lowering in the raw data.

For extent 129, cold compilation including host FFI realization is 1.911 s,
the memory hit is 5.206 ms, and the fresh-process disk hit is 5.842 ms.
Corruption recovery takes 542.2 ms and invokes lowering once while reusing
device-code cache. These exclude package imports.

On `sm_86`, compiling the original GEMM/attention PrimFunc mutates its
serialized script hash. Reusing that object re-lowers it. Reconstructing the
original frontend function hits the initial cached object for all five
workloads without lowering or CUDA compilation. Persist/key canonical input
IR before compilation, and do not treat a mutated object as canonical source.

## CPU provider contract (E6)

The registered CPU manifest is re-registered through the public API and
executes extents 1/127/128/129/1025 with zero error using target `c`, Cython,
and `tirx.disable_vectorize=True`. This demonstrates a working non-CUDA
execution model using the existing provider machinery.

Three restrictions were measured:

- Default vectorization of the aligned ReLU emits a vector comparison that
  g++ rejects. Scalar lowering is an explicit configuration, not a default
  backend success.
- CPU target `c` with the TVM FFI execution adapter rejects device compilation.
  The Linux wheel also reports LLVM execution disabled.
- A new manifest claiming `c` is rejected because both manifests must define
  target predicates and the built-in CPU manifest does not. Re-registering
  the existing manifest does **not** prove a new third-party provider can
  take over that target kind. No private registry mutation is used.

The literal new-provider experiment therefore has a negative result. The
provider ABI remains provisional; upstream registration/codegen boundaries
need work before claiming unrestricted extensibility.

## Native host and ABI inventory (E7)

A standalone C++ executable loads a typed TVM FFI module wrapping the actual
TileLang CPU library, passes borrowed `DLTensor` views, checks 129 output
values, and verifies that a wrong dtype raises an error. Its linked
dependencies include neither Python nor PyTorch. This is an explicit P0
wrapper, not a working built-in CPU TVM FFI adapter.

A second native invocation loads the compiler and TileLang shared libraries,
deserializes the symbolic TIRx JSON through `node.LoadJSON`, and serializes it
byte-identically through `node.SaveJSON`. **Loading IR does not prove a native
end-to-end compiler pipeline.** Rust hosting was not tested.

The inventory in the raw report separates exercised calls, buffer ownership
and errors from unresolved foreign-stream, provider-neutral event, signal,
and memory-ordering contracts. These are not silently inherited from an FFI
function-call interface. See the primary
[TVM FFI C++ guide](https://tvm.apache.org/ffi/guides/cpp_lang_guide.html) and
[stable C ABI](https://tvm.apache.org/ffi/get_started/stable_c_abi.html).

## Framework boundary (E8)

A narrow `torch.compile` FX backend recognizes exactly `relu(2*a+b)` on
contiguous CPU float32 vectors and lowers it to the measured CPU kernel.
Extents 129, 129, and 130 produce exactly two specializations and agree with
PyTorch. Unsupported graphs retain the FX reference executor.

A separate AOTAutograd backend captures one normalized forward and one
backward graph across symbolic extents 129 and 257. Boxed FX reference
execution checks outputs and gradients. A deliberate graph break produces
two graph fragments. **AOT capture/reference execution is not Tensor training
compilation.** These results support the proposal's inference-first raw-FX
integration and provide an observed boundary for later AOT work, consistent
with [PyTorch's custom-backend documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html).

## Symbolic tiling and artifact boundary (E4b / E4)

One serialized/reloaded dynamic elementwise function is compiled once and
executed at extents 1/127/128/129/1025. One GEMM with static tiles is compiled
once and executed at row extents 1/31/32/33/65. Both give zero maximum error
in these checks. A symbolic GEMM tile is rejected during frontend
construction: `T.gemm requires static tile dimensions, but M is symbolic`.
Runtime dimensions and compile-time scheduling tiles are different contracts.

The GEMM post-`LowerTileOp` artifact is 222,661 bytes. Serialization succeeds,
but feeding it back into the standard source pipeline raises
`InternalError: bad optional access`. The inspected public namespaces expose
no name containing `fus`. This rejects the proposed shortcut of treating
post-lowering IR as a freely re-lowerable artifact. It does not prove fusion
is impossible, that a new IR is necessary, or that two modules can be composed.
Keep frontend IR as a pinned source tier and scope the MVP to opaque execution.

## Full compilation and GPU baselines (E9 / E11)

The matrix attempts all five workloads on `sm_80`, `sm_86`, `sm_90`,
`sm_100`, `sm_90a`, and `sm_100a`. Timers exclude package imports and include
lazy host FFI compilation. CUDA compiler time is nested within lowering time;
the stage times must not be added together.

| A10G workload | Cold full compile (s) | Repeated original object (ms) |
|---|---:|---:|
| fused elementwise | 1.962 | 1.939 |
| GEMM + bias + ReLU | 3.663 | 1750 |
| row sum | 1.930 | 2.007 |
| gather | 1.904 | 1.963 |
| attention | 5.113 | 5117 |

These are one cold sample and one repeated call per cell. The long repeated
calls are measured re-lowering, not successful cache hits; rebuilding the
canonical frontend function avoids them. Raw timing and call counts are
preserved for every target.

Actual nvcc compilation narrows the earlier source-emission result:

| Target | Compiled workloads | Measured restrictions |
|---|---:|---|
| sm_80 | 5/5 | compilation only |
| sm_86 | 5/5 | available execution/performance hardware |
| sm_90 | 3/5 | GEMM and attention require architecture-specific features |
| sm_100 | 2/5 | GEMM/attention feature requirements; gather vector-pack error |
| sm_90a | 5/5 | compilation only |
| sm_100a | 4/5 | gather `cutlass::half_t` to CUDA half conversion rejected |

GPU performance uses 20 warmups, 100 operations captured in a CUDA graph,
five CUDA-event samples, and the median per operation. Graph replay avoids
Python submission starvation for small kernels. Inputs remain resident;
the candidate reuses its output and baseline tensor lifetimes are captured.
Correctness is checked before timing. Baselines are TorchInductor fused
pointwise/GEMM, PyTorch CUDA sum/index_select, and forced FlashAttention SDPA.
The GEMM baseline uses a vendor-selected matrix multiply plus an epilogue;
it is not evidence of matching vendor epilogue fusion.

| Workload | Candidate (µs) | Baseline (µs) | Baseline / candidate |
|---|---:|---:|---:|
| fused elementwise, 1024×1024 fp16 | 5.059 | 4.178 | 0.826× |
| GEMM + bias + ReLU, 1024³ fp16 | 43.694 | 46.131 | 1.056× |
| row sum, 256×1024 fp32 | 1.894 | 3.666 | 1.935× |
| gather, 1024×128 fp16 | 1.597 | 2.109 | 1.321× |
| attention, 8×128×64 fp16 | 4.936 | 8.622 | 1.747× |

Elementwise is slower than the selected fused baseline. Retain that result
alongside the favorable comparisons; these shapes do not justify a general
speedup claim.

These are five fixed, untuned workload comparisons on an A10G, not a broad
performance claim. `sm_80`/`sm_90`/`sm_100` hardware performance is deferred
from Phase 0. Symbolic-versus-static throughput remains an unrun A10G check.

## Two-host executable transfer (E12)

[GitHub Actions run 36595583781](https://github.com/kreasof-ai/tensor/actions/runs/36595583781)
builds the boundary matrix on a separate Ubuntu host using the locked Python
environment and SHA-256-verified NVIDIA build components. The consumer runs
the downloaded cubins in fifteen fresh NumPy-only processes on the A10G.
The transfer audit requires different hostnames, matching clean Git revisions,
source and lock hashes, all five boundary sizes, compiler import guards, and
absence of TileLang/TVM FFI/PyTorch package installations on the consumer.

Startup is measured from consumer module entry to downloaded first result.
It excludes interpreter startup. Warm host launch plus synchronization is
reported separately from the CUDA-event throughput measurements.

All fifteen transfer executions pass with zero maximum absolute error. First
result medians are 247.9–250.7 ms across the five sizes, from consumer module
entry. Producer Python is 3.12.3 and consumer Python is 3.12.14; the exact
source/lock identity matches while compiler packages are absent on the
consumer. The locked GPU-enabled regression suite passes **40 tests with
zero skips in 54.31 s** on the same clean experiment checkout.

## Reproduction

```bash
uv sync --locked
uv run --locked python tools/bootstrap_cuda.py --out experiments/p0/out/cuda-12.9
export CUDA_HOME="$PWD/experiments/p0/out/cuda-12.9"
uv run --locked python -m experiments.p0.validation --out experiments/p0/out/new-validation-run
```

Use a new output directory. Each worker saves its JSON and log before the
aggregate report. Compiler rejections are explicit per-row results; the
aggregate `completed_with_limits` means the probes ran, not that every
architecture or architectural claim passed. The Actions workflow is manual
so large producer dependencies are installed only for requested builds.
