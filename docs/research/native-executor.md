# C++ PyTorch executor

Implemented and validated on 2026-09-30 at source commit
`ca3446b3ff245c1208a34b6a257637f7526395b9`. The separate adapter now
optionally executes compiled FX regions and fixed prepared calls in C++.
On A10G, the isolated allocating launch path is **2.13× faster** and the
20-case full-call suite is **1.27× faster** than the Python adapter path.

[Build and use the native wheel](../pytorch.md#native-execution).
[Raw timings, source/binary hashes, tests and CI](data/native-executor-metrics.json).
The original [Phase 4 acceptance](phase4-exit.md) remains a record of the
portable adapter; these measurements cover the subsequent C++ optimization.

## Execution and packaging

`executor.cpp` reads tensors through PyTorch's C++ API, validates shape/stride,
dtype/device and alignment, allocates outputs with ATen, updates kernel
arguments, obtains the current stream, records allocator usage, and submits
through the CUDA driver. Prepared calls retain fixed tensors and copied scalar
arguments and validate metadata/storage on each invocation. FX plans retain
specifications and argument storage, without retaining their initial tensors.
Native argument updates and submission are serialized; the GIL is released
during allocation and submission. Closing the adapter invalidates its calls.
The follow-up `839fee6d2e78e0af3a0db10d2e298da5112afd2d` preserves the core
`CudaError` exception contract; it does not change the measured success path.

Initial DLPack/ABI validation, artifact loading, FX specialization, ordinary
custom-operator calls and the outer Dynamo wrapper remain in Python. The
kernel binaries and NVRTC compilation path are unchanged. Direct PTX remains
experimental after v1.

The optional C++20 build uses installed PyTorch 2.14 headers and libraries,
with no CUDA toolkit headers or direct toolkit linkage. Its CPython/Torch ABI
requires a `cp312-cp312` wheel. A matching versioned executor module is selected
automatically; other supported Torch versions and `TENSOR_TORCH_NATIVE=0` use
the portable path. The independent `cp312-abi3` portable wheel still builds
using Python headers alone. Consumers of either wheel need no host compiler.

## Measured results

Python 3.12, PyTorch 2.14.0+cu130, NVIDIA A10G (`sm_86`), driver 595.91.07.
The installed local native wheel is used for both paths; only executor
selection changes. Both paths execute identical cached `.tbin` images.

| Measurement | Python path | C++ path | Speedup |
|---|---:|---:|---:|
| Allocating launch, without Dynamo | 15.80 µs | 7.43 µs | 2.13× |
| Fixed prepared submission | 9.13 µs | 4.11 µs | 2.22× |
| 20-case full-call geometric mean | — | — | 1.27× |

The isolated benchmark rotates four providers in one process across 12 batches
of 1,000 calls. Allocating calls include output allocation; fixed prepared calls
allocate outside timing. Completion is excluded from host submission timing.
All providers are warmed up and checked numerically.

Full calls use `phase4_benchmark.py` with its original eager/Inductor baselines,
output allocation, stream completion and default `torch.compile` settings.
Two complete 20-case runs per path are collected in Python/C++/C++/Python order.
For each case, the median of its two run medians is used before computing the
geometric mean of speedups. First-case timing varied, and all samples are retained.
The two individual paired-run speedups are 1.33× and 1.22×; the combined 1.27×
corresponds to **21% lower full-call latency**.

Against the respective baselines in the native runs, full-call geometric-mean
speedup is **1.71× vs Inductor** and **0.62× vs eager** (1.61× slower than eager).
The GPU-bound 4M pointwise case remains about 2.39× faster than eager, with
essentially unchanged latency after the executor rewrite. C++ reduces adapter
overhead; outer Dynamo dispatch and GPU execution still determine complete calls.
These are measured microbenchmarks on one GPU, rather than whole-model claims.

All artifacts are already cached in these comparison runs. Their
`cold_graph_seconds` fields measure graph capture/first execution, rather than
fresh NVRTC compilation. Original Phase 4 compilation timings remain separate.

## Verification

The final GPU-enabled suite passes **145 tests, zero skips, in 174.90 seconds**.
Existing numerical, fallback, autograd and stream tests execute with the native
executor. New checks cover prepared scalar copies, input/output metadata
invalidation, current-stream selection, graph replay, pointer rebinding,
concurrent calls, alignment fallback and closed sessions.
An invalid-launch regression verifies `CudaError` and subsequent valid execution.

[Linux and Windows CI](https://github.com/kreasof-ai/tensor/actions/runs/36709145756)
build and import native and portable wheels without a system toolkit, and
verify backend discovery, CPU fallback, FakeTensor contracts and compiler-free
installation. The uploaded Linux C++ wheel also executes all 20 cached FX
graphs on A10G with compiler imports blocked. Three Linux-produced and three
Windows-produced profiles execute through both functional custom ops and native
prepared calls. Windows native GPU execution itself has not been measured.

Reproduce the full comparison by running `tools/phase4_benchmark.py` twice
with `TENSOR_TORCH_NATIVE=0` and twice with native execution enabled, using the
same installed wheel and populated cache. To isolate adapter overhead:

```sh
python tools/phase4_executor_benchmark.py \
  --artifact path/to/cached-pointwise-129.tbin --out build/executor-timing.json
```
