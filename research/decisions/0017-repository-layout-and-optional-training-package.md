# ADR 0017: Grouped core and optional training distribution

Status: accepted implementation direction.

## Decision

Organize core implementations under `runtime`, `providers`, `compiler`,
`artifacts` and `cli`. Runtime providers own execution; compiler providers own
lowering/building. Shared native headers and reference hosts retain their
installed paths. Compiler/framework dependencies stay out of ordinary consumer
requirements, and compilation imports remain deferred.

Keep `ManualFunction` and `BackwardContext` in core. Move static NN templates and
the nanoGPT plan into the optional `tensor-nn` distribution, beside `tensor-torch`.
The development workspace installs NN for its tests; the runtime wheel depends
only on NumPy. A training consumer installs Tensor, Tensor NN and NumPy.

Group workloads under `benchmarks/inference` and `benchmarks/nanogpt`, artifact
acceptance commands under `scripts/validation`, and plotting under `scripts/plots`.
Core tests mirror implementation responsibilities; optional package tests live
with their package. Keep Phase 0 experiments and measured evidence historical.

## Compatibility

The initial migration retained flat core aliases, a forwarding `tensor.nn`
namespace and old tool wrappers. A subsequent user-requested cleanup removes
those compatibility paths. Internal imports use grouped modules, NN imports use
`tensor_nn`, and commands use `benchmarks` or `scripts`. The public
`tensor.build(...)` function remains available and imports its implementation
from `tensor.compiler.build`.

The generic artifact schemas, native descriptor layouts and public workbench
signatures do not change. Training bundles are separately implementation-bound:
schema `tensor.manual-nanogpt.v2` records hashes of canonical modules from both
installed distributions, including shared runtime/provider code. V1 training
bundles use their original wheels or must be rebuilt. Exact source matching is
retained, rather than silently rebinding an old executable to changed code.

Historical measurements and hashes are not rewritten. Fresh bundles, core-only
and training consumers, full CUDA/WebGPU regressions and Linux/Windows producers
validate the migration. The [development guide](../development.md) describes
canonical paths and installation; the [migration report](../research/reorganization.md)
records checks and evidence.
