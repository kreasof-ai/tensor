# Repository reorganization acceptance

Core implementations now live under `tensor.runtime`, `tensor.providers`,
`tensor.compiler`, `tensor.artifacts` and `tensor.cli`. Manual backward remains in
core; NN templates and the static nanoGPT plan ship in the separate `tensor-nn`
distribution. Benchmark, validation and plot code lives outside the runtime.
See the [development guide](../development.md),
[ADR 0017](../adr/0017-repository-layout-and-optional-training-package.md) and
[path mapping](data/reorganization-paths.json).

The migration changes ownership and import paths. Generic `.tbin`/`.tpack`
schemas, native descriptor layouts, public workbench signatures and kernel math
retain their existing contracts. Flat imports alias the same module objects,
preserving shared globals and monkeypatches; `tensor.build(...)` stays callable.
Legacy script paths forward to grouped commands and work outside the checkout.
Optional training symbols report a missing `tensor-nn` installation explicitly.

## Local verification

**199 tests pass, zero skips, in 189.44 s**, with every CUDA/WebGPU opt-in enabled.
The full run includes the previous 195 cases and four migration contracts for
import/global identity, a core with optional imports blocked, legacy executable
paths and training schema/fingerprint rejection. The small nanoGPT specialization
also completes ten reference-checked updates. Results are retained as
[JUnit](data/reorganization-regression.xml) and
[diagnostic validation](data/reorganization-diagnostic-validation.json).

A fresh full-model NVRTC build selects 54 kernel specializations, tunes 18
GEMM/fused-GEMM shapes and takes 212.46 s. All selected source files are checked
against regeneration from the moved templates. The
[producer manifest](data/reorganization-nanogpt-producer.json) records schema
`tensor.manual-nanogpt.v2` and canonical implementation hashes spanning both wheels.

Ten full-model numerical updates pass, checking every parameter gradient before
reference replacement, then optimizer weights/moments/variances on identical
checked gradients, as in Phase 6 acceptance. Maximum loss error remains
0.0000123941 and worst parameter-gradient relative L2 error is 0.003449. This
uses the same 123,980,544-parameter architecture and precision contract, including
tied weights. The [numerical report](data/reorganization-nanogpt-validation.json)
retains per-parameter errors and final-state samples; this migration does not
restate those checks as independent parameter-trajectory equality.

The installed [training consumer](data/reorganization-training-consumer.json)
contains exactly **NumPy, Tensor NN and Tensor** and repeats all ten full-model
updates, loss/norm checks and sampled optimizer-state checks with framework and
compiler imports blocked. The [core-only consumer](data/reorganization-core-consumer.json)
contains exactly **NumPy and Tensor**, blocks NN, Torch, compiler and wgpu imports,
executes manual backward on CPU, and executes a packaged CUDA kernel. Wheel
inspection checks that repository tooling, optional NN implementations and stale
flat package-collision files are absent from the core wheel.

## Scope and evidence

The original benchmark figures and physical AMD acceptance remain historical
records with their original source hashes and wheels. No new throughput or
physical AMD/Apple claim follows from this file-layout migration. Generic
inference artifacts retain ABI/target compatibility. Implementation-bound
training and WebGPU validation bundles need their matching source/wheels;
training v1 bundles require original wheels or a fresh v2 build.

The [verification record](data/reorganization-verification.json) binds source,
wheel, module and evidence hashes. All three workflows pass at implementation
commit `75acca2`, including their Linux/Windows jobs:

- [NN producer](https://github.com/kreasof-ai/tensor/actions/runs/36779120801),
  retained [status](data/reorganization-nn-ci.json).
- [NVRTC runtime/module and native executor](https://github.com/kreasof-ai/tensor/actions/runs/36779120793),
  retained [status](data/reorganization-nvrtc-ci.json).
- [WebGPU producer/transfer](https://github.com/kreasof-ai/tensor/actions/runs/36779120766),
  retained [status](data/reorganization-webgpu-ci.json).

The NN jobs compile all 54 diagnostic kernels without a GPU and verify both
installed wheels. Their actual Linux and Windows bundles then each pass ten
diagnostic updates on the separate A10G in clean three-distribution consumers:
[Linux](data/reorganization-ci-linux-consumer.json) and
[Windows](data/reorganization-ci-windows-consumer.json). Numerical fixtures copy
the local Torch-validated diagnostic values unchanged, with explicit binding to
the equivalent independent compilation after config, all kernel specializations,
all 11 implementation sources, both wheels and artifact/module checksum checks.
[Linux verification](data/reorganization-ci-linux-verification.json) and
[Windows verification](data/reorganization-ci-windows-verification.json) record
line-ending normalization, original/derived fixture hashes and wheel identities.
Derived fixtures and producer manifests are retained beside those records.
These are transferred diagnostic runs, not additional full-scale benchmarks.
