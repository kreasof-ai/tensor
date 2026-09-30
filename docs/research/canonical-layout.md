# Canonical import cleanup

The flat core module aliases, `tensor.nn` forwarding namespace and 30 relocated
`tools/*.py` wrappers have been removed. Core internals use grouped modules,
training imports use `tensor_nn`, benchmarks run from `benchmarks` and validation
and plot commands run from `scripts`. Dependency bootstrap commands remain in
`tools`. The public `tensor.build(...)` function and manual backward API remain
available. See the [development guide](../development.md) and
[replacement paths](data/reorganization-paths.json).

Repository callers, tests, documentation commands and workflows use the new
paths. The native Torch executor now imports `tensor.providers.cuda`; its C++
extension was rebuilt for verification. Historical raw measurements are retained.

A freshly built core wheel contains none of the retired modules or NN namespace.
A clean Tensor + NumPy consumer imports the core and runs the CLI. The new core
and NN wheels match all eleven implementation hashes in the existing v2 training
bundle. A fresh Tensor + Tensor NN + NumPy consumer completes ten full nanoGPT
updates against the existing reference, checking losses, gradient norms and final
parameter/optimizer samples with framework/compiler imports prohibited.
[Wheel and layout checks](data/canonical-layout.json) and
[clean training consumer](data/canonical-training-consumer.json) retain the evidence.

The full CUDA/WebGPU suite passes **199 tests, zero skips, in 186.84 s**. This
includes NVRTC compilation, CUDA and WebGPU execution, manual training, Torch
stream/capture behavior and the rebuilt native executor.
[JUnit results](data/canonical-regression.xml) retain the complete run.
