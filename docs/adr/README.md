# Architecture decisions

[Documentation](../README.md) · [Roadmap](../plan/roadmap.md) · [Original proposal](../architecture/proposal.md)

Each numbered record states a decision, its evidence, and its consequences.
Read the current guides for usage; these records explain how the design arrived
there. Earlier phase decisions retain their original scope, including CPU as an
initial validation provider before WebGPU became the second GPU provider.

## Runtime, modules, and optional packages

| Record | Decision |
|---|---|
| [0011](0011-runtime-call-abi-and-nvrtc.md) | Separate runtime calls from compiler IR; use bundled NVRTC for CUDA |
| [0012](0012-phase2-executable-and-workspace-contract.md) | Define executable/event identities, lifetimes, and workspace requirements |
| [0013](0013-phase3-offline-module-system.md) | Resolve offline modules with pinned closures and explicit compilation |
| [0014](0014-pypi-module-transport.md) | Use PyPI wheels to transport verified Tensor modules |
| [0015](0015-webgpu-provider.md) | Add portable WebGPU inference and opaque buffer handles |
| [0016](0016-manual-training-and-bounded-autotuning.md) | Expose manual backward and validate bounded training/autotuning |
| [0017](0017-repository-layout-and-optional-training-package.md) | Group core implementation and separate optional training templates |

## Initial design and acceptance

| Record | Decision |
|---|---|
| [0001](0001-no-product-code-before-phase0.md) | Validate the architecture before product implementation |
| [0002](0002-cpu-backend-as-second-provider.md) | Use CPU to exercise the initial provider boundary |
| [0003](0003-pin-toolchain-treat-internals-private.md) | Pin compiler versions and keep frontend internals private |
| [0004](0004-no-gpu-local-remote-split.md) | Separate GPU-free research from remote execution |
| [0005](0005-pytorch-as-client-side-adapter.md) | Ship PyTorch integration as a client adapter |
| [0006](0006-prototyping-surface-not-tensor-library.md) | Define the kernel-author workbench and its initial scope |
| [0007](0007-opaque-artifact-spike.md) | Validate opaque executable transfer before expanding modules |
| [0008](0008-phase0-evidence-boundaries.md) | Bound the product contract by measured behavior |
| [0009](0009-complete-phase0-with-scoped-provider-and-composition.md) | Accept Phase 0's provider and composition profile |
| [0010](0010-complete-phase1-cuda-cli.md) | Accept the single-device CUDA CLI profile |
