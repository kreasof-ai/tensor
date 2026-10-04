# Support and compatibility

[Documentation](../README.md) · [Installation](installation.md)

Tensor is pre-1.0. The package version `0.1.0` identifies the current packaging
baseline; it is not a promise of a stable API or artifact format. Until release
versions distinguish revisions, record the Git revision with experimental bundles
as well as the package version. Keep producer and consumer wheels together.

The proposed [1.0 roadmap](../plan/v1.md) describes the compatibility guarantees
and support-tier qualification work to complete before a stable release. Those
future commitments are not yet implemented or current support claims.

## Current validation scope

| Path | Environment and evidence | Boundary |
|---|---|---|
| CUDA producer | Linux and Windows pinned Python 3.12 / TileLang 0.1.14 / TVM FFI 0.1.12, NVRTC 12.9 | Exact target required on GPU-free hosts |
| CUDA runtime | NVIDIA A10G execution and compiler-free transfer | Other GPUs/drivers need validation; exact SM must match |
| WebGPU runtime | Windows RX 6700 XT through Vulkan and Linux software Vulkan CI | Software tests do not establish physical GPU performance; features/limits vary |
| CPU provider | Linux x86-64 host compilation/runtime contracts | Validation provider; no general CPU performance claim |
| macOS WebGPU | Native wgpu can use Metal | No retained physical Apple acceptance result; treat as unvalidated |
| Optional training/inference | Specific nanoGPT and LFM2 workload profiles | No general model/framework coverage |

The [research index](../research/README.md) links the retained hardware reports.
Use the [WebGPU guide](webgpu.md) for adapter selection and platform prerequisites.
Documentation for a possible backend is not evidence that every device is supported.

## Upgrading

- Pin exact Tensor and optional-package versions. For unreleased work, also pin
  the repository revision and keep the dependency lock.
- Rebuild `.tbin` files for a different CUDA SM. Portable TIRx can be recompiled
  only in a compatible producer environment.
- Read [CHANGELOG.md](../../CHANGELOG.md) before upgrading. Runtime ABI and
  capability checks reject incompatible artifacts when loaded; they do not
  replace application correctness checks.
- Training/inference bundles can fingerprint implementation modules. A source
  change can require a rebuild even when the Python version number is unchanged.
  Pre-reorganization nanoGPT v1 bundles require their original wheels; current
  training bundles use `tensor.manual-nanogpt.v2`.
- Re-run the workload's numerical validation after changing compiler versions,
  schedule profiles, arithmetic precision, or hardware.

See the [runtime ABI](../reference/runtime-abi.md) for format and lifetime rules,
and the [development guide](../development.md) for bundle provenance.
