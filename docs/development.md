# Development

## Environment

```sh
uv sync --locked
uv run --locked python -m pytest
```

The default groups install the pinned compiler, pytest, and the workspace's
`tensor-nn`. Install the optional PyTorch binding separately when needed:

```sh
uv pip install --no-deps -e packages/tensor-torch
```

Torch/Triton and wgpu device checks are optional. Use `uv run --no-sync` to retain
additional local dependencies. Hardware tests require their documented opt-in
flags; skipped tests do not establish device support.

## Source reading order

Follow [Architecture](architecture.md) for a complete kernel path and ownership
map. Core tests live beside corresponding responsibilities under `tests/`.
Package tests live in `packages/tensor-nn/tests` and `packages/tensor-torch/tests`.
Historical contracts live with their implementation under `research/phase0/tests`.

```text
src/tensor/       reusable runtime, providers, compiler, artifacts, modules, CLI
packages/         PyTorch binding and native NN templates/examples
examples/         small complete kernel demonstrations
benchmarks/       kernels, framework adapters, nanoGPT training and references
tools/bootstrap/  dependency setup
tools/validation/ artifact/provider/registry/distribution acceptance commands
research/         historical experiments, reports, evidence, plans and decisions
```

A focused runtime check:

```sh
uv run --locked python -m pytest tests/runtime tests/integration/test_layout.py
```

Bootstrap NVRTC as described in [Quickstart](quickstart.md). Build native WebGPU
submission with `TENSOR_BUILD_WEBGPU_NATIVE=1`; ordinary core wheels remain
Python-only. Optional binding native builds are documented in the package README.

## Distribution and consumer checks

```sh
uv run --locked --no-default-groups python tools/validation/distribution_check.py
```

This builds core, NN, and Torch distributions and verifies metadata, included
sources, CLI entry points, and compiler-free installed consumers. It uploads
nothing. See [Releases](releases.md).

Kernel factories use `tensor.compiler.frontend.source.export_source` to include
implementation fingerprints in specialization files. Rebuild compiled bundles
when those implementation inputs change. The NN bundle checks both the runtime
and its templates; changing an import path also changes its fingerprint.
Generic `.tbin`/`.tpack` formats and native ABI layouts remain separate contracts.

## Benchmarks and historical experiments

Current measurements live under `benchmarks/kernels`, `benchmarks/adapters`, and
`benchmarks/training/nanogpt`. The NN example stays in
`packages/tensor-nn/examples/nanogpt`; PyTorch supplies reference checks only.
Model inference measurements and serving belong to [AKAS](https://github.com/kreasof-ai/akas).

```sh
uv run --locked python -m research.phase0.harness --list
uv run --locked python -m pytest research/phase0/tests
```

Historical reports describe the revisions and environments measured there.
Evidence files retain their original bytes and hashes. Read
[Research](../research/README.md) for the archive organization.

## Changes

Keep library dependencies independent of CLI presentation. Initializers expose
imports; implementations belong in named modules. Update the example or usage
page when behavior changes and run checks appropriate to the affected contract.

The [migration ledger](migration.md) records source ownership and verification.
See [Contributing](../CONTRIBUTING.md) for the review process.
