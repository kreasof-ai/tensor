# Contributing to Tensor

Tensor is pre-1.0. Small fixes, clearer examples, hardware validation, and
reproducible bug reports are useful contributions. For a new backend, API, or
large compiler change, describe the intended behavior in an issue first so the
scope and compatibility implications can be discussed.

## Set up a checkout

Install [uv](https://docs.astral.sh/uv/) and use 64-bit Python 3.12:

```sh
git clone https://github.com/akbar2habibullah/tensor.git
cd tensor
uv sync --locked
uv run --locked tensor --version
uv run --locked python -m pytest
```

The default environment installs the pinned compiler, pytest, Tensor NN, and
Tensor LLM. The ordinary suite needs no physical GPU. Tests for real devices,
NVRTC, native hosts, or optional frameworks may skip unless their dependencies
and opt-in flags are present. Skips do not establish hardware support.

For a quicker runtime-only change, start with:

```sh
uv run --locked python -m pytest tests/runtime tests/integration/test_layout.py tests/integration/test_doctor.py
```

For actual CUDA builds, follow the [quickstart](docs/guides/quickstart.md) to
bootstrap NVRTC. For optional adapters, native builds, provider flags, and the
source map, see the [development guide](docs/development.md). Use
`uv run --no-sync` when preserving additional local Torch/Triton/wgpu installs.

## Make and verify a change

1. Create a branch and keep the change focused on a concrete behavior.
2. Run the tests for the affected runtime, compiler, provider, or optional package.
   Add a regression test when fixing a behavior that could break again.
3. Update the usage guide or example when a user-facing API or command changes.
4. Add a concise entry under `Unreleased` in [CHANGELOG.md](CHANGELOG.md), including
   upgrade or rebuild requirements when applicable.
5. Open a pull request explaining the problem, resulting behavior, and validation.
   State any hardware-dependent checks you could not run.

Keep runtime imports independent of compiler and optional framework packages.
Keep reusable code in `src/tensor` or the relevant package, small examples in
`examples`, and workload-specific measurements in `benchmarks`. Document current
usage in `docs/guides`; retain historical evidence in `docs/research` with its
original revision and provenance. Consult the existing
[architecture decisions](docs/adr/README.md) for changes to contracts.

Package changes should also pass the [distribution checks](docs/releases.md).
The core distribution name is intentionally still `tensor-workspace`; changing
it requires migrating package dependencies and artifact recipes together.

## Report a problem

Use [GitHub issues](https://github.com/akbar2habibullah/tensor/issues). Include
the Tensor version or Git revision, OS, Python version, exact command, traceback,
and a small reproduction. For provider issues, include GPU/driver information and
`tensor doctor --json` (or `tensor doctor --provider webgpu --json`). Remove local
secrets and private paths from diagnostic output before sharing it.

Contributions are distributed under the repository's [MIT license](LICENSE).
