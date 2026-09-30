# Repository layout and development

The runtime wheel contains reusable execution, compilation entry points and
artifact/module tooling. Optional framework adapters and training templates have
separate distributions. Benchmark workloads and acceptance scripts stay in the
repository rather than entering the runtime wheel.

```text
src/tensor/
  runtime/       buffers, executables, ABI, signatures, DLPack, manual backward
  providers/     CUDA, CPU and WebGPU execution; shared WebGPU metadata contract
  compiler/      build/cache, NVRTC/nvcc, CUDA/CPU/WebGPU lowering, tuning
  artifacts/     formats, portable IR, module packages, registry transport
  cli/           argument handling, run/bench/inspect and diagnostics
  include/       installed native ABI headers
  native/        installed C++/Rust reference hosts

packages/
  tensor-torch/  PyTorch adapter, native submission code and adapter tests
  tensor-nn/     manual training templates, nanoGPT plan and training tests

benchmarks/
  inference/    Torch/TileLang/Triton comparisons, attention and WebGPU scaling
  nanogpt/      producer, independent reference, validation, consumer, benchmark

scripts/
  validation/   artifact transfer, registry and provider acceptance commands
  plots/        standalone figures from retained benchmark data

tools/           dependency bootstrap scripts
tests/           runtime/compiler/artifact/provider/integration/legacy contracts
examples/        small kernels and modules
experiments/p0/  historical architecture experiments
docs/research/   historical measurements and their original evidence
```

Use the grouped implementation paths: `tensor.runtime.abi`,
`tensor.providers.cuda`, `tensor.compiler.build` and `tensor.artifacts.modules`.
Flat imports such as `tensor.abi`, `tensor.cuda`, `tensor.build` and
`tensor.modules` have been removed. Import build internals from
`tensor.compiler.build`; the public `tensor.build(...)` function remains callable.

The public workbench remains `import tensor as tx`. Manual backward stays in
core: `tx.ManualFunction` and `tx.BackwardContext`. Training uses the optional
`from tensor_nn import GPTConfig, NanoGPT`; the `tensor.nn` namespace has been
removed. The core wheel depends only on NumPy and can import and run the CLI
without NN, Torch, wgpu or compiler packages.

## Development environment

```sh
uv sync --locked
uv run --locked python -m pytest
```

The uv workspace includes `packages/tensor-nn`, installed through the development
group. Compiler packages retain the existing pinned versions. Torch/Triton and
native wgpu are separately installed dependencies for their opt-in validation
paths. Use `uv run --no-sync` when preserving those development additions.
The Torch adapter remains independently installable:

```sh
uv pip install --no-deps -e packages/tensor-torch
```

Core tests mirror the core groups. Package tests live beside their implementation;
the root pytest configuration includes both optional packages. Phase 0 contracts
are grouped under `tests/legacy`, with the experiments themselves preserved.

## Wheels and standalone training

```sh
uv build --wheel --out-dir build/wheels
uv build --wheel packages/tensor-nn --out-dir build/wheels
uv venv --python 3.12 build/consumer
uv pip install --python build/consumer/bin/python \
  build/wheels/tensor_workspace-0.1.0-py3-none-any.whl \
  build/wheels/tensor_nn-0.1.0-py3-none-any.whl
```

Omit the second wheel for a core-only consumer. On Windows use
`build/consumer/Scripts/python.exe`. A standalone training consumer has exactly
three installed distributions: Tensor, Tensor NN and NumPy. Tensor NN adds no
framework/compiler dependency. Existing inference consumers keep their previous
dependency sets.

With the producer/reference environment and NVRTC bundle prepared:

```sh
uv run --no-sync python benchmarks/nanogpt/producer.py --out build/training --target sm_86 --tune
uv run --no-sync python benchmarks/nanogpt/validate.py --bundle build/training --out build/reference.json
build/consumer/bin/python benchmarks/nanogpt/consumer.py \
  --bundle build/training --reference build/reference.json --out build/consumer.json
```

Training manifest schema `tensor.manual-nanogpt.v2` hashes canonical implementation
modules from both installed wheels. Consumers reject a mismatched core or NN
implementation before kernel loading. This protects implementation changes as
well as file moves. The reorganized runtime deliberately rejects v1 training
bundles; run those with their original wheels or build a new v2 bundle. Generic
`.tbin`/`.tpack` formats and native descriptor layouts have not changed.

Relocated `tools/*.py` wrappers have been removed. Run benchmark commands from
`benchmarks`, validation and plot commands from `scripts`, and dependency
bootstrap commands from `tools`. The [path mapping](research/data/reorganization-paths.json)
lists replacements for retired imports and commands.

Historical reports retain the distribution counts and source hashes they
actually measured. Their original wheels remain the reproducible consumer for
fingerprint-bound artifacts; import cleanup does not rewrite past evidence.
