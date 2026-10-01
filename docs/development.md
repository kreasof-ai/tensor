# Repository layout and development

[Documentation](README.md) · [Quickstart](guides/quickstart.md)

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
  tensor-llm/    GGUF/tokenizer and standalone LFM2 inference templates

benchmarks/
  inference/    Torch/TileLang/Triton comparisons, attention and WebGPU scaling
  nanogpt/      producer, independent reference, validation, consumer, benchmark
  lfm2/         pinned GGUF download, producer, references, consumer, comparison

scripts/
  validation/   artifact transfer, registry and provider acceptance commands
  plots/        standalone figures from retained benchmark data

tools/           dependency bootstrap scripts
tests/           runtime/compiler/artifact/provider/integration/legacy contracts
examples/        small kernels and modules
experiments/p0/  historical architecture experiments
docs/research/   historical measurements and their original evidence
```

Documentation is organized by purpose: `docs/guides` for current usage,
`docs/reference` for contracts, `docs/architecture` for the original proposal,
`docs/adr` for numbered decisions, `docs/plan` for scope and validation plans,
and `docs/research` for measurements and retained data. The
[documentation index](README.md) is the entry point.

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

The uv workspace includes `packages/tensor-nn` and `packages/tensor-llm`, installed
through the development group. Compiler packages retain the existing pinned
versions. Torch/Triton and native wgpu are separately installed dependencies for their opt-in validation
paths. Use `uv run --no-sync` when preserving those development additions.
The Torch adapter remains independently installable:

```sh
uv pip install --no-deps -e packages/tensor-torch
```

Core tests mirror the core groups. Package tests live beside their implementation;
the root pytest configuration includes all optional packages. Phase 0 contracts
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

## Standalone GGUF inference

The optional [Tensor LLM package](../packages/tensor-llm/README.md) runs the
LFM2.5-2.6B hybrid convolution/attention architecture from packed GGUF weights
and an NVRTC-produced kernel bundle. Its consumer has four distributions:
Tensor, Tensor LLM, NumPy and regex. Workload recipes and the independent
llama.cpp/Torch references stay under `benchmarks/lfm2`; neither enters core.
See the [inference comparison](research/lfm2-inference.md) for the measured profile.

## Retained Phase 0 experiments

The initial architecture harness stays under `experiments/p0`. These commands
exercise historical experiments; current product builds use the NVRTC
[quickstart](guides/quickstart.md).

```sh
uv run --locked python -m experiments.p0.harness --list
uv run --locked python -m experiments.p0.harness --only codegen
uv run --locked python -m experiments.p0.validation --out experiments/p0/out/new-validation-run
```

Use a fresh output directory for each validation run. Reports record the Git
revision, source and lock hashes, and installed package versions. The historical
native CUDA probes require nvcc and a host compiler; the C++/Rust host probes
also need their respective compilers. Follow the
[original validation runbook](plan/opaque-artifact-validation.md) and
[Phase 0 exit report](research/e16-phase0-exit.md) for the full setup and commands.

## Documentation changes

Keep current setup and API instructions in the guides. Link benchmarks to their
retained data and specify the hardware, precision, and timing protocol behind
performance claims. When moving a document, update inbound links and relative
links inside it. Update the relevant index when adding a report or decision.
Preserve historical measurements and provenance rather than rewriting them to
look like results for the current revision.
