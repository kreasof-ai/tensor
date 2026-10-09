# Tensor documentation

Start with [installation](guides/installation.md) to choose a runtime or producer
environment, then follow the [quickstart](guides/quickstart.md) to build and
execute your first kernel. Tensor is pre-1.0; read
[support and compatibility](guides/compatibility.md) before transferring or
upgrading artifacts. Commands run from the repository root unless stated otherwise.

## Guides

| I want to… | Read |
|---|---|
| Try the CLI or install a compiler-free runtime | [Installation](guides/installation.md) |
| Check tested platforms and upgrade expectations | [Compatibility](guides/compatibility.md) |
| Install a producer, build a kernel, and run it without compiler packages | [Quickstart](guides/quickstart.md) |
| Bring an existing TileLang kernel into Tensor | [From TileLang to Tensor](guides/from-tilelang.md) |
| Discover schedules and reuse measured producer settings | [Schedule search](guides/schedule-search.md) |
| Work with device buffers, loaded kernels, timing, and CUDA streams | [Python runtime](guides/runtime.md) |
| Package exports, pin dependencies, and publish through PyPI | [Tensor modules](guides/modules.md) |
| Use Tensor kernels with `torch.compile` or custom operators | [PyTorch integration](guides/pytorch.md) |
| Train LLT and use persistent shared-KV attention | [LLT operators](guides/llt.md) |
| Measure vLLM, SGLang and llama.cpp with the same request load | [LLM serving benchmarks](guides/llm-serving-benchmarks.md) |
| Build and run portable WGSL artifacts | [WebGPU provider](guides/webgpu.md) |
| Write explicit forward and backward callbacks | [Manual backward](guides/manual-backward.md) |
| Run the standalone nanoGPT training template | [Tensor NN](../packages/tensor-nn/README.md) |
| Generate text with LFM2.5-2.6B GGUF weights | [Tensor LLM](../packages/tensor-llm/README.md) |

## Reference and development

- [Contributing](../CONTRIBUTING.md): first setup, focused checks, and pull requests.
- [Release preparation](releases.md) and [changelog](../CHANGELOG.md).
- [Path to 1.0](plan/v1.md): proposed stability boundary, remaining work, and acceptance criteria.
- [1.0 benchmarks and demonstrations](plan/v1-benchmarks.md): requested comparison matrix and reproducible evaluation plan.
- [Unified Tensor LLM engine](plan/unified-llm-engine.md): staged batch-1 and batch throughput development in the existing optional package.
- [Tuning workbench proposal](plan/tuning-workbench.md): human/agent profiling, refinement and verified kernel reconstruction.
- [Runtime ABI](reference/runtime-abi.md): descriptor layouts, capabilities,
  scalar and buffer rules, executable/event lifetimes, and native hosts.
- [Development](development.md): source layout, environments, wheel builds,
  testing, and retained experiments.
- CLI reference: `tensor --help` and `tensor COMMAND --help` describe the
  installed command options.

## Results and project history

| Collection | Purpose |
|---|---|
| [Research and benchmarks](research/README.md) | Measured results, methods, reproduction commands, and retained raw data |
| [Roadmap and validation plans](plan/README.md) | Accepted phase scopes, completion criteria, and validation runbooks |
| [Architecture decisions](adr/README.md) | Numbered decisions with their evidence and tradeoffs |
| [Original proposal](architecture/proposal.md) | Initial architecture and longer-term ideas; implemented scope is in the roadmap |

Research reports describe the code, artifacts, and environments they measured.
Historical test counts, package layouts, and commands belong to those revisions.
Use the guides for current usage and the reports for reproducing a specific
measurement. Data under `research/data/` retains its original provenance.
