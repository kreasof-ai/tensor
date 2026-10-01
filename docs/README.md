# Tensor documentation

Start with the [project README](../README.md) for an overview, then follow the
[quickstart](guides/quickstart.md) to build and execute a kernel. Commands in the
guides run from the repository root unless stated otherwise.

## Guides

| I want to… | Read |
|---|---|
| Install a producer, build a kernel, and run it without compiler packages | [Quickstart](guides/quickstart.md) |
| Work with device buffers, loaded kernels, timing, and CUDA streams | [Python runtime](guides/runtime.md) |
| Package exports, pin dependencies, and publish through PyPI | [Tensor modules](guides/modules.md) |
| Use Tensor kernels with `torch.compile` or custom operators | [PyTorch integration](guides/pytorch.md) |
| Build and run portable WGSL artifacts | [WebGPU provider](guides/webgpu.md) |
| Write explicit forward and backward callbacks | [Manual backward](guides/manual-backward.md) |
| Run the standalone nanoGPT training template | [Tensor NN](../packages/tensor-nn/README.md) |
| Generate text with LFM2.5-2.6B GGUF weights | [Tensor LLM](../packages/tensor-llm/README.md) |

## Reference and development

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
