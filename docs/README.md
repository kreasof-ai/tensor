# Tensor documentation

Read [the quickstart](quickstart.md), then [the architecture](architecture.md)
to follow a kernel through source, compilation, artifact loading, and execution.
Commands run from the repository root unless stated otherwise.

| Document | Purpose |
|---|---|
| [Installation](installation.md) | One-line CLI setup and Python project dependencies |
| [Quickstart](quickstart.md) | Installation and one complete kernel |
| [Architecture](architecture.md) | Responsibilities, dependencies, and source reading order |
| [Compilation](compilation.md) | TileLang exports, targets, schedules, and caching |
| [Runtime](runtime.md) | Buffers, calls, streams, timing, and explicit backward |
| [Modules](modules.md) | Named exports, dependencies, packaging, and distribution |
| [Backends](backends.md) | Provider-specific setup and WebGPU execution |
| [Compatibility](compatibility.md) | Tested scope, versions, and artifact upgrades |
| [Runtime ABI](reference/runtime-abi.md) | Native descriptors and ownership rules |
| [Development](development.md) | Environment, tests, native helpers, and package checks |
| [Releases](releases.md) | Local release preparation |

[Tensor Torch](../packages/tensor-torch/README.md) documents the PyTorch binding.
[Tensor NN](../packages/tensor-nn/README.md) documents native module templates and
the nanoGPT example. Model inference and serving belong to
[AKAS](https://github.com/kreasof-ai/akas).

For command options, use `tensor --help` and `tensor COMMAND --help`.
[Research](../research/README.md) retains old experiments, measurements, plans,
and decisions. Those documents describe their measured revisions.

