# Tensor

Build, package, and run tensor kernels with a small runtime and one CLI.

Tensor uses TileLang and TIRx to compile kernels into `.tbin` artifacts. You can
ship those artifacts to a separate machine and execute them without the compiler
stack. Named modules add reusable exports, pinned dependencies, and offline or
PyPI distribution.

CUDA is the primary backend. Compilation uses a local NVRTC bundle; execution
needs only the Tensor wheel, NumPy, and an NVIDIA driver. An optional native
WebGPU provider runs a bounded inference profile through wgpu.

## How it works

```text
TileLang kernel → TIRx lowering → provider compiler → .tbin artifact
                                                       ↓
                                             Tensor runtime → GPU
```

The build environment owns compiler dependencies. The runtime owns buffers,
kernel loading, argument validation, streams, and execution. CUDA artifacts
target one exact GPU architecture; portable TIRx can be recompiled on a producer
for another target. WebGPU artifacts contain WGSL and negotiate adapter features
and limits when loaded.

## Quickstart

From a checkout, install [uv](https://docs.astral.sh/uv/) and prepare the pinned
Python 3.12 compiler environment. On Linux, with an NVIDIA GPU:

```sh
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
uv run --locked tensor doctor
uv run --locked tensor build examples/elementwise.py --out build/elementwise.tbin
```

Run the compiled kernel from Python:

```python
import tensor as tx

with tx.Device() as device:
    kernel = device.load("build/elementwise.tbin")
    a = device.arange(129)
    b = device.ones((129,))
    result = kernel(a, b)  # relu(2 * a + b)
    print(result.to_numpy())
```

Use `uv run --locked python` for this example in the checkout. A GPU-free build
host can select an explicit target with `--target sm_86`; the consuming GPU must
match that target. Output paths must be new.

The [full quickstart](docs/guides/quickstart.md) covers Windows, CLI execution,
benchmarking, and installing a separate compiler-free consumer.

## What is available

| Component | Current scope | Guide |
|---|---|---|
| Core runtime and CLI | CUDA kernels, contiguous buffers, symbolic dimensions, typed scalars, DLPack, streams, inspection, and caching | [Runtime](docs/guides/runtime.md) |
| Modules | Named exports, exact versions, verified `.tpack` archives, lockfiles, and PyPI transport wheels | [Modules](docs/guides/modules.md) |
| PyTorch adapter | `torch.compile` inference regions and installed kernels as custom operators; optional C++ executor | [PyTorch](docs/guides/pytorch.md) |
| WebGPU provider | Elementwise, GEMM/linear, device-resident MLPs, and forward attention | [WebGPU](docs/guides/webgpu.md) |
| Manual training | Public backward callbacks and optional `tensor-nn` nanoGPT templates | [Manual backward](docs/guides/manual-backward.md) |
| GGUF inference | Optional `tensor-llm` single-sequence LFM2.5-2.6B inference with F16, Q4_0, and Q4_K_M weights | [Tensor LLM](packages/tensor-llm/README.md) |

The core wheel depends only on NumPy. Compiler packages, PyTorch, training and
model templates, and wgpu are installed separately for the paths that use them.
A Linux x86-64 CPU provider also exercises the shared runtime contract.

## Status and measurements

Phases 0–6 are complete within the profiles defined in the
[roadmap](docs/plan/roadmap.md). CUDA execution has been measured on NVIDIA A10G;
WebGPU transfer and execution have been validated on Windows RX 6700 XT through
Vulkan. Other hardware requires its own validation.

The [research index](docs/research/README.md) collects reproducible benchmarks:
Torch/TileLang/Triton/WebGPU latency scaling, ten-update nanoGPT training, and
LFM2 inference against llama.cpp CUDA. The ordinary LFM2 runner offers an opt-in
`optimized` CUDA profile with packed decode, fused projections, tuned prefill,
and shared K/V attention at long context. See the
[F16/Q4_0/Q4_K_M measurements](docs/research/lfm2-cuda-formats.md).
The CUDA algorithms now remain visible in TileLang/TIRx, with shared CUDA/WebGPU
schedule discovery and producer profiles. The
[compiler cleanup report](docs/research/lfm2-compiler-cleanup.md) records the
fresh correctness checks and comparison with the frozen native implementation.

Standalone autograd, arbitrary graph fusion, complete TileLang/TIRx coverage on
WebGPU, and general GGUF model support remain future work. Direct PTX compilation
remains experimental work after v1.

## Documentation and development

- [Documentation index](docs/README.md): setup, usage guides, contracts, and evidence.
- [Development guide](docs/development.md): repository layout, optional packages, and tests.
- [Runtime ABI reference](docs/reference/runtime-abi.md): descriptors, ownership, and compatibility.
- [Architecture decisions](docs/adr/README.md): the reasons behind the current design.
- [Original proposal](docs/architecture/proposal.md): design background and longer-term goals.
