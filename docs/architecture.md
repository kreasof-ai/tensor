# Architecture

Tensor compiles kernels into `.tbin` files and runs those files through device
providers. `tensor-torch` adapts those primitives to PyTorch. `tensor-nn` shows
native module composition and explicit training. Model engines belong to AKAS.

```text
compiler ──writes──→ artifact ──loaded by──→ provider
                                               │
                                            runtime
modules ──packages/resolves──→ artifacts
CLI ──calls──→ the relevant library
```

## Read one kernel through the system

1. [The kernel](../examples/elementwise.py) exports a TileLang PrimFunc.
2. [Build dispatch](../src/tensor/compiler/build.py) selects a compiler backend.
3. [CUDA construction](../src/tensor/compiler/cuda/build.py) selects implementations
   for opted-in logical operations through
   [prelowering](../src/tensor/compiler/cuda/prelower.py), lowers the kernel,
   compiles source, and writes a verified artifact. Explicit kernels retain their schedule.
4. [Artifact validation](../src/tensor/artifacts/format.py) checks metadata and
   payload integrity.
5. [CUDA Device](../src/tensor/providers/cuda/device.py) owns allocation, streams,
   image loading, and launches through the [driver](../src/tensor/providers/cuda/driver.py).
6. [Executable](../src/tensor/runtime/executable.py) binds inputs, checks shapes
   and resources, allocates declared outputs, and submits a call.
7. [The runner](../examples/run_elementwise.py) checks the result against NumPy.

## Ownership

| Component | Owns |
|---|---|
| `runtime/` | Buffer/executable/event lifetimes, call descriptors, signatures, DLPack, manual backward |
| `providers/` | Hardware allocations, transfers, streams, and submission |
| `compiler/frontend/` | Source factories, frontend restoration, signatures, and IR operator inventory |
| `compiler/cuda/`, `compiler/webgpu/`, `compiler/cpu.py` | Backend lowering and artifact construction |
| `compiler/cache.py`, `search.py`, `tuning.py` | Build reuse and correctness-gated schedule exploration |
| `artifacts/` | Executable format, integrity checks, portable frontend preflight, inspection |
| `modules/` | Manifests, content, pinned graphs, storage, archives, Project/Module, registry transport |
| `cli/` | Command arguments and presentation |
| `diagnostics.py` | Environment and capability checks |

Runtime/provider imports stay independent of compiler and framework packages.
Frontend deserialization and source execution happen only during an explicit
build or compile request. CLI imports remain lightweight; command handlers load
on demand. Implementation belongs in named modules; package initializers expose
small public surfaces.

WebGPU algorithms live in `compiler/webgpu/schedules/`. Each file contains one
schedule family and its DSL implementation. `passes/` contains transformations
and barrier validation. Provider-specific prepared submission lives in
`providers/webgpu/plan.py`.

CUDA's explicit Hopper ring pipeline lives in `compiler/cuda/hopper.py`.
`compiler/cuda/schedules.py` supplies the pipeline, storage and geometry search
space with legality checks. Native TMA signatures use TileLang's launch adapter;
the ordinary artifact ABI continues to accept pointer/scalar launches only.
The Hopper producer can fuse K-partition reduction through cluster shared memory.
`compiler/cuda/sliced.py` instead splits K across warps and reduces within a block,
using ordinary MMA on SM80 and newer. Both retain FP32 accumulation and make their
parallelism and reduction strategy explicit search choices.

## Collections of kernels

[Project](../src/tensor/modules/project.py) resolves installed module exports.
[Module](../src/tensor/modules/module.py) selects or explicitly compiles them.
Manifest/content validation, caches, lock graphs, archives, and registry
transport have separate owners. Packaging code does not execute source while
installing or inspecting a module.

## Outside the core

`packages/tensor-torch` binds framework buffers, streams, custom operators and
supported inference graphs. `packages/tensor-nn` contains module templates,
matching kernel factories, and the worked nanoGPT example. `benchmarks/` owns
measurements; `tools/` owns environment setup and acceptance commands.

Historical experiments and evidence live in `research/`, grouped by study.
LLT-specific material belongs to Loop Latent Transformer; LFM2/Qwen inference
and serving material belongs to AKAS.
