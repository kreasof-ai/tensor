# Device providers

Tensor builds separate artifacts for each provider. The runtime selects a device
and validates the artifact's ABI, target, and required capabilities before loading.
See [compilation](compilation.md) for producers and [runtime](runtime.md) for buffers,
streams, sessions, and explicit backward.

| Provider | Artifact | Consumer requirements |
|---|---|---|
| CUDA | CUDA image | NVIDIA driver; compiler/toolkit needed only on the producer |
| WebGPU | WGSL (`webgpu-portable-v1`) | Tensor's optional `webgpu` extra and a supported native adapter |
| CPU | CPU implementation | See the tested scope in [compatibility](compatibility.md) |

```python
import tensor as tx
cuda = tx.Device(provider="cuda")
webgpu = tx.Device(provider="webgpu")
```

Use `tensor doctor --provider cuda --json` or
`tensor doctor --provider webgpu --json` to inspect the environment.

## WebGPU contracts

The producer emits WGSL without a GPU or CUDA toolkit. Native wgpu creates the
consumer's shader pipelines; measure pipeline creation separately from warm calls.
The optional dependency is pinned to wgpu 0.29.0. Dependency upgrades require
revalidation of native encoding and resource lifetime checks.

| Workload | Supported profile |
|---|---|
| Elementwise | FP32/FP16 affine and ReLU; scalar arguments and symbolic dimensions |
| GEMM/linear | FP16/FP32 inputs, FP32 accumulation, transposed RHS, tails, bias/ReLU |
| Composition | Device-resident two-layer MLP and prepared execution plans |
| Attention | FP16 operands/output, FP32 online softmax, causal/noncausal, D=64/128 |

These profiles have no dropout, arbitrary attention masks, GQA, backward kernels,
external streams, or DLPack pointer borrowing. Schedules must meet adapter limits.
Tensor checks workgroup storage (at most 32 KiB) and buffer binding limits (128 MiB
by default). Larger buffers require an explicit supported device request, such as
`tx.Device(provider="webgpu", max_buffer_size=268435456)`. FP16 requires
`shader-f16`; unsupported features fail before execution.

GEMM uses 2×2 register microtiles by default. Function attributes
`tensor.webgpu.gemm_microtile`, `tensor.webgpu.gemm_accumulation`,
`tensor.webgpu.loop_unroll`, and `tensor.webgpu.reduction` select supported lowering
choices. The schedule factories and their DSL implementations live together in
[`compiler/webgpu/schedules`](../src/tensor/compiler/webgpu/schedules/).
Changes to floating-point accumulation order require application numerical checks.
See the [accumulation study](../research/studies/webgpu-gemm-accumulation/reports/webgpu-gemm-accumulation.md)
and [outer-product study](../research/studies/webgpu-outer-product/reports/webgpu-outer-product-gemm.md).
Model-specific profile selection and measurements belong to
[AKAS](https://github.com/kreasof-ai/akas/tree/main/docs/research).

Prepared plans validate resource lifetimes and cache bind groups. Readback arrays
own their memory. Setting `TENSOR_BUILD_WEBGPU_NATIVE=1` when building a wheel
adds the optional C encoder; the Python path remains available without it.
The native fast path requires its supported wgpu version and 64-bit ABI.

## Build and consume

Producer Python 3.12 needs Tensor's pinned compiler extra. No GPU, CUDA toolkit,
NVRTC or native host compiler is needed to emit WGSL:

```bash
uv sync --locked
uv run --locked tensor build examples/webgpu_gemm.py --provider webgpu --out build/webgpu-linear.tbin
uv run --locked tensor inspect examples/webgpu_gemm.py --provider webgpu --stage target
uv build --wheel --out-dir build/webgpu-wheels
```

For a local consumer, install the wheel and its WebGPU extra into a separate
Python 3.12 environment:

```sh
uv venv --python 3.12 build/webgpu-consumer
uv pip install --python build/webgpu-consumer/bin/python \
  'build/webgpu-wheels/tensor_workspace-0.1.0-py3-none-any.whl[webgpu]'
build/webgpu-consumer/bin/tensor doctor --provider webgpu --json
```

On Windows use executables under `build/webgpu-consumer/Scripts/`. The Python
runtime selects this provider with `tx.Device(provider="webgpu")`. CUDA and
WebGPU artifacts are separate builds; this example uses WebGPU-specific tiling.

## Transfer the validation suite

Build the acceptance workload bundle on the producer:

```sh
uv run --locked python tools/validation/webgpu_validation.py --build build/webgpu-transfer
```

The generated directory contains `suite.json`, executable WGSL `.tbin` files and
`validation.tpack`, with original portable TIRx/source included. Transfer the
directory and `tools/validation/webgpu_validation.py` to the other machine. Install the
prepared Tensor wheel and its optional WebGPU extra into a separate Python 3.12
environment. This resolves the native wgpu library appropriate to that machine.

```bash
python -m pip install 'tensor_workspace-0.1.0-py3-none-any.whl[webgpu]'
python webgpu_validation.py --consume webgpu-transfer --require-second-gpu --out webgpu-result.json
```

For the prepared `webgpu-validation.zip`, extract it, change into its
`webgpu-validation` directory, install the included
wheel with the command above, and run the consumer. The current suite has **33
checks**, including 1M-element affine, 256/512-square GEMM and attention lengths
65/129/512. Return `webgpu-result.json` from the GPU machine. Audit it against the
transferred suite with:

```bash
python webgpu_audit.py --suite webgpu-transfer/suite.json --result webgpu-result.json
```

The audit rejects missing/duplicate cases, changed artifacts or consumer sources,
compiler imports/packages, invalid timing samples, same-host evidence and software
adapters. `--allow-software` checks CI evidence while keeping the physical GPU gate
open.

`--device N` selects another adapter. `tensor doctor --provider webgpu --json`
reports available features and limits for the selected adapter. Linux requires a
working Vulkan loader and GPU ICD; CI uses Mesa lavapipe explicitly as software
validation. macOS uses Metal and Windows uses the available native wgpu backend.
FP16 artifacts require the actual adapter's `shader-f16` feature; no FP32 fallback
silently changes the artifact's buffer representation. Unsupported adapters fail
with a feature/limit error.

On the tested Windows RX 6700 XT, ordinal 0 selected Vulkan with `shader-f16`;
ordinal 1 selected D3D12 without it. Recheck ordinals with `tensor doctor` before
selection. If the suite is on another drive from the Windows temporary directory,
set `TEMP` and `TMP` to a directory on the suite's drive before consumption; the
[Windows report](../research/studies/webgpu-validation/reports/webgpu-rx6700xt.md#reproduce-on-windows) records the
cross-drive module-path error and complete PowerShell reproduction commands.

Consumers import only Tensor, NumPy and wgpu. The validation command blocks
TileLang/TVM/TVM-FFI/Torch/Triton imports, restores the packaged module into a fresh
project, runs NumPy-reference comparisons and transfer checks, and records adapter
identity/features/limits, package versions, producer and artifact hashes, cold
pipeline creation, host enqueue and completed-call timing. MLP intermediates stay
on the device. Timing uses preallocated outputs and excludes uploads/downloads
and first pipeline creation. It measures host submission plus queue completion,
not isolated GPU timestamps. The [matched A10G scaling benchmark](../research/studies/latency-scaling/reports/latency-scaling.md)
instead allocates each output inside the timed call, matching the CUDA baseline
protocol, and compares copies of the same input tensors.

## Validation and evidence

`TENSOR_WEBGPU=1 python -m pytest tests/providers/test_webgpu.py` enables native
adapter checks. Ordinary tests cover GPU-free lowering and artifact validation.
The transferred suite distinguishes software validation from physical GPU evidence.
Historical [Windows AMD validation](../research/studies/webgpu-validation/reports/webgpu-rx6700xt.md)
and [Linux-to-Windows transfer](../research/studies/webgpu-validation/reports/webgpu-rx6700xt-transfer.md)
record the measured adapters, revisions, and strict audit results.

For the version-specific headless NVIDIA Vulkan setup used in latency measurements,
see the [original backend guide](../research/studies/webgpu-validation/reports/backend-guide-before-refactor.md#headless-nvidia-vulkan-in-a-compute-container).
Use [compatibility](compatibility.md) for current tested scope.
