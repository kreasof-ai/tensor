# Native WebGPU provider

Phase 5 implements portable inference through native wgpu. Its physical AMD/Apple
exit gate is still open. The local software Vulkan adapter passes the inference
suite, including compiler-free module consumption; its timings are software CPU
execution and must not be interpreted as AMD/Apple GPU performance.

## Build and consume

Producer Python 3.12 needs Tensor's pinned compiler extra. No GPU, CUDA toolkit,
NVRTC or native host compiler is needed to emit WGSL:

```bash
uv sync --locked
uv run --locked tensor build examples/webgpu_gemm.py --provider webgpu --out linear.tbin
uv run --locked tensor inspect examples/webgpu_gemm.py --provider webgpu --stage target
uv run --locked python tools/webgpu_validation.py --build build/webgpu-transfer
```

The generated directory contains `suite.json`, executable WGSL `.tbin` files and
`validation.tpack`, with original portable TIRx/source included. Transfer the
directory and `tools/webgpu_validation.py` to the other machine. Install the
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

Consumers import only Tensor, NumPy and wgpu. The validation command blocks
TileLang/TVM/TVM-FFI/Torch/Triton imports, restores the packaged module into a fresh
project, runs NumPy-reference comparisons and transfer checks, and records adapter
identity/features/limits, package versions, producer and artifact hashes, cold
pipeline creation, host enqueue and completed-call timing. MLP intermediates stay
on the device. Timing uses preallocated outputs and excludes uploads/downloads
and first pipeline creation. It measures host submission plus queue completion,
not isolated GPU timestamps.

## Supported scope

| Profile | Coverage |
|---|---|
| Elementwise | FP32 dynamic affine/ReLU and FP16 affine/ReLU; 32-bit scalars and symbolic dimensions |
| GEMM/linear | FP16/FP32 inputs, FP32 accumulation, FP16/FP32 outputs, transpose-B, M/N/K tails, optional bias/ReLU |
| Composition | Device-resident two-layer MLP, shared executable/event lifetime rules |
| Attention | FP16 Q/K/V/output, FP32 online softmax, causal/noncausal, batching/heads, sequence tails, D=64/128 |

These are bounded inference profiles, not arbitrary TileLang/TIRx compatibility.
There is no dropout, custom attention mask, GQA, backward pass, vendor matrix
instruction lowering, external stream or DLPack pointer borrowing in this profile.
The CUDA PyTorch adapter continues to use CUDA buffers and its existing C++ executor.

`examples/webgpu_gemm.py` uses 32×32 output tiles and K=16. The suite specializes
the existing attention source to query tiles of 8 and key tiles of 16, preserving
its online-softmax algorithm without a global score matrix. A large CUDA schedule
can exceed WebGPU workgroup limits even for the same logical operation. Tensor
rejects schedules above 32 KiB of workgroup storage; smaller adapter limits are
also checked. Buffer bindings are limited to the requested/device maximum, up to
128 MiB each. Dispatch x can be packed over x/z with the verified POD grid bound;
batch/head dimensions are flattened onto y. Subgroup width 32 is never assumed.

The artifact target is `webgpu-portable-v1`, kind `wgsl`. Portability comes from
WGSL plus explicit adapter feature/limit negotiation, rather than a CUDA SM or
toolkit version. Native wgpu still compiles/translates the shader when creating a
pipeline. Pipeline creation is reported separately from warm execution. The
Python binding is pinned to wgpu 0.29.0, bundling wgpu-native 27.0.2.0; it is an
optional consumer dependency, with its own transitive dependencies and native
library footprint. The previously measured wgpu-native v29 stripped-library size
is not a measurement of this entire Python distribution.

## Validation gates

`TENSOR_WEBGPU=1 python -m pytest tests/test_webgpu.py` enables native-adapter
contract checks. Ordinary tests exercise GPU-free lowering and artifact
validation. The dedicated workflow builds on Linux and Windows and transfers the
Windows-produced package to a separate Linux software consumer. Software success
does not close the physical GPU gate. Run the transferred suite with
`--require-second-gpu` on AMD or Apple and retain `webgpu-result.json` as acceptance
evidence; performance and vendor-specific issues must be assessed from that run.

The portable SIMT lowering prioritizes complete profile correctness. No CUDA,
native TileLang, Triton or vendor-library performance parity is claimed. Kernel
tuning and subgroup/matrix acceleration can build on this provider after real
hardware measurements establish a baseline.
