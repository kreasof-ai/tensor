# Windows RX 6700 XT WebGPU validation

**Acceptance update:** the subsequent
[Linux-to-Windows RX 6700 XT transfer](webgpu-rx6700xt-transfer.md) passes all 33
checks and the strict two-host audit, completing Phase 5. The same-host run
documented below is the earlier hardware-validation baseline.

On **2026-10-01 (Asia/Jakarta)**, the native wgpu provider passed all **33
inference/composition checks** on this machine's physical **AMD Radeon RX 6700
XT through Vulkan**. The WebGPU contract and evidence-audit suite also passed
**28 tests, zero failures, zero skips**, with native GPU checks enabled, in
15.17 seconds. No provider or lowering code was changed for this validation.

The same GPU's **D3D12 backend passed six FP32 affine checks**, then rejected
the first FP16 artifact because `shader-f16` was absent. D3D12 did not complete
the suite; GEMM, attention and MLP are not validated on that backend by this run.
Vulkan is the verified backend for the full inference profile on this setup.

A subsequent [same-size scaling sweep](latency-scaling.md#windows-rx-6700-xt-at-the-same-workload-sizes)
also passes all 17 A10G workload sizes and two repeats, including 64M pointwise,
GEMM 4096³ and attention S=8192 with eight heads. It verifies identical WGSL
and uses the allocating-call protocol, with a separate cross-system latency table.
The [utilization derivation](latency-scaling.md#rx-6700-xt-compute-and-bandwidth-utilization)
estimates 5.03% of FP32 peak for GEMM 4096³ and 5.94% effective useful-data
bandwidth utilization for 64M pointwise; actual DRAM counters were not measured.

## Environment and evidence

The measured source was `ed1f8ea8f72dba06e85aa96a84e3962353250acf` (the
[A10G comparison](latency-scaling.md) revision), with a clean checkout before
recording these results. The producer built the current suite and Tensor wheel
locally. A separate Python 3.12 environment installed that wheel and its WebGPU
extra, without compiler/framework packages.

| Property | Observation |
|---|---|
| OS | Windows 11, `Windows-11-10.0.26200-SP0` |
| CPU | AMD Ryzen 5 5600, 6 cores |
| GPU | AMD Radeon RX 6700 XT, `DiscreteGPU`, vendor `0x1002`, device `0x73df` |
| Windows driver | `32.0.21043.19003` |
| Vulkan driver description | `26.6.2 (AMD proprietary shader compiler)` |
| Python / Tensor | 3.12.13 / tensor-workspace 0.1.0 |
| NumPy / wgpu / wgpu-native | 2.5.3 / 0.29.0 / 27.0.2.0 |
| Consumer distributions | Tensor, NumPy, wgpu, CFFI, pycparser, rendercanvas |
| Vulkan device features requested | `shader-f16` |
| Device allocation / storage binding limits | 128 MiB each |
| Workgroup storage limit | 32 KiB |

The consumer restored the packaged module into a fresh project, checked
producer/consumer source and artifact hashes, checked exact upload/download
round trips, compared results with NumPy, and exercised device-resident MLP
composition and event handling. The import guard blocked TileLang, TVM,
TVM-FFI, Torch and Triton. The result records no compiler/framework imports and
no such installed packages. Native tests additionally exercised symbolic
dispatch, opaque/session handle validation, FP16 tails and feature rejection,
large-buffer limit negotiation, and packed x/z dispatch masking.

Retained evidence:

- [Full successful Vulkan result](data/webgpu-rx6700xt-vulkan.json): all 33 checks,
  numerical errors/tolerances, completed-call and enqueue medians, pipeline
  creation times, adapter identity, package versions and hashes.
- [Producer suite](data/webgpu-rx6700xt-suite.json): 32 kernel cases; the consumer
  adds the MLP composition check.
- [Verification metadata](data/webgpu-rx6700xt-verification.json): both adapter
  capability probes, captured D3D12 failure, audit results, Windows workaround,
  source revision and evidence/bundle hashes.
- [JUnit test result](data/webgpu-rx6700xt-tests.xml): all 28 contract/audit tests.

## Correctness and latency

Coverage includes dynamic FP32 affine lengths 1/127/128/129/4097/1M; FP16
elementwise tails; FP16/FP32 GEMM/linear, bias/ReLU, transpose-B and M/N/K tails;
dynamic GEMM rows 1/31/32/33/65; square GEMM 256/512; two-layer MLP composition;
and causal/noncausal FP16 attention with D=64/128 and S=65/129/512, including
batch/head dimensions. Every check passed its existing tolerance.

| Vulkan case | Median completed call | Maximum absolute error |
|---|---:|---:|
| FP32 affine, 1M | 0.544 ms | 0.000000954 |
| FP32 linear, M=33/N=65/K=37 | 0.644 ms | 0.000002384 |
| FP16 GEMM, 256 square | 1.267 ms | 0.0625 |
| FP16 GEMM, 512 square | 2.453 ms | 0.0625 |
| Attention, D64/S512 noncausal | 2.666 ms | 0.000244141 |
| Attention, D64/S512 causal | 1.220 ms | 0.000976563 |
| Attention, D128/S512 noncausal | 1.337 ms | 0.000244141 |
| Attention, D128/S512 causal | 1.363 ms | 0.001953125 |

Each timed case used **15 samples after two warmups**, with preallocated
outputs. Medians include Python host submission and queue completion; uploads,
downloads and first pipeline creation are excluded. Initial pipeline creation
for the 23 distinct exports measured approximately 72–185 ms, separately from
warm calls. The MLP chain is a correctness check without its own timing.

FP32 tolerances were `atol=3e-5, rtol=2e-4`; FP16 kernel tolerances were
`atol=0.015, rtol=0.01`; the MLP chain used `atol=0.03, rtol=0.01`. GEMM's
maximum absolute error must be assessed together with relative tolerance and
output magnitude, rather than against absolute tolerance alone.

These are sequential samples on a desktop GPU, with visible host/driver timing
variation. They are a functional baseline, not a throughput or vendor parity
claim. The [A10G scaling benchmark](latency-scaling.md) uses different shapes,
input values and an allocating-call protocol, so its latencies are not a
matched cross-GPU comparison with this table. Larger scaling workloads were
measured separately in the [scaling follow-up](latency-scaling.md#windows-rx-6700-xt-at-the-same-workload-sizes).

## Reproduce on Windows

Run from the repository root in PowerShell. Use fresh build directories, since
`--build` intentionally refuses to replace an existing suite.

```powershell
uv sync --locked --extra webgpu
.venv/Scripts/python.exe tools/webgpu_validation.py --build build/webgpu-rx6700xt
uv build --wheel --out-dir build/webgpu-rx6700xt/wheel
uv venv --python 3.12 build/webgpu-rx6700xt-consumer
uv pip install --python build/webgpu-rx6700xt-consumer/Scripts/python.exe 'build/webgpu-rx6700xt/wheel/tensor_workspace-0.1.0-py3-none-any.whl[webgpu]'

# Keep the fresh consumer project on the suite's drive.
New-Item -ItemType Directory -Force build/webgpu-rx6700xt-temp | Out-Null
$env:TEMP = (Resolve-Path build/webgpu-rx6700xt-temp).Path
$env:TMP = $env:TEMP

# Confirm adapter identity/backend before choosing its ordinal.
build/webgpu-rx6700xt-consumer/Scripts/python.exe -m tensor doctor --provider webgpu --device 0 --json
build/webgpu-rx6700xt-consumer/Scripts/python.exe tools/webgpu_validation.py --consume build/webgpu-rx6700xt --device 0 --require-second-gpu --iters 15 --out build/webgpu-rx6700xt-vulkan.json
build/webgpu-rx6700xt-consumer/Scripts/python.exe tools/webgpu_audit.py --suite build/webgpu-rx6700xt/suite.json --result build/webgpu-rx6700xt-vulkan.json --allow-software

$env:TENSOR_WEBGPU = '1'
.venv/Scripts/python.exe -m pytest tests/test_webgpu.py tests/test_webgpu_audit.py -o addopts='' -q --junitxml=build/webgpu-rx6700xt-tests.xml
```

In the measured enumeration, ordinal 0 was Vulkan, ordinal 1 was D3D12, and
Microsoft Basic Render Driver was a software adapter. Ordinals should be
rechecked on another setup. For the recorded D3D12 attempt, the consumer used
`--device 1` and the same suite, isolation and timing arguments. It returned
exit code 1 at `elementwise_f16_tail` with:

```text
tensor.artifact.ArtifactError: WebGPU adapter lacks required features: ['shader-f16']
```

This is the explicit feature gate; the artifact was not silently converted to
FP32. The remaining cases did not execute and no successful D3D12 result JSON
was written. This observation applies to the pinned wgpu/native library and
driver combination, not every D3D12 implementation.

The initial Vulkan attempt failed before kernel execution because the suite
was on D: and the default Windows temporary project was on C:.
`tensor.modules.add` called `os.path.relpath(source, project)` across drives
and raised `ValueError: path is on mount 'D:', start on mount 'C:'`. The
`TEMP`/`TMP` setting above worked around it; cross-drive module handling itself
was not changed.

## Acceptance boundary

**Physical AMD execution is verified. This run alone left the formal Phase 5
two-host transfer gate open.** The producer and consumer in this run both used hostname
`pc-gaming`. `--require-second-gpu` verifies a physical AMD/Apple adapter, while
the evidence auditor additionally requires distinct producer/consumer hosts.

Auditing these results with `--allow-software` passed all 33 coverage,
hash/isolation and timing checks and correctly reported
`physical_second_gpu=true`, `two_hosts=false`, `phase5_hardware_gate="open"`.
The option relaxes the formal gate; it does not classify this Vulkan GPU as
software (`software_adapter=false`). A strict audit rejected the same-host
evidence with `Phase 5 needs a transferred suite on a physical AMD or Apple GPU`.
The subsequent [Linux-to-Windows AMD transfer](webgpu-rx6700xt-transfer.md)
consumes a matching suite and wheel produced on another host and passes the
strict audit, closing that gate. The earlier Windows-to-Linux software transfer
and physical A10G results remain separate evidence.
