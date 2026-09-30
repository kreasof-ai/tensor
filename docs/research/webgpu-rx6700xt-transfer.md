# Linux-to-Windows RX 6700 XT WebGPU acceptance

**Phase 5 is complete.** The WebGPU suite built on Linux from source
`f2b056b` passed all **33 inference/composition checks** on the Windows
RX 6700 XT through Vulkan. The strict evidence audit passed on Windows and
was independently rerun on the producer against the original bundle. It reports
`physical_second_gpu=true`, `two_hosts=true`, and
`phase5_hardware_gate="passed"`.

The consumer result is timestamped **2026-09-30 18:35:37 UTC**
(2026-10-01 in Asia/Jakarta). The producer hostname is `default`; the consumer
is `pc-gaming`. This closes the transfer requirement that the earlier
[same-host AMD validation](webgpu-rx6700xt.md) left open.

## What was transferred and checked

The single ZIP contained 23 WGSL `.tbin` artifacts, the packaged validation
module, the matching Tensor wheel, Windows x64 runtime dependency wheels,
consumer/audit scripts, and a byte-hash manifest. The consumer installed into
a separate Python 3.12 environment using the supplied wheels without an index.
It restored the packaged module and executed the transferred artifacts without
rebuilding them through TileLang or TVM. Native wgpu performs shader translation
and pipeline creation on the GPU machine.

The returned manifest exactly matches the original manifest. Both ZIP archives
passed integrity checks, and all 60 original bundled file hashes were verified.
The strict audit confirms matching suite, module, artifact and consumer-source
hashes, complete coverage, valid timings, compiler import isolation, and a
physical AMD adapter on a different host. Its independently recomputed output
exactly matches the returned audit.

| Property | Returned evidence |
|---|---|
| OS / Python | Windows 11 `10.0.26200-SP0` / 3.12.13 |
| Adapter | AMD Radeon RX 6700 XT, discrete GPU, Vulkan |
| Vendor / device | `0x1002` / `0x73df` |
| Driver description | `26.6.2 (AMD proprietary shader compiler)` |
| Tensor / NumPy | 0.1.0 / 2.5.3 |
| wgpu / wgpu-native | 0.29.0 / 27.0.2.0 |
| Requested feature | `shader-f16` |
| Buffer / workgroup storage limits | 128 MiB / 32 KiB |
| Installed distributions | Tensor, NumPy, wgpu, CFFI, pycparser, rendercanvas, pip |
| Compiler/framework imports | None; import guard enabled |
| Coverage / timing | 33 checks; 32 timed cases, 15 samples after two warmups |

Coverage includes dynamic FP32 affine, FP16 pointwise, FP16/FP32 GEMM and linear
with tails, transpose-B, bias/ReLU, dynamic rows, device-resident two-layer MLP,
and batched causal/noncausal forward attention with D=64/128 and S=65/129/512.
All cases passed the suite's existing numerical tolerances.

## Measured latency

| Case | Median completed call | Maximum absolute error |
|---|---:|---:|
| FP32 affine, 1M | 0.545 ms | 0.000000954 |
| FP32 linear, M=33/N=65/K=37 | 0.619 ms | 0.000002384 |
| FP16 GEMM, 256 square | 1.223 ms | 0.0625 |
| FP16 GEMM, 512 square | 2.465 ms | 0.0625 |
| Attention, D64/S512 noncausal | 2.652 ms | 0.000244141 |
| Attention, D64/S512 causal | 1.124 ms | 0.000976563 |
| Attention, D128/S512 noncausal | 1.346 ms | 0.000244141 |
| Attention, D128/S512 causal | 1.381 ms | 0.001953125 |

These timings include host submission and queue completion with preallocated
outputs; uploads, downloads and initial pipeline creation are excluded. They
validate the transfer suite's latency reporting. The separate allocating-call
[scaling comparison](latency-scaling.md) remains the performance baseline.

## Retained evidence and reproduction

- [Original Linux-produced suite](data/webgpu-rx6700xt-transfer-suite.json)
- [Returned AMD consumer result](data/webgpu-rx6700xt-transfer-result.json)
- [Strict audit result](data/webgpu-rx6700xt-transfer-audit.json)
- [Original bundle manifest](data/webgpu-rx6700xt-transfer-bundle-manifest.json)
- [Windows run log](data/webgpu-rx6700xt-transfer-validation.log)
- [Producer-side verification and archive/file hashes](data/webgpu-rx6700xt-transfer-verification.json)

From the repository root, rerun the strict evidence audit without a relaxation:

```bash
python tools/webgpu_audit.py \
  --suite docs/research/data/webgpu-rx6700xt-transfer-suite.json \
  --result docs/research/data/webgpu-rx6700xt-transfer-result.json
```

Git attributes preserve these evidence files byte-for-byte across platforms,
including the original suite's LF line endings, so its recorded byte hash
remains reproducible after checkout.

Acceptance covers the bounded inference profile. D3D12 FP16 support, arbitrary
TIRx coverage, training, accelerated matrix lowering, and performance tuning
retain the scope documented in [ADR 0015](../adr/0015-webgpu-provider.md).
