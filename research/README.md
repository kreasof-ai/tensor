# Tensor research

Historical experiments, measurements, decisions, and planning baselines live here.
Reports describe their measured revisions; evidence retains its original bytes.
Use [current documentation](../docs/README.md) for installation and APIs.

## Tensor studies

### Clblast rx6700xt

- [CLBlast versus generated WebGPU GEMM on RX 6700 XT](studies/clblast-rx6700xt/reports/clblast-rx6700xt.md)

### Cuda artifact transfer

- [Phase 1 CLI baseline — A10G](studies/cuda-artifact-transfer/reports/phase1-cli-baseline.md)
- [Phase 1 exit — CUDA CLI, runtime dimensions and GPU interop](studies/cuda-artifact-transfer/reports/phase1-exit.md)

### Direct backend comparison

- [Direct Tensor, TileLang and Triton comparison](studies/direct-backend-comparison/reports/direct-backend-comparison.md)

### Ecosystem

- [Ecosystem Survey and Packaging Precedents](studies/ecosystem/reports/ecosystem-and-precedents.md)

### Flash attention

- [FlashAttention through Tensor's NVRTC compiler and artifact runtime](studies/flash-attention/reports/flash-attention-demo.md)

### Latency scaling

- [Larger-shape latency scaling](studies/latency-scaling/reports/latency-scaling.md)

### Module transport

- [Phase 3 exit — offline tensor modules](studies/module-transport/reports/phase3-exit.md)
- [Phase 3 PyPI transport acceptance](studies/module-transport/reports/phase3-pypi.md)

### Nanogpt training

- [Phase 6: standalone nanoGPT training](studies/nanogpt-training/reports/phase6-nanogpt.md)

### Phase0

- [E13 — Opaque CUDA artifact prototype](studies/phase0/reports/e13-opaque-artifact-prototype.md)
- [E14 — NVIDIA execution and compiler-free startup](studies/phase0/reports/e14-cuda-execution.md)
- [E15 — Remaining Phase 0 validation](studies/phase0/reports/e15-phase0-validation.md)
- [E16 — Phase 0 exit validation](studies/phase0/reports/e16-phase0-exit.md)
- [E4 — Artifact Shape and Re-targetability](studies/phase0/reports/e4-artifact-shape.md)
- [E4a — Artifact Versioning and the Compatibility Surface](studies/phase0/reports/e4a-artifact-versioning.md)
- [E4b — Symbolic Shapes](studies/phase0/reports/e4b-symbolic-shapes.md)
- [1. The headline: most of Phase 0 does not need the GPU](studies/phase0/reports/phase0-ground-truth.md)

### Pytorch adapter

- [Phase 4 — Inference-first PyTorch adapter acceptance](studies/pytorch-adapter/reports/phase4-exit.md)

### Pytorch executor

- [C++ PyTorch executor](studies/pytorch-executor/reports/native-executor.md)

### Repository layout

- [Executed migration and audit](../docs/migration.md).

- [Canonical import cleanup](studies/repository-layout/reports/canonical-layout.md)
- [Repository reorganization acceptance](studies/repository-layout/reports/reorganization.md)

### Runtime abi

- [Phase 2 exit — stable runtime ABI and NVRTC](studies/runtime-abi/reports/phase2-exit.md)
- [Phase 2 — runtime call ABI and bundled NVRTC](studies/runtime-abi/reports/phase2-validation.md)

### Rx6700xt history

- [RX 6700 XT Vulkan: a history of Tensor kernel and inference improvements](studies/rx6700xt-history/reports/rx6700xt-vulkan-history.md)

### Webgpu gemm accumulation

- [WebGPU whole-loop GEMM accumulators on RX 6700 XT](studies/webgpu-gemm-accumulation/reports/webgpu-gemm-accumulation.md)

### Webgpu outer product

- [Chasing CLBlast with staged outer products](studies/webgpu-outer-product/reports/webgpu-outer-product-gemm.md)

### Webgpu validation

- [Phase 5 WebGPU implementation and hardware validation](studies/webgpu-validation/reports/phase5-webgpu.md)
- [Linux-to-Windows RX 6700 XT WebGPU acceptance](studies/webgpu-validation/reports/webgpu-rx6700xt-transfer.md)
- [Windows RX 6700 XT WebGPU validation](studies/webgpu-validation/reports/webgpu-rx6700xt.md)

## Original experiments and plans

- [Phase 0 implementation](phase0/) and [tests](phase0/tests/).
- [Architecture decisions](decisions/README.md).
- [Planning baselines](plans/README.md).
- [Original proposal](design/proposal.md).

## Independent projects

LFM2/Qwen inference reports and evidence are maintained in [AKAS](https://github.com/kreasof-ai/akas/tree/main/docs/research).
LLT qualifications and evidence are maintained in [Loop Latent Transformer](https://github.com/kreasof-ai/loop-latent-transformer/tree/main/docs/tensor).
