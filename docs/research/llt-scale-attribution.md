# nanoGPT-scale training attribution after projection optimization

The full-sized nanoGPT profile exercises 124,475,904 parameters, 12 layers,
width 768, 12 heads, context 1024 and vocabulary 50,304. The Tensor projection
optimization and optional serving parameter cache are qualified separately in
[the optimization study](llt-optimization.md) and
[the cache contract](llt-inference-weight-cache.md).

CUPTI captures one batch-1 training step after three warmups. The diagnostic
includes full-vocabulary loss, backward, gradient clipping and AdamW with FP32
master weights/moments. Numerical Tensor operations use explicit artifacts;
PyTorch controls autograd, views, tied-gradient accumulation and optimizer
orchestration. The Torch attribution control uses scalar AdamW; the model study
also measures the stronger fused AdamW control. Raw model latency/memory results belong to the
[LLT repository](https://github.com/kreasof-ai/loop-latent-transformer).

The unprofiled Tensor nanoGPT training median is slower than the standard Torch
control. The captured GPU kernels, however, sum to 21.61 ms for Tensor and
25.41 ms for Torch. Those sums come from Chrome trace events whose category is
`kernel`; GPU user annotations must not be added to those events because they
would double-count covered work. The measurements are from an instrumented
step, so they do not establish an exact idle-time decomposition of the separate
unprofiled training medians.

Tensor's largest exclusive operation ranges are AdamW (5.42 ms, 148 updates),
parameter/activation casts (3.01 ms, 366 calls), and projection GEMMs. Attention
backward dK/dV totals .84 ms and dQ .63 ms across 12 layers. The profile captures
1,497 GPU kernels for Tensor and 1,798 for Torch. Runtime/kernel submission,
autograd scheduling, copies and synchronization therefore deserve attribution
before another attention-kernel rewrite.

The subsequent LLT model study validates complete actual nanoGPT training graph
replay, including backward, clipping and optimizer. Batch-1 Tensor/Fused-Torch
graph medians are 20.698/20.882 ms, and batch-4 medians are 44.839/44.550 ms.
Every parameter device step counter reaches 105 and final loss is finite. This
demonstrates that fixed-shape capture mitigates the submission gap without a new
Tensor numerical implementation. Variable-shape execution and native argument
binding/dispatch remain optimization candidates. Detailed memory/latency samples
and capture accounting are retained in the
[LLT report](https://github.com/kreasof-ai/loop-latent-transformer/blob/main/benchmarks/L40S_NANOGPT.md).

[Retained attribution](data/llt-scale-attribution) includes operation and kernel
rows, source identities, artifact coverage and compact trace totals. The full
Chrome traces remain build artifacts. The pinned upstream nanoGPT and numerical
adapter are identified in the linked LLT experiment sources.

```sh
TENSOR_NVRTC_HOME=build/nvrtc-12.9 \
  .venv/bin/python -m benchmarks.llt.scale_attribution
```

Run this diagnostic after, rather than concurrently with, GPU performance
measurements. Profiler overhead is excluded from all model benchmark comparisons.
