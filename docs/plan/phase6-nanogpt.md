# Phase 6: standalone nanoGPT training

The acceptance workload is ten complete optimizer updates of a pinned nanoGPT
architecture on one A10G: 12 layers, 12 heads, width 768, vocabulary 50,304,
batch 2, sequence 512, tied token/output weights, causal attention, dropout zero
and no linear/LayerNorm bias. A small specialization exercises the same path
for gradient diagnostics. FP16 compute uses FP32 LayerNorm/softmax reductions,
master weights, parameter gradients and AdamW state. Loss scaling is fixed and
identical across implementations; overflow is an explicit validation failure.

Tensor executes forward, manually authored backward and optimizer kernels
without Torch, Triton, TileLang or TVM on the consumer. PyTorch supplies a
separate numerical reference and eager/compiled training baselines. Standalone
autograd is deferred. The public manual interface defines saved-buffer
lifetime, gradient metadata and context consumption; callbacks own accumulation
and recomputation.

After the [repository reorganization](../development.md), templates ship in the
optional `tensor-nn` distribution. Consumers install matching Tensor/Tensor NN
wheels and NumPy; fresh v2 bundles bind both implementations. Original Phase 6
measurements retain their original two-distribution wheels and v1 bundle hashes.

Implementation covers embedding/scatter gradients, linear gradients,
LayerNorm, GELU, residuals, causal attention, cross-entropy, global gradient
clipping and AdamW. The static training plan reuses buffers and performs bounded
fusion (GEMM epilogues, normalization, softmax derivatives and optimizer update).
Arbitrary user-module fusion remains a separate extension. Kernel autotuning
uses correctness-checked candidates through Tensor's NVRTC artifacts/runtime,
and retains measurements and selected configurations for the measured GPU.

Acceptance compares every parameter gradient independently with Torch autograd,
then compares complete optimizer updates and state on identical checked gradients
to isolate optimizer arithmetic. An additional independent eager ten-update loss
trajectory must agree at recorded tolerances. Coverage includes tied-weight
accumulation; ten finite losses and updates; a clean compiler-free consumer;
and repeated ten-step timing windows with weights/state restored after warmup.
Report milliseconds/update, tokens/second, owned device memory, compile/tune
time and first-ten-update time separately. Compare identical model, data,
precision, loss scaling and optimizer settings. No convergence or published
speedrun result follows from ten updates. Performance is measured before an
explicit speed gate is adopted; fallback must not hide missing Tensor work.

Implementation and local GPU acceptance are complete; see the
[Phase 6 report](../research/phase6-nanogpt.md) for raw numerical, consumer,
benchmark and regression evidence. Tensor measures 36.58 ms/update versus
38.34 ms/update for compiled Torch with native SDPA and fused AdamW on A10G.
WebGPU training and Flash-style
attention backward optimization are later portability/performance extensions;
Phase 6's initial training workload targets CUDA, while preserving Phase 5's
inference provider scope.
