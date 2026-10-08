# Prepared inference weight casts

Full-vocabulary inference keeps FP32 master weights and FP32 embeddings. Torch
autocast reuses BF16 parameter copies within the serving context. The explicit
Tensor path previously converted every projection weight at every invocation.

`Operators(cache_inference_weights=True)` now caches parameter casts only under
`torch.inference_mode()`. It is opt-in. Ordinary training, including the
no-grad body of an `autograd.Function.forward`, continues to cast directly and
propagates gradients to the FP32 master parameters. Only `nn.Parameter` inputs
are retained, so changing activations are never cached.

Entries use parameter identity, storage address and the Torch version counter.
Versioned in-place updates refresh the prepared copy on the next eager call.
Weak references discard entries when parameters die, preventing model storage
retention or pointer-reuse collisions. Use versioned updates rather than `.data`
writes. Create master parameters outside inference mode so they have counters.

Prepared BF16 copies increase persistent inference memory, which is counted in
the LLT profile. `clear_inference_weight_cache()` releases them. Dispose CUDA
graphs referencing those copies before clearing, changing masters or destroying
the model; a graph replay cannot execute Python version checks. Reprepare and
recapture after updates. This is a static-weights serving cache.

The CUDA regression checks repeated reuse, refresh after an update, training
gradients with caching enabled, and weak-reference lifetime cleanup:

```sh
TENSOR_NVRTC_HOME=build/nvrtc-12.9 TENSOR_LLT_CUDA=1 \
  TENSOR_LLT_CACHE_DIR=build/llt-optimization/artifacts \
  .venv/bin/pytest packages/tensor-torch/tests/test_llt.py \
  --override-ini addopts='' -q -k inference_weight_cache
```

The full-size LLT/naive and actual nanoGPT output and all-parameter gradient
checks completed against the optimized GEMM implementation before this serving
cache was enabled. The subsequent inference profiles explicitly enable the
Tensor cache to match warmed Torch autocast weight preparation. Those model
results live in the
[LLT repository](https://github.com/kreasof-ai/loop-latent-transformer), rather
than this Tensor development record.
