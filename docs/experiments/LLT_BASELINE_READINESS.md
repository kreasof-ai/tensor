# Recurrent architecture kernel support for LLT profiling

This is Tensor's development record for the additional recurrent/cache-sharing
workloads in [Loop Latent Transformer](https://github.com/kreasof-ai/loop-latent-transformer).
The LLT repository owns architectural adaptations and measurements; Tensor owns
these numerical operations and their qualification.

## Added native entry points

`tensor_torch.recurrent.RecurrentOperators` extends `Operators` without changing
its existing attention, decode, projection, or optimizer entry points:

- `window_attention`: causal sliding windows, optionally preceded by a globally
  causal shared bank. Shared/local entries use one softmax. Window size includes
  the current token. Fully masked tiles are skipped before GEMMs in forward and
  backward; no dense score matrix or score mask is materialized.
- `shared_decode`: one supplied query reads distinct persistent global/local KV
  banks, with device-resident valid lengths and one normalization. Global history
  is not copied into every local bank. Partitioned FP32 merging supports CUDA graphs.
- `silu`, `sigmoid`, `multiply`, `blend`: native pointwise forward/backward for
  SwiGLU and recurrent gates, including mixed FP32/BF16 storage where needed.

Model layout, cache orchestration, RNG generation, and checkpoint scheduling stay
with Torch. These operations dispatch directly through Tensor's native executor;
there is no implicit numerical fallback.

## Qualification

On NVIDIA L40S, the new CUDA tests pass **18 cases** covering BF16 window/union
attention outputs and Q/K/V gradients, non-aligned tails, query offsets, window=1,
shared-head reduction, exact checkpoint recomputation, FP32/BF16 pointwise gradients,
and two-bank cached decode with CUDA-graph replay and partially filled banks.
The original causal attention entry point is checked too.

```bash
TENSOR_LLT_CUDA=1 TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
  .venv/bin/python -m pytest packages/tensor-torch/tests/test_recurrent.py -q
```

The original LLT numerical implementation files and native runtime binaries are
unchanged. New specializations use separate template modules and artifact identities.
Architecture-level and full-shape comparisons belong to the LLT experiment records.
