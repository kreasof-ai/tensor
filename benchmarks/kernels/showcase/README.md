# CUDA kernel showcase

This producer-side benchmark measures real GPU execution of Flash Attention,
dense absorbed MLA decode, FP16 matmul and fused GGUF dequantization/matmul.
It reuses the GGML kernels from the sibling `akas` checkout. It does not require
changes to the compiler's normal installation dependencies.

The measured L4 run and interpretation live in
[`research/studies/cuda-showcase`](../../../research/studies/cuda-showcase/README.md).

## Comparisons

Every case includes Tensor's default schedule, Tensor's searched schedule,
a fixed native TileLang schedule, native TileLang with Tensor's selected
schedule, searched Triton, and a relevant PyTorch baseline. Optional isolated
processes measure upstream FlashAttention-2 and unmodified GGML CUDA.

Matmul goes through Tensor's checked semantic CUDA prelowering. The other
workloads use explicit TileLang kernels and producer schedule search. There is
currently no semantic attention, MLA or GGUF prelowering pass. The native
TileLang matmul control emits the explicit equivalent of the selected semantic
operation, since bare TileLang cannot schedule the logical operation itself.

Native TileLang uses NVCC and TVM-FFI. Tensor uses its regular NVRTC artifact
builder and runtime. The packed AKAS GEMV calls an external scalar U16-load
helper; the native TileLang compiler callback receives the same helper definition
before calling its ordinary NVCC compiler. No device binary is substituted.

Triton has independent implementations and its own schedule search, including
a reduction-based GEMV for single-row GGUF decode. The search budgets and every
candidate are recorded. These are measured implementations, not a claim to have
exhausted every possible implementation in any language.

## Run

Use Python 3.12 with `tilelang==0.1.14`, `apache-tvm-ffi==0.1.12`, PyTorch,
Triton, NumPy, and a complete CUDA toolkit. The recorded producer environment
uses the existing `../akas/.venv`; `PYTHONPATH=src` selects this Tensor checkout.
Provide a regular Tensor NVRTC bundle as described in the repository quickstart.

```sh
export PYTHONPATH="$PWD/src"
export TENSOR_NVRTC_HOME="$PWD/../akas/build/nvrtc-12.9"
../akas/.venv/bin/python benchmarks/kernels/showcase/benchmark.py \
  --out build/showcase --akas ../akas \
  --candidates 16 --iterations 200 --repetitions 11
```

`--quick` runs one shape per workload plus both GGUF phases. `--workload`
selects a workload; `--case` selects one exact ID. Output directories must be
new. All artifacts, exported sources and incremental results are retained.

The full matrix has four matmul cases, four attention cases (including a causal
tail), three MLA cases and six GGUF cases: Q4_0, Q4_K and Q6_K, each with M=1
and M=32. GGUF N=K=1024. No weights are dequantized globally in the fused kernels.

To measure upstream GGML, build the benchmark-only bridge against an existing
CUDA build of llama.cpp. Match `USE_CUDA_GRAPH` to the library's configuration;
the bridge reads its private context solely to obtain its stream for events.

```sh
llama_root="$PWD/../akas/build/baselines/llama.cpp"
nvcc -arch=sm_89 -std=c++17 -shared -Xcompiler -fPIC -DUSE_CUDA_GRAPH \
  -I"$llama_root/ggml/include" -I"$llama_root/ggml/src" \
  benchmarks/kernels/showcase/ggml_bridge.cu \
  -L"$llama_root/build-cuda/bin" -lggml-cuda -lggml-base \
  -o build/showcase-ggml.so
```

Add `--ggml-library build/showcase-ggml.so` to both commands below. Ensure
`LD_LIBRARY_PATH` includes the GGML and matching CUDA libraries. The bridge
captures the entire `ggml_mul_mat` operation, including activation quantization
and any auxiliary kernels. Internal GGML graph capture is disabled to allow
the benchmark's repeated-operation graph. The upstream kernels are unchanged.

For upstream FA2, install a compatible official wheel in an isolated environment
and add `--fa-python path/to/environment/bin/python`. The measured wheel is
`flash-attn==2.8.3.post2`, built for Torch 2.8 / CUDA 12 / Python 3.12, whereas
the main producer uses Torch 2.14.1 / CUDA 13. Those versions are recorded
separately. Inputs are identical, transferred outside timing. FA2 reads the
BHSD inputs through stride-only BSHD views and returns native BSHD output
storage viewed as BHSD; no output-layout copy is charged.

After search completes, remeasure the selected schedules in a separate process:

```sh
../akas/.venv/bin/python benchmarks/kernels/showcase/finalize.py \
  --run build/showcase --akas ../akas --iterations 200 --samples 21 \
  --fa-python path/to/fa/environment/bin/python \
  --ggml-library build/showcase-ggml.so
```

This captures each selected provider once, warms all providers, then interleaves
their replay samples in randomized order. FA2 and GGML retain isolated-process
measurements. Do not run other GPU jobs concurrently. `finalize.py` preserves
initial measurements and writes `showcase.png` and `showcase.svg` using Matplotlib.

## Qualification and scope

Each candidate must pass a reference check before its timing enters search.
The top three Tensor schedules receive independent selection measurements.
Final samples never select a schedule. Selected kernels and native controls
replay the same graph on random, changed-sign/scaled and zero-query inputs,
checking fresh scratch and output writes. Tolerances and errors are saved.
GGUF prefill's oracle independently rounds the decoded weight and activation
to FP16, then accumulates in FP32, matching the MMA operand contract. Errors
against original FP32 operands are retained separately. GEMV uses FP32 operands.
GGML's default Q8_1 activation approximation receives a separate relative-L2
check; it does not inherit the FP16 candidates' tolerance.

Times are warm CUDA events around repeated operations in CUDA graphs. Allocation,
compilation, autotuning, host submission, transfers and validation are excluded.
The small resident buffers fit the L4's L2 cache; these are hot-cache kernel
measurements, not streaming HBM or end-to-end model measurements. No backward,
dropout, quantized KV, model projections, graph construction or serving is timed.

MLA is one-query dense MQA with QK=576, V=512 and the values stored in the first
512 channels of the shared latent cache. Both split computation and merge are
timed. The cache is contiguous; there is no page-table lookup or ragged batching.
It is not a complete reproduction of the upstream paged FlashMLA interface.
The [V3 dense upstream kernel requires SM90](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7),
so it cannot be measured on the SM89 L4.

PyTorch uses forced Flash SDPA for attention, `torch.mm` for cuBLAS matmul,
and materializing FP16 MQA for MLA. For GGUF it uses a resident, already
dequantized weight with activation/output conversions. That GGUF comparison
excludes weight decoding and is a lower-bound diagnostic, not a fused baseline.

The fixed TileLang comparison quantifies the benefit of searching a schedule.
The same-schedule control limits the claim attributable to Tensor's compiler
or runtime. Keep both controls, all losing shapes, failed candidates and raw
samples when sharing results.

Run `python benchmarks/kernels/showcase/audit.py build/showcase` to verify
artifact members, timing statistics and identical-cubin controls, and export
the selected `ScheduleProfile` and candidate records.
