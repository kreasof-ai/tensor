# Compilation

The compiler converts an exported TileLang PrimFunc into a verified `.tbin`.
Consumers load the artifact without importing the compiler. Set up the producer
with [Quickstart](quickstart.md).

## Native Hopper pipeline schedules

`tensor.compiler.cuda.hopper.staged_matmul` generates an explicit SM90a
TMA/WGMMA pipeline. Its knobs include shared-memory ring depth, the number of
outstanding WGMMA groups, weight orientation and K partitions. The producer
prefills the ring, waits for the oldest required MMA group, then refills only
the buffer whose reads have completed. A final drain precedes the epilogue.

```python
import tilelang
from tensor.compiler.cuda.hopper import staged_matmul

operation = staged_matmul(
    256, 4096, 8192,
    block_m=128, block_n=64, block_k=128,
    stages=2, queue_depth=1,
    transposed_weights=True, out_dtype="bfloat16",
)
kernel = tilelang.compile(
    operation,
    target={"kind": "cuda", "arch": "sm_90a"},
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": True,
        "tl.disable_wgmma": False,
    },
)
```

Inputs are BF16 `x[M,K]` and weights `[K,N]` when transposed, or `[N,K]`
otherwise. By default outputs are `[splits,M,N]`; multiple partitions require
FP32 output and an explicit reduction. `paired=True` accepts interleaved hi/lo BF16 weight rows,
retains separate FP32 accumulators and sums each adjacent pair in the epilogue.
The caller owns any required precision-preserving weight preparation.

These kernels use TileLang's native launch adapter because their signature
contains tensor-map descriptors. The ordinary `.tbin` pointer/scalar launch ABI
is unchanged. Disabling automatic warp specialization here is intentional: the
kernel already manages its asynchronous buffer ownership explicitly.

`hopper_space` and `hopper_legal` in `compiler.cuda.schedules` expose these
choices to `ScheduleSearch`. Legality checks reject premature buffer reuse,
unsupported tile shapes, misaligned partitions and excessive shared memory.
Compile, check numerical correctness, time graph replays and retain the winning
schedule on the target GPU. `native_async=False` provides a synchronous mirror
for numerical checks on older CUDA GPUs; it does not validate Hopper execution.

Set `cluster_reduce=True` with 2, 4 or 8 K partitions to combine their FP32
contributions in distributed shared memory. This emits one clustered launch and
an output of shape `[1,M,N]`, allowing either BF16 or FP32 final output without a
global partial buffer or a second reduction kernel. `hopper_cluster_space`
exposes this choice to search. Cluster reduction requires native Hopper execution;
its communication and scheduling costs must be measured against the two-launch
alternative.

Each cluster initializes its barriers before use, completes all MMA reads before
exchanging partials, then waits again before any block exits. The middle barrier
also protects operand buffers that the memory planner may reuse for the exchange.
Rank zero waits for all bulk shared-memory transfers and sums FP32 contributions.
The caller must use TileLang's cluster-aware native launch adapter.

`tensor.compiler.cuda.sliced.sliced_matmul` provides another single-launch
alternative for SM80 and newer. Warps compute different K slices using Tensor
Core MMA, store FP32 partials in block-local shared memory and reduce them before
the final write. Its knobs are output tile size, warp count, K chunk size and
pipeline stages. It accepts weights `[N,K]`, or interleaved hi/lo rows when paired,
and writes `[M,N]`. This trades extra padded MMA work for greater parallelism on
small output tiles. Numerical checks cover tails, sparse rows, outliers and fresh
graph inputs; target-device measurements determine whether the tradeoff wins.

`register_epilogue=True` replaces the shared-memory epilogue with direct
accumulator stores. It currently supports 64 physical weight rows per tile
(`block_n=32` for paired routers, 64 otherwise), whole 8-column token tiles and
the separate K-partition reduction. Paired outputs use an inline PTX XOR-4 warp
shuffle to combine adjacent hi/lo rows in FP32. Every lane executes the exchange
before output-tail guards. The WGMMA queue is drained and accumulator ownership
is fenced before any register is read.

`hopper_register_space` exposes shared and register epilogues as matched search
choices. The main TMA/WGMMA computation and K reduction stay identical within a
geometry pair. Direct stores remove shared staging and two epilogue barriers,
but their different store pattern can offset that saving. Compare producer-only
and complete graph replays on the target device. The synchronous numerical mirror
uses MMA with an explicit row warp policy to reproduce the WGMMA register map;
its timings do not predict native Hopper performance.

For paired register outputs, `coalesced_epilogue=True` exchanges row groups
within each warp before storing them. Eight adjacent FP32 output channels fill
one 32-byte sector. All lanes participate in the shuffles before the tail guards.
This halves the output-store instruction count for the 64×32 accumulator tile
while adding an indexed shuffle. Search it against the uncoalesced register path
with the same main computation; the instruction tradeoff needs target timings.

## Work-centric CUDA GEMM

`tensor.compiler.cuda.streamk.streamk_matmul` partitions the flattened output-tile
and K-tile iteration space into balanced contiguous worker ranges. It accepts
BF16 `x[M,K]`, weights `[N,K]`, FP32 scratch `[workers,2,block_m,block_n]` and a
final BF16 or FP32 output `[M,N]`. Accumulation and partial reductions use FP32.
The decomposition follows the work-centric principle in
[Stream-K](https://arxiv.org/abs/2301.03598).

Each worker writes complete owned tiles directly to output. A worker can have at
most two incomplete boundary tiles; these use its two scratch slots.
`streamk_reduce` computes the contributing worker range for each boundary tile
and sums those slots in a deterministic second launch. Every slot consumed by the
reducer is overwritten in the same invocation. Fresh graph inputs and weights
require no scratch clearing or persistent output state.

Use the same shape, worker count and tile parameters for both factories. With
`tiles = ceil(M/block_m) * ceil(N/block_n)`, a worker count dividing `tiles` assigns
only complete tiles, so the caller may omit the reduction launch. Worker count is
a search choice: an SM-sized grid can balance work, while a nearby aligned count
can eliminate boundary reductions. Uniform K partitions provide a control by
choosing `workers = tiles * splits` when K tiles divide evenly by `splits`.

This implementation uses ordinary pointer arguments and a separate deterministic
fixup kernel. Native Hopper compilation can use WGMMA with asynchronous shared
copies; older CUDA GPUs provide an MMA numerical comparison. Benchmark complete
operations against the incumbent, and use matched producer/full graph timings to
understand scheduling costs. These graph ratios include replay overhead.

## Export an existing kernel

A source file supplies a zero-argument `tensor_export()` function:

```python
import tilelang.language as T


def make_add(n, block=128):
    @T.prim_func
    def add(a: T.Tensor((n,), "float32"),
            b: T.Tensor((n,), "float32"),
            out: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for lane in T.Parallel(block):
                i = bx * block + lane
                if i < n:
                    out[i] = a[i] + b[i]
    return add


def tensor_export():
    return {"kernel": make_add(1025), "outputs": ["out"]}
```

The kernel must be an uncompiled PrimFunc. A TileLang JIT wrapper may return a
compiled callable; expose its original factory instead. `outputs` contains
buffer names, not parameter indices. Omit outputs when the caller will pass
every buffer explicitly through `launch`.

Build and check the included factory:

```sh
uv run --locked tensor build examples/tilelang_add.py --out build/add.tbin
uv run --locked python examples/run_tilelang_add.py build/add.tbin
```

A Python export executes source during an explicit build. Installation and
artifact inspection read bytes and metadata. Consumers execute the compiled
image. Use the pinned producer versions; a successful JIT run with another
TileLang version does not establish compatibility.

The Python API follows the same path:

```python
import tensor as tx

tx.build("examples/tilelang_add.py", "build/add.tbin", target="sm_89")
```

Choose the target for your consumer. CUDA accepts an exact SM; NVRTC is the
default compiler. `compiler="nvcc"` selects the offline toolchain. WebGPU uses
`provider="webgpu"`, `compiler="wgsl"`; see [Backends](backends.md).

## Arguments and lowering

Tensor extracts typed buffer/scalar arguments, symbolic dimensions and the
lowered launch configuration. Shared dimension values are inferred from buffers
or provided explicitly. The runtime validates shapes, dtypes, alignment and
launch limits; it does not silently reshape or cast input buffers.

Optional explicit launch metadata must match the lowered launch. CUDA artifacts
contain their device image, frontend IR, signatures, runtime requirements and
redistribution notices. WebGPU artifacts contain WGSL and adapter requirements.
Portable frontend rebuilding verifies recorded versions and operators before
restoring IR; it still requires the producer stack.

```sh
uv run --locked tensor inspect examples/tilelang_add.py --stage tirx --target sm_89
uv run --locked tensor inspect build/add.tbin
uv run --locked tensor cache
```

Use `tensor inspect --help` for target source and pass tracing. Use new output
paths. Cache hits reuse checked compiled images, not outputs or device buffers.

## CUDA operation prelowering

For a standalone projection, describe the operation and let the CUDA producer
build its Tensor Core implementation:

```python
from tensor.compiler.cuda.matmul import matmul

def tensor_export():
    kernel = matmul(
        256, 4096, 8192, dtype="bfloat16",
        schedule={
            "block_m": 256, "block_n": 32, "block_k": 64,
            "stages": 2, "threads": 128, "orientation": "weight",
        },
    )
    return {"kernel": kernel, "outputs": ["out"]}
```

The signature is `x[M,K], w[N,K] -> out[M,N]`, computing `x @ w.T`.
Inputs are FP16 or BF16; accumulation is FP32, with matching input/output dtype
or an explicit `out_dtype="float32"`. Dimensions are positive static integers.

`block_m` always counts tokens and `block_n` counts output features.
`orientation="activation"` gives the hardware GEMM its usual token-major
orientation. `orientation="weight"` computes `w @ x.T` inside each tile and
stores the transposed result. Wider token tiles can reuse weights across more
tokens; their register, storage and occupancy costs must be measured.

The producer verifies the complete logical computation before substituting an
implementation. An operation annotation on different computation is rejected.
Existing explicit kernels pass through unchanged. This stage does not retile
arbitrary fused kernels, insert extra kernel launches or change precision.
Floating-point reduction order can differ between schedules; applications must
validate their numerical tolerance when selecting a schedule.

The artifact retains the original logical IR and records the selected geometry
under `cuda_prelowering`. Its cache identity includes the transformed IR and
prelowering implementation. CUDA artifacts retain their pointer/scalar launch
contract; Hopper tensor-map and cooperative launch support remain separate
runtime capabilities.

For more parallelism without rereading weights for another token tile,
`matmul_partials(M, N, K, splits, ...)` explicitly produces FP32
`[splits, M, N]` partial sums. `matmul_reduce(M, N, splits, out_dtype)` combines
them into the final output. Export these as two kernels, allocate the partial
buffer, and submit both in one graph. Each K partition must align to `block_k`.
Both accumulation and reduction stay FP32; no atomics or persistent output
values are required. The additional launch and scratch traffic belong in
candidate timing.

`matmul_space` and `matmul_legal` in `compiler/cuda/schedules.py` expose orientation,
tile sizes, pipeline depth and thread count to the search API below. A default
schedule is a legal starting point, not a measured performance guarantee.
Use `matmul_space(..., partitioned=True)` to explore partition counts and supply
`depth=K` to `matmul_legal` when checking those candidates.

## Schedule discovery

Schedule search proposes settings. The caller owns compilation, correctness,
timing, budgets and result persistence. Search does not choose a configuration
merely because it compiled.

The runnable example explores block sizes, validates against NumPy, times only
accepted candidates, and saves a target-bound profile:

```sh
uv run --locked python examples/search_tilelang_add.py --out build/add-search --candidates 3 --seconds 60
```

For WebGPU, add `--provider webgpu`. The time budget is checked between trials;
it does not interrupt compilation. Keep allocations/transfers and correctness
checks outside the timed loop.

```python
from tensor.compiler.search import ScheduleSearch

spaces = {"add": {"block": (64, 128, 256)}}
search = ScheduleSearch([{"family": "add", "block": 128}], spaces=spaces, width=2)
config = search.next()
# Compile, execute and check config before recording a positive finite timing.
search.record(config, measured_seconds)
```

`next()` raises `StopIteration` when the finite space is exhausted. Optional
`legal` and `coupled` callbacks supply backend constraints and coordinated
moves. Rejected candidates remain seen; do not record invalid timings.
`ScheduleProfile` selects by operation, semantic parameters, provider and target;
conflicting equally specific matches are rejected.

Reusable factories should produce specialization files through
`tensor.compiler.frontend.source.export_source`. Declare every implementation
module in `dependencies`; changing a factory or schedule then invalidates its
source identity. Consumers receive artifacts, not schedule-search code.

## Implementation reading order

[Dispatch](../src/tensor/compiler/build.py) selects
[CUDA](../src/tensor/compiler/cuda/build.py),
[WebGPU](../src/tensor/compiler/webgpu/build.py), or
[CPU](../src/tensor/compiler/cpu.py).
[Frontend extraction](../src/tensor/compiler/frontend/signature.py),
[cache storage](../src/tensor/compiler/cache.py),
[search](../src/tensor/compiler/search.py), and
[tuning](../src/tensor/compiler/tuning.py) have separate owners.

WebGPU passes and named schedule/DSL families live under
`compiler/webgpu/passes` and `compiler/webgpu/schedules`. Model-specific kernel
selection and measured model profiles belong to AKAS.
