# Compilation

The compiler converts an exported TileLang PrimFunc into a verified `.tbin`.
Consumers load the artifact without importing the compiler. Set up the producer
with [Quickstart](quickstart.md).

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

On Hopper, GEMM lowering normally chooses warp-group MMA when the tile permits
it. An operation can request synchronous warp MMA with
`kernel.with_attr("tensor.cuda.gemm_mode", "mma")`; `"auto"` retains the default.
This choice preserves the target and pointer/scalar ABI and is recorded in the
artifact's compiler pass settings and cache identity.

Static Hopper kernels can opt into TMA copies and warp specialization with
`kernel.with_attr("tensor.cuda.tma", 1)` when targeting `sm_90a`. The compiler
records tensor-map geometry and its source buffer in the artifact. Buffer
bindings retain their existing public names and types; the CUDA launch adapter
encodes aligned descriptors once per bound call, before graph capture. Artifacts
require the `cuda_tensor_maps` runtime capability. This initial implementation
accepts contiguous static geometry and one descriptor per source buffer.

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

FP32 inputs use tiled SIMT FMA with FP32 shared storage and accumulation.
FP16 and BF16 inputs use Tensor Cores. FP32 matmul does not cast operands to
TF32 or FP16. Pass `dtype="float32"` to `matmul_space` and `matmul_legal` when
exploring its schedule family; shared-memory estimates include its padded tile.
`reduction_lanes` selects 1/2/4/8/16/32 strided FMA chains before their reduction.
Keep it fixed when qualifying schedules that must preserve a reduction order.

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

## Bounded dynamic CUDA shapes

An export can attach a checked domain to its symbolic dimensions:

```python
import tilelang.language as T
from tensor.compiler.cuda.matmul import matmul

def tensor_export():
    return {
        "kernel": matmul(T.dynamic("rows"), 1536, 2048, dtype="float16"),
        "outputs": ["out"],
        "shape_domain": {
            "version": 1,
            "bounds": {"rows": {"min": 1, "max": 256}},
            "relations": [],
        },
    }
```

The existing CUDA matmul prelowering handles dynamic M with static N/K and
static hardware tiles. It keeps live row counts in signatures and launch grids.
The compiler inserts guarded TIR assumptions before TileLang lowering; the
executable validates the same domain before output allocation or submission.
Each symbol needs finite positive bounds. Optional `multiple_of` is a real
admission constraint, not a request to pad. Relations contain `left`, `op`
(`eq` or `le`) and `right`, using the signature's checked integer expressions.
They validate explicit/inferred bindings without solving ambiguous products.

Bounded CUDA exports use artifact format 4 so older consumers reject mandatory
guards. Static and unrestricted symbolic exports retain their existing formats.
Portable rebuilding preserves bounds. Module variants must agree on the domain
as well as their logical signature. CPU/WebGPU bounded producers are not yet
implemented; their existing execution paths are unchanged.

`tensor.compiler.tuning.tune_shapes` accepts an existing `ScheduleSearch`, shape
points, a resource-owning build context and an independent qualification/timing
callback. It builds each candidate once, visits shapes sequentially and rejects
the candidate if any shape fails. Results retain per-shape measurements and
winners. `select_regions(report, symbol="rows", bounds=(1, 256), max_regions=4)`
selects contiguous dispatch regions from that shared bank, using dynamic
programming and a per-measured-point slowdown limit. It does not build new
images. Region boundaries lie between measured points; unmeasured integers
have dispatch coverage but carry no measured performance claim.

For a streaming workload, `cuda_working_set(device, weight_bytes, out=path)`
from `tensor.compiler.tuning` supplies an independent canonical matrix product
and a serializable description. Use its launch callback in measurement's
`reset` callback to touch that working set outside timing. Select a footprint
appropriate to the device cache and retain the description with the result;
this controls workload residency without promising hardware cache invalidation.

Version-2 module exports package these regions under `portfolio`, with
`symbols`, `shape_domain`, ordered `regions` and optional `evidence` file.
Each region has `bounds` (inclusive min/max for every symbol) and the existing
`artifacts`, `portable` or `source` fields. Several regions may share an image.
Installation checks complete coverage over the bounded integer box, accounting
for divisibility, and verifies that every packaged image guards assumptions
compatible with its region. Relations are checked at selection and binding;
coverage conservatively includes cells excluded by relations. Overlaps select
the first declared region. Validation is bounded to 64 regions and 4096
uncovered cells and rejects contracts exceeding that budget.

`module.resolve("matmul", target="sm_89", dimensions={"rows": 7})` selects an
AOT image without importing a compiler. Use `module.load` with the same
`dimensions` to load it. Engines own loaded-image reuse and call rebinding;
module resolution performs no JIT or implicit capture. The artifact retains
its own mandatory guards. The evidence file and region map participate in the
package identity. Version-1 exports retain their existing resolution behavior.

Run the generic projection study with:

```sh
python -m benchmarks.kernels.comparison.dynamic_shapes \
  --out build/dynamic-shapes --columns 1536 --depth 2048
```

It retains exact/dynamic artifacts, independent NumPy checks, timing samples
and producer costs. Exact controls use the selected dynamic schedule at each
point; this does not establish an advantage over separately tuned static code.

## Schedule discovery

### Explore schedules

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
