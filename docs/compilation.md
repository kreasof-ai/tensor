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
