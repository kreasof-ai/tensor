# Discover and reuse kernel schedules

[Documentation](../README.md) · [From TileLang to Tensor](from-tilelang.md) · [Python runtime](runtime.md)

Tensor's producer-side discovery layer explores explicit schedule choices for
your TileLang factories. You define the choices, compile each proposed candidate,
reject incorrect results, and feed measured latency back into the search. A
schedule profile lets later builds reuse the settings you measured.

Start with a kernel that already builds and passes a correctness check through
Tensor. The [TileLang migration guide](from-tilelang.md) covers that first step.
This guide adds Tensor's discovery workflow around the factory. It does not
require replacing any TileLang autotuning harness you already use.

## What the layer provides

| API | Responsibility |
|---|---|
| `tensor.compiler.search.ScheduleSearch` | Propose configurations using seeds, beam exploration, family retention, and deterministic restarts |
| `tensor.compiler.search.ScheduleProfile` | Validate and select producer settings by operation, semantic parameters, provider, and target |
| `tensor.compiler.tuning.measure_cuda` | Measure an allocation-free CUDA callback with CUDA events and completed-call timing |
| `tensor.compiler.tuning.tune` | Validate and rank an already built, fixed set of CUDA candidates |
| `tensor.compiler.cuda_schedules` / `webgpu_schedules` | Supply workload-specific spaces, legality checks, or coupled moves |

The generic search opens no device and performs no compilation, validation,
measurement, or automatic IR rewriting. Your factory must apply configurations
to its tiling, thread count, pipeline stages, algorithm variant, or other explicit
knobs. There is currently no generic `tensor search` command or `tensor build --tune`
switch. Use a Python harness, such as the runnable example below.

## 1. Run a small complete search

[examples/search_tilelang_add.py](../../examples/search_tilelang_add.py) searches
the `block` argument of the existing
[addition factory](../../examples/tilelang_add.py): 64, 128, or 256 threads per
block. It uses 65,537 elements to include a partial final block.

Prepare the pinned producer and NVRTC bundle using the
[quickstart](quickstart.md), then run on your NVIDIA GPU:

```sh
uv run --locked python examples/search_tilelang_add.py --out build/add-search --candidates 3 --seconds 60
```

Use a fresh output directory each time. The example starts with block 128,
builds each candidate for the selected device, checks against NumPy, and scores
only correct candidates. Compilation, allocation, transfers, and correctness
checks are outside the timed region. CUDA scores use median CUDA-event latency.

The same factory can be searched through WebGPU without NVRTC:

```sh
uv sync --locked --extra webgpu
uv run --locked --extra webgpu python examples/search_tilelang_add.py --provider webgpu --out build/add-search-webgpu --candidates 3 --seconds 60
```

WebGPU scores use median launch-plus-queue-synchronization time from `tx.bench`;
these are host/completion measurements, not isolated GPU timestamps. The two
providers' scores measure different quantities and should not be compared as
the same latency metric. Use `--device N` to choose a different device.

The output directory contains:

| File | Purpose |
|---|---|
| `candidate-*.py` and `candidate-*.tbin` | Specialized exports and successfully compiled artifacts |
| `tilelang_add.py` | Copy of the factory used by those exports |
| `report.json` | Attempted candidates, rejections, scores, protocol, device, package versions, source hashes, and stop reason |
| `profile.json` | A schedule selector for this operation, shape, dtype, provider, and target |
| `selected.tbin` | Copy of the best correctness-checked artifact from this run |
| `a.npy`, `b.npy`, `reference.npy` | The exact inputs and independent reference |

The wall-time budget starts before device creation and is checked between
trials, including compilation. An in-progress build or measurement can finish
after the deadline. A candidate limit bounds attempted trials; beam width does
not. The selected result is the best observed candidate in this run, not a
guarantee of an optimal schedule or a statistically significant improvement.

## 2. Define your own space and seeds

Import the shared discovery API:

```python
from tensor.compiler.search import ScheduleSearch

spaces = {"add": {"block": (64, 128, 256)}}
seed = {"family": "add", "block": 128}
search = ScheduleSearch([seed], spaces=spaces, width=2)

config = search.next()
print(config)  # {'family': 'add', 'block': 128}
```

Every configuration has a `family` selecting a named space. The remaining keys
are schedule knobs you interpret in your factory. For this example:

```python
kernel = make_add(n=65537, block=config["block"])
```

`make_add` is the plain factory in `examples/tilelang_add.py`. The `family` label
is discovery metadata; do not forward it to a factory that does not accept it.

For your kernel, replace `block` with parameters the implementation actually
uses. Keep semantic parameters such as shape, dtype, quantization, and epilogue
separate from schedule parameters. Search one semantic case at a time and give
different algorithm implementations separate families when appropriate.

Each axis needs a nonempty ordered sequence of hashable values. Seeds must name
a family and use values present in its axes. Include every axis in each seed so
the configuration identity is complete. Neighbor moves step to adjacent values
in each sequence, so their order affects exploration. `width` must be positive;
it controls the ranked beam, not the number of candidates evaluated.

Known-good production settings make useful seeds and comparison baselines.
Include them in any restricted space instead of silently excluding your control.
You can also start with no seeds; deterministic restarts enumerate configurations.

## 3. Drive trials and record only valid scores

Your loop should follow this order:

1. Check candidate/time budgets and obtain `search.next()`.
2. Apply the configuration to your factory and build into a new artifact path.
3. Load the artifact; allocate/upload data outside measurement.
4. Launch and check required outputs against an independent reference.
5. Warm up, measure a consistent metric, and call `search.record(config, seconds)`.
6. Retain the configuration, artifact, timing samples, protocol, and any rejection.

`next()` raises `StopIteration` when the finite space is exhausted. A candidate
becomes seen when considered; it can remain unscored after compilation or
correctness rejection. Continue without recording that candidate.

`record()` accepts positive finite scores and ignores zero, negative, NaN, or
infinite scores. It does **not** run correctness checks or establish that the
candidate was measured. Lower scores rank better; seconds are the convention.
Always pass the same metric for every candidate in a search.

The engine explores neighbors of leading candidates while retaining candidates
from each family. When no unseen neighbors remain, deterministic restarts keep
exploring the finite space. Finding a winner does not automatically stop search;
your budget or space exhaustion does that.

The example initializes a fresh output to NaN before each correctness check.
This catches unwritten elements and prevents a partially writing candidate from
inheriting correct output from the previous trial. For a mutating kernel, reset
all input/state buffers before each check and timed repetition as required by
its semantics. Check saved intermediates as well as final outputs when they are
part of the kernel contract. Device/driver faults may require ending the run;
do not continue benchmarking a session that is no longer usable.

For CUDA, `measure_cuda(device, callback, warmup=3, samples=7, repeats=10)` returns
per-launch `gpu_seconds`, `completed_seconds`, and their medians. The callback
should launch into preallocated buffers without allocation or transfer. For
WebGPU, `tx.bench(kernel, buffers, warmup=3, iters=7)` reports host enqueue and
launch-plus-synchronization medians; the example ranks by the latter.

The example uses `tensor.compiler.entry.export_source` to fingerprint its
imported factory and writes the helper beside each export. List additional
implementation modules in `dependencies` when using this helper for a larger
factory. This ties candidate sources to the implementation used during search.

## 4. Filter illegal schedules and add coupled moves

Inside an open device session, pass `legal=` to reject configurations before
compilation:

```python
max_threads = device.limits["block"][0]
if device.info["provider"] == "webgpu":
    max_threads = min(max_threads, device.info["limits"]["max-compute-invocations-per-workgroup"])
search = ScheduleSearch(
    [{"family": "add", "block": 128}],
    spaces={"add": {"block": (64, 128, 256)}},
    width=2,
    legal=lambda config: config["block"] <= max_threads,
)
```

Obtain `max_threads` from the device limits for your workload. CUDA/CPU
`device.limits["block"]` describes per-axis launch limits; WebGPU also exposes
negotiated limits in `device.info["limits"]`. Account for total workgroup
invocations, shared memory, and additional kernel-specific restrictions.

A legality predicate should be a cheap filter for rules you know in advance.
Passing it does not prove compiler acceptance, correctness, or device support.
The compiler can still reject a configuration because of layout, lowering, or
resource constraints. Illegal proposals are seen but never returned for a trial,
so report attempted trials separately from `len(search.seen)`.

Some legal moves require several axes to change together. A
`coupled(config, family_space)` callback can yield such configurations in addition
to ordinary single-axis neighbors. Keep yielded configurations within the
declared family and axis values. For example, changing a register microtile can
require changing the thread count to preserve output ownership.

Existing provider helpers are starting points for their specific workloads:

- `cuda_schedules.projection_space` and `projection_legal` describe projection
  families; the legality helper has an SM86-oriented shared-memory default.
  Supply the appropriate limit and constraints for another target.
- `webgpu_schedules.SPACES` and `coupled_moves` describe existing partitioned and
  staged projection families. Their keys must map to a compatible factory or
  schedule helper; they do not automatically apply to arbitrary kernels.

Start with a small custom space before adopting a larger workload-specific one.

## 5. Persist and select a producer profile

`ScheduleProfile` uses the schema `tensor.schedule-profile.v1`. The example
writes a file similar to this illustrative CUDA profile:

```json
{
  "schema": "tensor.schedule-profile.v1",
  "provider": "cuda",
  "target": "sm_86",
  "entries": [
    {
      "operation": "add",
      "parameters": {"n": 65537, "dtype": "float32"},
      "schedule": {"block": 128}
    }
  ]
}
```

Load and select settings in your producer:

```python
import json
from pathlib import Path
from tensor.compiler.search import ScheduleProfile

profile = ScheduleProfile(json.loads(Path("build/add-search/profile.json").read_text()))
schedule = profile.select(
    "add", {"n": 65537, "dtype": "float32"},
    provider="cuda", target="sm_86",
)
if not schedule:
    raise ValueError("no measured schedule for this case")
print(schedule)
print(profile.sha256)
```

Use your actual build's provider and target when selecting, rather than copying
`sm_86` blindly. For the WebGPU example, select from
`build/add-search-webgpu/profile.json` with `provider="webgpu"` and
`target="webgpu-portable-v1"`.

Pass selected knobs into your existing factory/export:

```python
def tensor_export():
    return {"kernel": make_add(65537, **schedule), "outputs": ["out"]}
```

Build that export through `tensor build` or `tx.build` as usual. Define or import
`make_add` in the export file's producer environment. Profiles do not compile
kernels or apply themselves automatically to generic builds. Generic `.tbin`
builds also do not automatically embed the profile hash: record `profile.sha256`
and selected settings in your bundle or report if you need that provenance. The
example does this in `report.json`.

Selectors can be partial: every listed parameter must match, and the matching
entry with the most parameters wins. Duplicate selectors are invalid. Conflicting
matches at equal specificity raise an ambiguity error. An unmatched operation or
case returns `{}`; choose an explicit fallback or fail, as above.

Provider/target mismatches raise an error. A CUDA SM is not a unique GPU model,
and every WebGPU adapter shares the portable target identifier. These checks
protect selector usage, not performance portability across devices. Keep actual
GPU identity, compiler versions, source hashes, shape, dtype, reference tolerance,
and the timing protocol in provenance. Revalidate after any change.

You can retain the profile and referenced factory source instead of the compiled
winner, then rebuild using the selected knobs without rerunning search. The
producer toolchain and source dependencies must remain available; schedule knobs
alone do not reconstruct an unknown implementation. `profile.sha256` hashes
settings, not compiled code. The proposed [kernel recipe workflow](../plan/tuning-workbench.md)
adds explicit source/toolchain closure, fresh-build identity checks, numerical
verification and performance replay; those interfaces are planned, not current APIs.

## 6. Deploy the winner and confirm it helps

The selected artifact is already compiled. A consumer executes it without
importing discovery or installing compiler packages. Using the corresponding
[consumer environment](installation.md), run from the checkout:

```sh
build/consumer/bin/python -m tensor run build/add-search/selected.tbin --input a=build/add-search/a.npy --input b=build/add-search/b.npy --out-dir build/add-search/results
```

On Windows use `build/consumer/Scripts/python.exe`. For the WebGPU search,
install the consumer's `webgpu` extra and replace the directory with
`build/add-search-webgpu`; the CLI infers the provider from the artifact.
Verify `results/out.npy` against the retained `reference.npy`.

Repeat discovery or replay finalists in a fresh run before adopting a small
timing improvement. Compare with the current production schedule under the same
protocol, then validate and measure the full application. A faster isolated
kernel may not improve a model once allocations, transfers, host overhead,
neighboring operations, or numerical changes are included.

For a fixed precompiled CUDA set, the shorter `tune` helper is available:

```python
from tensor.compiler.tuning import tune

result = tune(
    {"block64": kernel64, "block128": kernel128},
    inputs=(a, b), output=out, reference=a_host + b_host,
    rtol=1e-5, atol=1e-8,
)
print(result["selected"])
```

Here `kernel64` and `kernel128` are loaded CUDA executables on the same device;
`a`, `b`, and `out` are already allocated. This helper appends one output after
the inputs, checks correctness, and chooses median CUDA-event latency. Use a
custom harness for other argument orders, multiple outputs, state-reset rules,
or WebGPU. Neither `tune` nor the example performs runtime autotuning for consumers.

For larger examples, see the
[CUDA projection search](../../benchmarks/lfm2/cuda_format_search.py),
[WebGPU outer-product search](../../benchmarks/inference/webgpu_outer_product_search.py),
and [measured producer profiles](../../benchmarks/lfm2/profiles/README.md).
Their references, hardware assumptions, and numerical gates are workload-specific.
