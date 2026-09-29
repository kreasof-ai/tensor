# ADR 0005 — PyTorch support ships as a client-side adapter, not a compiler dependency

**Status:** Accepted · 2026-09-29
**Evidence:** `experiments/p0/harness.py --only torch_dependency`, `experiments/p0/link_check.py`

## Context

torch is **71% of the installed footprint** (495 MB of 695 MB; 118 MB of the ~230 MB
download). It is a *hard* dependency of TileLang, and proposal §23 makes "steps from
download to first kernel" and "number of manually installed dependencies" primary success
metrics.

The question: can Tensor support `torch.compile(backend="tensor")` without carrying torch
in its main binary?

## Evidence

**Measured, three probes** (compile a kernel to CUDA source in an isolated process):

| Probe | Result |
|---|---|
| `normal` (torch working) | ✅ compiles, 23 lines of CUDA emitted, torch loaded eagerly |
| `torch_blocked` (`import torch` raises) | ❌ fails inside `import tilelang` |
| `torch_stubbed` (importable, no functionality) | ❌ fails in `tvm_ffi/_optional_torch_c_dlpack.py:193` at `torch.cuda.is_available()` |

**Native libraries are torch-free.** Parsing the PE import directory of the shipped binaries:

| Library | Size | torch/c10 in import table |
|---|---|---|
| `tvm_compiler.dll` | 60.0 MB | **none** |
| `tvm_runtime.dll` | 3.4 MB | **none** |
| `tilelang_cython_wrapper.pyd` | 0.1 MB | **none** |

So the dependency is **entirely at the Python layer**, not a binary link.

The specific culprit is one line — `tilelang/__init__.py:135`:

```python
@contextlib.contextmanager
def _lazy_load_lib():
    import torch  # preload torch to avoid dlopen errors
```

That comment describes a **dlopen-ordering precaution**. Given the import tables are clean,
it is a precaution against a failure mode the shipped binaries do not appear to have. There
are also 17 modules with top-level `import torch` (including `cuda/target.py` and
`language/dtypes.py` — core paths, not just interop).

## Decision

1. **The compiler never links torch.** It may not `import torch` on any path needed to
   compile, inspect, or emit code. The `tensor` binary and its runtime are torch-free.
2. **The tensor ABI is DLPack, not torch-shaped.** DLPack is the zero-copy interchange
   standard; `torch.Tensor.__dlpack__` is stable and torch-agnostic. TileLang already
   depends on `torch-c-dlpack-ext` (4.7 MB) for exactly this.
3. **PyTorch integration is a separate wheel, `tensor-torch`**, that declares a
   `torch_dynamo_backends` entry point. The user who wants PyTorch **already has torch
   installed**; Tensor never ships it. This is the same mechanism that lets third parties
   plug into `torch.compile` with zero patching of PyTorch.
4. **Contribute the fix upstream.** Make TileLang's torch import lazy and move `torch` to
   an extra. This benefits TileLang's own users and is the single change that converts a
   torch-free *binary* into a torch-free *install* (~695 MB → ~200 MB).
5. **Escape hatch, not a plan:** Tensor can drive `apache-tvm-ffi` (3.4 MB, torch-free)
   directly if the TileLang Python package cannot be made torch-optional.

## Consequences

- **Honest caveat:** until (4) lands, `pip install tensor` still pulls torch, because
  tilelang's metadata declares it. A torch-free *binary* is achievable immediately; a
  torch-free *install* depends on the upstream change. Do not claim the latter before (4).
- `tensor` gains an execution backend that is not `torch`; TileLang already offers
  `tvm_ffi` (3.4 MB, torch-free) and `cython`, so this needs no new work — only a
  decision not to default to torch.
- The compiler's import graph gets a standing constraint: **no torch on the compile
  path.** This is testable, and is exactly what the `torch_dependency` experiment
  automates. It should stay green.
- The `tensor-torch` wheel must keep working across torch versions. Isolating it in its own
  package with its own version line is the point — not an accident.
