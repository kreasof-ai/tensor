# Tensor modules

A module names reusable kernel exports. `tensor.json` describes them;
`tensor.lock` pins the complete installed dependency graph. Package operations
need only Tensor and NumPy. Source or portable-TIRx compilation requires the
pinned producer environment and, for CUDA, the NVRTC bundle.

## Manifest

```json
{
  "formatVersion": 1,
  "name": "my-ops",
  "version": "0.1.0",
  "tensorAbi": 1,
  "exports": {
    "affine": {
      "source": "src/affine.py",
      "portable": "artifacts/affine.tbin",
      "artifacts": ["artifacts/affine.tbin"]
    }
  },
  "dependencies": {
    "base-ops": {"path": "../base-ops", "version": "0.1.0"}
  },
  "capabilities": ["contiguous", "scalars"],
  "files": ["src/helper.py", "README.md"]
}
```

Source-only exports may use the shorthand `"affine": "src/affine.py"`.
`portable` is a verified `.tbin` whose bundled frontend TIRx can be re-lowered;
it is not automatically selected as a native image unless also in `artifacts`.
Each native variant must have the same logical signature and outputs. If source
is present, variants must match its source hash. Two binaries for the same
provider/target are ambiguous and rejected.

Module versions are exact three-part versions. Names are lowercase words and
digits separated by hyphens; export names are ASCII identifiers. `tensorAbi`
declares the runtime ABI major, while each `.tbin` declares its own required
minor/capabilities. Package paths stay inside the module and use forward slashes.
Exports plus explicit `files` are the complete package file list. Include helpers
and notices explicitly; unlisted files, symlinks and case-colliding paths do not
become part of a package. Empty exports are useful for an application project.

Dependencies identify a local directory or `.tpack`, an exact version, and
optionally an expected module `sha256`. A hash-only dependency uses an existing
verified cache snapshot. Resolution pins all transitive hashes; conflicting
versions or contents under one name and dependency cycles fail. A `.tpack`
contains the whole resolved closure, so its original directory paths are not
needed on another host.

## Build and package the included examples

```bash
tensor install --project examples --module-cache build/modules
tensor build tensor-examples::dynamic_affine --project examples \
  --module-cache build/modules --target sm_86 --out build/affine.tbin
tensor pack examples --out build/examples.tpack --module-cache build/modules
```

The included manifest is source-only. To ship a compiler-free consumer package,
place built `.tbin` files inside your module and list them under each export's
`artifacts`, optionally also `portable`. `tensor pack` validates the files and
creates a deterministic archive with their content hashes and pinned closure.
It creates a new output exclusively. It does not publish to a registry.

## Install and resolve

Create an application `tensor.json` with schema 1, its name/version/ABI, and
`"exports": {}`. Add a module archive or directory:

```bash
tensor add ./my-ops.tpack --project app --module-cache build/modules
tensor install --project app --module-cache build/modules --frozen
tensor resolve my-ops::affine --project app --module-cache build/modules --target sm_86
tensor inspect my-ops::affine --project app --module-cache build/modules --target sm_86
tensor cache --module-cache build/modules
```

`add` writes exact version/hash pins and installs the closure. `install` updates
the lock after resolving changes; `--frozen` requires the existing lock to match
and never rewrites it. Installed closures can be restored from their cache even
after a source archive is removed. A changed source at an available dependency
path is checked against its pin. Updating a dependency added with an explicit
pin requires `tensor add` again. Source-only unpinned development dependencies
can be updated with `tensor install`.

Commit `tensor.json` and `tensor.lock`. Serialize operations that mutate a
project. Module snapshots live under `TENSOR_MODULE_CACHE` or
`~/.cache/tensor/modules`; compiler cache settings remain independent.
Modified/incomplete snapshots fail verification. Reinstall from a verified
source repairs them; generated-image corruption triggers a rebuild only when
compilation is allowed.

## Execute an export

```bash
tensor run my-ops::affine --project app --module-cache build/modules \
  --input a=inputs/a.npy --input b=inputs/b.npy --scalar scale=2.5 --out-dir results
tensor bench my-ops::affine --project app --module-cache build/modules \
  --input a=inputs/a.npy --input b=inputs/b.npy --scalar scale=2.5
```

Resolution chooses an exact-target packaged binary, then a verified compiled
cache image. If both are missing, the command gives an actionable error. Add
`--compile` to run/bench/resolve/inspect to allow local compilation, or use
`tensor build`. Portable TIRx takes precedence over source and requires its
exact frontend versions and registered operators. CUDA compilation still uses
NVRTC by default. A previously compiled cache image can later run without those
compiler dependencies. CPU exports use `--provider cpu` and the scoped native
compiler profile. CUDA execution targets the selected device's exact SM.

```python
import tensor as tx

project = tx.Project("app", cache_dir="build/modules")
module = project.module("my-ops")
with tx.Device() as device:
    kernel = module.load("affine", device)
    result = kernel(device.arange(129), device.ones((129,)), scale=2.5)
    print(result.to_numpy())
```

Use `compile=True` on `load()` or `resolve()` when local compilation is intended.
`resolve()` returns the selected path, target, module identity and whether the
image was packaged, cached or newly compiled. Kernel and buffer lifetimes remain
the [runtime contract](runtime-abi.md).

This phase supports local paths, exact versions and opaque one-kernel exports.
Network registries, version ranges, fusion and multi-image CUDA compatibility
are later extensions. The [decision record](adr/0013-phase3-offline-module-system.md)
states the supported boundary.
