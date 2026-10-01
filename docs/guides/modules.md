# Tensor modules

[Documentation](../README.md) · [Quickstart](quickstart.md)

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

Dependencies identify a local directory, `.tpack` or Tensor transport wheel, an exact version, and
optionally an expected module `sha256`. A hash-only dependency uses an existing
verified cache snapshot. Resolution pins all transitive hashes; conflicting
versions or contents under one name and dependency cycles fail. A `.tpack`
contains the whole resolved closure, so its original directory paths are not
needed on another host.

PyPI dependencies additionally pin an origin containing the Simple Index URL,
Python distribution name and wheel SHA-256. `tensor add pypi:...` writes these
fields automatically. The module's `sha256` identifies its Tensor content;
`pypi.sha256` identifies the transport wheel. The project manifest and lock
together pin the registry source and complete dependency closure.

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

## Publish and install through PyPI

`tensor publish` wraps a verified `.tpack` and its complete dependency closure
in a deterministic, data-only Python wheel. Source directories are first packed
through the same validator. The default Python distribution is
`tensor-module-<module-name>`; `--distribution` selects another available PyPI
project name. This prefix is a naming convention, not a reserved namespace.
The module version is also the Python distribution version.

Prepare and inspect the wheel without credentials, Twine or network uploads:

```bash
tensor publish ./my-ops --dry-run --out-dir build/pypi-preview
tensor inspect build/pypi-preview/tensor_module_my_ops-0.1.0-py3-none-any.whl
```

From a checkout, build the core wheel and install its optional publisher
dependencies, then upload to TestPyPI or PyPI:

```bash
uv build --wheel --out-dir build/wheels
python -m pip install 'build/wheels/tensor_workspace-0.1.0-py3-none-any.whl[publish]'
# Configure Twine credentials using its environment variables or keyring.
tensor publish ./my-ops --repository testpypi --out-dir build/testpypi
tensor publish ./my-ops --repository pypi --out-dir build/pypi
```

Twine checks the prepared wheel and uploads noninteractively. Its standard
`TWINE_USERNAME`, `TWINE_PASSWORD`, keyring and publishing authentication apply.
Preparation rejects existing outputs; failed uploads retain the verified wheel,
which can also be uploaded with `python -m twine upload PATH.whl`. Custom upload
services use `--repository-url`; this is distinct from the download index URL.

Consumers need only Tensor and NumPy:

```bash
tensor add pypi:tensor-module-my-ops==0.1.0 --project app
tensor install --project app --frozen
tensor install --project app --frozen --offline
```

Use `--index-url https://test.pypi.org/simple/` on `add` for TestPyPI or provide
another public Simple Index URL. One explicitly selected index is used and
persisted with the wheel hash. JSON (PEP 691) and HTML (PEP 503) index responses
are supported; Tensor selects its exact `py3-none-any` transport wheel, requires
a SHA-256, and validates wheel metadata, RECORD hashes and the embedded package
before updating the project. New adds exclude yanked releases; restoration of
an already pinned wheel can use a yanked release with the exact pinned hash.
Changed wheel hashes or module identities fail. Registry URL credentials and
non-HTTPS remote endpoints are rejected; HTTP loopback supports local indexes.
Authenticated private downloads are not yet supported. Upload authentication is
handled by Twine.

An intact cached closure avoids network access. A missing or corrupt registry
closure can be restored with `install`, including `--frozen`, into a fresh cache
without changing the lock. `--offline` prevents registry requests and requires
local sources or the verified cache. `resolve`, `run` and `bench` read installed
snapshots and never download packages implicitly.

The wheel is only a transport envelope: Python compatibility tags do not encode
GPU architecture, CUDA driver or Tensor ABI requirements. Tensor still checks
those when selecting/loading an export. Installing the wheel with pip places
its payload at `tensor_module_payloads/<normalized_distribution>/module.tpack`
in site-packages, but does not mutate a Tensor project. Tensor can consume the
wheel directly using `tensor add ./PACKAGE.whl`, or consume that installed
`.tpack`. No compiler or Python dependencies are declared by transport wheels.

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
the [runtime contract](../reference/runtime-abi.md).

This implementation supports local paths, PyPI transport, exact versions and
opaque one-kernel exports. Version ranges, fusion, autotuning and multi-image
CUDA compatibility remain extensions. The original
[offline decision](../adr/0013-phase3-offline-module-system.md) and
[PyPI transport decision](../adr/0014-pypi-module-transport.md) state the boundaries.
