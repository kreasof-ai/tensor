# ADR 0014: PyPI as a transport for verified Tensor modules

Status: accepted, 2026-09-30. Extends ADR 0013's offline module profile.

## Decision

Use existing Python indexes for publication and retrieval. Keep `.tpack` as the
Tensor package format and put the complete pinned closure in a deterministic
data-only wheel. Do not introduce a hosted Tensor registry, package install
hooks, eager compiler imports or a Python dependency graph for transport wheels.

`tensor publish` prepares the wheel, checks it with Twine and uploads to PyPI,
TestPyPI or a custom upload endpoint. `--dry-run` prepares/verifies bytes without
Twine or upload. Twine is an optional publisher extra; credentials and upload
authentication remain Twine's responsibility. It is never needed by consumers.

Default distribution names are `tensor-module-<module-name>` with an explicit
override. The Python distribution and Tensor module share the exact three-part
version. Wheel metadata carries a descriptor binding the distribution/version
to a payload path, payload hash, module name and module content hash. The wheel
contains only that payload and its metadata, including valid RECORD hashes;
extra files, Python dependencies and executable installation files fail the
Tensor reader. The data envelope uses `py3-none-any`; actual provider, Tensor
ABI and GPU compatibility stay inside Tensor artifacts and runtime negotiation.

`tensor add pypi:distribution==version` uses one explicit Simple Index (default
PyPI), with JSON or HTML responses. Download and wheel SHA-256, RECORD, metadata,
payload and closed graph are verified before manifest/lock mutation. Persist
the normalized distribution, index URL and wheel hash beside the exact module
version/hash in `tensor.json`. These fields participate in the existing project
identity and lock graph without changing the lock schema. Hashes are content
integrity checks, not publisher identity or execution sandboxing.

Prefer a verified cached closure for registry dependencies. If a cached closure
is missing or corrupt, fetch the exact pinned wheel and restore it; frozen
installation still requires the graph and project identity to match the lock.
Offline installation forbids registry requests. Execution/selection never
downloads or compiles implicitly. Reject yanked releases on new adds; allow
restoring an already pinned exact wheel when yanked. No cross-index fallback.
HTTPS is required remotely; HTTP loopback permits local development/acceptance.
Authenticated download indexes and version ranges remain separate work.

## Validation

Tests exercise deterministic wheels, installation through a real wheel installer,
relocated dependency closures, JSON/HTML indexes, exact wheel/module pins, fresh
frozen restoration, corrupt-cache recovery, offline behavior, yanking, rejected
tampering and atomic project updates. A real Twine subprocess uploads to a local
multipart endpoint and its received wheel is installed through the resolver.

Linux/Windows CI run those tests with the publisher extra, then prepare a wheel
carrying the five existing CUDA profiles. A clean Tensor/NumPy consumer restores
it through a local index into a new cache, repeats frozen installation offline
and selects all five packaged exports under a compiler import guard. The same
acceptance tool can execute the installed exports on the existing A10G. Public
PyPI/TestPyPI publication requires a publisher account and is not an acceptance
test or performed automatically by the development workflow.
