# ADR 0013: Offline modules with pinned closures and explicit compilation

Status: accepted, 2026-09-29.

## Context

Phase 2 provides runtime ABI 1.1, a compiler-free executable consumer and NVRTC
production. Proposal §11 needs reusable named exports, dependencies, module
packages and a resolver. Phase 3 should use the existing executable/frontend
representations and establish reproducibility before a registry exists.

## Decision

Use `tensor.json` schema 1, exact `major.minor.patch` module versions and Tensor
ABI major 1. Exports declare Python source, a verified `.tbin` carrying portable
frontend TIRx, and/or exact-target executable artifacts. Capabilities apply to
the module and are checked against the selected runtime provider. Executable
artifacts retain their independent envelope/runtime versions and requirements.

Resolve local directory or `.tpack` dependencies, reject cycles and conflicting
identities, and pin every transitive module by its content SHA-256 in
`tensor.lock`. A module identity includes normalized manifest bytes, declared
files and the pinned dependency identities. One version/content identity per
module name is permitted in a project. No version ranges or network registry
resolution are introduced in this phase.

`tensor pack` creates a deterministic ZIP containing the whole dependency
closure and a hashed index. It selects only exports and explicit `files`;
unlisted helpers are not included. Packages reject duplicate/external/nonportable
paths, conflicting variants, undeclared members and invalid graphs. Native
artifacts are verified before packaging or installation. Hashes identify bytes;
they are not publisher authentication.

`tensor add` records an exact version and identity, installs the closure, and
updates the manifest and lock after graph validation. `tensor install --frozen`
requires the existing graph to match, restores verified snapshots and leaves
the lock unchanged. Snapshots and generated artifacts occupy a separate module
cache (`TENSOR_MODULE_CACHE`). Cached bytes are verified before use; compilation
cache entries still use the existing full frontend/compiler/header identity.
Operations that mutate one project should be serialized by the caller.

Export selection prefers a packaged exact-provider/target artifact, then a
verified locally compiled image. A missing image fails without compiler imports
unless compilation is explicitly requested (`--compile`, `tensor build`, or
`Module.resolve/load(..., compile=True)`). The producer then prefers bundled
TIRx over Python source. TIRx hashes, exact TileLang/FFI versions and its declared
operator set are checked before deserialization; operator availability is
checked against the registered pinned frontend. An incompatible portable carrier
fails explicitly rather than silently using another representation.

Cached export identity includes module closure, export, provider, target,
compiler family and supported runtime ABI. A valid image can subsequently be
used without the producer environment. Compiler versions remain provenance on
that consumer path. Source helpers execute from the verified snapshot with
local import isolation and bytecode writes disabled.

Expose `Project.module().resolve/load()` in Python and `module-name::export`
references in build, inspect, run and bench. `tensor pack` creates an artifact
for distribution; publishing to a remote registry belongs to later ecosystem
work. No publication service is needed for Phase 3's offline resolver.

## Scope and acceptance

Exercise package relocation, deterministic archives, transitive closures,
cycles/conflicts, frozen locks, failed updates, corruption recovery, capability
rejection and compiler-free CLI paths. Validate actual CPU portable compilation,
NVRTC source compilation and sm_80-to-sm_86 TIRx retargeting for affine and GEMM.
Linux and Windows CI package all five profiles, install with only Tensor/NumPy,
and inspect exact-target exports under an import guard; a separate A10G executes
those installed exports and module CLI references.

This is an offline module system with one-kernel exports. Scheduling/fusion,
autotuning metadata, public registries, cross-SM binary compatibility, optimized
CPU and additional GPU providers remain later work. Direct PTX stays experimental
after Tensor v1; portable reuse still lowers through TileLang and NVRTC.
