# Phase 3 exit — offline tensor modules

Phase 3 is complete for the offline, exact-version, one-kernel-export profile.
Acceptance on 2026-09-30 used implementation commit
`fce946c651a38cebe2d1b7361159f0c9c312b956`, Linux x86-64 and A10G `sm_86`.
The producer and consumer recorded clean checkouts at the same revision.
[ADR 0013](../adr/0013-phase3-offline-module-system.md) defines the boundary;
the [module guide](../modules.md) describes the API and CLI.

## Delivered gates

| Phase 3 requirement | Implemented and validated |
|---|---|
| `tensor.json` | Schema 1, exact versions, named source/portable/native exports, capabilities and explicit package files |
| Module resolver | Local directories and `.tpack` closures; exact name/version/hash checks; cycles and conflicting identities rejected |
| Artifact cache | Verified immutable package snapshots and generated exports; full existing compiler/header cache retained; corruption recovery |
| Exports | `Project.module().resolve/load()` and `module-name::export_name` in build/inspect/run/bench |
| Versioned Tensor ABI | Module ABI major 1; independent artifact envelope/runtime minor and capability negotiation; legacy buffer semantics preserved |

`tensor add` records exact dependency pins and updates the lock after graph
validation. `tensor install --frozen` requires the same graph, keeps the lock
unchanged and can recover a closure from verified cache snapshots after its
source archive is removed. `tensor pack` creates a deterministic ZIP containing
every transitive dependency and a hashed file index. Only exports and explicit
files are included. External paths, symlinks, case collisions and ambiguous
variants are rejected.
Source is not executed during package, install or binary-selection operations.

Export resolution prefers an exact-provider/target packaged artifact, then a
verified generated image. Missing images fail unless compilation is explicitly
requested. Compilation prefers bundled frontend TIRx over source; it checks
hashes, exact TileLang/FFI versions, operator-set integrity and availability
before deserialization. NVRTC remains the default CUDA executable compiler.
Local helper imports are isolated and do not write bytecode into snapshots.

## Results

| Check | Observed result |
|---|---|
| Full GPU-enabled regression suite | **104 passed, zero skips**, 68.08 s |
| Linux producer CI | **85 passed, 19 skipped**, 21.02 s |
| Windows producer CI | **75 passed, 29 skipped**, 14.26 s |
| Each remote package → separate A10G | **21 numerical/interop cases + eight diagnostics** through installed module exports |
| Module CLI | Named-export run and bench passed for both transferred consumers |
| Relocation/frozen closure | Source package removed after installation; frozen reinstall and all five selections passed from snapshots |
| Binary-only consumer | Exactly Tensor 0.1.0 and NumPy 2.5.3; compiler imports prohibited; generated-image cache remained empty |
| Existing direct-artifact baseline | Repeated 21 cases and eight diagnostics per platform before module acceptance |
| Portable reuse | CPU affine recompiled; sm_80 affine/GEMM retargeted to sm_86 through NVRTC and numerically executed |
| Legacy compatibility | Equivalent v1/current signatures resolve; a real CUDA v1 kernel executes through a module export |

The [successful final CI run](https://github.com/kreasof-ai/tensor/actions/runs/36648206335)
built elementwise, GEMM + ReLU, dynamic affine, dynamic GEMM and int64 offset.
It repeated NVRTC cold/warm-cache and corruption checks, then packaged
`tensor-ops` and its `tensor-base` dependency. Each producer packed twice and
verified identical archive bytes. Its clean consumer installed, froze and
selected all exports under a compiler import guard.

GPU tests skip on the Actions runners. Native CPU tests also skip on Windows
because that producer profile is Linux x86-64. Windows-produced wheels and
packages were numerically executed on the separate Linux A10G; GPU execution
on Windows itself is not claimed.

Downloaded archives matched GitHub's SHA-256 digests. Kernel/source hashes were
checked against the exact producer revision, accepting its verified LF or CRLF
checkout bytes. Module archive hashes matched their producer reports and the
hashed package graph. Installed consumer package files matched the wheel in
each downloaded archive. Both producer and consumer revisions were equal and
clean, and producer/consumer hostnames differed.

The closure archive is **141,672 bytes on Linux** and **141,275 bytes on Windows**:
two modules, five exports, source, frontend TIRx and native images. These are
package sizes, not compiler distribution sizes. The producer's Python/frontend
dependencies and NVRTC/header bundle remain separate and substantial.

Raw evidence: [exit and source hashes](data/phase3-exit.json),
[CI jobs/artifact identities](data/phase3-ci.json),
[Linux transfer](data/phase3-transfer-linux.json), and
[Windows transfer](data/phase3-transfer-windows.json).
The [runbook](../plan/phase3-validation.md) reproduces the checks.

## Remaining scope

The resolver uses local sources and exact three-part versions with one content
identity per module name. Public registry/publisher authentication and version
ranges are later work. Packages provide one-kernel exports; general fusion,
autotuning and scheduled-representation selection are later extensions.

CUDA binaries still require an exact SM and a compatible driver. Successful
frontend retargeting is local compilation, not cross-SM binary compatibility.
CPU remains a Linux x86-64 validation provider. Runtime/provider contracts stay
at ABI 1.1; this phase does not introduce a C provider-plugin table or native
artifact parser. **Direct PTX stays experimental after Tensor v1.**
