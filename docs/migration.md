# Tensor migration map

This migration has been executed and audited against the preserved dirty-checkout baseline. Paths in the manifests are relative to `/home/sagemaker-user`; `tensor/`, `akas/`, and `loop-latent-transformer/` identify the owning repository.

The rule is complete ownership: every existing file has explicit destinations. Splitting and merging are allowed. A split lists the definitions and statement ranges that move to each target. A merge lists every incoming source. A file marked `retain` has an intentional unchanged destination. No old implementation is kept as a fallback or a second copy after its migration succeeds.

## Complete inventory

- [File destinations](../research/studies/repository-layout/migration/migration-files.csv): all 2,304 existing tracked or nonignored untracked repository files, including package code, tests, examples, native code, configuration, documentation, evidence, and historical source snapshots. Each row contains source, targets, action, rationale, source SHA-256, and checks of existing destination files.
- [Code ownership](../research/studies/repository-layout/migration/migration-units.csv): 605 definition, state, statement-block, or whole-file routes. Split definitions include their original line ranges; nonsplit implementation files have whole-file routes.
- [References to update](../research/studies/repository-layout/migration/migration-references.csv): original imports and detected path/fingerprint references, with source line numbers. This is an editing checklist, not proof that string-based references are exhaustive. Migration verification must check the resulting tree and run consumers.
- [Coverage and incoming sources](../research/studies/repository-layout/migration/migration-coverage.json): counts, baseline revision, all merged targets and their incoming files, and explicit conflicts.

This baseline includes the uncommitted NN rewrite and AKAS extraction already in the checkout. Files already deleted before this inventory are not represented as existing sources. Generated or ignored builds, environments, caches, and package metadata are outside the source migration; this plan neither deletes nor moves them. Historical source snapshots and evidence are treated as immutable historical bytes rather than live implementations.

## Target responsibilities

```text
src/tensor/
  runtime/              Buffer, Executable, Session, Event, ABI, signatures,
                        DLPack, dtypes, manual backward and timing
  providers/
    cuda/               driver, device execution and graph capture
    webgpu/             adapter probing, opaque buffers, device execution,
                        prepared native encoding/submission plans
    cpu.py              CPU execution
  compiler/
    frontend/           source exports, portable IR restoration,
                        signature extraction and IR operator inventory
    cuda/               CUDA lowering, NVRTC/nvcc, toolchain selection,
                        CUDA build pipeline and schedule space
    webgpu/
      passes/           intrinsics, explicit unroll, matmul transformation,
                        uniform-barrier validation
      schedules/        register matmul, outer products, packed outer products,
                        packed integer matmul, partitioned matmul and GEMV
      build.py          WGSL artifact construction
    build.py            backend dispatch only
    cache.py            compiled-image identities, reads and writes
    licenses.py         redistributed compiler/header notices
    inspect.py          frontend/target/pass inspection
    errors.py           build errors
    search.py           device-independent schedule exploration/profiles
    tuning.py           correctness-gated timing and selection
  artifacts/            .tbin validation, portable preflight,
                        inspection and WebGPU artifact metadata
  modules/              manifests, content, cache, lock graphs, archives,
                        resolution, Project, Module and registry transport
  cli/                  parser, dispatch and command presentation
  diagnostics.py        environment/capability checks
  include/              native ABI header
  native/               standalone hosts and native encoding helper
packages/
  tensor-torch/         existing PyTorch binding; explicit retain rows
  tensor-nn/            existing native templates and nanoGPT examples;
                        explicit retain rows
benchmarks/
  kernels/              WebGPU, CLBlast, attention and backend comparisons
  adapters/torch/       FX producers/consumers and executor comparison
  training/nanogpt/     independent training/reference measurements
  measurement.py        shared measurement recording
examples/               existing complete kernel demonstrations
research/
  phase0/               original experiment implementation and its tests
  studies/              related reports, evidence and figure generators
  decisions/            historical architecture decisions
  plans/                preserved planning baselines
  design/               original architectural proposal
```

Directories follow actual implementation responsibilities. There is no new generic `utils`, `common`, or model-specific implementation bucket. Small CPU implementations remain files rather than receiving empty symmetric hierarchies.

## Explicit splits and merges

The CSV contains every definition; these are the major routes.

| Existing file | Destinations and ownership |
|---|---|
| `src/tensor/runtime/__init__.py` | `runtime/buffer.py` → Buffer; `executable.py` → Executable; `session.py` → Session and identity counter; `event.py` → Event; `errors.py` → TensorRuntimeError; `timing.py` → bench. Initializer retains exports only. |
| `src/tensor/artifacts/modules.py` | `modules/manifest.py` → manifest/path rules and limits; `content.py` → canonical encoding, content records and directory validation; `cache.py` → verified storage; `lock.py` → lock graph validation; `archive.py` → archives/pack; `resolver.py` → resolve/install/add; `project.py` → Project/reference selection; `module.py` → loaded Module; `errors.py` → ModuleError. |
| `src/tensor/compiler/build.py` | `compiler/build.py` → provider dispatch; `cuda/build.py` → existing CUDA pipeline; `cache.py` → image cache/hash helpers; `frontend/signature.py` → launch validation, merged with the existing signature lowering file; `frontend/ir.py` → IR operator inventory; `licenses.py` → notices; `errors.py` → BuildError. |
| `src/tensor/compiler/__init__.py` | `cuda/nvcc.py` → NvccCompiler; `cuda/toolchain.py` → compiler selection; initializer retains explicit exports only. |
| `src/tensor/compiler/webgpu_lowering.py` | `webgpu/passes/{intrinsics,unroll,matmul,barriers}.py` → corresponding transformations; schedule factories move to their named algorithm files. |
| `src/tensor/compiler/webgpu_templates.py` | Each DSL implementation moves beside its corresponding schedule factory in `webgpu/schedules/`; scalar-register allocation helpers move to `registers.py`. The CSV lists both inputs to every merged algorithm file. |
| `src/tensor/providers/cuda.py` | `cuda/driver.py` → errors and Driver; `cuda/device.py` → Device. Graph implementation moves from `cuda_graph.py` to `cuda/graph.py`. |
| `src/tensor/providers/webgpu.py` | `webgpu/adapter.py` → enumeration/probe; `buffer.py` → opaque buffer/handle counter; `device.py` → Device; `plan.py` → PreparedPlan. |
| `src/tensor/artifacts/portable.py` | Artifact preflight remains `artifacts/portable.py`; executable frontend restoration moves to `compiler/frontend/portable.py`. |
| `src/tensor/cli/doctor.py` | Checks move to `diagnostics.py`; nvcc location resolution joins `compiler/cuda/toolchain.py`; rendering stays in `cli/doctor.py` alongside its command handler. |
| `src/tensor/cli/__init__.py` | Parser construction → `cli/parser.py`; command branches → `cli/{modules,doctor,build,cache,inspect,run}.py`; dispatch → `cli/main.py`; initializer exports `main` only. The exact existing statement ranges are enumerated. |
| `src/tensor/cli/commands.py` | Execution/input preparation → `cli/run.py`, merged with run/bench handling; artifact inspection → `artifacts/inspect.py`. Source inspection moves from `cli/inspect.py` to `compiler/inspect.py`. |

Imports, module docstrings, and comments follow the definitions they describe; import-only initializers and the backend/CLI dispatch adapters are explicit glue. They must not accumulate implementation. Existing class methods stay with their class unless a later, separately enumerated ownership map assigns them elsewhere.

## Current documentation

Each documentation source also has a CSV row. The merged destinations have explicit topic ownership:

- `docs/quickstart.md`: installation/environment selection and the complete first-kernel walkthrough.
- `docs/compilation.md`: bringing TileLang kernels and exploring schedules.
- `docs/runtime.md`: calls, buffers, streams and explicit backward contexts.
- `docs/backends.md`: provider-specific setup and WebGPU execution.
- `docs/modules.md`: packaging, pinned dependencies and distribution.
- `docs/compatibility.md`: tested scope and upgrades.
- `docs/reference/runtime-abi.md`, `docs/development.md`, `docs/releases.md`, and `docs/README.md` retain their current roles.
- PyTorch usage merges into `packages/tensor-torch/README.md`. NN instructions remain in `packages/tensor-nn/README.md`.

Repeated setup sections should be reconciled when merging, rather than concatenating the old documents. Stale imports, ownership claims and commands must be corrected. Original plans and decisions are preserved as history rather than presented as current implementation. [Architecture](architecture.md) describes the verified implementation ownership.

## Transfers to independent projects

LFM2/Qwen engine reports and evidence map to their exact paths in AKAS. LLT-specific guides, plans, reports and evidence map to explicitly enumerated paths in `loop-latent-transformer/docs/tensor/`. Figure scripts also have explicit owning-project targets.

1,595 source files were verified byte-for-byte at their AKAS target before retirement. Their source and destination checksums are recorded. Those are duplicate-copy retirements, not overwrites. Verify the recorded destination checksum again before retiring a Tensor copy.

Three targets contain different bytes:

| Tensor source | Target | Required resolution |
|---|---|---|
| `benchmarks/qwen35/dense_ablation.py` | `akas/benchmarks/qwen35/dense_ablation.py` | AKAS has the newer engine integration. Compare behavior and retain any Tensor-only functionality explicitly; do not overwrite it with the stale imports. |
| `benchmarks/qwen35/dense_ablation_report.py` | `akas/benchmarks/qwen35/dense_ablation_report.py` | Reconcile report functionality and updated reproduction paths against the AKAS version. |
| `docs/guides/llm-serving-benchmarks.md` | `akas/docs/guides/llm-serving-benchmarks.md` | Tensor holds a forwarding note; AKAS holds the substantive guide. Preserve the AKAS guide and link to it from the Tensor index. |

No conflicting target can be replaced simply because its source has a migration row.

## Migration and verification order

1. Establish baseline checks for the existing dirty checkout; preserve unrelated work. Resolve the three external conflicts before retiring their sources.
2. Move/split artifact, module, runtime and provider implementations according to the unit map. Update exports and imports while preserving the public API and ABI.
3. Move frontend/compiler implementations and merge each WebGPU schedule factory with its matching DSL implementation. Preserve generated kernel source and numerical behavior; compare against the baseline.
4. Move CLI, benchmarks, validation tools, historical experiments/tests and CI entry points. Update repository-root calculations and relative paths, not just imports.
5. Merge current docs; transfer verified model material and retain Tensor history by study. Update indexes and local links.
6. Run focused CPU tests, compiler-free import/installed-consumer checks, package builds, applicable real-device checks, CLI/examples and affected benchmark smoke checks. Update implementation fingerprint inputs and verify rebuild requirements for consumers in Tensor NN, Tensor Torch and AKAS. Historical evidence hashes are not rewritten to match new source files.
7. Reconcile every manifest row: targets exist, retained sources are intentional, moved sources are gone, all split units are accounted for, merged targets have every incoming unit, no old implementation aliases remain, and no conflicting target was silently overwritten. Record results against this source-checksum baseline.

## Execution audit

All 2,304 baseline file rows have existing targets. All 605 code-unit routes have
existing targets; every named split definition and state binding was found in its
assigned file. [Unit results](../research/studies/repository-layout/migration/migration-unit-results.csv)
include current target line numbers. Whole-file and statement-block routes are
checked by destination presence and the implementation tests below.

The preserved baseline matches every recorded source checksum. All 1,595 duplicate
retirements match their target checksums, and every mapped evidence file retains
its baseline bytes. Retired source paths are absent, except paths explicitly reused
as targets. The three conflicts retain the newer AKAS implementations and guide.

New implementation glue consists of import-only package initializers and WebGPU
`identity.py`, which hashes the split passes and schedules. Current architecture
and documentation indexes describe the resulting tree. The original backend guide
is additionally preserved under the WebGPU validation study. Compiler/source
fingerprints now cover the new implementation owners; rebuild existing bundles
when consuming this revision. Historical evidence fingerprints remain unchanged.

Validation: Tensor's default suite passed 205 checks, historical tests passed 33,
and CUDA runtime/compiler checks passed 14. Tensor Torch passed 28 CUDA checks;
nanoGPT passed 12 native training checks. AKAS passed 264 default checks and 28
CUDA request checks; LLT passed 17 tests and 1,053 subtests. Hardware-dependent
checks outside these enabled suites remain skipped. Three wheels and three source
archives passed metadata, source rebuild, and isolated compiler-free consumption
checks. Current documentation links and core import lint pass.

Of 92 split definitions, 89 retain the same AST excluding imports/docstrings.
Two constructors lose annotations that referred to the Device factory; one
NVRTC diagnostic updates its bootstrap path. The coverage file records the
explicit additions and validation results. Ignored generated builds,
environments, and caches are outside the baseline inventory.

