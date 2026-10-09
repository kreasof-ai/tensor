# Changelog

User-facing changes belong here. The [research archive](docs/research/README.md)
contains the detailed experimental record. The package version is currently
`0.1.0`; this file does not imply that a public release has been published.

## Unreleased

- Add target-aware Qwen producers and a bounded Modal H200 replay runner with
  persistent weights/artifacts, native AR/speculative client measurements and
  retained numerical checks. Exact `sm_90` CUDA builds use pointer-compatible
  copies and warp MMA; compiler identities record the Hopper pass configuration.
  Retain physical H200 C8 32K/16K rates of 588.2 tok/s AR and 911.5 tok/s
  MTP plus output lookup; numerical qualification still fails.
- Add experimental native Qwen3.5 FP8 execution with chunked prefill, embedded MTP,
  batched speculative verification and output-history proposals. Retain the
  completed L40S C8 32K/16K client replay at 735.3 output tok/s, together with its
  failed numerical qualification; this is not a qualified throughput claim.
- Organize `tensor-llm` into shared utilities, model-independent speculative
  helpers, LFM2 and Qwen3.5 subpackages. Preserve top-level class imports and
  update benchmark producers; internal paths and source identities changed,
  requiring inference bundles to be regenerated.
- Add bounded independent LFM2 request handles sharing weights, kernels and
  executor scratch, with isolated sequence state and reset/close, failed-allocation
  cleanup and buffer accounting. Calls remain serial; existing inference bundles
  require an explicit producer rebuild because implementation fingerprints changed.
- Order LFM2 pageable metadata/state copies before nonblocking CUDA execution;
  host return from a transfer alone does not guarantee its device DMA completed.
- Add an independent vLLM, SGLang and llama.cpp serving harness with immutable
  token-ID workloads, concurrency sweeps, streamed latency/throughput metrics,
  retained failures and telemetry, and labeled comparison plots.
- Qualify all three serving adapters on L40S with pinned Qwen3-0.6B, shared
  requests, real GPU plots and a retained raw run.
- Run the bounded Qwen3.5-35B-A3B 32K/16K stress profile on L40S across all three
  baselines, retaining plots, raw requests, GPU residency and explicit FP8/GGUF
  differences; broader sweeps and numerical model gates remain open.
- Plan one shared `tensor-llm` engine with staged batch-1, batching, cache and
  serving development, retaining all three inference baselines.
- Set a planned L40S stress goal: beat matched FP8 baselines, qualify eight
  resident requests, and pursue 600 output tok/s at concurrency 8 with bounded
  kernel/scheduler search and full-replay acceptance evidence.
- Add ABI 1.3 CUDA BF16 storage, DLPack import/export, capability checks, and
  dtype-specialized Torch kernels while preserving existing numeric dtype IDs.
- Add explicit LLT attention forward/backward, decoupled RoPE, BF16 training
  operators, full-vocabulary loss, FP32 AdamW, and persistent batched KV caches.
- Fix the pinned NVRTC shim's missing integral trait for warp-reduction helpers.
- Fix prepared/native alignment validation to honor frontend arguments and allow
  explicitly mutable manual plans with no outputs. Add L40S qualification tools.

- Plan live progress bars and retained BEAM/agent refinement trajectories from
  shared event logs, including budgets, lineage and independent winner verification.
- Propose shared human/agent profiling reports, visual tuning tools, bounded
  kernel refinement and verified reconstruction from stored kernel recipes.
- Extend the 1.0 demonstration plan to the full pinned llama.cpp quantization
  lineup, with preset/type coverage, packed execution, quality and memory checks.
- Plan the 1.0 Qwen inference and FA/MLA/GDN kernel comparisons, sustained
  1B-token nanoGPT training, image/audio demonstrations, and reproducible evidence.
- Record CPU/GPU composition and SSD offloading as proposed post-1.0 addons.
- Add a proposed 1.0 roadmap with public-contract, compatibility, discovery,
  hardware-validation, onboarding, and release-candidate acceptance milestones.
- Add a schedule discovery guide and bounded CUDA/WebGPU example with
  correctness checks, measured scores, producer profiles, and a deployable winner.
- Add a migration guide and runnable examples for existing TileLang kernels,
  covering factory exports, output mapping, validation, and standalone consumption.
- Add runtime-only installation, a verified elementwise runner, contributor
  onboarding, compatibility guidance, and a release preparation workflow.
- Add `tensor --version` for installation checks and bug reports.
- Add MIT licensing and distribution metadata to the core and optional packages.
- Check source distributions and wheels on Linux and Windows, including a
  compiler-free installed runtime outside the checkout.
- Include the optional Torch executor source in source distributions so native
  builds can be prepared from an archive as well as a checkout.

## Before this changelog

The existing implementation includes CUDA artifact execution, modules and
transport wheels, the optional PyTorch adapter, native WebGPU, manual training,
and bounded LFM2 inference. See the [roadmap](docs/plan/roadmap.md),
[architecture decisions](docs/adr/README.md), and research reports for their
accepted scope and measured evidence. Historical entries have not been assigned
invented release dates or tags.
