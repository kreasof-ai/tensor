# Changelog

User-facing changes belong here. The [research archive](docs/research/README.md)
contains the detailed experimental record. The package version is currently
`0.1.0`; this file does not imply that a public release has been published.

## Unreleased

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
