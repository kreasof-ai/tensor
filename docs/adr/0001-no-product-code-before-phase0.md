# ADR 0001 — Do not build product packages before Phase 0 concludes

**Status:** Satisfied by [ADR 0009](0009-complete-phase0-with-scoped-provider-and-composition.md) · 2026-09-29

## Context

The proposal describes a CLI, a runtime ABI, a provider model, and a module system. It is
tempting to scaffold `src/tensor` and `src/tensorc` immediately so the work has visible shape.

But the proposal's own Phase 0 question is *how much of `tensorc` already exists*. Ground
truth found that TileLang already ships: a per-pass IR dumper, a backend registration
manifest, six registered backends, a cache, an autotuner, and inspection tooling. A package
layout written now would be a guess about which of those Tensor wraps, owns, or replaces —
and §3.4's central argument is that Tensor must *not* let TIRx's shape become its own.

## Decision

No `src/` directory until experiment E4 (`artifact_shape`) is answered and Phase 0 exit
criteria are met.

## Consequences

- Visible output is `experiments/`, not product code. Correct, but it will feel slow.
- Structural decisions get made once, with evidence, instead of twice.
- Cost: if Phase 0 takes longer than expected, there is no skeleton to show. Accepted.
