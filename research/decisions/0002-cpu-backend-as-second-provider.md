# ADR 0002 — Use the CPU backend as the second provider, not AMD

**Status:** Accepted · 2026-09-29

## Context

Proposal Risk 3: *"Do not stabilize provider ABI until at least two substantially different
hardware implementations exist."* Phase 5 plans to add AMD or Metal as that second
implementation.

The local machine is a Windows box with an **RX 6700 XT (Navi 22, gfx1031, RDNA2)**. Ground
truth §6 established that no supported TileLang path to it exists: ROCm is Linux-only,
Windows wheels exclude it, `gfx1031` is absent from TileLang's issue tracker, and upstream
ROCm CI is disabled for lack of AMD machines.

So the tempting move is to plan Phase 5 around the local GPU. That would mean writing a
backend from scratch and spending the phase on a GPU with no matrix cores, no TMA, no
`cp.async`, and a de-facto EOL software stack.

## Decision

Validate the provider ABI against TileLang's **`cpu` backend** (`c` and `llvm` target
kinds), locally, in Phase 0/1. AMD support stays a Phase 5 decision, evaluated on rented or
institutional hardware — not on this machine.

## Consequences

- The Risk 3 test becomes cheap and immediate: `cuda` + `cpu` are both available today.
- A CPU backend is a genuinely different execution model — no warp hierarchy, no MMA, no
  async copy — so it is a real test of whether the provider boundary is semantic or
  CUDA-shaped. It is not a weak proxy.
- It does not prove the ABI works for ROCm. A second *GPU* provider remains required before
  the ABI is frozen; this ADR only moves the *first* test earlier and cheaper.
- RDNA2 is explicitly off the roadmap. If AMD support is wanted later, the options are a
  rented MI300X or RDNA4 box, or writing and owning a backend.
