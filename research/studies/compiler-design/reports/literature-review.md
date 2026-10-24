# Compiler design literature for bounded dynamic AOT execution

Review date: **2026-10-18**. Status: selective first-pass review and proposed
experiments. Primary publisher, author, project and vendor sources are linked
below. Abstracts and selected relevant design sections were inspected. This is
not an exhaustive survey, implementation or replication of published speedups.

The review addresses Tensor's [dynamic-shape design](../../../plans/dynamic-shapes-and-batch-invariance.md)
and [compilation-latency plan](../../../plans/compilation-latency.md), including
AKAS benchmarking and validation memory. It complements the historical
[ecosystem survey](../../ecosystem/reports/ecosystem-and-precedents.md), whose
version/package observations belong to its measured revision.

## Findings that affect the proposed design

1. **Bounded AOT profiles have a practical precedent.** Distinguish legal bounds,
   optimization workload and allocation capacity. One broad interval does not
   establish good performance throughout it.
2. **Evaluate shared search before fixing independent per-region tuning.** Regions
   can be the dispatch result of shared kernel exploration, avoiding repeated
   search, compilation and reference work.
3. **Ragged execution needs a costed metadata path.** Consider packed activations,
   offsets, request maps, local tile padding and preparation/upload cost together.
4. **Compilation is an incremental dependency problem.** Separate deciding whether
   work is stale from scheduling work that actually needs execution.
5. **Numerical contracts constrain these optimizations.** Dynamic scheduling
   research does not automatically establish bitwise invariance of its variants.

These are recommendations inferred for Tensor/AKAS from the sources below.
They do not imply adopting another compiler or replacing TileLang/TVM.

## First reading set

| Priority | Primary source | Design question |
|---|---|---|
| 1 | [DietCode: Automatic Optimization for Dynamic Tensor Programs, MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html) | Can a shape-generic search space and shared cost model optimize many shapes with less repeated tuning? |
| 2 | [MiKPoly: Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization, ASPLOS 2024](https://cuihuimin.github.io/papers/mikasplos.pdf) | What can be tuned once offline and selected/parameterized at runtime? |
| 3 | [TensorRT dynamic-shape optimization profiles, official documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html) | How should minimum, optimization and maximum shapes form the producer/runtime contract? |
| 4 | [BladeDISC: Optimizing Dynamic Shape Machine Learning Workloads via Compiler Approach, PACMMOD 2023](https://jamesthez.github.io/files/bladedisc-sigmod24.pdf) | Which shape relationships survive without concrete values and enable fusion/code generation? |
| 5 | [The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding, MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/file/afe8a4577080504b8bec07bbe4b2b9cc-Paper.pdf) | How do ragged layouts, local padding and metadata preparation interact with efficient kernels? |
| 6 | [Build Systems à la Carte, ICFP 2018](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/build-systems-final.pdf) | How should dependency discovery, cache validity and task scheduling be separated? |

For the immediate benchmarking complaint, read item 6 alongside items 1–2.
For bounded shapes and padding, read items 1–5. TensorRT is an implementation
precedent rather than a historical paper; version-qualify its checked-on-this-date
documentation during implementation.

### DietCode: share optimization across shapes

The paper constructs a shape-generic search space and shared cost model, using
tile-sized micro-kernels as building blocks. Its design includes adaptation
costs such as occupancy and padding and produces shape-to-kernel dispatch.
The [authors' implementation RFC](https://discuss.tvm.apache.org/t/rfc-dietcode-an-auto-scheduler-for-dynamic-tensor-programs/12757)
and [artifact](https://github.com/UofT-EcoSystem/DietCode) add implementation context.

**Application hypothesis:** reuse Tensor's authored schedule families as a shared
candidate bank across the domain. Reuse compatible compilation and measurements;
derive regions from the winners across shapes. Begin with measured tables and
simple ranking before adding a learned model. Region assumptions that alter
emitted code still need separate variants and identities.

**Limit:** this is not a ready-made optimizer for Tensor's pinned frontend,
quantized kernels or invariant mode. Sparse measurements need held-out checks
and explicit coverage.

### MiKPoly: move reusable optimization offline

The design separates offline micro-kernel generation from runtime selection and
parameterization. Section 4 describes binaries with runtime addresses and loop
counts; runtime composition does not mean rerunning the full device compiler.
It also exposes potentially substantial offline tuning cost.

**Application hypothesis:** compare a small reusable binary family plus dynamic
loop/grid parameters against region-specific generated kernels. Keep runtime
selection bounded and measured. Explore producer reuse across models sharing a
semantic operator/layout/target, so equivalent tuning is not charged per model.

**Limit:** composition may change launch count, data movement and accumulation.
Not every technique fits one Tensor executable or its numerical contract. The
paper's operator/hardware evaluation is not AKAS serving evidence.

### TensorRT: separate valid ranges from optimization shapes

Profiles declare minimum, optimization and maximum dimensions. Profiles can
overlap; the runtime selects one before setting concrete input dimensions. The
documentation also notes that profile dimensions affect memory and some tactics
require identical minimum/optimization/maximum values.

**Application hypothesis:** keep explicit domains and representative optimization
points/distributions, with singleton regions where measured useful. Report memory
and per-region performance instead of extrapolating from one optimum shape.

**Limit:** Tensor needs its own operation/shape and artifact contracts. This
profile model requires neither TensorRT as a dependency nor backend parity.

### BladeDISC: carry shape relationships through compilation

Sections 3–4 distinguish shape relationships, including dimension equality and
reshape element-count relationships, from properties such as divisibility and
ranges. Symbolic information is preserved across passes for dynamic optimization
and fusion.

**Application hypothesis:** AKAS graph transforms retain equality/product
relationships instead of resolving every shape to concrete integers. Tensor
lowering consumes validated applicable assumptions. Start with a small expression
and constraint subset, invalidating facts when rewrites change them.

**Limit:** a full graph compiler's fusion architecture is broader than Tensor's
kernel compiler. AKAS owns graph propagation/rewrites; arbitrary fusion and a new
general solver are not prerequisites for the first dynamic batch.

### CoRa: optimize ragged metadata and useful local padding

CoRa supports ragged operators with layout/scheduling transformations and partial
padding. Sections 7.3–7.4 discuss parallelism and metadata/copy overhead; appendix
D.7 identifies redundant per-operator preparation that can be shared in a graph.

**Application hypothesis:** compute sequence offsets and reusable request maps
once per batch or compatible plan and reuse them across layers. Compare tile and
padding choices using total preparation, transfer and execution cost. Small local
tile tails can be worthwhile when they improve parallelism; minimizing padded
scalar operations alone is insufficient.

**Limit:** transformer-encoder experiments do not establish autoregressive serving
with KV/GDN state. Preserve request isolation and separate per-batch metadata
preparation from compiler latency in reports.

### Build Systems à la Carte: separate validity from scheduling

The paper separates task scheduling from rebuilding decisions and compares
dependency/reuse strategies, including incremental and content-based reuse.

**Application hypothesis:** represent export, lowering, device compilation,
qualification and packaging as dependency-tracked tasks with separate identities.
Scheduling enforces CPU/memory limits; reuse decides which tasks are already
valid. Preserve source/header/toolchain/numerical-policy inputs.

**Limit:** Python exports may use undeclared files/environment. Early reuse needs
a sufficient declared input closure; retain export/IR verification otherwise.
Timings are noisy and hardware-scoped, so tuning evidence needs its own validity
policy rather than only an executable-cache key.

## Older foundations and supporting reading

| Source | Transferable idea and boundary |
|---|---|
| [Cousot and Cousot, Abstract Interpretation, POPL 1977](https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml) | Sound analysis over approximated values. Interval/modular facts must justify transformations; a bounded variable is not a concrete constant. This is conceptual groundwork, not an analyzer rewrite request. |
| [Jones, Gomard and Sestoft, Partial Evaluation and Automatic Program Generation, 1993](https://raspi.itu.dk/people/sestoft/pebook/preface.html) | Separate specialization-time knowledge from runtime values. Keep model/tile constants distinct from batch/token extents. Range analysis and partial evaluation are related, not interchangeable. |
| [Halide, PLDI 2013](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) | Separate algorithm semantics from schedules. Explore tiling/storage/execution choices while retaining the operation contract; floating-point schedule changes still require equivalence checks. |
| [Ansor, OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng) | Hierarchical search, cost-guided exploration and task scheduling. Allocate a fixed tuning budget across important operations before adding a new large search engine. |
| [Efficient Reproducible Floating Point Summation and BLAS, Berkeley 2016](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.html) | Order-independent reproducible accumulation is an alternative to fixed canonical reductions, with different arithmetic/performance costs. Do not make it the default by analogy. |
| [Nimble, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/5b47430e24a5a1f9fe21f0e8eb814131-Abstract.html) | Dynamic types, optimizations and lightweight execution. Useful compiler/runtime boundary reading; its broader dynamic-control/data-structure scope does not imply AKAS needs a VM. |
| [BladeDISC++, 2024 preprint](https://arxiv.org/abs/2412.16985) | Symbolic-shape scheduling/rematerialization for memory. Relevant to activation planning, but it does not diagnose NumPy leaks or replace validation-cache budgets. |
| [Finding and Understanding Bugs in C Compilers, PLDI 2011](https://users.cs.utah.edu/~regehr/papers/pldi11-preprint.pdf) | Generate legal tests and compare independent implementations. Adapt to constrained tensor/shape cases with explicit floating-point policies; differential agreement alone is not proof. |

Retain the DeepSeek and Thinking Machines sources in the
[numerical-policy plan](../../../plans/dynamic-shapes-and-batch-invariance.md).
These compiler sources do not supersede operation-level invariance contracts.

## Proposed experiments before freezing architecture

| Experiment | Alternatives | Evidence needed |
|---|---|---|
| LIT-01 — Shared dynamic search | Independent region searches versus a shared schedule bank and joint measurements | Equal producer budgets; unique compilations, references, wall time and memory; per-region/weighted execution on held-out shapes |
| LIT-02 — Constraint propagation | Runtime extents alone versus guarded interval/modular/equality facts | IR/source changes, actual live work, boundary correctness, code size, build cost and execution latency; explicit unsupported transformations |
| LIT-03 — Ragged preparation reuse | Per-operator versus shared batch/plan metadata; several local tile choices | Packing/preparation/upload/kernel time, metadata bytes, state isolation and mixed-length throughput |
| LIT-04 — Incremental compilation DAG | Current path versus separately cached export/lowering/image/artifact tasks | Cold/warm/new-output/single-change turnaround, invalidation correctness, actual work skipped and peak memory |
| LIT-05 — Numerical scheduling freedom | Canonical invariant kernels versus dispatch/composition variants | Exact output/state across positions, region boundaries and advertised chunk/phase scopes; target/toolchain qualification |

Begin LIT-04 with a stage trace of one representative AKAS benchmark invocation.
That is still needed to identify the actual bottleneck. LIT-01/02 can use a
projection and attention/reduction case before model-wide search. All experiments
are proposed; none was run as part of this review.

Keep compile time, memory, artifact count/size and validation budgets alongside
runtime performance. Report measured tradeoffs for interactive builds and
deliberate deep tuning. Published speedups are not Tensor release targets, and
independent correctness checks remain necessary.

## Decisions supported now and still open

Retain bounded symbolic execution, static hardware tiles, packed request metadata,
guarded dispatch, qualified numerical modes and compiler-free consumers as the
working design. Add shared search/compilation reuse as an alternative to separate
region searches, and include metadata sharing in packed prefill.

Leave cost models, partition boundaries/counts, solver strength and composition
mechanisms open until experiments establish their value. Instrument the current
pipeline first: the literature does not establish Python as the main latency
source or justify a replacement compiler stack.
