# ADR-099: Linux CPU Embedding GEMM, Parallel Execution, and Resource Budgets

**Status**: Proposed (2026-10-12)\
**Date**: 2026-10-12\
**Crates**: lattice-inference, lattice-embed\
**Scope**: f32 CPU embedding on Linux x86_64 and aarch64; shared CPU serving admission\
**Requirements**: fast CPU embedding on Linux cloud hosts such as Fly and Hetzner, AMD x86_64
first; use multiple cores where that pays, within CPU and memory limits that protect the host;
keep Apple silicon and edge devices (Linux ARM boards such as the Raspberry Pi 5) supported,
each qualified on its own hardware\
**Related**: ADR-070, ADR-081, ADR-087, proposed ADR-088\
**Source baseline**: `61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0`

## Context

Linux embedding has two distinct performance problems. Bulk ingestion needs high token
throughput from packed batches. Query embedding needs low latency for one text, including
texts too short to activate the current GEMM row split. Increasing batch size addresses
the first problem and does not establish an improvement to the second.

The source baseline already packs BERT batches without padding. Position-wise products
use the sum of real sequence lengths as their row count. A batch containing one text
instead calls the single-sequence `encode` path. Both paths use `matmul_bt` for dense
projections and attention products. Qwen and EmbeddingGemma 2 service calls still process
texts individually. These are source findings, not new measurements.
[BERT source](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/model/bert.rs#L421),
[embedding service](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/embed/src/service/native.rs#L30).

On macOS, `matmul_bt` selects Accelerate. On Linux x86_64, its large full tiles use an
unpacked 4 by 8 AVX2 dot-product kernel, with K tiles of 128 and horizontal reductions
after each K tile. AVX-512 is available to the direct path, including small row
remainders, rather than those full tiles. Rows split over Rayon only when `m >= 32` and
`m*n*k >= 117,440,512`. Linux aarch64 uses tiled NEON without that row split.
[Dispatcher](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/forward/cpu/matmul.rs#L58),
[tile definitions](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/forward/cpu/tiled.rs#L16).

The packed attention loop is serial and retains one head's score matrix at a time.
Its comment records a macOS Accelerate experiment where outer parallelism lost. It
explicitly requires a new measurement on other backends. Parallelizing Linux attention
therefore has a plausible benefit and an explicit memory cost.
[Attention implementation and rationale](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/attention/standard.rs#L445).

### Measurements and their limits

The following measurements were taken before this decision was written; none was rerun
while writing it. The clean Linux host was a Fly.io Machine with 8 dedicated AMD EPYC (Zen 4)
vCPUs, 32 GiB of memory and no SMT. H0 r2 and H1 used baseline `7b2e0878`, which differs from
the source baseline above. H1 changed the x86 `matmul_bt` branch to `gemm` 0.18.2 and
used ABBA ordering. The embedding corpus contained 1,024 synthetic texts, 55,693 tokens,
lengths 1 through 256, with CLS pooling.

| Evidence                                | Recorded result                                                                                                        | What it establishes                                                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| H0 r2, 18 BERT GEMM shapes, one thread  | Lattice 25.9 to 42.5 GFLOP/s; `gemm` 91.2 to 118.1; `matrixmultiply` 82.7 to 112.4. Best-reference ratio median 2.96x. | Substantial kernel headroom on this x86 host.                                                                     |
| H1, BGE-small, one thread, batch 32     | Base 767 / 767 tokens/s; swap 2328 / 2199; reported ratio 2.95x.                                                       | A kernel replacement can improve the complete workload. Swap A/A spread was 5.9%.                                 |
| H1, BGE-small, eight threads, batch 32  | Base 2964 / 3023; swap 5004 / 4837; reported ratio 1.64x.                                                              | Multi-core end-to-end gain, smaller than the single-thread gain.                                                  |
| H1, BGE-small, eight threads, batch 128 | Base 2979 / 3059; swap 5504 / 5561; reported ratio 1.83x.                                                              | Larger packed batches benefit on this workload.                                                                   |
| H0 r2, BGE-small                        | Batch 1 at eight threads: 794 tokens/s; batch 32 at one thread: 776.                                                   | Extra available cores are poorly exploited for this batch-1 corpus. This is not a latency percentile measurement. |
| H0 r2, BGE-small, batch 32              | Thread counts 1, 2, 4, 8: 776, 1369, 2286, 2847 tokens/s.                                                              | Scaling is incomplete before the replacement.                                                                     |
| H1, eight-thread arms                   | Average process use about 4.3 to 4.4 cores, including model load.                                                      | A reason to profile remaining work, not proof that half of steady-state inference is serial.                      |

H0 r2 reports foreign load at most 0.03 cores and steal of 0.17%; H1 reports host steal
of 0.22%. H1's BGE-base comparison included a contaminated base arm and is excluded
from quantitative acceptance. Six-decimal printed agreement is not a numerical gate.
The raw run artifacts and complete build flags are not yet published. Attach those to the
implementation PR before making release performance claims. No Linux aarch64 speedup has been measured in this evidence set.

One later measurement on the same host bears on F3. With rustc 1.94.1,
`std::thread::available_parallelism()` cost about 35.5 µs per call (2,000 calls after
warm-up, three repeats: 35.6, 35.5 and 35.5 µs). At `72817c66`, WordPiece batch
tokenization calls it on every call, batch 1 included. In one ABBA run of the tokenizer
benchmark on that host, the batch-1 median was 50.6 µs with that call and 14.7 µs without
it (in-run A/A spreads 0.9% and 1.2%); the sequential tokenization code was otherwise
identical between the two arms.

## Refutation and priority

**GEMM is the first batch-throughput lever, not a universal bottleneck declaration.**
H1 is an intervention in the complete workload, stronger evidence than a microbenchmark
alone. The attack on prioritizing GEMM for BERT ingestion fails on the clean
BGE-small evidence above. It succeeds against extending that ranking to query latency, long
sequences, concurrent serving, or other architectures without measurement.

The widening per-shape gap with larger M and K is consistent with packing and reuse
problems. It does not isolate them from microkernel register pressure, dispatch, or
packing overhead. Source-level accumulator counts do not prove generated register
spills. Measure operator time and packing separately before claiming a specific cause.
For each candidate operator, compare recoverable wall time `t * (1 - 1/s)`, where `t`
is its measured time and `s` its demonstrated speedup. Change the ordering when an
alternative's end-to-end gain passes the gates below and exceeds GEMM's gain.

The current row threshold is model-dependent. For `H=384, I=1536`, the FFN
projections become eligible at 200 tokens; for `H=768, I=3072`, at 50 tokens.
Thus the batch-1 observation does not mean every single text always uses one core.
Likewise, the H0 rates imply Amdahl serial fractions of approximately 0.134, 0.119
and 0.169 at two, four and eight threads respectively. A single 0.12 to 0.13 serial
fraction does not explain the eight-thread point. These are calculations from the
source threshold and recorded rates, not a measured operator decomposition.

**A pure Rust dependency is not automatically an admissible hot-path dependency.**
The published `gemm` 0.18.2 implementation creates packing storage within GEMM, using
`MemBuffer::new`; its public GEMM entry accepts no caller workspace. A reusable L2 slab
does not eliminate the separate per-call allocation. The package also puts AVX-512
behind its `nightly` feature. These are source findings, not allocation measurements.
**SINGLE-SOURCE upstream code chain,
accessed 2026-10-12:** [gemm 0.18.2 archive](https://static.crates.io/crates/gemm/gemm-0.18.2.crate),
[gemm-common 0.18.2 archive](https://static.crates.io/crates/gemm-common/gemm-common-0.18.2.crate),
[dyn-stack 0.13.0 archive](https://static.crates.io/crates/dyn-stack/dyn-stack-0.13.0.crate).

Version 0.19.0 changes the feature surface to `x86-v4`, so the older nightly limitation
must not be generalized. Its checked public signature still has no workspace parameter
and its packing path retains the same allocation. **SINGLE-SOURCE, accessed 2026-10-12:**
[gemm 0.19.0 archive](https://static.crates.io/crates/gemm/gemm-0.19.0.crate),
[gemm-common 0.19.0 archive](https://static.crates.io/crates/gemm-common/gemm-common-0.19.0.crate).

The repository requires preallocated inference hot paths. Consequently H1 demonstrates
the value of fast packed GEMM, but does not authorize shipping that substitution unchanged.

## Decision

### F1: own a narrow packed f32 core; use the dependencies as references

Select **F1-A**, scoped to the measured f32 work. Integrate a Lattice-owned packed
outer-product implementation with reusable, explicitly charged scratch. Audit and adapt
permitted upstream Rust code where that reduces work; preserve its license and source
provenance. Vendoring transfers maintenance and unsafe review to Lattice. It is not a
correctness proof or a reason to copy an entire multi-dtype GEMM package.

Start with scalar correctness and x86 AVX2/FMA packing and panel reuse. Choose `mc`,
`kc`, `nc` and microtiles by measurement. The proposed 6 by 16 AVX2, 8 by 32 AVX-512,
and 8 by 12 NEON shapes are experiment candidates, not mandated geometry. Add AVX-512
only after an incremental gain over the packed AVX2 implementation passes its gate.
Linux NEON follows the separate F4 qualification. Preserve small-M and scalar fallbacks.

Keep the public `matmul_bt` signature and release-active shape checks. An internal
workspace-aware entry carries the execution policy and leased scratch from model
orchestration. BERT single and packed forwards adopt that entry. A low-level caller
without a workspace keeps the existing allocation-free kernel; the compatibility wrapper
must not allocate secretly. Decoder adoption requires its own path-proven greedy gate.

| Alternative                                   | Disposition and rationale                                                                                                                                                                                                                             |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B: stock `gemm`                               | Reject the inspected 0.18.2 and 0.19.0 integration under the existing allocation rule. Keep 0.18.2 as the measured reference. Current-pool Rayon coupling is usable, but it does not solve scratch ownership.                                         |
| C: stock `matrixmultiply`                     | Retain as an independent timing reference. Its comparable recorded single-thread rates warrant attention, but do not establish superior end-to-end behavior or allocation compliance. Do not enable its independent threading inside the shared pool. |
| D: dependency for `Auto`, owned kernels later | Reject as written: `Auto` is not exempt from preallocation. Reopen only for an audited release exposing reusable workspace, or an explicit amendment permitting bounded hot-path allocation. Either changes this decision's premises.                 |
| Keep the present kernel                       | Keep as fallback and control, not the ingestion performance strategy; H1 provides a material measured opportunity.                                                                                                                                    |

`matrixmultiply` 0.3.11 documents optional threading with its own thread-count control;
its BLIS-style implementation also supplies an independently developed comparison.
**SINGLE-SOURCE, accessed 2026-10-12:**
[matrixmultiply documentation](https://docs.rs/matrixmultiply/latest/matrixmultiply/)
(page identified itself as 0.3.11).

**Cost accepted.** Lattice takes on hand-written AVX2/FMA microkernels and panel packing
first, then AVX-512 and NEON only after their own gates; a manual unsafe review of each;
and their maintenance across toolchains and targets. The alternative declined here is one
audited heap allocation per GEMM call inside stock `gemm`, which needs an explicit
exception to the preallocation rule.

**Reversal threshold.** If two bounded kernel/blocking candidates of the owned AVX2 core
both miss the packed-core gate in the measurement plan (at least 2.0x geometric-mean
single-thread gain over the parent on the original 18 shapes, and at least 80% of the
faster reference's geometric-mean rate), this decision reverses. The reversal adopts
stock `gemm` behind the internal workspace-aware entry with exactly one audited
allocation per call, charged to the memory envelope, through an amendment to F1 that
records the exception. The allocation, memory, compatibility and performance gates are
then rerun against that integration.

**MAY:** reuse audited f32 microkernels, add bounded panel packing, tune shape dispatch.
**MAY NOT:** require nightly, C/C++ builds, an ML runtime, per-GEMM heap allocation,
unbounded packed-weight caches, or a second compute pool. No new leaf-crate dependencies
on inference. **Verify by:** allocation counters inside repeated kernel/layer execution,
manual unsafe review, scalar equivalence, build/feature gates, and the F1 timing gate.

### F2: select one level of parallel work for each phase

Select **F2-c**, a small measured policy, with intra-operation execution as the first
candidate. Policy inputs are architecture, actual matrix dimensions, sequence lengths,
batch size, granted CPU width, and scratch capacity. Version the crossover table and
choose it before executing the phase. Do not learn thresholds from live request timing.

| Workload                        | Mechanism                                                                                                                                                                             | Measurement                                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Packed batch throughput         | Keep fused projections over all real tokens. Use parallel GEMM for large projections; compare coarse sequence/head tasks for attention and row chunks for fused norm/activation work. | Fixed-corpus tokens/s and texts/s at batch 8, 32, 128, plus CPU-seconds/token and peak memory.                   |
| One-text latency                | Fast serial GEMM first. Then compare intra-GEMM widths and parallel heads within the single-sequence path; row chunks only where their dispatch cost pays.                            | Batch 1, one outstanding request, p50/p95/p99 by token length, with queue time excluded and separately reported. |
| Many short independent requests | Concurrent width-one forwards in the same pool, or a measured small width per request.                                                                                                | Throughput and response percentiles under a fixed offered-load trace, including admission and refusal rates.     |

Attention tasks cover independent `(sequence, head)` pairs, with score/softmax/context
computed together. Inner GEMMs run serially while those tasks are parallel. Give each
active task disjoint scratch and output; interleaved head output may require a bounded
staging buffer and a separate scatter. Sharing a mutable score buffer is forbidden.
Never split one sequence into independent encodes: that changes its attention semantics.

Parallel layer normalization splits complete rows, retaining the reduction order inside
each row. Bias/GELU operates on disjoint row chunks. Mean pooling may parallelize across
sequences; parallel reductions within a short vector and parallel CLS copies are not
default work. Embedding lookup, QKV layout copies and normalization remain profile-driven
candidates rather than mandatory new tasks.

Also benchmark **F2-b**: token-balanced sub-batches, each running a complete serial
forward, sharing immutable weights and owning scratch. It may exploit serial regions
but reduces GEMM M, duplicates packing, and increases concurrent working sets. Select it
only in cells where it beats the packed intra-op policy by the registered margin.
It cannot improve the latency of the sole text in a batch. Reject unconditional
intra-op threading and unconditional per-core sub-batches for these respective overheads.

**MAY:** install one winning mode per measured shape class and retain serial execution
below crossover. **MAY NOT:** put parallel GEMM beneath parallel heads/sub-batches,
add automatic cross-request batching delay, or infer batch-1 success from tokens/s at
batch 32. **Verify by:** both public BERT entry paths, length-stratified latency,
packed-throughput A/B, disjoint-output tests, and charged concurrent scratch.

### F3: one process resource domain governs CPU work and memory

Use one shared resource domain for Lattice-owned CPU inference. It owns the Rayon pool,
CPU-width permits, retained-memory accounting and transient-work reservations. Services
and serving routes share it. Library hosts can supply the domain explicitly; compatibility
constructors share a single default domain rather than creating one pool per model.
Place the primitive in inference so embed can consume it without an upward dependency.

**CPU discovery.** Compute the budget once, when the domain is built or its limits are
rechecked, and cache it. Never call `available_parallelism` per request, per batch or
per tokenization: on the measured host one call cost about 35.5 µs, more than twice a
whole batch-1 WordPiece tokenization without it. Start from `available_parallelism`. Rust 1.93's
Linux implementation already combines affinity and detected cgroup quota, rounding quota
down and returning at least one worker. It is an estimate, with discovery limitations;
it does not measure shared-host contention. **SINGLE-SOURCE Rust project, accessed
2026-10-12:** [1.93 implementation](https://raw.githubusercontent.com/rust-lang/rust/1.93.0/library/std/src/sys/thread/unix.rs),
[API limitations](https://doc.rust-lang.org/std/thread/fn.available_parallelism.html).

Resolve cgroup membership and visible controller ancestors using procfs and mount
metadata; do not assume the process is at `/sys/fs/cgroup/`. Retain quota as a rational
core capacity for diagnostics. For known affinity count `A`, finite effective quota
`Q`, and optional positive user cap `U`, use `P = max(1, min(A, floor(Q), U))`, omitting
absent limits. The default is the discovered estimate, tightened by resolved limits.
When discovery is incomplete, report that fact and use an explicit cap, or one worker
if there is no reliable estimate. Overrides may lower capacity; an override above a
known affinity/quota limit must not silently bypass it. A fractional quota below one
still has one worker and remains subject to OS throttling.

**Execution.** Enter the pool for the complete CPU job, including serial phases,
tokenization, packing and pooling. A fixed-size pool bounds workers across concurrent
jobs; `RAYON_NUM_THREADS` alone does not provide admission. Explicit per-job policy must
travel into Rayon closures, not rely on initiating-thread local state. No separate
WordPiece OS-thread fan-out, private GEMM pool or CPU forward on a Tokio worker.
Use an explicitly small serving control runtime, initially one worker, and admit native
work before submitting it. This budgets Lattice compute, not every OS thread or unrelated
code in a host application. The OS quota remains the process CPU-time enforcement.

Rayon documents that nested work uses the pool entered through `install`.
**SINGLE-SOURCE, accessed 2026-10-12:**
[Rayon ThreadPool documentation](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html).

**Admission.** Each job reserves a fixed useful width `w` in `1..=P`, selected by F2;
the sum of admitted widths cannot exceed `P`. Descendants consume that width, never
reacquire it. For identical jobs the CPU-derived concurrency ceiling is `floor(P/w)`;
byte reservations can lower it. CPU chat and embedding share the same permits, while
route caps are optional lower ceilings. The initial HTTP policy refuses excess work
with the existing busy response instead of building an unbounded wait queue. This
replaces fixed independent caps such as four embedding jobs or two CPU-chat jobs with
one accounting rule. Reconcile the changes associated with #1923 and #1924 at integration.
Admission precedes SSE commitment and expensive input materialization. Job-owned permits
survive client disconnection until native work actually exits.

**Memory.** Set a finite process envelope from the user limit and effective visible
cgroup/host ceiling, using the tightest known ceiling. The proposed default leaves 20%
outside Lattice's accounted envelope. Reserve resident model weights, fused QKV copies,
retained packed weights if any, thread stacks, persistent scratch, caches and bounded
runtime overhead first. Reserve request input, tokenization, activations, packing,
attention scratch and outputs before their corresponding allocations. If discovery
cannot establish a finite envelope, the server requires an explicit memory limit;
library callers may supply one. Use checked byte arithmetic and typed admission errors.

Token count alone is insufficient: the estimate also includes individual sequence
lengths and simultaneously active heads. The current packed BERT arrays alone contain
`4*T*(10*H + I)` bytes, before per-sequence scratch, tokenization and outputs (`T` real
tokens, `H` hidden width, `I` FFN width). Active attention score matrices add
`4*sum(s_i*s_i)` bytes over active head tasks, plus their Q/K/context and transpose
buffers. Derive the final estimator from actual buffer lifetimes. The single-sequence
`AttentionBuffers` currently retains additional full-head score storage and must have
its own estimate. Reuse bounded workspaces rather than growing them inside layer loops.
[Packed buffers](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/model/bert.rs#L782),
[single-sequence buffers](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/attention/standard.rs#L41).

Use a bounded input/tokenization reservation first, then refine it after tokenization
and before forward allocation. Reject an oversized single sequence; split oversized
batches only at sequence boundaries, preserving order and documented truncation.
Maintain a ledger for owned memory and a separate pressure check from current usage;
do not count the same allocation as both live RSS and an additional reservation.
Read ancestor headroom as well as the leaf because sibling activity can consume a
shared ancestor's remaining capacity. Such observations cannot reserve sibling memory.

Linux distinguishes CPU bandwidth limits, memory hard limits, current memory use and
memory pressure thresholds. **SINGLE-SOURCE, accessed 2026-10-12:**
[Linux cgroup v2 interfaces](https://docs.kernel.org/admin-guide/cgroup-v2.html).
Support cgroup v1 equivalents where discoverable; test `max`, missing files, malformed
values, nested limits and restricted visibility. A shared cgroup cannot provide a
no-OOM guarantee against unrelated consumers.

**Contention.** Keep worker counts bounded when steal or throttling rises; do not
respond by adding threads. Expose granted width, active work, rejections, queue/service
time, RSS, quota throttling and pressure. Operators may lower the budget on shared vCPUs.
Recheck resource limits between requests on a bounded cadence. A reduced limit stops
new admissions, drains incompatible work and rebuilds a smaller pool without overlapping
old and new pools; increases require explicit reconfiguration. Existing non-preemptible
kernels may finish. A new automatic contention controller is deferred.

Reject three alternatives: independent per-route pools multiply compute capacity;
an environment-only thread limit cannot bound jobs or bytes; unrestricted adaptive
thread growth under contention consumes more of the resource already in shortage.

**MAY:** reduce concurrency or batch tokens to fit admitted resources. **MAY NOT:**
block a pool worker waiting for descendants' permits, release a running job's lease
on waiter cancellation, or silently fall back to unlimited work. **Verify by:** cgroup
integration tests, simultaneous chat/embedding/tokenization load, cancellation floods,
byte-boundary tests and the resource acceptance gate below.

### F4: share the design, qualify Linux aarch64 independently

Use the same packed-core and resource-domain architecture on Linux aarch64. Preserve
NEON/scalar execution until a NEON implementation passes numerical and performance
qualification on a quiet Linux ARM host. A shared CI runner can validate correctness;
it cannot establish a stable performance claim. Apple Accelerate measurements cannot
stand in for Linux NEON measurements.

Reject immediate rollout inferred from x86 speedups. Also reject a separate ARM
scheduler and budget implementation: they duplicate contracts without measured need.
Keep architecture-specific microtiles, packing thresholds and scheduling crossovers.
Qualify a cloud ARM machine and a Raspberry Pi 5 separately before claims about each;
record model, cache topology, frequency, memory and thermal conditions. Pi-specific
limits may select serial execution and smaller token batches.

No dedicated Linux aarch64 timing host, cloud machine or Raspberry Pi 5, is provisioned
for this work at the time of writing. The ARM rows of the measurement plan wait on one.
Until then Linux ARM keeps its current NEON/scalar paths, checked for correctness by the
existing aarch64 CI jobs only.

**MAY:** stage x86 and ARM enablement separately. **MAY NOT:** publish a generic ARM
speedup or enable an unmeasured ARM default. **Verify by:** the same ABBA matrix on
Linux aarch64, using 1, 2, 4 and available higher thread counts, plus both drift gates.

### F5: change `Auto`; retain ADR-088's independent identity contract

All performance enablement in this ADR applies to legacy/`Auto` execution. The existing
`BertCpuKernelPolicy` is metadata only; no constructor accepts a pinned profile at the
source baseline. Naming an external crate version, or owning a new kernel, does not
implement pinned execution.
[Policy seam](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/inference/src/forward/cpu/bert_policy.rs#L71).

Proposed ADR-088 requires versioned Lattice-owned output-affecting kernels and
`CanonicalPerItemV1`: the same text must be bit-identical alone and in every admitted
batch position. Preserve that boundary. A future pinned adoption must bind kernel and
dispatch revisions, build/dependency identity, FP policy, packing/reduction order and
any output-affecting thread policy. If permitted worker counts change arithmetic, fix
that order or distinguish the profiles. Do not promise bit identity across different
ISAs. Shared scheduling may govern resources without selecting a different numeric
profile. Ambient resource discovery happens before constructing explicit prepared
budgets; it must not bypass ADR-088's no-ambient-input preparation rule.

Reject admitting dependency dispatch merely by labeling it in a descriptor; that
requires an ADR-088 amendment. Reject postponing `Auto` improvements until prepared
mode is complete. **MAY:** reuse an audited owned kernel later. **MAY NOT:** route
prepared calls through packed batching or claim the cosine drift gate proves exact
identity. **Verify by:** policy-route exclusion now; exact per-item, position,
permutation and restart goldens before any future pinned adoption.

### F6: keep precision and decoder batching separate

This decision changes f32 execution and scheduling. Defer bf16/int8 compute, quantized
weight formats and retrieval-quality tradeoffs to a separate decision with retrieval
evaluation. Defer fused Qwen/EmbeddingGemma 2 batching to a separate model-aware change.
Their individual calls still share CPU admission; any common-kernel changes they reach
remain subject to decoder/embedding numerical gates.

Reject combining precision changes with GEMM tuning because it obscures both quality
and speed attribution. Reject combining decoder batching with BERT scheduling because
the state and masking contracts differ. **MAY:** record separate future experiments.
**MAY NOT:** count their projected benefits toward this ADR's acceptance. **Verify by:**
an f32-only diff and a call-path inventory identifying all consumers of changed kernels.

## Acceptance criteria

All criteria apply before enabling the affected path by default.

1. **Numerics.** Both existing x86_64 and aarch64 embedding-drift jobs succeed against
   unchanged frozen references, with maximum per-vector `1-cos < 1e-3`. Missing models,
   missing fixtures, non-finite outputs and absent execution markers fail. Preserve the
   existing quartet, and add BGE-base plus any newly enabled model as supplemental
   coverage. Exercise serial/parallel policies, mixed lengths, batch positions and odd
   dimensions. Retain packed-versus-sequential checks of cosine at least 0.99999 and
   maximum absolute difference at most 1e-4. Do not regenerate references to pass a
   kernel change. [Gate wiring](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/.github/workflows/e2e-parity.yml#L1474),
   [threshold](https://github.com/ohdearquant/lattice/blob/61f5f25e89f5cff14b904f2cc62a4eb1377ed7f0/crates/embed/src/drift.rs#L10).
2. **Decoder coverage.** Inventory decoder callers through changed `matmul_bt` routes.
   Run the existing greedy-token parity contract, including the first three HF tokens
   and the existing two-token long-prefill exception, on Linux for each enabled ISA.
   Require counters proving the candidate path ran on relevant prefill shapes, and
   compare complete fixed greedy traces with the parent for affected decoder fixtures.
   Preserve macOS parity. A macOS Accelerate-only pass does not validate a Linux kernel;
   a model whose CPU path never reaches it is not a substitute for one that does.
3. **Kernel contract.** Preserve overflow-first validation, oversized-scratch acceptance,
   zero-dimension behavior and output overwrite semantics. Test tails, `m=1`, short K,
   alignment, and scalar versus each reachable ISA with scale-aware error bounds.
   Any upstream adapter must review stride order and coefficient conventions explicitly.
   Require manual SIMD/raw-pointer review and the repository safety inventory update.
4. **Allocation and resources.** Zero new heap allocations within provisioned GEMM and
   parallel layer/head loops, including dependency code; output/request setup is measured
   separately. No uncharged scratch growth, CPU width above the domain, deadlocks,
   permit leaks or early release on cancellation. Under the registered resource stress
   workloads, zero OOM kills and no accounted-byte overshoot. Test quota 0.5, 1, 2 and 8
   core equivalents, cpuset restrictions and mixed requests at offered concurrency
   1, 2, 4, 8 and 16. Insufficient model memory must produce a typed refusal.
5. **Compatibility.** MSRV 1.93 and the pinned development toolchain pass required tests,
   clippy, formatting, packaging and target checks on dedicated machines. Preserve
   wasm32 embedding and SIMD128 parity; Linux packed kernels and native resource discovery
   remain target-gated. Preserve macOS Accelerate dispatch and its performance. Audit the
   selected source/dependency closure, license notices and unsafe changes before release.
6. **Performance disposition.** Every implementation PR carries the repository's required
   bench-compare disposition plus paired end-to-end evidence below. Search all declared
   benchmark and timed binary targets for relevant coverage. Unrelated default benches
   cannot certify this change, and dependency/build-input changes cannot claim a runtime
   reachability waiver. Attach test and benchmark output, run conditions and raw arms.

## Measurement plan and registered kill points

These are **proposed thresholds fixed before the runs they judge**, not measured
results. Changing one requires a recorded rationale before a new run; a failed run
cannot be retroactively accepted by moving the threshold.

Run x86_64 builds and timing on the H0/H1 host class: a Fly.io Machine with 8 dedicated
AMD EPYC (Zen 4) vCPUs, 32 GiB of memory and no SMT. A Hetzner AMD host may be added as
a second x86_64 host; report each host's results separately and never pool them. Linux
aarch64 rows wait on a dedicated host (F4). Use a dedicated Apple silicon host for macOS
regression checks. Run no heavy work on a
developer laptop. Build both immutable revisions and download inputs before the quiet
window. Record CPU/ISA, topology, OS, toolchain, target flags, Cargo features, lockfile,
pool width, affinity, cgroup limits, corpus/model hashes, cache state and active policy.

For acceptance, collect at least five independent ABBA blocks for each primary cell:
`base, candidate, candidate, base`. Warm each arm identically, run identical fixed work,
and report load time separately from steady-state encoding. Show raw values and both
in-run A/A spreads, defined as `abs(x1-x2)/mean(x1,x2)`. Reject a block if either spread
exceeds 5%, mean foreign load exceeds 0.05 cores, any foreign-load sample exceeds 0.10
cores, or mean steal exceeds 1%. Sample within every arm, not only at boundaries;
require at least ten samples. Missing telemetry makes a run inconclusive. Retain all
rejected blocks with reasons. Three invalid attempts stop that cell until conditions
are corrected. The historical H1 5.9% arm remains motivation, not a pass under this bar.

Use one block-level candidate/base ratio per block, geometric aggregation, and a
95% interval resampled over blocks. For throughput, the interval's lower bound must
meet the gain threshold. For latency, its upper bound must meet the reduction
threshold. A wide interval is inconclusive and cannot enable the candidate. Do not
treat per-token samples from one run as independent host replicates.

| Experiment, in order              | Primary cells and measurements                                                                                                                                                                            | Promotion / kill point                                                                                                                                                                                                                                                                                |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reproduce and profile             | Rebase the isolated swap experiment onto the source baseline; BGE-small/base, batch 1/32/128; tokenize, allocation, packing, projections, attention, norm/GELU, pooling, queue and load times separately. | Reorder work if another isolated lever yields a qualifying end-to-end gain exceeding the GEMM intervention by at least 5 percentage points. Without a valid current-baseline reproduction, do not publish the historical ratios as current performance.                                               |
| Packed f32 core                   | The 18 measured BERT shapes, plus short M 1/2/4/8/16/32/64, tails, fused QKV and attention shapes; threads 1/2/4/8. Compare owned core with parent and both Rust references, packing included.            | Require at least 2.0x geometric-mean single-thread gain over parent on the original 18 shapes and at least 80% of the faster reference's geometric-mean rate. After two bounded kernel/blocking candidates miss, stop expanding the ISA implementation; F1 reverses as its reversal threshold states. |
| BERT ingestion enablement         | Warm, cache-miss BGE-small and BGE-base, batch 32 and 128, widths 1 and 8; original synthetic corpus plus a frozen representative document corpus.                                                        | Each primary model/corpus at eight threads must gain at least 20% throughput; no supported primary cell may regress more than 5%. The contaminated BGE-base H1 result supplies no waiver.                                                                                                             |
| One-text latency                  | Exactly one outstanding text, lengths 8/32/64/128/256/512 where admitted, both models, widths 1/2/4/8; model resident, cache disabled.                                                                    | Enable multi-core latency policy only if p50 and p95 improve at least 10% at both 128 and 256 tokens versus the fast serial core; p99 and short-text p50/p95 must not regress more than 5%. Otherwise keep serial in those cells and report the latency objective still open.                         |
| Non-GEMM threading                | Isolate attention, then fused norm/activation, then their composition against the accepted fast core. Measure batch 32/128 and the latency cells separately.                                              | Retain a batch policy only for at least 10% complete-encode throughput gain, or a latency policy meeting the preceding latency gate. No microbenchmark-only promotion.                                                                                                                                |
| Sub-batches versus intra-op       | Same total texts/tokens, weights, CPU width and memory envelope; token-balanced serial sub-batches versus packed intra-op.                                                                                | Select sub-batches only for at least 10% throughput gain over intra-op, within the envelope and without more than 5% p95 regression in the matched service-load test. Otherwise keep packed intra-op.                                                                                                 |
| AVX-512 refinement                | Compare against accepted packed AVX2 on the same AMD host, including long sustained runs and batch 1.                                                                                                     | Require at least 10% additional end-to-end throughput and at most 5% regression elsewhere; otherwise retain AVX2.                                                                                                                                                                                     |
| Linux ARM qualification           | Repeat kernel, ingestion, latency and resource matrices on a quiet cloud ARM host; qualify Pi 5 separately with sustained thermal reporting.                                                              | Apply the same relevant gain and regression thresholds. No qualifying evidence means no new default or speed claim for that platform.                                                                                                                                                                 |
| Resource and service policy       | Finite memory envelopes at 1.05x/1.25x/2x measured resident-model charge; fractional quotas; cancellations; mixed chat/embedding; declared foreign-load stress runs.                                      | Zero accounting violations/OOMs/deadlocks. On quiet underloaded cells at most 5% p95 regression. At fixed successful throughput under overload, p95 must be no worse than the independently capped baseline; disclose rejected requests and completed work so refusal cannot masquerade as a speedup. |
| Footprint and protected platforms | Same clean build inputs and target profiles; stripped binary size, clean compile time, peak RSS; macOS and wasm checks.                                                                                   | Hold for review if binary growth exceeds the larger of 10% or 2 MiB, or clean compile time grows over 20%. Any wasm correctness failure blocks; a reproducible macOS throughput/latency regression over 5% blocks default rollout.                                                                    |

The representative corpus, model revisions, raw 18-shape list, latency request counts
and service offered-load traces must be committed to the measurement manifest before
running. Use at least 1,000 measured batch-1 requests per length/cell for tail reporting.
Host-contention stress is a separate experiment with declared load, never certified as
a quiet performance run. Compare scheduling candidates within the same CPU and memory
budget. If a resource envelope cannot load a model safely, report a refusal test rather
than assigning it zero throughput in the speed comparison.

## Consequences

The recorded evidence supports improving GEMM first for bulk BERT work. The allocation
contract makes the quickest measured substitution a reference rather than the production
implementation. Owning a narrow packed core costs more review and maintenance, but gives
scratch and dispatch ownership without weakening the resource rules. That ownership
does not automatically make it suitable for prepared identity.

Parallel work becomes a measured choice with a memory charge. A throughput win can ship
while the one-text latency objective remains open; the release must say which improved.
Shared resource limits bound admitted work but cannot create guaranteed CPU time on
shared vCPUs or eliminate races with unrelated consumers of a cgroup.

## Ordered follow-up issues

1. **Publish reproducible Linux CPU baseline and operator profile**: freeze artifacts,
   feature flags, shape/corpus manifests and ABBA thresholds; verify the isolated swap.
2. **Add shared CPU execution and memory admission domain**: unify tokenization, CPU
   chat and embeddings, including cancellation-safe leases and cgroup discovery.
3. **Add reusable f32 GEMM workspace and audited packed AVX2 core**: preserve the public
   boundary, scalar oracle, small-M fallback, provenance and allocation counters.
4. **Qualify packed GEMM on BERT single and batch paths**: pass drift, affected decoder
   parity, wasm/macOS protection and end-to-end ingestion gates before default dispatch.
5. **Reduce one-text BERT latency with measured intra-op policy**: qualify short-M GEMM
   widths and bounded head parallelism on the actual single-sequence entry point.
6. **Parallelize remaining packed BERT phases within the budget**: test attention and
   fused row operations, then compare serial sub-batches under the same resource cap.
7. **Qualify packed NEON on Linux cloud ARM and Pi 5**: produce separate correctness,
   throughput, latency, memory and sustained thermal evidence before each rollout.
8. **Evaluate incremental AMD AVX-512 packed GEMM**: proceed only after AVX2 acceptance
   and retain it only if the incremental threshold is met.
9. **Bind future pinned CPU kernels to ADR-088 identities**: keep canonical per-item
   execution and require exact invariance before prepared adoption.
10. **Evaluate reduced-precision embedding compute separately**: require retrieval
    quality evidence and a new precision decision.
11. **Design model-aware decoder embedding batching separately**: address per-text
    Qwen/EmbeddingGemma 2 execution without folding projected gains into this work.

## What acceptance commits to

Accepting this ADR accepts the owned-kernel maintenance commitment, the F1 reversal
threshold and the proposed measurement thresholds, including the 20% memory reserve.
Enabling a dependency immediately under a bounded allocation exception is the declined
alternative; it becomes the decision only through the F1 reversal, with its gates rerun.

This draft does not accept ADR-070 or ADR-088, weaken their existing gates, certify
SIMD correctness, or record completed implementation work. ADR-081 supplies a model for
measurement-based priority, not transferable Metal performance results. ADR-099 was
unallocated in `docs/adr/` at the source baseline; recheck numbering when integrating.
