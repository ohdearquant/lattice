# ADR-071: GDN Recurrent State as a First-Class Serveable Object — Priority and Scope

**Status**: Proposed
**Date**: 2026-07-08
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own source and
> measured results (the **Measured / source-verified reality** table) — much of the GDN
> state machinery is already built, and this ADR ranks what is left rather than proposing a
> greenfield subsystem. A parallel external prior-art survey (Q-Mamba state quantization,
> recurrent-state roofline arithmetic, snapshot/branch precision analysis) has landed and is
> folded as the **[prior, unvalidated on our hardware]** section. The survey's roofline bound
> on state-compression speedup is a general inequality this ADR adopts; its *fraction estimate*
> (f = 2.5–7.8%) is an external computation, not a measurement from this repo, and is held
> apart from the pinned per-step numerator this engine actually reports. The top-level ordering
> is fixed by what is already shipped versus what is not, not by the survey.

## Context

Gated-DeltaNet (GDN) layers carry a **recurrent state** (a per-layer causal-conv cache plus an
S-matrix) that, unlike the KV cache, is a fixed-size function of model config rather than of
sequence position. That state is the object a serving stack must observe, snapshot, persist,
quantize, and roll back to support cross-turn caching, agentic branch/rollback, and failure
detection. Issue #481 tracks turning it into a first-class serveable object; a family of child
issues (#462, #486, #491, and the compression proposals) hangs off it.

The purpose of this ADR is to **sequence that family against what the codebase already
contains**. The recon that preceded it found substantially more shipped than the research brief
assumed, so the ranking below leads with wiring and typing the machinery that exists, puts the
one cheap decisive measurement ahead of every compression proposal, and records which
compression theses are closed by arithmetic so they are not re-opened.

Target is Metal-only (no CUDA), f32/f16 compute, unified memory. Any numerics change (state
quantization, chunk-scan rewrite) is gated by greedy-token agreement plus a PPL delta budget
(≤0.3 conservative, ≤1.0 aggressive), and any snapshot/restore path is gated by bit-identical
restored logits at a deterministic boundary.

## Measured / source-verified reality (this engine)

Each row is tagged by how it is known: **runtime-measured** (an A/B or bench on this hardware),
**unit-test-pinned** (a value asserted by a committed test), or **source-read** (a structural
fact read from merged code), with its most durable public pointer (merged PR/commit, tracking
issue, or `file:line`). Internal profiling with no merged artifact says so.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| S1 | GDN state = **20,201,472 bytes (19.27 MiB)** active for Qwen3.5-0.8B (18 GDN layers × per-layer 1,122,304 B = conv 73,728 B + S-matrix 1,048,576 B). Fixed function of config. | unit-test-pinned | `metal_qwen35.rs:28190` (`report.decode.read_bytes == 20_201_472`); shape formula `:27938-27962` |
| S2 | Per **decode step**, logical state traffic = **40,402,944 B (read+write, 20.2 MB each)** and is **context-invariant** — fixed by config, does not grow with context (unlike KV bytes/token). | unit-test-pinned + source-documented | `metal_qwen35.rs:28190-28191`; `examples/bench_gdn_state.rs` docstring |
| S3 | **Cross-turn GDN checkpoint replay is SHIPPED**: `ReplayFromCheckpoint` fully wired for the CLI chat path via a `MetalGdnCheckpoint` ring (`CROSS_TURN_GDN_CHECKPOINT_CAP=3`). | source-verified (merged) | PR #590/#635, commit `53a726b14` (on `main` 2026-07-04); `chat_metal.rs` (8 cross-turn sites) |
| S4 | **HTTP serve path is NOT wired** to cross-turn state cache — `lattice_serve.rs` has **0** cross-turn sites and re-prefills the whole conversation each turn. | source-verified (grep 0 vs 8) | #462 (open); `bin/lattice_serve.rs` |
| S5 | GPU state is **not a first-class type** — two parallel `Vec<Buffer>` (`gdn_gpu_conv_bufs`, `gdn_gpu_s_matrices`) held by index discipline, no wrapping object/metadata. | source-read | #481 (tracking, open); `metal_qwen35.rs` session fields |
| S6 | **Observability instrumentation is partially built** behind the `gdn-state-counters` feature: `GdnStateTrafficShape/Bucket/Report/Counters`, with `record_layer_access`/`record_copy` **wired into the production decode path**, and `examples/bench_gdn_state.rs` is documented as "the #491 bandwidth-share decision gate" harness. | source-verified | `Cargo.toml:36,188`; `metal_qwen35.rs:3579,3586`; `examples/bench_gdn_state.rs` |
| S7 | **#491 is UNRUN**: the per-step state *numerator* (S2) is pinned, but the *fraction* f = state ÷ (state + KV + weight) has **no committed measurement** — it needs the bench run on real hardware to supply the denominator. | source-verified (issue open) | #491 (open) |
| S8 | C32 chunked-scan = 5 MSL kernels (materialize / solve / residual-output / state-update / norm-silu), `GDN_CHUNK_SIZE=32`; already meets ≥5×@4K and is structurally ~5× capped. | source-read | `metal_qwen35.rs:902` (chunk size), `:1205-1209` (C32 pipeline stages) |
| S9 | **Naive tile-growth on the GDN chunk scan regresses**: the tiled-B64 chunk-scan variant measured **−24%** prefill. Tile-growth is the wrong direction for this kernel. | runtime-measured (2026-07-08) | #175 (result recorded there) |
| S10 | In decode, GDN is the **cheapest mixer** (~336 µs/layer); the decode lever is GEMV/MLP weight-bandwidth, and latency is linear in context. Corroborates that fixed-size state traffic is a small decode fraction. | internal profiling, no merged artifact | prior decode profiling; #491 is the public ground-truth gate |

## Prior / unvalidated on our hardware

Folded from the external survey as **data**. Adopted where it is general arithmetic; flagged
where it rests on an unmeasured fraction.

- **First-class exact state object + observability layer** — CONFIRMED-BY-ANALYSIS as
  prerequisites for quantization, routing, branch/rollback, and failure detection. *This repo
  already contains both in partial form (S5 buffers, S6 counters); the survey confirms the
  direction, not a greenfield build.*
- **State compression for single-request decode SPEED** — REFUTED by roofline. With state
  fraction f and compression ratio r, the optimistic upper bound is
  `speedup ≤ 1 / (1 − f + f/r)`, ignoring dequant/kernel/sync overhead. At f = 10% and a perfect
  r = 4, that is ≤ **8.11%**; at the survey's f = 2.5–7.8% it is **1.3–6.2%**. The survey's f is
  *independently derived*, not a repo measurement (S7), but the bound is robust across the whole
  plausible single-digit-to-low-double-digit range, and S2's context-invariance means f *shrinks*
  as context grows. Compression is not a decode-speed lever.
- **Snapshot-only int8 for STORAGE / branch artifacts** — NEEDS-EXPERIMENT. The resident-memory
  arithmetic (`bytes = B_layer · L_gdn · C · (1 + S) · q`) makes int8/int4 material only at high
  concurrency, many retained branches, or persisted snapshots — an admission-control / storage-tier
  question, distinct from decode speed. GDN quantization drift across turns is unmeasured. Kill:
  PPL delta > 0.3 or top-1 disagreement after restore, controlling for FP non-associativity.
- **Online int8/int4/fp8 resident state**, **state-as-GQA-router**, **observer-derived
  confidence** — NEEDS-EXPERIMENT, no direct prior art; each carries a falsifiable kill condition
  (recurrent error compounding; Recall@16 < 0.80 or NDCG@16 not beating locality baselines; not
  beating logit entropy/margin).
- **Low-rank / sketch state compression** — REFUTED first round: no prior art, no decode-bandwidth
  need, kilobyte-scale state. Revisit only if state bytes become an admission-control bottleneck.
- **gdnchunk B=128 (dense chunkwise algebra)** — algebra CONFIRMED (NumPy reference matches the
  serial scan to < 1e-5 at B=64 and B=128); Metal `simdgroup_matrix` speedup NEEDS-EXPERIMENT.
  B=128 needs ~448 KiB of live scratch → **device scratch, not threadgroup**, plus small
  `simdgroup_matrix` tiles. S9 already shows naive tiling loses; this is the one untried variant.

## Decision

Rank the #481 family by *shipped-ness × user-visible value*, with the one cheap decisive
measurement gating all compression work.

1. **P1 — Wire cross-turn state cache into the HTTP serve path (#462).** Highest user-visible
   win at lowest risk: the replay *engine* already shipped (S3) and is reachable from the CLI;
   the serve binary just needs the same wiring (S4). Closes the largest serving regression
   (serve re-prefills every turn). *Build, ready.*
2. **P2 — Promote the GPU state to a first-class typed object (#481 core).** Replace the
   index-disciplined parallel `Vec<Buffer>` (S5) with a typed state object carrying dims, layer
   count, byte size, and precision tag; fold in the traffic-accounting shape that already models
   the byte layout (S6). This is the confirmed prerequisite for observability lifecycle, precision
   experiments, snapshot APIs, and correctness checks. *Build, with/after P1.*
3. **P3 — Run #491 and record the state-bandwidth fraction.** The harness (`bench_gdn_state.rs`)
   and production counters already exist (S6); this is **measure-and-record, not build**. It turns
   the pinned numerator (S2) into the actual fraction f on real hardware and is the **decision gate
   for every state-compression proposal**. Expected outcome: f single-digit and shrinking with
   context → all decode-speed compression closed; only storage/branch compression survives to
   experiment. *Measure, ready today.*
4. **P4 — Formalize observability modes (Off / Canary / Debug) on the P2 object.** The counters
   exist (S6); the API contract that must be added is Off-mode-is-a-compile-time-no-op (kill: if
   Off cannot compile to a no-op, redesign the API; if Canary needs full state CPU readback per
   token, reject it). Prerequisite for downstream precision/routing experiments. *Build, after P2.*

**Deferred — NEEDS-EXPERIMENT (gated on the P2 object + the P3 measurement):**
snapshot-only int8 for **storage/branch artifacts** (the only compression thesis with a plausible
payoff, and it is a storage lever not a speed lever); state-as-GQA-router; observer-derived
confidence. **Agentic snapshot/branch/rollback (#486)** is a deferred *feature* that consumes the
P2 object and additionally needs KV page-table integration; its scope is decided once the object
lands.

**Closed by arithmetic (do not allocate kernel work):** state compression for **decode speed**
(roofline, robust across f); **low-rank/sketch** state compression.

**Closed by measurement:** **naive tile-growth on the GDN chunk scan** (S9, #175 tiled-B64 −24%).
The B=128 device-scratch + `simdgroup_matrix` variant is the *sole* untried chunk-scan sub-lever,
and it is **low priority**: decode profiling (S10) makes GDN the cheapest mixer, so the chunk scan
is a prefill-only concern already meeting ≥5×@4K (S8). Revisit only if prefill attention becomes
the top prefill lever (tied to ADR-070's deferred FA2 track).

## Consequences

- **Positive.** The first shipped item (#462) lands a serving-visible win on already-merged
  machinery. #491 settles the long-running compression question with a near-free measurement
  instead of an external estimate. The typed object (P2) unlocks the whole #481 family behind one
  clean refactor, and the observability contract (P4) is a formalization of code that already runs.
- **Cost.** P1/P2 are real Metal-adjacent work: serve-path wiring plus a type refactor across the
  buffer allocation and checkpoint-blit sites. Bit-exact restore must be preserved (kill: restored
  logits not bit-identical at a deterministic boundary under identical kernels/accumulation order).
- **Risk.** If #491 measures f in the low double digits at short context / high batch, the
  decode-speed-compression closure weakens in that regime — but the roofline still caps it at
  ≤ 8% with r = 4, so the closure holds as a *priority* call even then; only the storage-compression
  experiment could still pay off, and it is already the sole surviving compression track.

## Follow-ups

- #462 (serve re-prefills each turn) → **P1**, wire cross-turn state cache into `lattice_serve`.
- #481 (first-class serveable state object, tracking) → **P2**, typed object; tick the cross-turn
  sub-item as delivered via #590/#635.
- #491 (state-vs-KV-vs-weight bandwidth share) → **P3**, run `bench_gdn_state.rs`, record f, and
  attach the number to this ADR as a measured row.
- #486 (agentic snapshot/branch/rollback) → deferred feature on the P2 object.
- #175 (GDN chunk tiling) → closed-by-measurement (S9); the B=128 device-scratch variant is the
  sole untried sub-lever, low priority, gated on the prefill-attention track.
