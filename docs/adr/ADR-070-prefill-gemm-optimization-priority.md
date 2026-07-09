# ADR-070: Prefill GEMM Optimization Priority (Metal)

**Status**: Proposed
**Date**: 2026-07-08
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own measured
> profiling and A/B results (the **Measured** table). A parallel external prior-art
> survey (llama.cpp `mul_mm` per-family tiling, MLX `steel_gemm`, Indirect Command
> Buffers, Flash-Attention-2-on-Metal, roofline ceilings) has partially landed — its first
> tranche independently corroborated the M5/M6 GEMM results and the Amdahl profile; its
> remaining findings will refine *how* each priority is implemented — and may reorder the
> GEMM *sub-levers* — but not the top-level ranking. That the GEMM operator is attacked first is fixed by
> Amdahl's law over the measured operator profile and does not depend on the survey.
> Sections tagged **[prior, unvalidated on our hardware]** are explicitly held apart so
> the survey folds in without a rewrite.

## Context

Prefill (processing the prompt before the first generated token) dominates
end-to-end latency for agentic workloads (long prompt, short completion), and it
lags llama.cpp and MLX by a wide margin at realistic context lengths. Decode is
already competitive at short context. This ADR does not introduce a new algorithm;
it **ranks the remaining prefill levers** so engineering effort lands on the one the
measured profile says matters, and it records which levers are already closed so they
are not re-tried.

The target is Metal-only (no CUDA), f32/f16 compute (Apple GPUs expose
`simdgroup_matrix`, not a low-precision matrix unit), 32 KiB threadgroup memory on
Apple7+, unified memory. Any numerics change is gated by greedy-token agreement plus a
PPL delta budget (≤0.3 conservative, ≤1.0 aggressive).

## Measured evidence (this engine, this hardware)

Each row is a runtime measurement (M2/M3/M5/M6/M7), an internal profiler result
(M1/M8/M9), or a source-level finding (M4) from this repo, tagged with its most durable
pointer. Rows backed only by internal profiling say so and cite a tracking issue where one
exists rather than a merged artifact. Hardware unless noted: M2 Max, Qwen3.5-0.8B, steady-state.

| # | Finding | Number | Artifact pointer |
|---|---------|--------|------------------|
| M1 | **Prefill is GEMM-bound.** Per-operator profile at seq=512, post-#265. Q4: FFN (gate/up/down) 47% + Q/K/V/O proj 25% = **72% GEMM**; f16: 49% + 31% = **80% GEMM**. Sequence-mixing (attention + GDN) 19–27.5%; `lm_head` <0.2%. **Amdahl:** optimizing only the non-GEMM portion caps total speedup at ~1.4× (Q4, 1/0.72) / ~1.25× (f16, 1/0.80); optimizing the GEMM portion caps at ~3.6× (Q4, 1/0.28) / ~5.0× (f16, 1/0.20). GEMM is the only operator whose measured share is large enough to attack the ~4× gap (M2). | GEMM 72–80%; GEMM-opt cap 3.6–5× | Internal per-command-buffer profiling (`LATTICE_PROFILE_PREFILL`), reported in #268; measured 2026-06-23, post-#265 |
| M2 | **Prefill gap ~4× vs a device-tuned engine**, post-#265. Reference ~5,530 tok/s cold prefill @≈840 ctx vs this engine ~1,360 tok/s = 4.07× (unique prompts per run; cache-hit numbers discarded). Curve peaks @512 (1,415 tok/s), −16% by 2048 (quadratic attention starting to bite). | 4.07× | measured 2026-06-23, post-#265 |
| M3 | **GEMM-family gate fix = 1.62× prefill.** The tiled `simdgroup_float8x8` Q4 GEMM was gated behind `Apple9` and silently disabled on every M1/M2 (which report `Apple8`), falling back to a scalar path. Lowering the gate to `Apple7` gave 874 → 1,415 tok/s @512. Correctness: logit max_abs_diff 7.25e-5, greedy 400/400, Q4 PPL 17.034 unchanged. | 1.62× | PR #265 |
| M4 | **Tile geometry is already textbook.** Source-level comparison found the existing tiled kernel uses BM=64 / BN=32 / BK=32, 4 simdgroups, 128 threads, `simdgroup_float8x8` — **identical** to llama.cpp `kernel_mul_mm_q4_0_f32` and the same tile shape as MLX `steel_gemm`. The remaining structural deltas are dequant staging and writeback, not tile shape. | geometry parity (source read, not a runtime bench) | source read of llama.cpp `kernel_mul_mm_q4_0_f32` + MLX `steel_gemm` vs our kernel, 2026-06 |
| M5 | **Double-buffered threadgroup loads: KILLED, −20.7%.** 1,405 → 1,115 tok/s @512, logits bit-identical. Root cause: threadgroup memory grew 16.25 → 30.25 KiB, crossing the 32 KiB bank, dropping co-resident threadgroups 2→1 (2×→1× occupancy). llama.cpp and MLX do not double-buffer either. **Closed lever.** | −20.7% | commit 206f5fcf9 (`perf(inference): double-buffered Q4 GEMM tiled kernel (NEGATIVE RESULT)`); tracked #268 |
| M6 | **f32→half threadgroup staging tiles: SHIPPED, +29.6%.** Narrowing the *store* of the staging tiles (accumulators stay `simdgroup_float8x8`, dequant math stays f32) dropped threadgroup memory 16.25 → 9.75 KiB toward the ~8–9.75 KiB llama.cpp/MLX use, restoring occupancy. Correctness gate = greedy argmax parity 400/400 (max_abs_diff 0.0214 is f16 rounding, expected — *not* a bit-exactness gate). | +29.6% | PR #270 |
| M7 | **GDN chunked-prefill tiling: KILLED (both stages).** Stage-1 (tiled-C32 simdgroup solve) passed correctness and beat scalar 661.42 ms vs 728.50 ms @4K (1.10×, IQR<8%, n=9, cross-session stable). Stage-2 (tiled-B64) was ~24% *slower* than scalar-C32 (2,048.59 ms vs 1,649.33 ms @4096, interleaved n=9, non-overlapping ranges) — a 10% C32 tiling win cannot overcome scalar-B64's 1.74× penalty. Correctness all-green (oracle 1.16e-10, greedy 0-flip n=1..129, PPL delta 0.001, mutation-sensitive). **The GDN-chunk tiling family is closed for now.** | Stage-2 −24% | #175, commits dcee9c3d5 / fafe6d233 / 9f921fbec, 2026-07-08 |
| M8 | **Prefill TTFT is O(n²) in prompt length**, knee at 1024–2048 tokens: 122 ms @128 → 3,203 ms @4096 (Qwen3.5-0.8B-Q4, M2 Max). Confirms attention cost is not yet dominant below the knee but grows super-linearly above it. | O(n²), knee 1–2K | internal TTFT sweep, 2026-07-06 (M2 Max, Qwen3.5-0.8B-Q4); internal profiling, no merged artifact |
| M9 | **Dispatch count is ~378 per token** at ~5–15 µs fixed overhead each (≈1.9–5.7 ms/token in dispatch overhead alone); an empty command-buffer round-trip measured ~18 µs. Far above the fused per-layer graph a competitive engine issues. | 378 dispatches/token | internal dispatch-count profiling; tracked #172 |
| M10 | **BN=64 Q4 GEMM N-tile widening: KILLED, −3.5%.** Widening the tiled Q4 GEMM's N-tile 32→64 (doubling per-simdgroup `simdgroup_float8x8` accumulators 8→16) regressed prefill −3.5% @512 **and** @1024 (2,310 → 2,229 tok/s @512; interleaved, tight non-overlapping ranges, re-measured independently by a second run agreeing at −3.6%). Threadgroup memory was ample at 14.75 KiB, so this is **not** the M5 occupancy-memory cliff — the cause is register/accumulator pressure. Correctness bit-exact (greedy 706/706, PPL delta 0.0), which together with a consistent throughput drop proves the wider kernel actually ran. Second tile-growth attempt to regress after M5 → the productive direction is shrink/restructure, not grow. **Closed lever.** | −3.5% | internal same-binary runtime-flag A/B (`LATTICE_GEMM_BN64`), 2026-07-08 |

## Prior / analyzed levers — **[prior, unvalidated on our hardware]**

These are designed or externally-evidenced but **not yet measured in this engine**.
They are the search space the in-flight survey informs; none may be treated as decided
until measured here.

- **GEMM structural sub-levers** (surfaced by the external survey; none measured here).
  With tile *geometry* now exhausted on the measured side (M4 parity, M5 double-buffer
  closed, M6 f32→half shipped, M10 BN=64 widening closed), the untried GEMM knobs are
  structural: packed gate/up and packed QKV GEMMs (fewer, larger matmuls per block) and an
  MLX-`steel_gemm`-style static tile-family / split-K router that selects a tile shape per
  GEMM shape. Expected single-to-low-double-digit percent each; each gated on a per-shape
  micro-bench before a build (M10 is the precedent that a plausible GEMM knob can regress).
  **Not measured here.**
- **Indirect Command Buffers (ICB)** to cheaply re-submit a fused, static command graph
  per token. Unknown-for-us: minimum graph-reduction size before ICB overhead pays off,
  resource-tracking-mode requirements, re-encode cost when the step-arguments struct
  changes.
- **Norm+residual fusion** into a single `RMSNorm(x)*w + residual` kernel — estimated
  ~48 dispatches/token removed. Designed, not implemented.
- **Flash-Attention-2-on-Metal.** External evidence (llama.cpp flash-attention beating
  MLX by seconds at ~8.5K context) suggests attention overtakes GEMM well past current
  profiled lengths; our own attention share is only 19–27.5% at seq=512. Crossover point
  **not measured here.**
- **`StorageModePrivate` / `MTLHeap` / `MTLResidencySet`** for read-only weight buffers
  (all ~91 buffers are `StorageModeShared` today). Flagged single-digit-percent, secondary
  to a compute-bound workload. Not measured.
- **`lm_head` two-stage top-k reduction** (local per-tile argmax → global reduce) to cut
  the ~124K-threadgroup naive projection (248K-token vocab) by >100× for greedy/top-k.
  The naive path was a shipped-then-reverted regression (measured problem); the two-stage
  fix is designed, not shipped. Tracked #171.

## Decision

Rank the prefill levers by the measured operator profile and the falsifiability of each
experiment. The measured evidence fixes the *operator*: GEMM is the lever (M1: 72–80% of
prefill; optimizing it caps at 3.6–5×, vs ~1.3× for everything else combined). It does
**not** by itself measure which GEMM *sub-lever* wins — the geometry already matches the
references at BN=32 (M4), and the tile-*geometry* knobs are now measured-exhausted: dequant
staging shipped (M6), double-buffer closed (M5), and N-tile widening closed (M10). That
leaves GEMM's remaining upside in *structural* changes (packing, shape-routing) — prior and
unmeasured, but the highest-ceiling lever — with the measured dispatch overhead (M9) a
smaller but ready parallel win. Therefore:

**Priority 1 — GEMM structural sub-levers (packing / shape-routing) — [prior, unvalidated;
measure before committing].** GEMM is the Amdahl-dominant operator (M1: 72–80% of prefill;
~3.6–5× ceiling), so it stays the first lever — but its tile *geometry* is now exhausted on
the measured side (M4 parity, M5 double-buffer closed, M6 f32→half shipped, M10 BN=64 widening
closed), so the next GEMM experiment is *structural*: packed gate/up and packed QKV GEMMs
(fewer, larger matmuls) and an MLX-`steel_gemm`-style static tile-family / split-K router.
This is the direct successor to the now-closed BN=64 slot. The external survey surfaced these;
none is measured here. Gate each on a per-shape micro-bench before a full build — M10 (BN=64)
is the cautionary precedent that a plausible GEMM knob can regress.
- Success criterion: per-shape micro-bench win before wiring; then ≥+10% prefill @512 with
  greedy parity and no PPL regression.
- Effort: medium.

**Priority 2 — Dispatch-count reduction: norm+residual fusion, then ICB.** M9 puts
1.9–5.7 ms/token in dispatch overhead. Norm+residual fusion (~48 dispatches removed) is
mechanical, numerically inert, and independently measurable; land it first, then wrap the
fused per-layer graph in an ICB. Its prefill share is modest — the ~378-dispatch fixed cost
amortizes across the whole prompt in one forward pass — but it is a low-risk ready win and
helps decode, where the per-token dispatch cost bites directly.
- Success criterion: measured dispatch-count drop **and** per-token latency improvement.
- Effort: low–medium.

**Priority 3 — `lm_head` two-stage top-k reduction (#171).** <1% of prefill, but the
naive path is a known shipped-then-reverted regression and the two-stage design is ready
and self-contained. Falsifiable, low blast radius.

**Deferred — Flash-Attention-2-on-Metal (#126). Gated on a measured expected-value
threshold, not scheduled now.** Attention is 19–27.5% of prefill at seq=512 (M1) and its
cost grows O(n²) (M8), so its value rises with context length. Before any FA2 work, run a
context-length sweep and start when attention's absolute time and a plausible FA2 speedup
exceed the expected value of the next-best GEMM/dispatch experiment — *not* merely when
attention's share crosses GEMM's (a large FA2 win on a 30–40% share can outrank a small
residual GEMM tweak even while GEMM stays the largest bucket). Building it speculatively,
before that sweep, risks large effort against a lever that is sub-dominant at our
workloads' current context lengths.

**Closed by measured evidence — do not revisit without a fundamentally different resource
layout.** Double-buffered threadgroup loads (M5, −20.7%), GDN chunked-prefill tiling
(M7, stage-2 −24%), and BN=64 Q4 GEMM N-tile widening (M10, −3.5%; register/accumulator
pressure, not a memory cliff). The external survey independently reached the same verdicts
on double-buffer (M5) and f32→half staging (M6), and had flagged BN=64's register-pressure
kill condition as a live risk before we measured it.

**Out of scope (a hardware exclusion, not a measured kill).** Tensor-core /
non-`simdgroup_matrix` GEMM paths: the M2 Max / Apple7–8 target exposes no low-precision
matrix unit, so this is excluded by hardware capability, not by an M5/M7-style negative
measurement. Reopen only if a future Apple family (e.g. Apple9 / M3+ cooperative-tensor)
plus a measured A/B justify it.

## Consequences

- Effort concentrates on the measured 72–80% (GEMM) and the measured dispatch overhead,
  not on attention or storage-mode work whose measured share is small today.
- The ~3.6× GEMM-optimization ceiling (M1, Q4) is the honest bound: even an infinitely
  fast GEMM kernel cannot fully close the measured 4.07× gap (M2) at Q4 precision; a
  structural attention change (FA2) is what unlocks the regime past the M8 knee, and it is
  deliberately deferred to when a measured expected-value threshold says it pays.
- **External survey (partially landed)**: its first tranche corroborated M5/M6 and the
  Amdahl profile and surfaced the P1 structural sub-levers. Remaining prior-art findings
  fold into the per-priority implementation notes in place — per-family tile params / split-K
  for P1, ICB gotchas for P2, FA2 SRAM-tiling for the deferred item — without changing the
  ordering. If a later roofline result shows a priority's headroom is near-exhausted, that is
  a recorded decision input, not a silent reversal.

## Follow-ups

- BN=64 measured and closed (M10); the next GEMM experiment is the P1 structural
  sub-levers, each gated on a per-shape micro-bench before a build.
- Fold the remaining external survey findings into the per-priority implementation notes above.
- Re-run the M8 context sweep to fix the FA2 gate threshold in tokens.
