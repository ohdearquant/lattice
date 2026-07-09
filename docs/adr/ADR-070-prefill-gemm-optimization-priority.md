# ADR-070: Prefill GEMM Optimization Priority (Metal)

**Status**: Proposed
**Date**: 2026-07-08
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own measured
> profiling and A/B results (the **Measured** table). A parallel external prior-art
> survey (llama.cpp `mul_mm` per-family tiling, MLX `steel_gemm`, Indirect Command
> Buffers, Flash-Attention-2-on-Metal, roofline ceilings) is in flight; its findings
> will refine the *implementation* of each priority below, not the *ordering*. The
> ordering is fixed by Amdahl's law over the measured operator profile and does not
> depend on the survey. Sections tagged **[prior, unvalidated on our hardware]** are
> explicitly held apart so the survey folds in without a rewrite.

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

All rows below are measurements taken in this repo, with a durable artifact pointer.
Hardware unless noted: M2 Max, Qwen3.5-0.8B, steady-state.

| # | Finding | Number | Artifact pointer |
|---|---------|--------|------------------|
| M1 | **Prefill is GEMM-bound.** Per-operator profile at seq=512, post-#265. Q4: FFN (gate/up/down) 47% + Q/K/V/O proj 25% = **72% GEMM**; f16: 49% + 31% = **80% GEMM**. Sequence-mixing (attention + GDN) 19–27.5%; `lm_head` <0.2%. **Amdahl: an infinitely fast non-GEMM path caps total speedup at ~3.6×** — the GEMM kernel *is* the remaining gap. | GEMM 72–80% → 3.6× cap | Profiler `perf/prefill-profile` branch (per-command-buffer instrumentation, `LATTICE_PROFILE_PREFILL`); measured 2026-06-23; tracked #268 |
| M2 | **Prefill gap ~4× vs a device-tuned engine**, post-#265. Reference ~5,530 tok/s cold prefill @≈840 ctx vs this engine ~1,360 tok/s = 4.07× (unique prompts per run; cache-hit numbers discarded). Curve peaks @512 (1,415 tok/s), −16% by 2048 (quadratic attention starting to bite). | 4.07× | measured 2026-06-23, post-#265 |
| M3 | **GEMM-family gate fix = 1.62× prefill.** The tiled `simdgroup_float8x8` Q4 GEMM was gated behind `Apple9` and silently disabled on every M1/M2 (which report `Apple8`), falling back to a scalar path. Lowering the gate to `Apple7` gave 874 → 1,415 tok/s @512. Correctness: logit max_abs_diff 7.25e-5, greedy 400/400, Q4 PPL 17.034 unchanged. | 1.62× | PR #265 |
| M4 | **Tile geometry is already textbook.** Source-level comparison found the existing tiled kernel uses BM=64 / BN=32 / BK=32, 4 simdgroups, 128 threads, `simdgroup_float8x8` — **identical** to llama.cpp `kernel_mul_mm_q4_0_f32` and the same tile shape as MLX `steel_gemm`. The remaining structural deltas are dequant staging and writeback, not tile shape. | geometry parity | source comparison, 2026-06 |
| M5 | **Double-buffered threadgroup loads: KILLED, −20.7%.** 1,405 → 1,115 tok/s @512, logits bit-identical. Root cause: threadgroup memory grew 16.25 → 30.25 KiB, crossing the 32 KiB bank, dropping co-resident threadgroups 2→1 (2×→1× occupancy). llama.cpp and MLX do not double-buffer either. **Closed lever.** | −20.7% | measured 2026-06-24 |
| M6 | **f32→half threadgroup staging tiles: SHIPPED, +29.6%.** Narrowing the *store* of the staging tiles (accumulators stay `simdgroup_float8x8`, dequant math stays f32) dropped threadgroup memory 16.25 → 9.75 KiB toward the ~8–9.75 KiB llama.cpp/MLX use, restoring occupancy. Correctness gate = greedy argmax parity 400/400 (max_abs_diff 0.0214 is f16 rounding, expected — *not* a bit-exactness gate). | +29.6% | PR #270 |
| M7 | **GDN chunked-prefill tiling: KILLED (both stages).** Stage-1 (tiled-C32 simdgroup solve) passed correctness and beat scalar 661.42 ms vs 728.50 ms @4K (1.10×, IQR<8%, n=9, cross-session stable). Stage-2 (tiled-B64) was ~24% *slower* than scalar-C32 (2,048.59 ms vs 1,649.33 ms @4096, interleaved n=9, non-overlapping ranges) — a 10% C32 tiling win cannot overcome scalar-B64's 1.74× penalty. Correctness all-green (oracle 1.16e-10, greedy 0-flip n=1..129, PPL delta 0.001, mutation-sensitive). **The GDN-chunk tiling family is closed for now.** | Stage-2 −24% | #175, commits dcee9c3d5 / fafe6d233 / 9f921fbec, 2026-07-08 |
| M8 | **Prefill TTFT is O(n²) in prompt length**, knee at 1024–2048 tokens: 122 ms @128 → 3,203 ms @4096 (Qwen3.5-0.8B-Q4, M2 Max). Confirms attention cost is not yet dominant below the knee but grows super-linearly above it. | O(n²), knee 1–2K | prefill-ttft baseline, 2026-07-06 |
| M9 | **Dispatch count is ~378 per token** at ~5–15 µs fixed overhead each (≈1.9–5.7 ms/token in dispatch overhead alone); an empty command-buffer round-trip measured ~18 µs. Far above the fused per-layer graph a competitive engine issues. | 378 dispatches/token | measured; tracked #172 |

## Prior / analyzed levers — **[prior, unvalidated on our hardware]**

These are designed or externally-evidenced but **not yet measured in this engine**.
They are the search space the in-flight survey informs; none may be treated as decided
until measured here.

- **BN=64 N-tile widening** of the Q4 tiled GEMM. Recomputed threadgroup footprint
  stays ~15 KiB (under the 32 KiB ceiling), doubling per-threadgroup compute. Identified
  risk: doubling accumulators from 8 to 16 `simdgroup_float8x8` registers per simdgroup
  may trigger compiler register spilling, which would silently negate the gain.
  **Not yet measured on hardware.**
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

Rank the prefill levers by measured expected value. The GEMM kernel is the lever
(M1: 72–80%, 3.6× Amdahl cap); the two closest tuning knobs on it are already resolved
(M5 double-buffer closed, M6 f32→half shipped); the tile geometry is already correct
(M4). Therefore:

**Priority 1 — BN=64 GEMM N-tile widening.** The single remaining unmeasured knob on
the dominant operator. Prototype the widened-tile Q4 kernel variant and measure both
throughput and register spill (via occupancy / throughput cliff).
- Success criterion: **≥+10% prefill throughput @512 with greedy parity 400/400 and no
  PPL regression.** A spill-induced flat-or-negative result closes the lever (like M5).
- Effort: low. `bench_gemm_variants.rs` A/B harness already exists.
- The survey's register-spill-mitigation and roofline-ceiling findings refine *how* to
  build and *how far* to push this; they do not change that it is first.

**Priority 2 — Dispatch-count reduction: norm+residual fusion, then ICB.** M9 puts
1.9–5.7 ms/token in dispatch overhead. Norm+residual fusion (~48 dispatches removed) is
mechanical, numerically inert, and independently measurable; land it first, then wrap
the fused per-layer graph in an ICB.
- Success criterion: measured dispatch-count drop **and** per-token latency improvement.
- Effort: low–medium.

**Priority 3 — `lm_head` two-stage top-k reduction (#171).** <1% of prefill, but the
naive path is a known shipped-then-reverted regression and the two-stage design is ready
and self-contained. Falsifiable, low blast radius.

**Deferred — Flash-Attention-2-on-Metal (#126). Gated on a measurement, not scheduled
now.** Attention is not the bottleneck below the M8 knee. Before any FA2 work, run a
context-length sweep and only start when a measured run shows attention share crossing
the GEMM share. Building it speculatively risks large effort against a lever that is
sub-dominant at our workloads' context lengths.

**Closed — do not revisit without a fundamentally different resource layout.**
Double-buffered threadgroup loads (M5), GDN chunked-prefill tiling (M7), and any
tensor-core / non-`simdgroup_matrix` approach (no such unit on Apple GPUs).

## Consequences

- Effort concentrates on the measured 72–80% (GEMM) and the measured dispatch overhead,
  not on attention or storage-mode work whose measured share is small today.
- The 3.6× Amdahl ceiling (M1) is the honest bound: kernel tuning alone cannot close the
  full ~4× gap (M2); a structural attention change (FA2) is what unlocks the regime past
  the M8 knee, and it is deliberately deferred to when a measurement says it pays.
- **Pending external survey**: when the prior-art findings land, each priority's
  implementation section is updated in place (exact per-family tile params for P1,
  ICB gotchas for P2, FA2 SRAM-tiling for the deferred item). The priority ordering in
  this ADR is not expected to change; if the survey's roofline result shows P1's headroom
  is already near-exhausted, that is itself a decision input recorded as a follow-up, not
  a silent reversal.

## Follow-ups

- P1 result (BN=64 measured) → its own perf PR with `make bench-compare` output.
- Fold external survey findings into the per-priority implementation notes above.
- Re-run the M8 context sweep to fix the FA2 gate threshold in tokens.
