# ADR-072: Weight Quantization Priority (Metal)

**Status**: Accepted
**Date**: 2026-07-09 (accepted 2026-07-14)
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own shipped source and
> measured perplexity results (the **Measured / source-verified reality** table). The core
> weight-quant machinery — group-wise Q4, per-row Q8, and the full offline QuaRot rotation
> pipeline — is already shipped, so this ADR ranks the *open* levers rather than proposing a
> greenfield quantizer. A parallel external prior-art survey (W3/SpinQuant/FP4/FP8/online-Hadamard
> literature) is folded as the **[prior, unvalidated on our hardware]** section; where its
> arithmetic assumed a different Q4 block size it is corrected against the shipped format. The
> ranking is fixed by the measured decode profile and the measured QuaRot quality result, not by
> the survey.

## Context

Decode on this engine is weight-bandwidth-bound: each generated token streams the full weight set
through the GPU, so weight bit-width is the primary decode-throughput lever (distinct from the KV
and GDN-state traffic addressed in ADR-071). Two quantization tiers already ship (Q4, Q8), and a
rotation-based tier (offline QuaRot) ships as infrastructure but measures as a net-negative on
quality. This ADR **ranks the open weight-quant levers** — W3 (#420), online-Hadamard rotation
(#703), FP4/FP8 load formats (#683/#684), SpinQuant (#421), and the role-aware precision policy
(#423) — so effort lands on the one the measured decode profile and the measured quality data say
matters, and records which paths are closed so they are not re-tried.

Target is Metal-only (no CUDA). Apple GPUs expose `simdgroup_matrix` but **no low-precision matrix
unit**, so sub-8-bit and FP4/FP8 formats are storage/bandwidth plays that dequantize to f16 for
compute — they cut weight traffic, not compute cost. Any new tier is gated by the armed Q4 PPL
quality gate (golden 16.589, tolerance 0.05) plus a per-tier PPL delta budget (≤0.3 conservative,
≤1.0 aggressive) and, per the repo's perf-PR rule, a `make bench-compare` decode-throughput A/B.

## Measured / source-verified reality (this engine)

Each row is tagged **runtime-measured** (a bench/eval on this hardware), **unit-test / gate-pinned**
(a value asserted by a committed test or CI gate), or **source-read** (a structural fact read from
merged code on `origin/main`), with its most durable public pointer. Internal profiling with no
merged artifact says so.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| W1 | **Q4 is shipped**: group-32, v2 format **20 B/block (16 B packed nibbles + f16 scale + f16 bias) = 5.0 effective bits/weight** (not the 4.25–4.5 of 64/128-size blocks), asymmetric by default; symmetric mode only for rotated tensors. Metal `gemm_q4_tiled` / `gemv_q4_decode`. | source-read | `weights/q4_weights.rs:3-71`; kernels `forward/shaders/gemm_q4_tiled.metal` |
| W2 | **Q8 is shipped in two distinct layouts**: the CPU weight representation is per-row symmetric int8 (one f32 scale per output row, near-zero metadata), while the Metal `gemm_q8_tiled` / `gemv_q8_decode` kernels consume the **Q8_0 block layout** (34 B/block = 2 B f16 scale + 32 int8). Same tier, two on-device forms. | source-read | `weights/q8_weights.rs` (CPU per-row); `forward/shaders/gemm_q8_tiled.metal:25` + `forward/metal_qwen35.rs:708-723` (Metal Q8_0) |
| W3 | **Offline QuaRot v0 (randomized Hadamard R1/R2) is shipped and wired into the serving path**: full convert/fuse/absorb pipeline, and the same `from_q4_dir` loader used for plain Q4 detects a QuaRot dir via the `quarot_seed` in `quantize_index.json` (`read_quarot_seed_from_index` → `quarot_seed_opt.is_some()`), then applies the rotation. The QuaRot object-form manifest (`ManifestFlavor::QuaRot`) is the on-disk descriptor, not the loader's detection key. CI exact-token composed-golden gate. | source-read | ADR-044/045; `quant/quarot/*`, `bin/quantize_quarot.rs`; `forward/metal_qwen35.rs:13546` (loader) + `:14287-14290` (seed detection); `quant/q4_manifest.rs`; `tests/quarot_q4_composed_golden.rs` |
| W4 | **Offline QuaRot v0 is a measured NET-NEGATIVE quality tier**: unrotated Q4 **16.589** vs QuaRot Q4 **19.007** PPL, delta **+2.418** @2048 tok (real GPU, WikiText-2). The ADR-044-era "−1.61 QuaRot better" was a pre-RoPE-fix artifact — forward-path fixes lifted the unrotated baseline ~4.5 PPL but QuaRot only ~1.8, so the old "win" was QuaRot compensating for since-fixed bugs. | runtime-measured | `docs/bench_results/perplexity.tsv` (commit `ae1f91cfc`); #616 (closed) |
| W5 | **The Q4 PPL quality gate is armed and required**: golden **16.589111** (q4-unrotated tier), tolerance 0.05, captured on CI paravirtual Metal; `q4-ppl-quality` → `parity-gate.needs` → repo required checks. | gate-pinned | `tests/fixtures/ppl_gate_v1/golden.json`; `.github/workflows/e2e-parity.yml:551-660` |
| W6 | **A runtime Hadamard-rotation call-site already exists**: `quarot_rotation: Option<RandomizedHadamard>` on the Metal state counter-rotates MTP-head activations (ADR-051). Not general R3/R4, but a working runtime `RandomizedHadamard::apply` precedent — #703 is an extension, not a greenfield wire-up. | source-read | `forward/metal_qwen35.rs:1481` (field), `:4952,:4982,:5330` (uses) |
| W7 | Decode is weight-bandwidth-bound; **MLP weight GEMMs are ≈35% of decode time**, GQA projections next, GDN cheapest. So MLP-weight bit-width is the decode-bandwidth lever. | internal profiling, no merged artifact | prior decode-bandwidth profiling; a `make bench-compare` on any W3 PR is the public gate |
| W8 | **W3 MLP-only is a measured NET-NEGATIVE quality tier** (added by Amendment 2; measured 2026-07-02 — *before* this ADR — but omitted from the original table): full-corpus `eval_perplexity` decode-loop (M=1 teacher-forced NLL), `wiki.test.raw` 310,034 tokens, window 512 / stride 256, serialized idle-GPU legs, same harness for all three rows — f16 **15.198** (CPU), Q4 **15.731** (Metal), W3 MLP-only **18.097** (Metal): **+2.366 PPL vs Q4 (+15.0%)**, ~10× this ADR's own ≤0.2–0.25 W3 budget, for −25% MLP payload bytes. | runtime-measured | PR #515 (draft, closed — the measurement record); issue #420 |

## Prior / unvalidated on our hardware

Folded from the external survey as **data**. Adopted where it matches the measured result;
corrected where its arithmetic assumed a different block size.

- **Offline-only rotation as a production quality path — REFUTED**, and this *matches* the measured
  W4 net-negative (cross-validated, not just asserted). **Online Hadamard rotation** is the survey's
  proposed mechanism to recover rotation's benefit at runtime.
- **W3 (3-bit), MLP-only — highest leverage.** The survey's weight-traffic arithmetic assumed 64/128
  blocks; corrected to the shipped **group-32**: W3@32 ≈ 4.0 bpw vs Q4's 5.0 → ~20% weight-traffic
  cut on the quantized tensors (the survey's 22–24% at larger blocks — direction holds, magnitude
  adjusts). MLP-only keeps the sensitive attention path at Q4.
- **FP8 (E4M3) / FP4 (MXFP4/NVFP4) — mechanical load formats**, dequant-on-read to f16. No compute
  speedup on Apple GPUs (no low-precision matrix unit); value is checkpoint compatibility + weight
  traffic, only when a target model ships in these formats.
- **SpinQuant (learned rotation) — last.** Only worth it if online rotation first proves rotation
  beats unrotated at all.
- **Role-aware mixed-precision policy — a framework layer** (which layers get which precision) that
  MLP-only W3 is the first concrete instance of.
- **Breadth surveys** (calibration-corpus sizing, fp16 failure modes, KV-cache compression) —
  tangential to weight quant; the KV points fold into a future KV-quant ADR, not this one.

## Decision

Rank the open weight-quant levers by *measured decode leverage × quality risk*, leading with the
lever the decode profile (W7) points at and gating the rotation question on a cheap equivalence check
before any kernel work.

1. **P1 — MLP-only W3 (#420).** Highest leverage: decode is weight-bandwidth-bound and MLP GEMMs are
   ~35% of it (W7); W3@group-32 ≈ 4.0 bpw vs Q4's 5.0 → ~20% MLP weight-traffic cut. It is
   **independent of the QuaRot net-negative** — it proceeds on the shipping unrotated-asymmetric-Q4
   baseline (W1). Validate: a `Q3Block` at the real group-32 (not 64/128) + the existing
   `eval_perplexity --q4-dir` harness (the #616 gate path), PPL delta budget < 0.2–0.25 vs the 16.589
   golden (W5), plus `make bench-compare` for the decode-time claim. *Build.*
2. **P2 — Online R3/R4 Hadamard rotation (#703).** This lever *decides whether QuaRot is worth
   shipping at all*: offline v0 is a measured net-negative (W4), so the shipped rotation
   infrastructure (W3) is dead weight as a quality tier until online rotation recovers it. Lower-risk
   than a greenfield wire-up given the runtime `RandomizedHadamard::apply` precedent (W6). Validate:
   **CPU/f16 rotation-equivalence first** (mean-abs ≤ 5e-4 vs unrotated) *before* any Metal kernel,
   then the PPL matrix — offline+online must land within ~+0.1 of the 16.589 unrotated baseline. Gate:
   if it cannot recover to within ~+0.1, offline QuaRot stays a research artifact, not a serving tier.
   *Build, gated on the equivalence check.*
3. **P3 — FP8 (E4M3) / FP4 (MXFP4/NVFP4) dequant-on-read loaders (#684/#683).** Lowest research risk
   (mechanical, golden-vector correctness, no PPL judgment) but **zero decode-throughput payoff on
   their own** — Apple GPUs dequant to f16 for compute, so these are load-format compatibility, not a
   speed lever. **Demand-gated**: implement when a target checkpoint actually ships in these formats
   (e.g. the DeepSeek-V4 line per the issue bodies), not speculatively. *Build, demand-gated.*

**Deferred — SpinQuant / learned rotation (#421).** Last; gated on #703 proving rotation beats
unrotated. If online Hadamard cannot recover QuaRot, learned rotations chase a refuted premise.

**Enabler (light framework) — role-aware mixed-precision policy table (#423).** The per-layer
precision-selection framework that "MLP-only W3" (P1) is the first instance of. P1 can ship
standalone; seeding the policy table as part of it keeps per-role precision clean. Not a separate
priority tier.

**Closed by measurement — offline QuaRot v0 as a quality tier** (W4, +2.418). The rotation pipeline
and loader stay shipped infrastructure (correct and gated, and still serving the ADR-051 MTP path),
but offline QuaRot is **not a recommended serving tier** until #703 recovers it — the ADR-044
conclusion is amended to this at the priority level (already reflected in the W5 gate's note).

## Amendment 1 — `Q3Block` format (P1, accepted 2026-07-14)

The P1 W3 build is split into a CPU format stage and a Metal kernel stage. This amendment fixes the
on-disk / in-memory `Q3Block` format so the two stages share one authoritative definition. Stage 1
(the format module + round-trip goldens, no engine change) implements exactly this; Stage 2 (the
Metal `gemm_q3` / `gemv_q3` dequant kernels) consumes it. Attention and GDN weights stay Q4/Q8 — W3
is MLP-only (gate/up/down).

**Block.** Group-32, **16 bytes**, 4.0 bpw:

| bytes | field | meaning |
|-------|-------|---------|
| `0..2` | `scale` | `f16` (stored as `u16` bits) |
| `2..4` | `bias` | `f16` (stored as `u16` bits) |
| `4..16` | `packed` | `[u8; 12]` — 32 3-bit codes, plane-split |

**Plane-split 2+1 packing.** The 32 codes are split into a low-2-bit plane and a high-1-bit plane
(chosen over a dense 3-bit bitstream because each plane byte-aligns, so pack/unpack is branch-free
and no code straddles a byte boundary):

- low-2-bit plane → `packed[0..8]`: `packed[b] = q2[4b] | q2[4b+1]<<2 | q2[4b+2]<<4 | q2[4b+3]<<6`
- high-1-bit plane → `packed[8..12]`: `packed[8+b] = Σ_{k=0..7} hi[8b+k] << k`
- dequant index `i`: `q[i] = ((packed[i/4] >> ((i%4)*2)) & 0x3) | (((packed[8 + i/8] >> (i%8)) & 0x1) << 2)`

**Quantization.** Asymmetric is the default; symmetric is available for zero-mean weights:

- asymmetric: `scale = (max - min) / 7`, `bias = min`, `q = clamp(round((w - min) / scale), 0, 7)`
- symmetric: `scale = abs_max / 3.5`, `bias = -4·scale`, `q = clamp(round(w / scale) + 4, 0, 7)`
- dequant (both): `w ≈ q · scale + bias`

**File.** Magic `KHQ3`, version `1`. Load fails closed on bad magic, non-finite scale/bias, or a
shape whose element count disagrees with the block count.

**Validation gates (pre-registered, binding).** PPL delta ceiling **0.2** vs the 16.589111 golden
via the `eval_perplexity --q4-dir` path (#616 gate), no post-hoc widening; a miss routes to
role-aware selective W3 (#423), documented honestly. Plus `make bench-compare` for the decode-time
claim. Stage 1 carries no runtime path, so its gate is the round-trip / mutation-sensitivity test
suite; the PPL and bench gates bind at Stage 2.

## Consequences

- **Positive.** The ADR leads with the single lever the measured decode profile says matters
  (MLP-only W3), and puts the QuaRot-recovery question (#703) behind a cheap CPU-equivalence gate
  before committing Metal kernel work. FP4/FP8 are correctly demand-gated, so no speculative kernel
  effort lands for formats no target checkpoint uses yet.
- **Cost.** W3 needs a new `Q3Block` format plus Metal `gemm_q3`/`gemv_q3` kernels — real work, and
  its PPL must be defended against the armed 16.589 gate (W5). Online rotation (#703) adds a runtime
  rotation apply on the decode hot path that must not regress decode throughput measurably.
- **Risk.** If W3-MLP PPL delta exceeds ~0.25, the ~20% bandwidth win is not worth the quality loss —
  the eval gate catches this before merge, so the downside is a closed experiment, not a shipped
  regression. If #703 cannot recover QuaRot to within ~+0.1, the rotation infrastructure stays unused
  as a quality tier (but is cheap to keep for the MTP path it already serves).

## Follow-ups

- #420 (W3 MLP-only) → ~~**P1**~~ **closed by measurement — see Amendment 2 (W8)**. Stage 1 (CPU format module) = #978, retained as dormant format infrastructure; Stage 2 (Metal kernels) not built.
- #703 (online R3/R4 Hadamard) → ~~**P2**~~ **P1 per Amendment 2**; CPU/f16 equivalence gate first (W6 precedent), then the recovery PPL matrix.
- #684 / #683 (FP8 / FP4 loaders) → ~~**P3**~~ **P2 per Amendment 2**, demand-gated on a target checkpoint shipping in-format.
- #421 (SpinQuant) → deferred, gated on #703.
- #423 (role-aware precision policy) → light framework; its seeding instance moves with Amendment 2 (rides whichever tier #703 proves out, or a future demand-gated format).
- ADR-044 / 045 / 051 (QuaRot family) → offline v0's quality-tier conclusion amended by W4; this ADR records the amendment at the priority level.

## Amendment 2 (2026-07-14): W3 closed by prior measurement; re-ranking

**What was wrong.** The original Decision ranked MLP-only W3 (#420) as P1 *Build*, and both the
Risk section and Amendment 1's pre-registered PPL gate treated a >0.2–0.25 PPL blowout as a future
possibility the eval gate would catch. In fact the experiment had already been run, and had
already failed, a week before this ADR was written: PR #515 (2026-07-02, draft, closed)
implemented a W3 MLP-only path (KHW3 format) and measured it at **+2.366 PPL vs Q4** on the full
corpus (evidence row W8, added by this amendment) — roughly ten times the tier's budget. The
original evidence table omitted that result, so the P1 ranking rested on the leverage arithmetic
(W7) without confronting the strongest available counter-evidence. This amendment corrects the
record and the ranking.

**Why the result transfers to `Q3Block` unchanged.** The `Q3Block` format merged in #978 (KHQ3,
`weights/q3_weights.rs`, specified by Amendment 1) has the *same quantization semantics* as the
KHW3 format #515 measured: group-32, 3-bit asymmetric min/max round-to-nearest (8 levels spanning
`[min, max]`), f16 scale + f16 bias, 16-byte block. The two differ only in bit-packing layout
(plane-split 2+1 vs sequential 3-bit fields) — packing changes neither the representable values
nor the rounding, so per-weight quantization error is identical and the W8 PPL result applies to
`Q3Block` as-measured. Re-running the experiment on `Q3Block` would not produce a materially
different number.

**Reopen conditions.** Any future W3 attempt must name a *quality mechanism that changes the
error distribution* — calibration-based quantization (GPTQ/AWQ-family), a proven online rotation
(#703) supplying outlier suppression, or finer groups at a bpw cost — and re-derive the byte
arithmetic for it. Naive RTN W3 at group-32 is closed. Amendment 1's "miss routes to role-aware
selective W3 (#423)" clause is narrowed the same way: selective W3 without a new quality
mechanism inherits the same per-tensor error and is not a reopening path by itself.

**Re-ranking.**

1. **P1 — online R3/R4 Hadamard rotation (#703)** (was P2). Unchanged in substance: CPU/f16
   rotation-equivalence gate first, then the recovery PPL matrix vs the 16.589 golden. Rotation
   headroom — flattening outliers — is precisely the mechanism whose absence the W8 result
   exposes at 3-bit, so #703 now also gates any W3 revival.
2. **P2 — FP8 (E4M3) / FP4 (MXFP4/NVFP4) loaders (#684/#683)** (was P3). Still demand-gated on a
   target checkpoint shipping in-format.
3. **Closed by measurement — MLP-only W3 (#420)** (was P1), per W8, joining offline QuaRot v0
   (W4) in the closed set. Reopen conditions above.

**Status of shipped W3 artifacts.** The `Q3Block` format module (#978) stays merged as dormant
format infrastructure: correct, tested, reachable from no engine path, and useful the day a
quality mechanism justifies a 3-bit tier. No Metal kernels, quantizer emit, or loader support for
it are to be built while #420 is closed. Amendment 1's format definition remains authoritative
for the dormant module; its Stage-2 build plan and binding validation gates are suspended with
the lane rather than repealed, so they re-arm as-written if the reopen conditions are ever met.

## Amendment 3 (2026-07-17): Stage-2 Metal kernels landed post-closure; dormant status recorded

**What happened.** PR #1014 (merged 2026-07-17T03:40Z) implemented the Stage-2 Metal kernels for
the `Q3Block` format — `gemv_q3_decode`, `gemm_q3_tiled`, the `Q3WeightBuf`/`mmap_q3_weight`
loader building blocks, and the CPU parity oracle — three days after Amendment 2 closed #420 and
stated that no Metal kernels, quantizer emit, or loader support were to be built while the issue
stays closed. The PR was driven from pre-Amendment-2 lane state: its body cites the original P1
ranking and Stage-1/Stage-2 plan with no mention of the closure.

**Disposition: retained, not reverted.** Every symbol #1014 added is `#[allow(dead_code)]` or
test-only and is reachable from no live engine path (`MetalFfnWeights::Dense` still carries only
`Q4WeightBuf` for MLP tensors); the kernels are differentially tested against the CPU oracle on
real GPU hardware, with mutation-sensitive coverage on both sides. Reverting correct, tested,
unwired code costs more than recording the truth. Stage 2 therefore joins Stage 1 (#978) under
the same status: **dormant format infrastructure**, correct and tested, wired to nothing.

**Prohibition reaffirmed, narrowed to the remaining surface.** With both format and kernels now
existing as dormant artifacts, the binding line moves to activation: no checkpoint-level routing,
no quantizer emit, and no live loader wiring of any Q3 path while #420 is closed. Amendment 2's
reopen conditions are unchanged — a quality mechanism that changes the error distribution (a
proven online rotation via #703, GPTQ/AWQ-class calibration, or finer groups) — and Amendment 1's
suspended Stage-2 validation gates re-arm as-written if a reopen ever occurs, now covering the
activation step rather than the already-built kernels.

**Process note.** Two same-day instances of stale lane state driving work (a queued Stage-1
assignment dispatched against the closed lane, and #1014 itself) establish this as a pattern:
build lanes must verify the governing issue state and the ADR's amendments at launch time, not
the queue entry that scheduled them.
