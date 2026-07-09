# ADR-073: KV-Cache Quantization Priority (Metal)

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own shipped source and one
> merged, measured result (the **Measured / source-verified reality** table). Unlike the weight-quant
> ADR (ADR-072), this is a **split verdict**: two levers are rankable now from measured evidence
> (f16 KV default-flip, and its port to the second cache), while the int8/int4 KV family has **zero
> lattice-measured quality or throughput data** and is gated behind a defined minimal experiment. A
> parallel external prior-art survey (KVQuant / KIVI / QuaRot-KV / FP8-KV literature) is folded as
> the **[prior, unvalidated on our hardware]** section; its central "KV ≈ 16% of decode bandwidth"
> premise is **refuted** by this engine's own measurement and corrected against the actual model
> shapes. The ranking is fixed by the measured decode profile, not the survey.

## Context

A KV cache stores per-token key/value projections so decode attends over the prefix without
recomputing it. KV-bit-width is, in the general dense-transformer literature, a decode-bandwidth and
memory-capacity lever. This ADR ranks the **open KV-quant levers** — f16 default-flip (#154), f16
for the paged cache (#148), and the int8/int4 family (#118 umbrella, #120 fused kernel, #122
pre-RoPE K, #123 int4 rotation) — against what this engine has actually shipped and measured, and
records which questions genuinely need an experiment so effort is not spent speculatively.

Three facts shape everything below:

1. **There are three independently-evolved KV caches.** `MetalKvCache` is the production serving path
   (both `lattice_serve` and `chat_metal` construct `MetalQwen35State` directly); it defaults to f32
   with an opt-in f16 mode. `FlatKVCache` is the CPU dense-`model::qwen` path, always f16, not
   production. `PagedKVCache`/`PrefixPageCache` is the continuous-batching engine (ADR-048), f32-only,
   **not wired into either serving binary**. Any lever proposal must name which cache it targets.
2. **f16 KV already shipped and measured decode-neutral on the production model.** PR #238 (merged)
   found f16 KV correct (PPL Δ≈4e-6) and decode-neutral (71.6 vs 72.8 tok/s @1k ctx) on Qwen3.5-0.8B,
   because the model is GDN-dominated: only 6 of 24 layers carry a KV cache, so KV reads are <1% of
   per-token weight-read bandwidth. Its own conclusion: *"The real decode lever is weight
   quantization, not KV dtype."* This reframes KV-quant as a **memory-capacity** lever, not a
   decode-throughput one, on this GDN-hybrid family.
3. **Apple GPUs have no low-precision matrix unit.** As with weight quant (ADR-072), int8/int4 KV is
   a storage/bandwidth play only — values dequantize to f16/f32 before the QK^T and P·V matmuls.
   There is no compute-cost reduction. And PR #238 already shows that even *halving* KV bandwidth
   (f32→f16, free of dequant complexity) bought ~0% decode at 1k context. So a fused int8/int4 kernel
   adds real complexity for a payoff capped below KV's actual bandwidth share — the one number nobody
   has measured (#491, open, unrun).

## Measured / source-verified reality (this engine)

Each row is tagged **runtime-measured** (a bench/eval on this hardware), **unit-test / gate-pinned**
(a value asserted by a committed test or CI gate), or **source-read** (a structural fact read from
merged code on `origin/main @ 454245434`), with its most durable public pointer. Internal profiling
with no merged artifact says so.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| KV1 | **Production serving uses `MetalKvCache`**: GPU-resident, separate `k_bufs`/`v_bufs: Vec<Buffer>` per full-attention layer, flat `[max_cache_len * kv_dim]`. Both serving binaries build `MetalQwen35State` directly — neither touches `FlatKVCache` or `PagedKVCache`. | source-read | `forward/metal_qwen35.rs:1166-1180`; `bin/chat_metal.rs:783`, `bin/lattice_serve.rs:989,1134` |
| KV2 | **`MetalKvCache` dtype defaults to f32; f16 is opt-in** via `LATTICE_KV_F16=1` at construction (read at 4 sites, must agree). Doc: *"store KV cache in f16 (default OFF, opt-in)"*; f32 is byte-identical to the pre-#154 path. | source-read | field `metal_qwen35.rs:1828-1835`; env reads `:3218-3223,:3481-3486,:14243-14251,:17358-17363` |
| KV3 | **f16 KV measured correct + decode-NEUTRAL (Qwen3.5-0.8B, real Metal)**: greedy-parity exact; PPL f32 4.3365 vs f16 4.3365 (Δ≈4e-6 NLL, under the 0.05 gate); decode 71.6 (f32) vs 72.8 tok/s (f16) @1k ctx, neutral within noise. PR conclusion verbatim: *"The real decode lever is weight quantization (Q4 vs Q8), not KV dtype."* **Artifact is the PR body, not a committed bench file.** | runtime-measured (PR-body artifact) | `gh pr view 238` (MERGED 2026-06-24), title "opt-in f16 KV cache for Metal GQA path (#154)" |
| KV4 | **CI exercises the f16 Metal KV path for execution-capability** (not perf): a test with `LATTICE_KV_F16=1`+path-proof enforce fails-closed unless all four f16 KV kernels actually dispatch. | gate-pinned | `gh pr view 558` (MERGED 2026-07-02, "cover f16 KV-cache Metal path (#252)") |
| KV5 | **#252's original bar (greedy first-N parity f16-vs-f32) is broader than KV4's dispatch-proof; #252 stays OPEN despite #558.** Whether the remaining gap is parity-test specificity or a self-hosted perf leg is **not independently verified here** — do not assume #558 fully closed it. | source-read (state) + flagged | `gh issue view 252` (OPEN); PR #558 body's own "does not cover" disclaimer |
| KV6 | **CPU dense path is a separate always-f16 cache** (`FlatKVCache`), used only by `generate()` (CPU `model::qwen`) and MTP CPU verification scratch — not production. No f32/f16 toggle. | source-read | `kv_cache/flat.rs:9-11`; `generate.rs:16-19,457-467`; `speculative.rs:495,654-664` |
| KV7 | **Continuous-batching engine KV cache is f32-only** (`PagedKVCache` `Vec<f32>`, `PrefixPageCache` `Arc<[f32]>`), consumed only by `batch/*`, **not wired into either serving binary** (zero `PagedKVCache`/`BatchWorker` hits in both). Third, currently-unserved path. | source-read | `kv_cache/paged.rs:14,146,148`; `kv_cache/prefix.rs:3,59-60`; `batch/worker.rs:28,600-630` |
| KV8 | **Cross-turn KV prefix reuse (#462/#619) is a logical handle over the live `MetalKvCache`** — owns no storage, is dtype-aware (`kv_f16: bool`), composes with whichever KV2 dtype is active. | source-read | `kv_cache/cross_turn.rs:59-76` |
| KV9 | **Per-token KV bytes = `num_full_attention_layers * 2 * full_kv_dim * dtype_bytes`**, `full_kv_dim = num_key_value_heads * head_dim`; tested formula backing the `doctor` memory-fit preflight. Worked example: 0.8B f16 = 6·2·512·2 = 12,288 B/tok ≈ 48 MiB @4096 ctx. | gate-pinned + source-read | `model/qwen35_config.rs:517-539,552-571` |
| KV10 | **Actual model shapes** (correct any prior-art assuming a generic GQA shape): 0.8B — 24 layers, `full_attention_interval=4` → 6 full-attn, `kv_heads=2`, `head_dim=256` → `full_kv_dim=512`. 27B — 64 layers → 16 full-attn, `kv_heads=4`, `head_dim=256` → `full_kv_dim=1024`. Interval=4 constant across every preset. | source-read | `model/qwen35_config.rs:212-260` (0.8b), `:302-338` (27b) |
| KV11 | **No int8/int4/fp8 KV path exists anywhere** — exhaustive grep of `crates/inference/src/` returns zero (`enum KvFormat` = 0 hits; the `KvFormat{F32,F16,I8,I4}` sketch in ADR-062 §Phase-4/5 was never implemented). Nearby "quant" hits are unrelated *weight* quantization. | source-read (absence, verified vs origin/main) | design sketch only: `docs/adr/ADR-062-metal-fa2-prefill.md:313-331` |
| KV12 | **No long-context retrieval harness exists** (needle/passkey/RULER/LongBench — zero hits). Any int8/int4 quality gate beyond a single-corpus PPL number must be built from scratch. | source-read (absence) | grep of `crates/inference/` |

**Derived (not measured) — memory-capacity arithmetic** from KV9/KV10: at 4096 ctx, 0.8B KV is 96 MiB
(f32) / 48 MiB (f16); 27B is 512 MiB / 256 MiB; int8 halves the f16 column again (27B: 128 MiB @4096,
2 GiB @32K). The **27B at long context** is where a memory-capacity argument actually bites; the 0.8B
stays small in absolute terms. This is derived from the tested KV9 formula, distinct from the
runtime-measured KV3 row.

## Prior / unvalidated on our hardware

Folded from the external survey (`fleet_atlas_lat_kvquant_001` harvest) as **data**. The packet was
written **without repository access** (its own MANIFEST §0), so its `patch.diff` is approximate and
its arithmetic assumes shapes this engine does not have.

- **REFUTED premise (the central one): "KV ≈ 16% of decode bytes."** Every "expected speedup" number
  in the survey is derived from this share, which it carries over from generic dense-transformer
  literature assuming `Hkv=8, head_dim=128`. This engine's shapes are `Hkv=2` (0.8B) / `Hkv=4` (27B),
  `head_dim=256`, and only 6-of-24 / 16-of-64 layers carry a KV cache at all (KV10). The measured
  share is **<1% at 1k ctx** (KV3). **Causal mechanism**: the GDN-hybrid architecture — most layers
  are GatedDeltaNet with no KV cache — structurally caps KV's bandwidth fraction far below the
  survey's generic assumption. The survey's 16%-share implies f32→f16 KV should buy ~1.09× whole-decode
  speedup at 8K; the real 1k measurement was neutral. (Not a clean apples-to-apples refutation — the
  survey's number was at 8K, ours at 1k — so the honest form is: **the premise must be re-derived from
  this engine's own long-context measurement, not the generic literature share**, before any
  int8/int4 "expected speedup" is trusted.)
- **f16 KV before any int8/int4 work** — the survey rates this CONFIRMED-BY-ANALYSIS/High, and it is
  *independently* corroborated by KV2/KV3 (f16 already proven correct and cheap, just not defaulted).
- **int8 before int4; int4 non-default** — survey CONFIRMED-BY-ANALYSIS; but the underlying quality
  numbers are rated **Low/Medium-low Qwen-transfer confidence by the survey itself** — no method in
  its 14-method table is High-confidence for Qwen3.5/3.6-hybrid transfer.
- **Pre-RoPE K storage helps per-channel K quant** (#122) — survey confirms it generically for
  RoPE-GQA but explicitly **NEEDS-EXPERIMENT for Qwen-family magnitude**. K is stored post-RoPE today.
- **int4 KV needs rotation** (#123) — cites Qwen2.5-7B unrotated int4 KV at +638 PPL (unusable);
  reuses the shipped QuaRot Hadamard machinery (ADR-072 W6). Deep in the ladder, not a first target.
- **FP8-KV not first on Metal** (no low-precision matrix unit) and **GDN state must not share the KV
  quant track** — both match this engine's constraints and ADR-071's separate GDN-state track.

## Decision

Rank by *measured leverage × quality risk*. Two levers are rankable now; the int8/int4 family is
gated behind an experiment because no lattice quality/throughput data exists for it.

1. **P1 — f16 KV default-flip (#154).** Rankable on measured evidence exactly as ADR-072 ranked
   MLP-only W3 #1: on an already-measured, already-shipped-but-gated result. KV3 proved correctness
   (Δ≈4e-6) and decode-neutrality, with a real memory win at long context (96→48 MiB @4096 on the
   0.8B, 512→256 MiB on the 27B — the f32→f16 halving; int8 would halve the 0.8B figure again to 24 MiB). The only blocker is KV5 (the CI parity-test gap) — an engineering
   completeness gate, **not** an open empirical question; close it by extending #558's existing probe
   into the first-N greedy-parity test #252 originally specified, then flip `use_kv_f16` default to
   ON. Validate: the existing f32-vs-f16 A/B (`ppl_metal` / `bench_decode_ab`) already produces the
   evidence; `make bench-compare` on the default path is byte-identical to main (the win is memory,
   not the A/B table). *Build — small, mostly a default-flip + parity-test.*
2. **P-low — f16 for `PagedKVCache` (#148).** A mechanical port of the KV2/KV3 pattern to the
   f32-only paged cache (KV7); the f32→f16 correctness argument transfers unchanged (same math,
   different buffer owner). Not a research question. **Priority tracks whether the continuous-batching
   engine is on a near-term roadmap at all** — it serves zero production traffic today (KV7). Do not
   rank above P1; do not do speculatively ahead of that engine being wired. *Build, roadmap-gated.*
3. **P2 — the int8/int4 KV family (#118 umbrella, #120 fused kernel, #122 pre-RoPE K, #123 int4
   rotation) is GATED behind a minimal experiment, not ranked on evidence it does not have.** There is
   zero lattice PPL or throughput data below f16 at any KV bit-width, and the central open question is
   not just quality but **whether KV bit-width moves decode throughput at all** on a GDN-dominated
   model whose KV is <1% of bandwidth (KV3). Run the experiment before committing kernel work:

   **Minimal experiment (the P2 gate).** Quantize **K only, int8, full-attention layers only**
   (per KV1's 6/16-layer count) — not V, not int4, not GDN-state (ADR-071 territory). Symmetric int8,
   per-(layer, kv_head, channel) K scales — **no rotation infrastructure** (#123 stays out) and **no
   fused Metal kernel** initially: a **CPU dequantize-to-f32-before-attention** prototype answers the
   quality question before any kernel investment. Harness: extend `bin/ppl_metal.rs` (the KV3 A/B
   harness — it already reads `LATTICE_KV_F16` at construction) or `eval_perplexity` with a CPU-side
   int8-K quantize/dequantize shim; the `ppl_gate_v1/golden.json` RECORD-bootstrap pattern (ADR-072
   W5) is the template for turning a chosen delta into a committed gate. **This KV harness does not
   exist yet** — building it *is* the experiment, which is exactly why P2 needs one and P1/P-low don't.

   The experiment settles: (a) whether int8-K PPL delta stays under a defensible budget (survey/#118
   suggest ≤0.02) on the **actual** 0.8B/27B presets, not the literature number; and (b) whether a
   fused int8 Metal kernel is worth building — i.e. whether **#491's bandwidth-share measurement**
   shows KV crossing into double-digit percent of per-token traffic at a realistic served context
   (KV3 says <1% at 1k on the 0.8B; the experiment must include ≥1 longer-context point, e.g. 8k/16k,
   before concluding the ceiling is low everywhere). #491 is a **prerequisite instrument** for the
   fused-kernel branch (#120), not just adjacent.

**Ordering rationale.** #491 (measure KV bandwidth share vs context) is the pivot: until it runs,
the *throughput* case for any int8/int4 kernel is unfounded on this architecture. The **memory-capacity**
case is real and derivable now (KV9/KV10 — multi-GB swings on the 27B at long context), so the P2
experiment should be framed as answering *memory-tier viability at acceptable quality*, with the
throughput case explicitly contingent on #491.

**Deferred — int4 KV + WHT rotation (#123), V-side and fused-kernel work (#120), pre-RoPE K
restructure (#122).** All behind the P2 int8-K experiment proving int8-K moves something worth the
complexity. int4 without rotation is unusable (+638 PPL literature); rotation reuses ADR-072 W6
machinery but only earns its place after int8-K is validated.

**Not a KV-quant lever — noted to prevent conflation.** #173 (fused 2Q-per-KV decode kernel + paged
head-major layout) is a dispatch-occupancy lever; #185/#126 (prefill O(n²), FA2) is compute-bound and
KV-quant does not address it (ADR-062). These stay out of this ADR's scope.

## Consequences

- **Positive.** The ADR leads with the one lever the measured profile supports (f16 default-flip),
  keeps the second cache's port correctly roadmap-gated, and — critically — refuses to rank the
  int8/int4 family on literature numbers whose own confidence for this model family is Low. It names
  the exact minimal experiment and its #491 prerequisite, so the int8-K question is settled by a
  cheap CPU prototype before any fused-kernel engineering.
- **Cost.** P1 needs the #252 parity-test closed and a default flip (small, plus a re-baseline of any
  memory-fit preflight now that the default halves). The P2 experiment needs a new int8-K PPL harness
  built (no KV quality harness exists, KV12) plus #491 run under GPU flock.
- **Risk.** If #491 shows KV's bandwidth share stays <1% even at long context on this family, the
  fused int8/int4 kernel branch (#120/#122/#123) is **closed by measurement** for these models —
  int8-K would survive only as a memory-tier option for the 27B at long context, not a throughput
  lever. That is a legitimate outcome the experiment is designed to reach cheaply, not a failure.

## Follow-ups

- #154 (f16 default-flip) → **P1**; close #252 parity-test (extend #558's probe), flip `use_kv_f16` default, re-baseline memory-fit preflight. GPU flock on any bench run.
- #148 (f16 PagedKVCache) → **P-low**, roadmap-gated on the continuous-batching engine being wired.
- #491 (KV/GDN/weight bandwidth-share vs context) → **prerequisite instrument** for the P2 throughput case; run under GPU flock, ≥1 long-context point.
- #118 / #120 / #122 / #123 (int8/int4 KV family) → **P2, gated** on the minimal int8-K CPU experiment + #491; do not start kernel work before both.
- #618 (long-context contract table) / KV12 (no retrieval harness) → the quality-gate substrate any int8/int4 KV gate needs beyond single-corpus PPL; note the dependency.
- ADR-062 (metal FA2 prefill, KvFormat sketch) → its Phase-4/5 `KvFormat` design is the never-implemented origin of this family; this ADR supersedes its KV-quant sequencing at the priority level.
