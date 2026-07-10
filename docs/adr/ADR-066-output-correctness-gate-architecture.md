# ADR-066: Output-Correctness Gate Architecture

- Status: Accepted (F-1 through F-4 resolved on 2026-07-01; dispositions recorded in D6)
- Date: 2026-07-01
- Depends on: ADR-064 (CI gate taxonomy), ADR-065 (feature promotion gates), ADR-063 (serving architecture)
- Supersedes: nothing (fills the gap ADR-064's verified-absence ledger and D6 table point at)

## Context

ADR-064 classified every *existing* gate and established the promotion policy (D7). ADR-065
governs *new feature admission*. Neither answers the question this ADR answers: **what output
invariants must the engine hold, and which gate layer enforces each one?**

Ground truth as of 2026-07-01:

- Branch protection is ruleset `16354934`: exactly 8 required status contexts (`CI` ×3,
  `feature-matrix` ×2, `bench-compile`, `cargo-deny`, `parity-gate`), **zero required PR
  approvals**, and RepositoryRole bypass actors with `bypass_mode: always`.
- The only *output*-correctness signal in the required set is `parity-gate`: greedy **token-ID**
  agreement with HF on 4 fixed prompts, match windows of 2–3 tokens, bf16 CPU path only.
- Everything else that runs is advisory to merge: `rustdoc-lint`, `unwrap-in-lib-lint`,
  cargo-deny advisories, the whole of `app-binaries.yml` (no aggregator, not required),
  `bench-update.yml` (no `pull_request` trigger at all), `embed-parity-release.yml`
  (release-cadence only).

Ten shipped-and-fixed bug classes were mapped against this gate set. Every one of them merged clean through
the gates above. They do not miss randomly — they cluster into five structural blind spots of
the current architecture:

| Blind spot | Why the gate is blind | Bug classes that exploited it |
|---|---|---|
| **B1. Token-ID-only comparison** | `e2e_parity_check.py` diffs `hf_ids` vs `lat_ids`, never decoded text or intermediate stream frames | detokenizer added-tokens swallow (#430); streaming UTF-8 mojibake (#196) |
| **B2. Single-path coverage** (bf16 CPU generate) | parity builds `qwen35_generate --features f16` only; Metal, Q4/QuaRot artifacts, kv_f16, and inlined bench-forward variants are never driven | RoPE pairing surviving in 3 alt-forward variants (#392/#393); Metal dispatch under-write (#384); open gaps #239/#320/#252 |
| **B3. Well-formed-input-only** | fixed friendly prompts; no adversarial numeric fields, no malformed schemas, no poisoned weights | serve DoS via unclamped `max_tokens`/`reasoning_budget` (#435); quantizer non-finite scale-fold (#452); mask sentinel `-10000` (#373/#374); softmax fail-open family (#409–#414) |
| **B4. Short-horizon windows** | 2–3-token match windows end before recurrent-state drift accumulates; no PPL or long-generation gate anywhere | GDN per-value-head decay indexing (#262/#427, coherent-early/garbage-late); any PPL regression preserving the first 3 greedy tokens |
| **B5. Skip-as-green signals** | capability-gated tests early-return pass on paravirtual-GPU runners; `metal_qwen35.rs:17635`'s own comment says "gate passes vacuously" | CI Metal paravirtual-GPU false-greens; any new Metal test added without `LATTICE_METAL_TEST_ENFORCE` |

The pattern across all ten classes: the forward pass and sampler were usually *correct* while a
sibling layer (detokenizer, dispatch geometry, quantizer load, streaming flush, serve boundary)
was silently wrong, and the one required output gate could not see that layer by construction.

## Decision

### D1. Four gate layers, each owning specific blind spots

Correctness gating is layered by cost and cadence. A defect class must be assigned to the
*cheapest layer that can structurally see it* — pushing everything into E2E parity is how the
current blind spots formed.

| Layer | What it is | Cadence | Owns blind spots |
|---|---|---|---|
| **L1 — Invariant contract tests** | Model-free, fail-closed unit tests asserting an invariant holds or an error is returned. No golden data, no weights. | Per-PR, inside the required `CI` contexts | B3 (input edges), B5 (via signal rules in D3) |
| **L2 — Differential parity** | Same computation, two implementations (HF reference vs lattice; CPU vs Metal; bf16 vs quantized artifact), compared at token AND rendered-text level. | Per-PR on engine paths, via the `parity-gate` aggregator pattern | B1, B2 |
| **L3 — Quality gates** | PPL and long-generation metrics against pinned goldens at literature-derived tolerances. Needs real hardware and minutes-to-hours. | Nightly/weekly scheduled, self-hosted (per #167); never per-PR | B4 |
| **L4 — Adversarial/abuse gates** | Malformed and hostile input against public boundaries (`lattice_serve` HTTP fields, grammar schemas, weight files). Cheap cases run as L1 tests; fuzz campaigns run scheduled. | Split: clamp/edge cases per-PR; fuzz scheduled | B3 (hostile half) |

### D2. The invariant catalog (L1) — every fixed bug class becomes a standing contract

Each historical class is now a named invariant with a required fail-closed test at every site
that implements the operation. New sites implementing these operations MUST ship the matching
contract test (this is enforceable in review by grepping the catalog):

1. **Softmax totality** (scope amended 2026-07-10, see below): at a **public boundary** — a
   softmax whose row feeds a sampler or is otherwise host-visible (the #611 sampling/serve
   class) — every row normalizes to Σ=1.0±ε or returns `InvalidInput`, never silent
   un-normalized output. For **internal attention softmax** (a softmax row that feeds a matmul
   inside the forward pass, never observed directly by a caller), the canonical fail-closed
   contract is ADR-080 C1's **zeroed row**: a non-positive or non-finite denominator (NaN, +inf,
   or all-`-inf` row) zeroes the row by direct assignment, and the forward call returns `Ok`. A
   zero row multiplies into zero attention output for that position — bounded and deterministic
   — and an all-masked row is a legitimate runtime state (every key excluded by causal/padding
   masking), not an input error to reject. See the amendment below for the full rationale and
   the tradeoff this scoping accepts.
2. **Mask exactness**: masked attention positions get `-inf` (post-softmax probability exactly
   0.0), never a finite sentinel.
3. **Quantizer scale sanity**: a scale computed from any non-finite weight fails the load;
   one test per sibling (q4 / q8 / neon-pack / metal) because the guard is per-site.
4. **Detokenizer totality**: `token_for_id` is total over base vocab ∪ `added_tokens`; every
   legally samplable ID decodes to a non-empty string (model-free tokenizer-JSON test).
5. **Streaming flush validity**: every intermediate streaming flush is complete UTF-8, verified
   token-by-token over a CJK/emoji corpus — not just the final concatenation.
6. **Dispatch coverage**: per-row-output Metal kernels prove threadgroup count ≥ ceil(rows/NR)
   via canary-fill tests at non-multiple-of-NR row counts (under-dispatch is the dangerous
   direction; guarded over-dispatch is safe).
7. **Untrusted size clamping**: every request-supplied size field is clamped to the KV/context
   window before any allocation or GPU-buffer sizing; abuse-path test asserts graceful 4xx,
   not abort.
8. **Convention single-sourcing**: RoPE pairing (and any similar layout convention) has one
   shared implementation; inlined copies in bench/alt-forward paths are a defect. Where
   duplication is unavoidable, each copy gets a differential micro-test against the reference
   convention (stride-half max-diff ~1e-6 vs interleaved ~67 — rejects in seconds).
9. **Recurrent-state indexing**: decay/gate parameters index by the reference architecture's
   cardinality (per-value-head for asymmetric GDN); verified against HF indexing, not
   self-consistency. (Full closure blocked on an asymmetric checkpoint — #262.)
10. **Fail-closed context bounds**: forward paths return errors, never assert-panic, when
    `prompt_len > max_context()`.

#### Amendment (2026-07-10): internal attention-softmax fail-closed scope (PR #794 review)

PR #794's codex round-1 review read invariant #1 literally against the live Metal
`fused_attention` finalize (`forward/shaders/flash_attention.metal`) and flagged a contract
conflict: the kernel zeroes an invalid row and the forward call returns `Ok`, while invariant #1
as originally worded says a softmax row "normalizes to Σ=1.0±ε or returns `InvalidInput`." Both
readings were independently correct against their own evidence — the review against this ADR's
literal text, the PR against ADR-080 C1's explicit zeroed-row contract (`docs/adr/ADR-080-consolidation-duplicated-contracts.md`,
section C1) and the two sites ADR-080 names as authoritative (`attention/gqa.rs`,
`attention/decode.rs`). This amendment resolves the conflict by scoping invariant #1 rather than
picking a winner by fiat:

- **Public-boundary softmax** (a row a caller can observe or that feeds a sampler directly — the
  #611 grammar/sampling class) keeps the original invariant #1 wording verbatim: normalize or
  return `InvalidInput`. A malformed row reaching a sampler is an input-validation failure the
  caller needs to see.
- **Internal attention softmax** (a row that feeds a matmul inside the forward pass and is never
  itself host-visible) follows ADR-080 C1's zeroed-row contract instead: zero the row by direct
  assignment on a non-positive or non-finite denominator, and let the forward call return `Ok`.
  This was already the actual behavior of every canonical site ADR-080 C1 identifies (CPU GQA,
  CPU decode, the Metal `fused_attention` finalize once #789 is fixed) before this amendment;
  the amendment brings the ADR-066 text into agreement with the contract ADR-080 already
  establishes, rather than changing behavior.

**Why zero-row is the right internal contract, not a weaker one.** An all-masked or
all-invalid attention row is a legitimate runtime state — every key excluded by causal or
padding masking, or (adversarially) a NaN/inf score reaching the row — and the finalize's job is
bounded, deterministic arithmetic: a zero row contributes zero to that position's output via the
downstream matmul, exactly like a fully-masked row should. Returning `InvalidInput` from inside
a Metal kernel has no cheap host-visible channel (the kernel has no per-forward status buffer
today), and even a future typed error would currently be swallowed by `QwenModel::forward`'s
silent CPU fallback (`model/qwen.rs`) rather than surfaced — building that plumbing to reject a
runtime state that is not itself an input-validation error is the wrong shape for this layer.

**The tradeoff this scoping accepts, stated plainly:** zeroing swallows numerical *defects*, not
just legitimate masked rows. A NaN produced by an upstream bug (a corrupted weight, a bad
kernel dispatch, a poisoned activation) that reaches this softmax becomes a silent zero row
instead of a visible failure — the forward pass keeps running and returns `Ok` with a locally
zeroed attention contribution. This ADR does not read as "NaN in internal attention is fine": it
is not fine, and this is exactly the defect class ADR-066's own D1 table assigns to the
differential/validation gate layer (D1's L2, `parity-gate`) and L3 quality gates, not to this L1
zero-row contract. The zero-row finalize is the **fail-closed arithmetic guarantee** (bounded,
deterministic, never propagates NaN/inf downstream); catching *why* a row went NaN in the first
place is a detection problem those higher layers own, and PR #794's own regression tests
(`fused_attention_fails_closed_on_nan_q_lane`, `fused_attention_fails_closed_on_inf_q_lane`)
verify the arithmetic guarantee, not defect detection.

This amendment does not reopen ADR-080 C1 (unchanged) and does not require the typed-error
plumbing the round-1 review requested for internal attention; it resolves the review's blocker
by making explicit what was previously only implicit in ADR-080's separately-landed text.

### D3. Signal-design rules (how gates report, not just what they check)

1. **No silent skip.** A capability-gated test must report *skipped* as a distinct, visible
   state or hard-fail when its capability probe fails — never early-return `ok`. New Metal
   tests default to the `LATTICE_METAL_TEST_ENFORCE` pattern instead of opting in per-test.
2. **The aggregator pattern is canonical.** Any path-filtered required check must follow
   `parity-gate`'s always-report aggregator design (e2e-parity.yml:205-238). `app-binaries.yml`
   currently violates this: it runs on nearly every engine PR, can show red, and blocks nothing.
   It gets an aggregator job; making that aggregator *required* is deferred pending sign-off (D6).
3. **Compare rendered text, not only token IDs.** `e2e_parity_check.py`'s `compare()` gains a
   decoded-text equality check alongside `hf_ids`/`lat_ids`. This single change makes the
   existing required gate structurally able to see bug classes #430 and (with a CJK prompt
   added) #196.
4. **Golden numbers are regenerated deliberately, never trusted stale.** The committed
   `docs/bench_results/perplexity.tsv` Q4 number (19.27) contradicts the post-RoPE-fix
   measurement (18.076, beating MLX 18.18). Before any L3 gate pins a baseline, the baseline is
   re-measured on current main and the regeneration command committed next to the golden —
   the `embed_parity_v1` JSON pattern already does this correctly and is the template.

### D4. Golden corpus (L2/L3 substrate) — extend existing patterns, invent nothing

- **Model**: Qwen3.5-0.8B — already auto-fetched and cached by `e2e-parity.yml`; the only
  zero-new-infrastructure choice. Larger models (27B) are weekly-tier candidates only after the
  self-hosted runner (D6) exists.
- **Prompts**: the 4 existing `e2e_parity_check.py` prompts, plus 1–2 CJK/emoji-heavy prompts
  (closes the B1 streaming/text blind spot at the prompt level).
- **Storage**: JSON goldens with metadata per `crates/embed/tests/fixtures/embed_parity_v1/`
  (HF-computed, committed, regenerated via documented `uv run` command); raw `.f32` logit
  vectors per `crates/inference/benchmarks/golden_logits/` where only numerics are needed. Both
  conventions already exist in-repo; new goldens slot in rather than adding a third pattern.
- **Tolerances** (literature-derived, already cross-confirmed by CLAUDE.md and #167):
  f16 Δ≤0.005 · bf16 Δ<0.05 · Q4 Δ≤0.30 PPL. Greedy token agreement stays exact-match.

### D5. Bug-class → gate mapping (the accountability table)

| Bug class (fixed in) | Invariant (D2) | Layer | Standing test exists today? |
|---|---|---|---|
| Softmax fail-open (#409–#414) | 1 | L1 | Yes — hardened across sites, incl. mutation test #406 |
| Mask sentinel (#373/#374) | 2 | L1 | Yes (flash + standard) |
| Quantizer non-finite (#452) | 3 | L1 | Yes — 4 load-time siblings |
| RoPE pairing variants (#392/#393) | 8 | L1/L2 | Partial — fixed, but no standing per-copy differential micro-test |
| Detok added-tokens (#430) | 4 | L1 + D3.3 | Partial — loader test yes; required gate still token-ID-only |
| Serve DoS (#435) | 7, 10 | L4/L1 | Partial — clamps shipped; no CI abuse-path test |
| Metal under-dispatch (#384) | 6 | L1 | Partial — fixed site tested; no canary-test convention for new kernels |
| Paravirtual false-green | D3.1 | signal | No — one opt-in env var on one test |
| GDN decay indexing (#262/#427) | 9 | L3 | No — needs long-horizon gate + asymmetric checkpoint (deferred) |
| Streaming UTF-8 (#196) | 5 | L1 + D3.3 | Partial — unit tests yes; no CJK prompt in the required gate |

Open issues #239 (Metal parity leg), #320 (rotated+Q4 composed golden), #252 (kv_f16 parity
run), #167 (self-hosted perf/quality gates), #153 (per-kernel micro-bench gate) are all
instances of this ADR's L2/L3 layers and inherit its design rather than each re-deciding shape.

### D6. Rollout — deferred vs immediately actionable

**Deferred** (each changes merge-blocking behavior or spends money; per ADR-064 D7 rule 5):

| # | Proposal | Cost/impact |
|---|---|---|
| F-1 | Add `app-binaries` aggregator to required contexts | Engine PRs blocked on app-binary build breaks (currently mergeable-while-red) |
| F-2 | Extend required `parity-gate` to include the Metal leg (#239) and Q4/QuaRot artifact leg (#320) once stable | Longer PR CI; macOS runner minutes |
| F-3 | Self-hosted M2 Max runner for L3 nightly (PPL tiers, long-generation, #167 perf gates) | Hardware + maintenance; the only path to non-paravirtual Metal signal |
| F-4 | Any change to review requirements or bypass actors (`required_approving_review_count: 0`, RepositoryRole bypass) | Governance; out of automated hands entirely |

**Dispositions (2026-07-01, approved as recommended):**

- F-1: **Approved.** `app-binaries` aggregator becomes a required context once the
  non-required job (item 5 below) exists and is green.
- F-2: **Approved, phased.** The Metal leg (#239) and Q4/QuaRot artifact leg (#320) land as
  non-required jobs first; promotion to required contexts after two weeks of green runs
  (review checkpoint 2026-07-15).
- F-3: **Approved, schedule-only.** The self-hosted runner workflow runs on `schedule` against
  `main` only, never on pull requests (public-repo fork-PR code execution risk). The workflow
  ships dormant; runner registration waits on designated hardware.
- F-4: **Keep as is.** Zero required approvals and RepositoryRole bypass remain while the repo
  is effectively solo; revisit when outside contributors arrive.

**Immediately actionable without further sign-off** (no required-context changes, no spend):

1. `e2e_parity_check.py`: add rendered-text comparison to `compare()` + 1 CJK/emoji prompt (D3.3).
2. Regenerate `docs/bench_results/perplexity.tsv` on current main; commit the command (D3.4).
3. Add the `kv_f16` on/off parity test behind `metal-gpu` (closes #252 at L1/L2, non-required).
4. Add the capability-probe assertion convention + canary-dispatch test template to AGENTS.md
   so new Metal tests inherit D3.1/D2.6 by default.
5. Add a non-required `app-binaries` aggregator job (making it required stays F-1).
6. Serve abuse-path regression tests (oversized `max_tokens`/`reasoning_budget` → 4xx) in the
   default test suite (D2.7).

## Consequences

- Positive: each of the ten shipped bug classes has a named owner-layer; a recurrence is a
  process failure, not bad luck. Review gains a grep-able invariant catalog. The gate set stops
  over-relying on one 3-token window.
- Negative: L1 catalog grows test count and review friction; L2 text-level comparison will
  surface benign HF/lattice whitespace-rendering differences that need one-time triage; L3 is
  blocked on hardware (F-3) and until then B4 remains open — this ADR documents, not solves,
  the long-horizon gap.
- Explicit non-goals: PR-time perf thresholds (ADR-064 D6 item 6, #167's domain), feature
  admission (ADR-065), MSRV/doc/publish gates (ADR-064 ledger items 2–4).

## Verification

- Inventory and corpus recon (2026-07-01) built from primary sources only — live ruleset
  JSON, workflow files, `git log`, `gh issue view`.
- Every "fixed in" claim in D5 maps to a merge commit verified in git history during recon.
