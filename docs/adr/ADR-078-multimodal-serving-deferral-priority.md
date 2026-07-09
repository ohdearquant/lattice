# ADR-078: Multimodal / Vision Serving Priority (Deferral)

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-inference

## Context

Eighth and final topic in the bundle sequencing set (siblings ADR-071…077). The research question
`lat_multimodal_001` asked whether multimodal / vision **serving** is a near-term lever worth
ranking among the active optimization priorities. The answer from the current head is **no** — but
for a reason that must be stated carefully, because vision is an active, roadmap-tracked development
lane (tracking issues #564 / #565 / #566), not a dead one.

This ADR records a **priority-sequencing** decision (where multimodal sits in the near-term lever
ranking), which is distinct from and does not re-litigate the **technical** vision design already
accepted in **ADR-069** (vision encoder recalibration for Qwen3.5-0.8B, merged). It slots the
multimodal topic into the bundle's ranking as *deferred, gated on the vision lane's scheduling*,
and pins the honest current state so the inert scaffold is neither mistaken for a shipped
capability nor read as a deprioritization.

Every load-bearing row below is source-read or issue-state against `origin/main @ 594b557be`.

## Measured / source-verified reality (`origin/main @ 594b557be`)

| # | Reality | Tag | Pointer |
|---|---------|-----|---------|
| R1 | **Serving fails closed on image input.** An OpenAI-style `image_url` content part returns HTTP 400 `"image input requires a vision-capable model"` — a truthful capability error, not a silent text-only degradation or a schema error. Tested at both unit and HTTP levels. Shipped via PR #656 (closing bug #641). | source-read + test-pinned | `crates/inference/src/bin/lattice_serve.rs:183` (message const), `:791,:799-800` (`content_text` reject arm), tests `:2169` (`message_content_image_url_rejected`), `:2478` (`chat_completions_image_url_400`); mirrored in the chat binary `lattice.rs:3063,:4786` |
| R2 | **The vision module is compiled but inert.** `pub mod vision` is unconditional (not feature-gated), but `VisionEncoder`/`generate_multimodal` are constructed and called **only** from their own `#[cfg(test)]` modules — no binary and no non-test library path builds a vision encoder, loads `model.visual.*` weights, or calls a real image encode. | source-read + absence(exhaustive-grep) | `crates/inference/src/lib.rs:44`; `generate_multimodal` at `metal_qwen35.rs:8788`, live callers only at `:20700,:21694` (test modules); `vision/mod.rs` `VisionEncoder::new` sole call site is a test helper |
| R3 | **The vision path is a CPU-only v0 scaffold.** Zero Metal references anywhere in the six `vision/*.rs` files; the only mentions state the Metal path is *not yet implemented*. | absence(exhaustive-grep) | `crates/inference/src/vision/vit.rs:12-14` (doc: "ADR-049 plans a Metal GPU forward pass; v0 implements the CPU path") |
| R4 | **The runtime model config carries zero vision fields.** `Qwen35Config` on head has no `vision_config`, image/video token ids, or M-RoPE fields; the parser drops every vision field the real 0.8B checkpoint carries, by design pending ADR-069 S1. | absence(exhaustive-grep) + source-read | `crates/inference/src/model/qwen35_config.rs` (comments `:211,:1533-1534` explicitly warn against letting vision fields leak into the text config) |
| R5 | **The technical vision design is already accepted, not open.** ADR-069 (vision encoder recalibration for the real Qwen3.5-0.8B VL checkpoint) is merged; it stages the work S1 (config parsing) → S6 (serve/Studio wiring) and calls the current scaffold "inert" in its own words. | source-read + issue-state | `docs/adr/ADR-069-vision-encoder-qwen35-recalibration.md:15` ("…is **inert**: no `model.visual.*` weights are loaded…"), Scope `:120-126`; merged PR #669 / commit `c2c23e444` (2026-07-05) |
| R6 | **S1 is already written and idle.** The config-parsing stage exists as an OPEN **draft** PR #670 (branch `s1-vision-config` → `feat/vision`, +296/−4, round-trip tests against the real 0.8B config fixture green), untouched since 2026-07-05 — additive, all new fields default `None`, forward pass unchanged. | issue-state | `gh pr view 670` (OPEN, draft, base `feat/vision`) |
| R7 | **Vision is an actively tracked development lane.** Three roadmap tracking issues are open (#564 vision, #565 audio, #566 Gemma-4); #564's first milestone is a CPU-only ViT-parity slice. The serve fail-closed of R1 is the *already-shipped API half* of #649. | issue-state | `gh issue view 564/565/566` (all OPEN); #649 OPEN (umbrella), #641 CLOSED (fixed by #656) |

## [Prior, unvalidated on our hardware] — the external survey

Folded from the external survey (`fleet_atlas_lat_multimodal_001` harvest) as **data**, not as a
ranking input. Its verdict — "REFUTED for current head" — is corroborated by R1–R4 above, which are
the load-bearing evidence. The survey adds no capability the repo does not already show; it is cited
only for completeness of the bundle's provenance chain.

## Decision — DEFERRED-NOT-RANKED (revisit gated on the vision dev lane)

1. **Multimodal serving does not enter the near-term lever ranking of this bundle set.** There is no
   served image capability to optimize at head: the vision path is inert scaffold (R2–R4), and the
   correct interim serving contract — a truthful fail-closed 400 (R1) — is already shipped. Ranking
   a lever against a capability that does not execute would be premature.

2. **This is a priority-sequencing decision, not a technical one.** The technical design is accepted
   in ADR-069 (R5) and is **not** re-litigated, amended, or superseded here. ADR-078 only places the
   multimodal topic in the bundle's ranking and pins the honest current state.

3. **The deferral is gated, not permanent.** It revisits the moment the vision dev lane (#564) is
   scheduled. The nearest concrete step is landing PR #670 (S1 config parsing, R6) onto `feat/vision`,
   then ADR-069 S2–S6. No new experiment is required to settle any of these claims — all are resolved
   by source and issue/PR state.

4. **Explicit non-deprioritization.** "Deferred in the near-term serving-lever ranking" is **not**
   "vision deprioritized." Vision is an active development lane (R7), and the model we already ship
   (the 0.8B checkpoint carries the vision weights on disk) can do more than we currently expose.
   When the lane is scheduled, ADR-069's staged plan and #564's first milestone are the entry points.

## Consequences

- The bundle's near-term lever ranking (prefill-GEMM priority, KV-quant, adaptive-reasoning, pruning
  validation, etc., per ADR-071…077) stands unchanged; multimodal is simply not in it yet.
- The fail-closed serve behavior (R1) remains the correct contract until decoder-side wiring lands;
  #649 stays open as the umbrella, #564 as the engine-side tracking issue.
- **Housekeeping (not done here):** a stale untracked duplicate of ADR-069 sits in the main
  working tree — a pre-review-fixup copy differing from the merged ADR only in corrected line
  anchors and hygiene wording. It carries no new information and is safe to delete; flagged for a
  separate housekeeping pass rather than folded into this docs PR.

## Activation path (for when the lane is scheduled)

Cheapest-to-land first, per ADR-069's Scope and #564's milestone:

1. **Land PR #670** (S1 config parsing) onto `feat/vision` — zero forward-pass risk.
2. **S2–S3** — load the `model.visual.*` tensors into `VisionWeights`, adapt the CPU ViT scaffold
   to the real depth-12/hidden-768 geometry, gate on cosine > 0.999 vs the HF ViT (= #564's first
   milestone, CPU-only).
3. **S4–S5** — merger + image-token expansion, then replace the decode-time drop-stub with real
   residual-stream injection + 3-axis M-RoPE confined to the 6 GQA `full_attention` layers, gated on
   end-to-end greedy first-N-token parity vs HF.
4. **S6 / #649** — wire the already-fail-closed `image_url` serve path through to a real vision encode
   once S1–S5 land.

The 0.8B checkpoint (Apache-2.0) already carries the needed weights in both fp16 and q4 forms; no new
model acquisition is required.

## S-row rider

When the vision lane is scheduled and multimodal serving lands, this deferral is superseded. Record
the activation as a status update / new row on this ADR whichever way the lever ranking then falls —
so the bundle's traceability chain shows idea → deferral → activation, not a silent flip.
