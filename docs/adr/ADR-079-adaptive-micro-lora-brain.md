# ADR-079: Adaptive Micro-LoRA Brain — the Composed Train → Govern → Compose → Route → Consume Loop

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-tune / lattice-inference / lattice-fann

## Context

Over five weeks (2026-06-20 → 2026-07-08) a set of independently-merged PRs built, one stage at a
time, a full adaptive multi-adapter LoRA system on top of the Qwen3.5 decoder: an exact-gradient CPU
backward trainer, a governed multi-adapter manifest, a decode-time mixture blend, a learned online
adapter-router, and seam-transparent consumers (generation and reranking). Each PR was scoped to its
own slice and reviewed on its own terms. **No single document describes the composed loop as one
system** — which stages are shipped, which are partial, where the real gaps are, and how the pieces
hand off to each other.

Seven existing LoRA ADRs (008 / 031 / 043 / 045 / 054 / 056 / 057) each cover a single-adapter
lifecycle slice or an orthogonal composition axis; none covers the composed *pool-of-adapters*
adaptive loop. ADR-056's own appendix admits the training pipeline it designed was never built and
points readers at the code that actually shipped. This ADR is the synthesizing record for that
composed loop: it names shipped-vs-partial-vs-gap per stage with `origin/main` file:line anchors,
references the prior ADRs without re-litigating them, and pins two record corrections that were
circulating as fact.

Every load-bearing row below is source-read or PR/issue-state against `origin/main @ b0604722d`,
read via a detached worktree at that commit.

## The composed loop

```
TRAIN            GOVERN              COMPOSE               ROUTE                 CONSUME
exact-grad  →   manifest +      →   decode-time       →   learned online    →   seam-transparent
CPU backward    fail-closed         weighted blend        adapter-router        generation / rerank
(#191)          loader (#444)       (#443)                (#448 + #453)         (#443 gen, #718 rerank,
                                                                                 ADR-057-D1 cross-enc)
```

The seam that ties CONSUME back to the rest is a single injection point: `Qwen35Model::set_lora`
installs a composed adapter, and every consumer (mixture-generate, query-likelihood rerank) reads it
transparently — no consumer-side code changes when an adapter is present.

## Source-verified reality (`origin/main @ b0604722d`)

| Stage | State | What shipped | Tag | Pointer (merge commit · file:line) |
|-------|-------|--------------|-----|-------------------------------------|
| **TRAIN** | shipped, one gap | Reverse-mode autodiff through Qwen3.5 gated-GQA attention (linear/lora/rmsnorm/rope/swiglu/cross_entropy VJPs), full-depth multi-layer tape, bounds-checked reusable wrapper. Verified on-par with MLX-LM on held-out NLL (PR body: lattice 4.9056→4.6600 vs MLX 4.9052→4.6897 over 30 steps; M1 lm_head NLL 5.1757→0.6103). | PR-merge + source-read | **#191** `7b46f6e773b6256f7168b627926655a2617a990c` (2026-06-20) · `crates/inference/src/backward/{ops.rs,attention_gqa.rs,gradcheck.rs,tape.rs}`; wrapper `crates/tune/src/lora/train_core.rs::forward_full:207` / `train.rs::train_micro_lora:209` (extracted from #191's inline code by #445/#488 `91e396409…`) |
| **TRAIN — the gap** | **GDN LoRA weight-grads NOT on main** | The GDN backward is **dx-only**: it propagates gradient *through* the frozen GDN layers into the lower GQA LoRA grads, but produces no GDN-layer LoRA weight gradients. `forward_full`'s `MixerKind::Gdn` arm takes **no** `lora_slot`; only the `MixerKind::Gqa` arm does. Today's trainer can only teach GQA-layer (`q_proj`/`v_proj`) adapters. | source-read + issue-state | `attention/gdn_backward.rs` (1228 LOC, "dx-only VJP", unchanged since #191); `train_core.rs::forward_full` Gdn arm `:259` (no LoRA) vs Gqa arm `:228` (`lora_slot`). Surface-B extension (`f34ed3b1…` + siblings) is **not** an ancestor of `origin/main` — abandoned branch; its PR **#202 is OPEN, unmerged** |
| **GOVERN** | shipped | `LoraManifest` (schema `version: u32`) + `ManifestEntry` (integrity_sha256 / base_model_rev / tokenizer_rev / rank / alpha / target_modules / dtype / status) + `AdapterStatus{Approved,Quarantined,Revoked}`; `load_adapters_from_manifest` runs ten ordered fail-closed checks (status → uri → existence → integrity-sha256 → …). This is the admissibility gate between "a trainer wrote a safetensors file" and "the blend/route stages may touch it." | PR-merge + source-read | **#444** `a729dad6ca6501ca79a49120070440bd685a698d` (2026-06-29, extended by #624) · `crates/tune/src/lora/manifest.rs:26,64,100`; `loader.rs:15,42,115` |
| **COMPOSE** | shipped, policy deferred here | Exact weighted-concat math folding N adapters' `alpha`/`rank` scales into one rank-Σr adapter (`blend_lora_adapters`), plus decode-time wiring into a **single Metal slot with zero kernel change** (`blend_lora_layer_data` → `generate_with_lora_mixture`). `AdapterRouter` does top-k selection with **constant, non-learned `1/k` weights** — selection *policy* is explicitly out of scope in this PR (it is what ROUTE supplies next). | PR-merge + source-read | **#443** `09a23c00f1bd998aa3cfdb2624939801b4e18180` (2026-06-29, closes #436/#437/#438) · `crates/tune/src/lora/blend.rs:73`; `crates/inference/src/forward/metal_qwen35.rs:1623,3897`; `crates/inference/src/mixture.rs::AdapterRouter:94` (behind `mixture` feature) |
| **ROUTE** | shipped, one mechanism inert, loop-close externalized | Model-agnostic RL substrate in `fann`: RLOO single-sample policy gradient (`RlooTrainer::step` + load-balance/z-loss) and diagonal Fisher (`DiagonalFisher`). LoRA-specific consumer `update_router` batches `FeedbackEvent`s, refits the gate via RLOO, damps updates to previously-important params via Fisher **null-space projection**, returns a `RouterDelta{network_bytes}` (a full serialized gate, not a param diff). | PR-merge + source-read | **#448** `0006ca67cc613da4feea74d2d92c33fb1b3b9c2a` (fann RL primitives, `online-router` feature) · `crates/fann/src/training/rloo.rs:21,49,445,462`, `ewc.rs:30`. **#453** `2dd2fc73f9d3611ae8d009166d70aff164d142d6` (2026-06-30, "capstone", issue #440) · `crates/tune/src/lora/router_update.rs:58,102,125,170,199,318` |
| **ROUTE — the partials** | inert-by-design + externalized loop | (a) The alternative EWC anchor-pullback penalty (`penalty_gradient`, gated by `ewc_lambda`) is **deliberately inert in v1** — a boundary test pins byte-identical output regardless of `ewc_lambda`; v1 anti-forgetting is Fisher null-space damping only. (b) Persisting `RouterDelta.network_bytes` back into a live `AdapterRouter` between requests is **left entirely to the caller**; nothing in-process closes the route→refit→route cycle inside lattice. | source-read + test-pinned | `router_update.rs` boundary test `ewc_lambda_is_inert_in_projection_path`; `RouterDelta` doc `router_update.rs:170-176` ("Load with `Network::from_bytes()` … pass as the `gate_bytes` argument to the next `update_router` call") |
| **CONSUME** | shipped, two independent siblings | Generation consumes a composed adapter transparently via `set_lora` (exercised by `generate_with_lora_mixture`). Reranking has **two** shipped siblings through the same seam: the cross-encoder/BERT query-likelihood path and the causal-LM query-likelihood path. #718's own doc comment: "scoring flows through the model forward, so a composed adapter installed via `set_lora` is applied transparently — no rerank-side changes are needed." | PR-merge + source-read | seam: `crates/inference/src/model/qwen35/model.rs::set_lora:69` (`lora` field `:18`). Causal-LM rerank **#718** `b0c9ed56bc9f5ba848a383a948271ca94f8068af` (2026-07-08) · `model/qwen35/rerank.rs:47,139`. Cross-encoder rerank **ADR-057-D1** (PR #65 `894dfdc3ce96b3ba358d736489f0611c3f42936b`, 2026-05-23, issue #59) · `model/cross_encoder.rs:76` |

### Record corrections (were circulating as fact)

1. **The merged CPU backward trainer is PR #191 (`7b46f6e77`), not commit `f34ed3b1`.** The
   "surface-B GDN LoRA weight-grad" commits (`f34ed3b1…` + two siblings, 2026-06-21/22) are **not
   ancestors of `origin/main`** — `git merge-base --is-ancestor <sha> HEAD` is false for all three;
   they live only on an abandoned `integration/engine-slice` branch, and the one PR for that work,
   **#202, is still OPEN and unmerged**. On head today, GDN-layer LoRA weight-gradients do not exist
   in mergeable code. Any "GQA+GDN LoRA grads shipped" claim overstates head: only GQA `q_proj`/
   `v_proj` adapters can be trained.

2. **PR #718 does not close issue #59.** #718 is the causal-LM query-likelihood reranker
   (`Qwen35Model::rerank`). Issue #59 ("Stable rerank API on lattice-inference with `LoraHook`
   injection") asked for `LoraHook` injection into the **cross-encoder** path and was closed
   2026-05-23 by **PR #65** (ADR-057-D1, `CrossEncoderModel::score_with_hook`) — five weeks before
   #718. #718's `closingIssuesReferences` is empty. Two independent rerank-via-seam consumers now
   exist; no document previously named them as siblings.

## Existing-ADR coverage and the gap

| ADR | Status | Scope (single slice / orthogonal axis) |
|-----|--------|-----------------------------------------|
| **008** lora-injection | Accepted | The `LoraHook` trait seam itself (single adapter, forward-only `apply`); why it lives in `lattice-inference` not `lattice-tune` (avoids a circular dep). |
| **031** lora-adapter-management | Accepted | Three-layer single-adapter representation + PEFT/MLX safetensors **import** (load only; no governance, export, or multi-adapter). |
| **043** lora-serving-verification | Accepted | CI coverage that the hook fires at all 12 projection sites (4 GQA + 5 GDN + 3 MLP) for **one** loaded adapter. |
| **045** quarot-lora-composition | Accepted | Counter-rotation math so a single LoRA composes with a QuaRot-rotated Q4 base at serving time. Orthogonal axis (quantization × LoRA), not multi-adapter. |
| **054** rolora-rotation-aware-lora | Proposed, not implemented | Training LoRA natively in the QuaRot-rotated basis; a different *training axis* (rotation-awareness), blocked on ADR-056. |
| **056** lora-tuning-pipeline | Proposed + status appendix | Designs a `LoraTrainLoop`/`train/lora/` pipeline. Its own appendix admits the design was **never built** and redirects readers to `train_grad_full.rs` / `train.rs` / `online.rs` — the clearest existing admission of the gap this ADR closes. |
| **057** lora-consumer-api | Accepted | Five lifecycle gaps for **one adapter, one request**: D1 cross-encoder hook (#59), D2 export (#60), D3 `adapt_step` per-event weight SGD (#61), D4 typed module names (#62), D5 docs (#63). No multi-adapter governance, no mixture, no learned routing. |

**The gap.** None of the seven describes the composed system: real-gradient multi-layer training that
*produces new adapters* (vs ADR-056's un-shipped design), governance/approval status across a *pool*
of adapters (vs ADR-031's single adapter), weighted *composition* of multiple adapters into one at
decode time, or a *learned selection policy* (RLOO + Fisher) over which adapter(s) to use per request
(vs ADR-057-D3's per-event weight nudge on one already-selected adapter). This ADR is that
synthesizing record; it references the seven above rather than re-opening any of them.

## Decision

1. **Record the adaptive micro-LoRA brain as one composed, mostly-shipped system.** The
   TRAIN → GOVERN → COMPOSE → ROUTE → CONSUME loop is real and merged at head, with the exact
   shipped/partial/gap state pinned above by PR-merge commit and file:line. This ADR is the canonical
   map; future LoRA-system work references it instead of re-deriving the loop from seven separate
   ADRs.

2. **Fix the record.** GDN-layer LoRA weight-gradients are **not** on main (PR #202 OPEN, not
   `f34ed3b1`); only GQA adapters train today. #718 is the causal-LM reranker and does **not** close
   #59 (that was #65). Both corrections are load-bearing for anyone building on the loop — the first
   bounds what the trainer can currently teach, the second disambiguates two real rerank consumers.

3. **Name the gaps as follow-on, each with a re-entry condition** (a gap without a named re-entry
   condition is a parking ticket). See Follow-on work.

4. **Do not re-litigate the prior ADRs.** ADR-008/031/043/045/054/056/057 stand as written; this ADR
   only composes and maps them, and supersedes nothing.

## Follow-on work (named gaps, each with a re-entry trigger)

- **G1 — GDN-layer LoRA weight-gradients.** Currently only GQA `q_proj`/`v_proj` train; the 18 GDN
  layers are frozen (dx-only pass-through). **Re-entry:** when a target task shows GQA-only adapters
  underfit (held-out NLL plateau above the MLX-LM reference on that task), promote PR #202's
  surface-B off its abandoned branch, rebase onto head, and gate on the surface-B gradchecks +
  on-par held-out NLL. Not required for GQA-adapter tasks.

- **G2 — In-process route→refit→route loop-closure.** `update_router` returns a fresh gate blob but
  nothing in lattice reloads it into a live `AdapterRouter` between requests; the caller owns
  persistence and reload. **Re-entry:** when a host runtime needs the refit gate to take effect
  without an out-of-process reload step, add an in-process `AdapterRouter::reload(gate_bytes)` path
  and a loop-closure integration test (route → collect feedback → refit → reload → route again). The
  boundary is deliberate today; this only fires if in-process closure becomes a requirement.

- **G3 — EWC anchor-pullback as a live anti-forgetting mode.** `penalty_gradient`/`ewc_lambda` is
  inert by design in v1 (Fisher null-space damping is the shipped mechanism). **Re-entry:** when
  multi-task router refits show measurable forgetting of earlier tasks under null-space damping
  alone, wire `ewc_lambda` live and A/B it against the projection-only path, with the existing
  inertness boundary test converted to a behavioral one.

- **G4 — Rerank-consumer disambiguation.** Two shipped rerank paths (cross-encoder ADR-057-D1;
  causal-LM #718) exist with no guidance on when to pick which. **Re-entry:** when a caller must
  choose between them for a real ranking task, add a short consumer-selection note (latency/quality
  trade-off, model-availability) — cheap, doc-only; deferred until a caller needs it.

## Consequences

- The seven prior LoRA ADRs are unchanged; this ADR adds a composition-level map above them and a
  corrected record, and supersedes none of them.
- The two corrections propagate: any downstream design that assumed GDN adapters train, or that #718
  satisfied #59's cross-encoder contract, must re-check against G1/G4.
- The QuaRot/RoLoRA composition axis (ADR-044/045/054) is orthogonal to this loop and unaffected —
  a rotation-aware trainer (ADR-054) would slot into the TRAIN stage without changing GOVERN/COMPOSE/
  ROUTE/CONSUME, since composition and routing operate on adapter blobs regardless of the basis they
  were trained in.
- G2's caller-owned loop-closure boundary is documented, not hidden: a host runtime wiring the
  adaptive loop knows it owns gate persistence/reload until G2 lands.

## S-row rider

When any Gn lands (most likely G1, surface-B GDN grads), record it as a status update / new row on
this ADR so the traceability chain shows idea → composed-record → gap-closure, not a silent flip.
