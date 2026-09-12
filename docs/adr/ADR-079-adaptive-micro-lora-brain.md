# ADR-079: Adaptive Micro-LoRA Brain — the Composed Train → Govern → Compose → Route → Consume Loop

**Status**: Proposed (amended 2026-09-10 — see Amendment 1)
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
lifecycle slice or an orthogonal composition axis; none covers the composed _pool-of-adapters_
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

| Stage                                                                  | State                                                                                  | What shipped                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Tag                       | Pointer (merge commit · file:line)                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TRAIN**                                                              | shipped, one gap                                                                       | Reverse-mode autodiff through Qwen3.5 gated-GQA attention (linear/lora/rmsnorm/rope/swiglu/cross_entropy VJPs), full-depth multi-layer tape, bounds-checked reusable wrapper. Verified on-par with MLX-LM on held-out NLL (PR body: lattice 4.9056→4.6600 vs MLX 4.9052→4.6897 over 30 steps; M1 lm_head NLL 5.1757→0.6103).                                                                                                                                                                                | PR-merge + source-read    | **#191** `7b46f6e773b6256f7168b627926655a2617a990c` (2026-06-20) · `crates/inference/src/backward/{ops.rs,attention_gqa.rs,gradcheck.rs,tape.rs}`; wrapper `crates/tune/src/lora/train_core.rs::forward_full:207` / `train.rs::train_micro_lora:209` (extracted from #191's inline code by #445/#488 `91e396409…`)                                                                                                                                                                           |
| **TRAIN — the gap (CLOSED 2026-07-10, see the note under this table)** | **GDN LoRA weight-grads WERE not on main when this was written; they are on main now** | As of 2026-07-09 the GDN backward was **dx-only**: it propagated gradient _through_ the frozen GDN layers into the lower GQA LoRA grads but produced no GDN-layer LoRA weight gradients, and `forward_full`'s `MixerKind::Gdn` arm took **no** `lora_slot` while the `MixerKind::Gqa` arm did, so the trainer of that day could teach only GQA-layer (`q_proj`/`v_proj`) adapters. **This is no longer true at head; see the supersession note below.**                                                     | source-read + issue-state | `attention/gdn_backward.rs` (1228 LOC, "dx-only VJP", unchanged since #191); `train_core.rs::forward_full` Gdn arm `:259` (no LoRA) vs Gqa arm `:228` (`lora_slot`). Surface-B commits (`f34ed3b1…` + siblings, from the closed/unmerged **PR #193** lineage) are **not** ancestors of `origin/main`; the current standalone GDN-grad **PR #202 is CLOSED and unmerged** (verified 2026-09-08: state CLOSED, draft, `mergedAt` null), its head commit (`5e0bec19…`) likewise not an ancestor |
| **GOVERN**                                                             | shipped                                                                                | `LoraManifest` (schema `version: u32`) + `ManifestEntry` (integrity_sha256 / base_model_rev / tokenizer_rev / rank / alpha / target_modules / dtype / status) + `AdapterStatus{Approved,Quarantined,Revoked}`; `load_adapters_from_manifest` runs eleven ordered fail-closed checks (status → uri → existence → integrity-sha256 → … → running base-model/tokenizer-revision match). This is the admissibility gate between "a trainer wrote a safetensors file" and "the blend/route stages may touch it." | PR-merge + source-read    | **#444** `a729dad6ca6501ca79a49120070440bd685a698d` (2026-06-29, extended by #624) · `crates/tune/src/lora/manifest.rs:26,64,100`; `loader.rs:15,42,115`                                                                                                                                                                                                                                                                                                                                     |
| **COMPOSE**                                                            | shipped, policy deferred here                                                          | Exact weighted-concat math folding N adapters' `alpha`/`rank` scales into one rank-Σr adapter (`blend_lora_adapters`), plus decode-time wiring into a **single Metal slot with zero kernel change** (`blend_lora_layer_data` → `generate_with_lora_mixture`). `AdapterRouter` does top-k selection with **constant, non-learned `1/k` weights** — selection _policy_ is explicitly out of scope in this PR (it is what ROUTE supplies next).                                                                | PR-merge + source-read    | **#443** `09a23c00f1bd998aa3cfdb2624939801b4e18180` (2026-06-29, closes #436/#437/#438) · `crates/tune/src/lora/blend.rs:73`; `crates/inference/src/forward/metal_qwen35.rs:1623,3897`; `crates/inference/src/mixture.rs::AdapterRouter:94` (behind `mixture` feature)                                                                                                                                                                                                                       |
| **ROUTE**                                                              | shipped, one mechanism inert, loop-close externalized                                  | Model-agnostic RL substrate in `fann`: RLOO single-sample policy gradient (`RlooTrainer::step` + load-balance/z-loss) and diagonal Fisher (`DiagonalFisher`). LoRA-specific consumer `update_router` batches `FeedbackEvent`s, refits the gate via RLOO, damps updates to previously-important params via Fisher **null-space projection**, returns a `RouterDelta{network_bytes}` (a full serialized gate, not a param diff).                                                                              | PR-merge + source-read    | **#448** `0006ca67cc613da4feea74d2d92c33fb1b3b9c2a` (fann RL primitives, `online-router` feature) · `crates/fann/src/training/rloo.rs:21,49,445,462`, `ewc.rs:30`. **#453** `2dd2fc73f9d3611ae8d009166d70aff164d142d6` (2026-06-30, "capstone", issue #440) · `crates/tune/src/lora/router_update.rs:58,102,125,170,199,318`                                                                                                                                                                 |
| **ROUTE — the partials**                                               | inert-by-design + externalized loop                                                    | (a) The alternative EWC anchor-pullback penalty (`penalty_gradient`, gated by `ewc_lambda`) is **deliberately inert in v1** — a boundary test pins byte-identical output regardless of `ewc_lambda`; v1 anti-forgetting is Fisher null-space damping only. (b) Persisting `RouterDelta.network_bytes` back into a live `AdapterRouter` between requests is **left entirely to the caller**; nothing in-process closes the route→refit→route cycle inside lattice.                                           | source-read + test-pinned | `router_update.rs` boundary test `ewc_lambda_is_inert_in_projection_path`; `RouterDelta` doc `router_update.rs:170-176` ("Load with `Network::from_bytes()` … pass as the `gate_bytes` argument to the next `update_router` call")                                                                                                                                                                                                                                                           |
| **CONSUME**                                                            | shipped, two independent siblings                                                      | Generation consumes a composed adapter transparently via `set_lora` (exercised by `generate_with_lora_mixture`). Reranking has **two** shipped siblings through the same seam: the cross-encoder/BERT query-likelihood path and the causal-LM query-likelihood path. #718's own doc comment: "scoring flows through the model forward, so a composed adapter installed via `set_lora` is applied transparently — no rerank-side changes are needed."                                                        | PR-merge + source-read    | seam: `crates/inference/src/model/qwen35/model.rs::set_lora:69` (`lora` field `:18`). Causal-LM rerank **#718** `b0c9ed56bc9f5ba848a383a948271ca94f8068af` (2026-07-08) · `model/qwen35/rerank.rs:47,139`. Cross-encoder rerank **ADR-057-D1** (PR #65 merge `8ac486dbf7bec91417a5d777500a1237ab09d07c`; D1 hook commit `894dfdc3`, 2026-05-23, issue #59) · `model/cross_encoder.rs:76`                                                                                                     |

### Record corrections (were circulating as fact)

> **Superseded on 2026-09-08 by re-verification at `origin/main` `3f41fbbc42e6b88a944a409a140f77b0bffd588f`.** G1 closed on 2026-07-10, ONE DAY after this record was written, and the record was never updated. PR #792 `fac5fa677f` ("GDN LoRA weight gradients via train_core (port of #202)", merged 2026-07-10T23:21:09Z) landed the GDN-layer LoRA weight gradients, and PR #1322 `b76c4278c0` (2026-08-06) wired them into `train_micro_lora` and added `crates/tune/tests/gdn_lora_wiring.rs`. Both are ancestors of `origin/main`; `5e0bec19` (#202's head) is not, and #202 is CLOSED precisely because its content was ported into #792. Evidence at main: `GdnGrads` carries ten LoRA weight-gradient vectors (`grad_a_qkv`/`grad_b_qkv`, `_z`, `_b`, `_a`, `_out`) beside `dx`, not `dx` alone; the driver calls `apply_gdn_adam_updates`; and the trainer prints its GDN slots at startup. Everything below in this row describes the tree as of 2026-07-09 and is kept for provenance, not as a statement about head.**

1. **The merged CPU backward trainer is PR #191 (`7b46f6e77`), not commit `f34ed3b1`.** The
   "surface-B GDN LoRA weight-grad" commits (`f34ed3b1…` + siblings, 2026-06-21/22) are **not
   ancestors of `origin/main`** — `git merge-base --is-ancestor <sha> HEAD` is false for each. Two
   separate PRs carried this work and neither is on head: **PR #193** (now CLOSED/unmerged) contained
   `f34ed3b1`; the standalone **PR #202** is CLOSED and unmerged, its head `5e0bec19…` a non-ancestor.
   **This paragraph's conclusion held for one day.** On 2026-07-10 PR #792 `fac5fa677f` landed the
   same capability as a port of #202 through `train_core`, which is why #202 was closed rather than
   merged, and PR #1322 `b76c4278c0` (2026-08-06) wired it into `train_micro_lora`. Both are
   ancestors of `origin/main`. On head today GDN-layer LoRA weight-gradients DO exist in merged code,
   and a "GQA+GDN LoRA grads shipped" claim is now correct rather than an overstatement.

2. **PR #718 does not close issue #59.** #718 is the causal-LM query-likelihood reranker
   (`Qwen35Model::rerank`). Issue #59 ("Stable rerank API on lattice-inference with `LoraHook`
   injection") asked for `LoraHook` injection into the **cross-encoder** path and was closed
   2026-05-23 by **PR #65** (ADR-057-D1, `CrossEncoderModel::score_with_hook`) — five weeks before
   #718. #718's `closingIssuesReferences` is empty. Two independent rerank-via-seam consumers now
   exist; no document previously named them as siblings.

## Existing-ADR coverage and the gap

| ADR                                | Status                     | Scope (single slice / orthogonal axis)                                                                                                                                                                                                                   |
| ---------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **008** lora-injection             | Accepted                   | The `LoraHook` trait seam itself (single adapter, forward-only `apply`); why it lives in `lattice-inference` not `lattice-tune` (avoids a circular dep).                                                                                                 |
| **031** lora-adapter-management    | Accepted                   | Three-layer single-adapter representation + PEFT/MLX safetensors **import** (load only; no governance, export, or multi-adapter).                                                                                                                        |
| **043** lora-serving-verification  | Accepted                   | CI coverage that the hook fires at all 12 projection sites (4 GQA + 5 GDN + 3 MLP) for **one** loaded adapter.                                                                                                                                           |
| **045** quarot-lora-composition    | Accepted                   | Counter-rotation math so a single LoRA composes with a QuaRot-rotated Q4 base at serving time. Orthogonal axis (quantization × LoRA), not multi-adapter.                                                                                                 |
| **054** rolora-rotation-aware-lora | Proposed, not implemented  | Training LoRA natively in the QuaRot-rotated basis; a different _training axis_ (rotation-awareness), blocked on ADR-056.                                                                                                                                |
| **056** lora-tuning-pipeline       | Proposed + status appendix | Designs a `LoraTrainLoop`/`train/lora/` pipeline. Its own appendix admits the design was **never built** and redirects readers to `train_grad_full.rs` / `train.rs` / `online.rs` — the clearest existing admission of the gap this ADR closes.          |
| **057** lora-consumer-api          | Accepted                   | Five lifecycle gaps for **one adapter, one request**: D1 cross-encoder hook (#59), D2 export (#60), D3 `adapt_step` per-event weight SGD (#61), D4 typed module names (#62), D5 docs (#63). No multi-adapter governance, no mixture, no learned routing. |

**The gap.** None of the seven describes the composed system: real-gradient multi-layer training that
_produces new adapters_ (vs ADR-056's un-shipped design), governance/approval status across a _pool_
of adapters (vs ADR-031's single adapter), weighted _composition_ of multiple adapters into one at
decode time, or a _learned selection policy_ (RLOO + Fisher) over which adapter(s) to use per request
(vs ADR-057-D3's per-event weight nudge on one already-selected adapter). This ADR is that
synthesizing record; it references the seven above rather than re-opening any of them.

## Decision

1. **Record the adaptive micro-LoRA brain as one composed, mostly-shipped system.** The
   TRAIN → GOVERN → COMPOSE → ROUTE → CONSUME loop is real and merged at head, with the exact
   shipped/partial/gap state pinned above by PR-merge commit and file:line. This ADR is the canonical
   map; future LoRA-system work references it instead of re-deriving the loop from seven separate
   ADRs.

2. **Fix the record.** As written on 2026-07-09: GDN-layer LoRA weight-gradients were not on main
   (PR #202 was never merged; `f34ed3b1` is not an ancestor), so only GQA adapters trained. **That
   half is now itself out of date** — see the supersession note under the stage table: #792 landed
   the gradients on 2026-07-10 and #1322 wired them on 2026-08-06. The second correction still
   stands unchanged: #718 is the causal-LM reranker and does **not** close #59 (that was #65).

3. **Name the gaps as follow-on, each with a re-entry condition** (a gap without a named re-entry
   condition is a parking ticket). See Follow-on work.

4. **Do not re-litigate the prior ADRs.** ADR-008/031/043/045/054/056/057 stand as written; this ADR
   only composes and maps them, and supersedes nothing.

## Follow-on work (named gaps, each with a re-entry trigger)

- **G1 — GDN-layer LoRA weight-gradients. CLOSED 2026-07-10, verified 2026-09-08.** The gap as
  described (GDN layers frozen, dx-only pass-through) was real on 2026-07-09 and was closed the next
  day by PR #792 `fac5fa677f`, a port of #202 through `train_core`; PR #1322 `b76c4278c0` wired it
  into `train_micro_lora` on 2026-08-06 with `crates/tune/tests/gdn_lora_wiring.rs`. No re-entry is
  needed and the trigger below is retired. **What this means for anyone sizing an adapter:** the
  trainer materialises GDN LoRA slots alongside GQA ones, so an adapter over a layer range is NOT
  q_proj/v_proj-only and its size is not the GQA-only figure. Read the trainer's own startup line,
  which names the GQA and GDN slot layers it actually materialised, rather than assuming from the
  range.

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
- Gate persistence stays caller-owned; `AdapterRouter::reload` closes the adaptive loop in-process.

## S-row rider

When any Gn lands (most likely G1, surface-B GDN grads), record it as a status update / new row on
this ADR so the traceability chain shows idea → composed-record → gap-closure, not a silent flip.

## Amendment 1 (2026-09-10) — G2's re-entry trigger has fired; the closure contract, fixed before it is built

**2026-09-12 — G2 landed:** atomic `AdapterRouter::reload` and route → feedback → refit → reload → route regression tests close the in-process loop.

**G2 has not landed. Its trigger has.** G2 was written to fire "if in-process closure becomes a
requirement", and it now has: the adaptive-loop work item wires feedback through `update_router`
inside the serving path, which is the host runtime G2 describes. This amendment records the trigger
firing and fixes the contract, so the implementation and this ADR agree on what G2 means rather than
discovering it afterwards. The S-row rider above asks for a status update rather than a silent flip,
and this is it.

### Source-verified surface (`origin/main @ 04c5d794ab`)

- `update_router(gate_bytes: &[u8], events, replay, fisher, config) -> Result<RouterDelta>` —
  `crates/tune/src/lora/router_update.rs:242`.
- `RouterDelta.network_bytes: Vec<u8>` — `router_update.rs:141-147`. Its own doc comment is explicit
  that this is "a complete network blob produced by `Network::to_bytes()`, not a parameter diff".
- `AdapterRouter { gate: Network }`, with `new`, `route`, `input_size`, `output_size` —
  `crates/inference/src/mixture.rs:95-208`. There is no `reload`, which is the gap as stated.

### The closure does not invert the dependency direction, and that is why it is cheap

`update_router` lives in `lattice-tune`; `AdapterRouter` lives in `lattice-inference`; inference does
not depend on tune. That asymmetry is the real reason the boundary has been caller-owned, and it
reads like an obstacle to closing it. It is not. `RouterDelta.network_bytes` is a FANN blob and
`AdapterRouter`'s field is a fann `Network`, so a reload path needs only `Network::from_bytes` —
already reachable from inference, which takes `lattice-fann` under `mixture`
(`crates/inference/Cargo.toml:51`). The closure is `inference + fann`, and the leaf-crate rule holds
unchanged.

**So the contract is:** `AdapterRouter::reload(&mut self, gate_bytes: &[u8]) -> Result<(), _>`,
accepting exactly the bytes `update_router` returns, in `lattice-inference` under `mixture`. The
caller still owns persistence; what it no longer owns is the out-of-process round trip.

### Dimension checking at reload is about blast radius, not about safety

The obvious thing to specify here would be that `reload` must reject a mismatched gate because
`route` would otherwise mis-select. **That is not true, and the reason it is not true is worth
recording**, because it is the kind of claim an implementer would accept from an ADR without
re-reading the routing body. `route` already validates both dimensions at use time and fails closed:
a context vector that does not match the gate's input width returns `RouterError::InputSizeMismatch`
(`mixture.rs:158-164`), and a gate narrower than the requested `k` returns
`RouterError::GateTooNarrow` (`mixture.rs:171-178`, whose comment says the check exists so that a
narrow gate "beats a panic inside `select_nth_unstable`"). A mismatched reload cannot produce a
wrong adapter selection. There is no memory-safety or correctness hole to close.

What a mismatched reload _does_ produce is a router that fails **every subsequent request** instead
of failing the one call that caused it. `reload` is the only moment at which the caller can still
choose to keep serving on the previous gate; after the swap, the old network is gone and each
inbound route pays the error. So `reload` validates `num_inputs`/`num_outputs` against the live gate
and returns an error **without mutating `self`** when they differ — the gate either swaps wholly or
not at all. The requirement is that a bad refit costs one failed reload rather than an outage, and
the no-partial-mutation clause is the part that makes that true.

One behaviour is deliberately left as it is. A gate narrower than the adapter pool, with `k` small
enough to stay inside `GateTooNarrow`, silently restricts routing to the first `num_outputs`
adapters (`n = available.len().min(scores.len())`, `mixture.rs:171`). That is pre-existing — a
caller can construct such a router today through `new` — and `reload`'s equal-dimension rule means a
reload cannot introduce it. Widening the pool remains what it already is: a new `AdapterRouter`, not
a reload.

### What the loop-closure test must show

G2 names the test as "route → collect feedback → refit → reload → route again". Stated as an
assertion rather than a sequence: the second `route` returns a **different** selection than the
first, for the same context vector and adapter pool, with the difference attributable to the
feedback. A test that merely runs the five steps and asserts no error would pass against a `reload`
that discards its argument, which is the failure this ADR's own follow-on discipline exists to
prevent. The refit must be mutation-visible at the router, or the loop is not closed. The
mismatch arm needs its own case: reload a wrong-width gate, assert the error, then assert `route`
still succeeds on the original gate — that is what pins the no-partial-mutation clause.

### Consequence for the earlier text

The Consequences line reading "a host runtime wiring the adaptive loop knows it owns gate
persistence/reload until G2 lands" is unchanged and still accurate: G2 has not landed. It stops
being accurate the moment `reload` merges, and should be revised in that same change rather than
left to decay.
