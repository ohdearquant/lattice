# ADR-077: Pruning + Distillation Priority (Metal + CPU)

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-inference, lattice-tune

> **Evidence basis**: The Decision below is grounded in this engine's own shipped source and its own
> status docs (the **Measured / source-verified reality** table). Like ADR-073/076, this is a **split
> verdict** across two families at different maturity levels — but here **both are NEEDS-EXPERIMENT**,
> because neither has a single committed quality measurement on top of its substrate. **Pruning** has
> real shipped substrate (a Block-Influence layer scorer wired to actual Metal forward-pass
> activations, plus in-memory layer-mask application) but its masks have **never been PPL-gated** —
> only checked against a 12-prompt logit-cosine + greedy-token smoke test (never perplexity), and unreachable from any CLI.
> **Distillation** is a **false cognate** in this repo: the one artifact named `DistillationPipeline`
> is a conversational-intent-label pipeline, not knowledge distillation; there is **zero**
> KL/soft-target/teacher-logit machinery anywhere. This ADR **builds on ADR-060** (the pruning
> toolbox, which honestly self-reports partial shipment) and does **not** amend it; the distillation
> false-cognate is likewise already corrected in issue #499's own body and audited by closed issue
> #302, so this ADR references those rather than minting a new correction. A genuinely repo-grounded
> external prior-art survey is folded as the **[prior, unvalidated on our hardware]** section and
> independently corroborates every source finding. The ranking is fixed by the shipped source and the
> already-filed experiment chain, not the survey.

## Context

Two model-compression families are in scope. **Pruning** = removing structure to shrink/speed the
model: depth pruning (drop whole layers/blocks, e.g. ShortGPT-style Block-Influence ranking), width
pruning (heads / FFN channels), sparsity. **Distillation** = training a smaller/faster student from a
teacher's outputs (KL / soft-target loss) — the object of a standing project to stand up a teacher
proxy feeding a distillation path in `lattice-tune`.

What ships today (table below): a Block-Influence (BI) scorer (`pruning.rs`, ADR-060 P1, PR #133
merged), a real in-memory `apply_layer_mask`, and a Metal `score_layer_importance` that scores layers
from **actual** forward-pass activations and produces a **type-balanced** GDN/GQA mask (it quotas by
layer type — so it does not naively ignore the hybrid architecture). But: the resulting masks have
**never** been evaluated against perplexity, only a tiny logit-cosine smoke test; no pruning surface
is reachable from any binary; and ADR-060's own status footer lists the calibration pipeline, width
pruning, `PrunePlan` serialization, and the PPL-gated workflow as **not shipped**. On the distillation
side, the `DistillationPipeline` targets a 6-class intent classifier with a hardcoded placeholder
teacher call (issue #11); no next-token KD exists. The only reusable training substrate is the CPU
LoRA NLL trainer, whose trainable range is hardcoded to the 0.8B top layer (`TOP_LAYER = 23`): it
runs on larger models but can only reach layers `0..=23`, so it cannot recover the top of the
64-layer 27B stack without parameterizing that range first.

## Measured / source-verified reality (this engine)

Tags: **runtime-measured** (a number from running this hardware), **unit-test/gate-pinned** (a
committed test or bench enforces it), **source-read** (a structural fact from merged code),
**internal-profiling-no-artifact** (measured but not committed durably). Pointers on
`origin/main @ 58e885ada`.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| R1 | **BI scorer is shipped** (ADR-060 P1): `cosine_similarity`, `BlockInfluence`, `BlockInfluenceAccumulator`, `score_from_hidden_states`, `pruning_rank`. Its own module doc scopes it: *"a standalone scorer utility, not the full ADR-060 calibration pipeline."* Public API. | source-read + unit-test-pinned | `crates/inference/src/pruning.rs:9` (doc), `:41,:59,:118`; `crates/inference/src/lib.rs:76` (`pub mod pruning`); PR **#133** MERGED |
| R2 | **In-memory layer removal is shipped**: `Qwen35Config::apply_layer_mask(mask)` (panics on wrong length / all-false), plus `is_layer_active`/`num_active_layers`/`num_active_{linear,full}_attention_layers`. Unit-tested. | source-read + unit-test-pinned | `crates/inference/src/model/qwen35_config.rs:638` (+ helpers above), tests `:1112,:1120` |
| R3 | **The Metal scorer runs REAL activations and is type-balanced**: `score_layer_importance(calibration_prompts, prune_layers)` computes per-layer mean cosine from actual traces via `forward_prefill_layer_traces_last_token`, and its recommended mask **quotas by layer type** (≈1 GQA removed per 3 GDN) with a warning past ~20% removal. Returns `LayerImportanceScore`/`LayerPruningPlan`. | source-read (no committed run output) | `crates/inference/src/forward/metal_qwen35.rs:12318` (traces), `:12397` (scorer), `:12437-12456` (type-quota), `:2607-2627` (types) |
| R4 | **Masks have never been PPL-gated.** The only committed quality check is a 12-prompt / ≤32-token logit-cosine + greedy-token-match smoke test that runs unconditionally (`bench_quality.rs`); the *optional* `score_layer_importance` scorer — gated behind `LATTICE_QUALITY_SCORE=1` — uses a separate 4-prompt / ≤8-token calibration subset. Neither is perplexity. A throughput example (`bench_pruning.rs`) exists but no committed markdown/PR carries any captured number. Issue #492 itself calls the scorer's calibration set "too small to trust." | source-read (absence) | `crates/inference/examples/bench_quality.rs:40,57-83,200-215`, `bench_pruning.rs`; no `BENCH_PRUNING_RESULT`/`mean_cos` in `docs/**/*.md` (grep, zero hits) |
| R5 | **No PPL gate, no `PrunePlan` artifact, no width/head/FFN pruning, no CLI surface.** `eval_perplexity.rs` exists (strided sliding-window PPL, CPU+Metal Q4) but has no `--layer-mask` flag; `CalibrationObserver`/`ForwardCtx`/`Wanda`/`SliceGPT` appear only in comments; no `head_prune`/`channel_prune`/`width_prun` anywhere; no `bin/` references pruning. | source-read (absence, exhaustive grep) | ADR-060 status footer `docs/adr/ADR-060-pruning-toolbox.md:680-684`; `crates/inference/src/bin/` (16 binaries, zero pruning refs) |
| R6 | **ADR-060 self-reports partial shipment** (2026-06-30): shipped = BI scorer + `apply_layer_mask` + Metal `score_layer_importance`/`LayerPruningPlan`; **not shipped** = D1 calibration observer, D3 SliceGPT, D4 Wanda, D5 GQA head pruning, D6 SwiGLU FFN pruning, D8 `PrunePlan` serialization, D7 PPL-gated workflow. | source-read (repo's own ground-truth doc) | `docs/adr/ADR-060-pruning-toolbox.md:680-684` |
| R7 | **`DistillationPipeline` is a false cognate** — a 6-class conversational-intent labeler (`IntentLabels`: continuation/topic_shift/…), with a hardcoded placeholder teacher call ("Placeholder: simulate labeling"), tracked as open issue #11. It is **not** knowledge distillation and would not produce next-token soft labels even once #11 wires the HTTP client. | source-read | `crates/tune/src/distill/pipeline/distill.rs:69` (placeholder), `:78` (IntentLabels); #11 OPEN |
| R8 | **Zero KD machinery exists.** No `kl_div`/`soft_target`/`teacher_logit`/`logit_distill`/`forward_kl`/`reverse_kl` anywhere in `crates/`. | source-read (absence, exhaustive grep, zero hits) | `git grep` over `crates/` → no matches |
| R9 | **The real reusable training substrate is the CPU LoRA NLL trainer** — pure position-wise NLL/cross-entropy (`position_nll`, `nll_and_grads` with softmax-CE gradient `prob − indicator`), no KD term. Prior lattice micro-LoRA runs are **plain-NLL adapter finetunes, not distillation-shaped** — the trainer has no soft-target/teacher-logit path, so it could not be KD even if repurposed. | source-read | `crates/tune/src/lora/train_core.rs:199,367-400`; `train.rs:209-230` |
| R10 | **The LoRA trainer's trainable range is hardcoded to the 0.8B top layer** (`TOP_LAYER = 23`): `train_micro_lora` iterates `first_layer..=23`, and its guard only rejects models with `num_hidden_layers <= 23` (a 48-layer model is explicitly accepted in tests). So on the 64-layer 27B the trainer **runs** but can only reach layers `0..=23` — it never trains the top 40 layers (`24..=63`), so top-layer recovery on the 27B is impossible without parameterizing that range. **No open issue named this** (now filed, #730). | source-read + unit-test-pinned (mutation-sensitive `TOP_LAYER` guards; 48-layer accept test) | `crates/tune/src/lora/train_core.rs:13`; guard+iteration `train.rs:120-135,248-251`; 48-layer accept test `train.rs:611-617`; `qwen35_config.rs:213` (0.8B=24), `:303` (27B=64) |

## Prior / unvalidated on our hardware

Folded from the external survey (`fleet_atlas_lat_prunedistill_001` harvest) as **data**. Unlike the
fully repo-blind packets folded by ADR-073/074/075, this packet is **genuinely repo-grounded**: it
cites a real commit (`f8c302f9e`, a verified ancestor of `origin/main`) and its file-level claims
cross-check true against the table above. It is used only as corroboration; the ADR's proof rests on
the repo-verifiable source/issue facts, not packet-internal text.

- **"`DistillationPipeline` is an intent-label pipeline with placeholder external calls" — matches R7
  exactly.** Independently confirms the false cognate, as does issue #499's own body.
- **"The reusable base is the NLL/CE LoRA trainer (`position_nll`, `nll_and_grads`)" — matches R9,
  including the function names.**
- **BI-on-hybrid is the unresolved question, both ways.** The packet found **no published paper
  validating ShortGPT/BI specifically on a Qwen-style GatedDeltaNet+GQA hybrid** (closest: Mamba-Shedder,
  a component-ablation study on Qwen3.5-0.8B — adjacent, not direct); its "Hybrid GDN/GQA validation?"
  column is "No" for every scoring method. It rates generic-BI-transfer-to-hybrid **0.35 confidence
  (low)** and recommends treating BI as "a candidate generator, not ground truth" with per-type
  z-score normalization + single-layer-ablation validation — a design recommendation, not evidence.
  This is exactly the open question issue #495 already frames.
- **Repo-verifiable divergences (not trusted):** the packet's `patch.diff` targets new files
  (`prune_gate.rs`, `prune_recovery_kl.rs`) that do not exist at the cited SHA (it flags these as
  approximate), and its reference `code/` crate was never compiled in its sandbox. Only the
  verdict-level corroboration above is used; its NEEDS-EXPERIMENT verdict matches this recon's.

## Decision

**Both families are NEEDS-EXPERIMENT. Rank the validation of the shipped scorer first; treat KD as
gated behind it. The pivot is a perplexity gate, not new machinery.**

**Rankable now (no experiment needed to scope):**

- **P1 — run the PPL-gate on the shipped scorer (#492).** This is the pivot: wire the already-shipped
  `score_layer_importance` to a CLI command with a real calibration corpus and a PPL gate, emitting a
  serialized `PrunePlan` artifact. It reuses 100% of the shipped scorer + the existing
  `eval_perplexity` PPL math behind a thin new `--layer-mask` flag — the only missing piece is the
  plumbing between them. #492 **blocks #495/#499/#502/#503 by its own text**, so it is the single
  unblocking step for the whole chain. The minimal experiment (below) is #492's own scope.
- **P2 — validate BI-ranking against PPL on the hybrid (#495).** With #492's harness, measure the
  scorer's recommended mask against the existing hand-picked masks (8-/12-layer removal) on
  Qwen3.6-27B by ΔPPL at matched removal counts. This is the ShortGPT-BI-on-hybrid question the survey
  flags as unresolved in the literature; #495's own promote bar (scorer must not lose to hand masks by
  >0.2 PPL) is inherited.
- **P3 — generalize the `TOP_LAYER = 23` hardcode (R10).** A small, well-scoped refactor
  (parameterize the trainable-layer range so the trainer targets the actual top layer(s) of any
  model) that is a hard prerequisite for **any** 27B recovery/KD work: today the trainer runs on the
  27B but can only reach layers `0..=23`. **Filed as #730** (surfaced by this ADR); it must land
  before #499 can run at 27B scale.

**Gated, not ranked:**

- **#499 (KL-distillation recovery loop) is not given a build priority yet.** It requires three things
  none of which exist: a KL-loss term added to the LoRA trainer (greenfield — R8), the P3
  `TOP_LAYER` generalization, and a pruned model worth recovering (so it is gated behind P1/P2
  producing a validated prune plan). Its own body already corrects the DistillationPipeline
  conflation; it graduates to a ranked priority only once a pruning result exists to recover from.
- **#502 (teacher-KV-prefix reuse) and #503 (iterative re-scoring) are conditional/secondary.** Both
  are speculative refinements of the KD-recovery loop; #503 defers itself behind #495/#499. Neither is
  worth prioritizing ahead of the primary PPL-gate.

**Not a compression lever — noted to prevent conflation.** `DistillationPipeline` / #11 is a
conversational-intent classifier, **not** knowledge distillation; wiring its HTTP client (#11) does
not advance model compression. The standing "teacher-proxy → distillation" project is **greenfield**
here (zero KL/soft-target/teacher-logit code, R8) and would build on the LoRA NLL trainer, not on
`distill/`. Width/head/FFN pruning (ADR-060 D3–D6) is likewise unbuilt and out of scope for this
priority pass, which ranks the depth-pruning path that has shipped substrate.

**Ordering rationale.** #492 is the one unblocking measurement: the scorer is shipped and wired
(R1–R3) but its masks are unvalidated (R4), so the highest-leverage next step is proving whether they
beat hand-picked masks — until that PPL number exists, ranking the KD-recovery loop (#499) or the
speculative refinements (#502/#503) is premature, and they are explicitly downstream. This applies the
experiment-gated discipline of ADR-073, with the ADR-076 refinement that the gate already exists as a
filed issue (#492) with its own thresholds — this ADR ratifies that gate and ranks running it. Unlike
ADR-076, the experiment here **validates an already-shipped primitive** (the scorer) rather than
deciding whether to build one.

## Consequences

- Pruning gets a clear next step (#492's PPL gate) that reuses shipped code, and an honest label: the
  scorer is built but unproven, so the work is validation, not construction.
- The distillation "substrate" is correctly identified as a false cognate; real KD is greenfield and
  correctly gated behind both a pruning result and two prerequisites (KL loss + `TOP_LAYER`
  generalization), so no one schedules a 27B KD run expecting `distill/` to help.
- The `TOP_LAYER = 23` prerequisite (the trainer reaches only layers `0..=23`, never the top of a
  64-layer model) is filed as **#730**, preventing a later 27B recovery attempt from stalling on it.

## Follow-ups

- **Commit the experiment artifact.** R4 shows the repo's pattern-of-failure: pruning masks have
  runnable harnesses but zero committed numbers. #492/#495 must land results as a committed
  `PrunePlan`-shaped artifact (model id, commit, config + corpus hash, ΔPPL), not example stderr.
- **`TOP_LAYER` generalization issue filed** (P3): #730.
- **GPU flock on all runs.** #492/#495 drive the Metal GPU on a 27B model; acquire
  `/tmp/lion-metal-gpu-test.lock` (machine-wide GPU test lock; contended GPU work corrupts timing and
  numerics, #628/#629).
- **Calibration corpus.** The scorer's current 4-prompt / ≤8-token set is too small (R4, #492); a real
  held-out corpus (disjoint from the PPL eval slice) must be assembled or reused before the scorer's
  mask is trusted.
- **If the scorer's mask loses to hand-picked masks** (#495 kill), record it as a new row on this ADR
  (measured closure of the generic-BI-on-hybrid question) rather than silently continuing — the same
  forward-instruction discipline the sibling ADRs carry.
