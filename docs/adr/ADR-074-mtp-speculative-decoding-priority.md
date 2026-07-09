# ADR-074: MTP / Speculative-Decoding Priority (Metal)

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own shipped source (the
> **Measured / source-verified reality** table) and the single throughput number that exists for
> this feature — a self-flagged-unreliable regression on one issue comment. Unlike the weight-quant
> ADR (ADR-072), and even more sharply than the KV-quant ADR (ADR-073), the verdict here is
> **NEEDS-EXPERIMENT**: the production path's *correctness* is settled and needs no ranking, but its
> *profitability* rests on one measurement whose own author flagged it unreliable, and the in-tree
> mechanism purpose-built to fix its measured root cause has **never been run**. A parallel external
> prior-art survey (EAGLE / Medusa / temperature-correct rejection-sampling literature) is folded as
> the **[prior, unvalidated on our hardware]** section; it independently reached NEEDS-EXPERIMENT but
> from a repo-blind vantage, and its central acceptance figure is an imported stale number the source
> issue itself caveats. The ranking is fixed by the measured decode profile and a defined minimal
> experiment, not the survey.

## Context

Multi-token-prediction (MTP) speculative decoding drafts candidate tokens with an auxiliary head and
verifies them against the target model in a single pass, trading extra compute for fewer sequential
decode steps. On a memory-bandwidth-bound decoder its payoff is strictly
`E[accepted tokens] / (verify_cost / baseline_decode_cost)` — acceptance in the numerator, verify
overhead in the denominator. This ADR ranks the **open MTP / spec-decode levers** — MTP profitability
and dynamic draft depth (#419), temperature-correct probabilistic acceptance (#388), tree-structured
verification (#591), GDN-snapshot rollback (#176), and the audit-flagged KV-drift concern (#293) —
against what this engine has actually shipped and measured, and records precisely which question
needs an experiment so effort is not spent speculatively.

Three facts shape everything below:

1. **The live MTP path is off by default, narrow, and already correct by test.** Both serving
   binaries construct `GenerateConfig` with `enable_mtp: None`; MTP activates only via the
   `LATTICE_MTP` env var. When enabled it fires only in a greedy-only, single-draft-token (K=1) mode
   that is regression-tested to be *exactly argmax-equivalent* to plain greedy decode. There is no
   open correctness question on the live path — so "MTP correctness" is not something this ADR ranks.
2. **The one throughput number is a net -64% regression, and its own author flagged it unreliable.**
   The only measurement in existence (issue #419, an *issue-comment* artifact, not a committed bench)
   is 175.7 → 63.0 tok/s free-form on Qwen3.5-0.8B Q4, whose analysis attributes the loss entirely to
   **verify cost, not draft quality**: the default verifier runs the full target model **sequentially
   twice** per round to check one draft token. Its 0% acceptance figure is self-caveated as possibly a
   *weight-basis* artifact (MTP tensors possibly not ADR-051 counter-rotated), not a real speculation
   signal. So there is currently **no trustworthy acceptance-rate measurement on any workload.**
3. **The fix for that root cause is already in-tree and has never been run.** A second verifier
   (`verify_tokens_batch_gemm`, opt-in via `LATTICE_MTP_BATCH=1`) batches the weight-projection GEMMs
   across verify tokens specifically to cut the sequential-forward cost — its own doc projects the
   cost multiplier dropping from ~2.0× to ~1.35× at full acceptance. It has **zero tests, zero bench
   data, zero references anywhere else in the repo.** Nobody has flipped the one flag that would tell
   us whether MTP is even viable on this GDN-hybrid architecture. That flag flip is the single
   cheapest, highest-value experiment this ADR identifies.

## Measured / source-verified reality (this engine)

Each row is tagged **runtime-measured** (a number from running this hardware), **unit-test /
gate-pinned** (a value asserted by a committed test), or **source-read** (a structural fact read from
merged code on `origin/main @ 853f7b1d2`), with its most durable public pointer. A runtime number
that was never committed as a durable artifact says so.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| M1 | **MTP is off by default**: both serving binaries build `GenerateConfig` with `enable_mtp: None`; it activates only via the `LATTICE_MTP` env var, never programmatically defaulted on. | source-read | `bin/lattice_serve.rs:870`; `bin/chat_metal.rs:819,908`; `forward/metal_qwen35.rs:8540-8542` |
| M2 | **The live MTP route is gated to greedy-only**: `mtp_present && mtp_enabled && top_k<=1 && temperature<=0.0 && !use_compact && grammar.is_none() && stop_strings.is_empty()`. At any temperature>0 or top_k>1, MTP is bypassed entirely regardless of the env var. | source-read + unit-test-pinned | route `metal_qwen35.rs:562-575`; tests `:628-644` |
| M3 | **Live draft depth is hard-wired K=1**: `generate_greedy_mtp` calls `mtp_forward_one` exactly once per round, then verifies `[pending_token, draft.token_id]` (2 tokens). `MTP_VERIFY_MAX_TOKENS=2` is the verify-buffer size, **not** a draft-chain cap — `MtpConfig.draft_length` (default 4) drives a *different, dormant* multi-token chain (M11/M12). Lifting K is unbuilt work, not a config flip. | source-read | `metal_qwen35.rs:900,7902,7913-7920` |
| M4 | **Two verify implementations; the DEFAULT is the expensive one.** `verify_tokens_batched` (default) is doc'd as "the H1 sequential path (V_2≈2.0)": one full target forward per token, sequentially. `verify_tokens_batch_gemm` (opt-in `LATTICE_MTP_BATCH=1`) batches projection GEMMs across verify tokens to cut this; doc: *"Expected V_2: ~1.35 at α=100%, ~1.60 at α=75%, ~2.30 at α=5%."* | source-read | `metal_qwen35.rs:4155-4160` (sequential); `:4197-4207` (batch-GEMM); `:7913-7920` (dispatch) |
| M5 | **The batch-GEMM verifier has never been measured or tested.** Repo-wide grep for `verify_tokens_batch_gemm` / `LATTICE_MTP_BATCH` returns only the definition + one dispatch site — zero unit tests, zero benches, zero PR/issue/doc hits. (It also hard-panics on a MoE FFN target layer in batch mode — irrelevant to the current dense targets, a landmine for a future MoE base model.) | source-read (absence, exhaustively grepped) | `metal_qwen35.rs:4197-4318`; grep of `crates/`, `docs/`, issues/PRs = 0 hits beyond def + call |
| M6 | **Measured (Qwen3.5-0.8B Q4, real Metal, default sequential verifier): net -64% free-form** (175.7 → 63.0 tok/s). Per-round counters: draft 1.72 ms, **verify 11.16 ms**, rollback 0.33 ms. Author's conclusion: *verify cost, not acceptance, is the bottleneck* — even 100% acceptance at K=1/cap=2 projects only ~92.8 tok/s (-47%). **Artifact is an issue comment, not a committed bench file.** | runtime-measured (issue-comment artifact) | `gh issue view 419` comment, 2026-06-27 |
| M7 | **M6's acceptance figure (0%) is self-flagged unreliable by its own source**: the comment states the restored MTP weights are *"likely raw bf16 rather than ADR-051 counter-rotated, making the draft garbage"* and the verify path *"may be running an unfused dispatch"* (a 2.8× penalty seen elsewhere in the engine). Both unresolved — **no trustworthy MTP acceptance number exists on any workload.** | runtime-measured, flagged unreliable by its author | same `gh issue view 419` comment |
| M8 | **Counter-rotation *infrastructure* is wired**: a `RandomizedHadamard` (`quarot_rotation`) is constructed at load time when MTP weights load AND the checkpoint index carries a `quarot_seed` — the ADR-051 shim is live, not stubbed. Whether the *specific* shipped-checkpoint MTP tensors are basis-compatible (M7's caveat) is a separate open question this construction does not settle. | source-read | `metal_qwen35.rs:14353-14367` |
| M9 | **ADR-051 counter-rotation correctness has a dedicated equivalence test** comparing counter-rotated MTP draft logits against a reference — independent of M7/M8's live-checkpoint question, and the method E2 below extends. | unit-test/gate-pinned | `metal_qwen35.rs:17177-17236`, `mtp_draft_logit_equivalence_with_quarot_counter_rotation` |
| M10 | **The live greedy MTP round is regression-tested for exact argmax-equivalence to plain greedy decode** (the #237 fix), enforced by a dedicated test module and a row in the engine's stop-token-contract coverage table. | unit-test/gate-pinned | `metal_qwen35.rs:28394-28397`; `stop_token_contract.rs:38` (row 8) |
| M11 | **The probabilistic acceptance path is production-unreachable.** `generate_greedy_mtp` calls `rejection_sample_draft(..., greedy=true, ...)` **directly**, never through the trait-level `mtp_verify_draft`. The ADR-050 probabilistic-sampling code is exercised only by its own unit tests. | source-read (absence of call site) | `metal_qwen35.rs:7947-7952`; `speculative.rs:1339-1650` |
| M12 | **`MtpConfig.draft_length` (default 4) drives a different, CPU/trait-level multi-token draft chain** (`draft_tokens_with_logits`, autoregressive over the MTP head) — the dormant path from M11, not the live K=1 Metal path (M3). | source-read | `speculative.rs:1204-1230,125-160` |
| M13 | **No `temperature` parameter exists in the acceptance math.** `rejection_sample_draft`'s probabilistic branch softmaxes raw logits directly (implicit T=1.0); the signature has no temperature argument. This is #388's still-open half — the **seed** half was fixed (#329). Moot for production today (M11: this path is unreached). | source-read | `speculative.rs:1917-1924,1302-1311`; #388 (OPEN), #329 (CLOSED) |
| M14 | **A second, independent speculative mechanism — `LATTICE_SELF_SPEC=1` — exists and needs no MTP head.** "GDN-first self-speculative decode" drafts 4 tokens via GDN-only forwards (cheap recurrence, no KV writes) and verifies in one batched pass; also greedy-only-gated. It has **zero measured acceptance or throughput numbers anywhere** and **no ADR** (only MTP has ADR-006/050/051). | source-read (mechanism) + source-read (absence of measurement/ADR) | `metal_qwen35.rs:8068-8078,954,8562-8577` |
| M15 | **Zero committed bench artifacts for MTP or spec-decode anywhere** in `docs/bench_results/`. The engine already emits the needed counters (`metrics.mtp_ms`, `metrics.verify_ms`, `metrics.verify_calls`, under `LATTICE_MTP_VERBOSE=1`) — the *harness* exists, the committed *result* does not. | source-read (absence) | `ls docs/bench_results/`; counters `metal_qwen35.rs:7901-7930`, print `:8042` |
| M16 | **MTP-local KV-cache post-accept drift is an OPEN, not-yet-runtime-traced correctness concern** — filed from a static audit at medium confidence, a plausible bug not a measured one. | issue-state | `gh issue view 293` (OPEN, "[unverified]") |

**Derived (arithmetic cross-check, not a new measurement)**: M6's 11.16 ms verify cost ÷ the implied
5.69 ms/tok baseline (from 175.7 tok/s) ≈ 1.96 — almost exactly M4's "2 sequential full forwards"
structural explanation. The measured bottleneck is the verifier's *structure* at K=1, a narrower and
cheaper problem than a tree-scan framing would imply.

## Prior / unvalidated on our hardware

Folded from the external survey (`fleet_atlas_lat_mtpspec_001` harvest) as **data**. The packet was
written **without repository access** — its `QUESTIONS.md` lists 30 numbered asks for exact source
paths and layouts a repo-connected pass would simply read, and its `patch.diff` is self-labeled
"approximate ... exact patch requires the pinned SHA." Treat every quantitative claim as
generic-literature arithmetic, not lattice measurement.

- **Survey's own top-level verdict: NEEDS-EXPERIMENT** (independently reached, matching this ADR).
  Its stated reasons: (1) "measured free-form MTP acceptance is only 4-5%" and (2) "GDN verification
  cost grows with speculative depth because the recurrent component is not a free parallel prefix."
- **Reason (1)'s 4-5% number is an imported stale figure, not lattice-current** — it comes from an
  internal note dated 2026-05-01 that the survey packet had no way to know is superseded and caveated
  by M7 (the *only* live acceptance number is 0% and is itself flagged as a possible weight-basis
  artifact). The honest
  form: **no acceptance figure — 4-5% or 0% — is trustworthy until E2 resolves the weight-basis
  question.** The survey's arithmetic built on 4-5% is therefore unranked, not refuted.
- **Reason (2) is directionally right but misidentifies the bottleneck.** The measured cost (M6) is
  the *sequential-forward* structure of the default verifier at the current K=1 — a mitigation
  (`verify_tokens_batch_gemm`, M4/M5) already exists in-tree, unmeasured, which the repo-blind survey
  could not have known. The survey frames GDN verify-cost growth as *inherent*; M4/M5 show lattice has
  an unrun fix for the K=1 case that is upstream of any depth-scaling argument.
- **Temperature-correct probabilistic verification required** — survey CONFIRMED-BY-ANALYSIS/0.99;
  matches M13 exactly, correctly identified sight-unseen. **Seedable deterministic replay required** —
  survey 0.95; *already shipped* in lattice (#329, M13's seed half) which the survey could not know.
- **Structured/code workloads may make MTP profitable** — survey NEEDS-EXPERIMENT/0.70; still open, no
  lattice structured-workload acceptance number exists (M7). **Tree verification improves
  accepted-tokens/verify-step** — survey 0.65; matches #591's own maintainer-written framing.
- **Its recommended architecture** (tree-causal mask, micro-chunked GDN verify-scan, counter-based
  replayable RNG, EWMA dynamic-depth controller) is a plausible extension of lattice's actual shape,
  but **every element is downstream of the verify-cost question (M5/M6) the survey had no visibility
  into.** Its profitability arithmetic assumes a configurable draft length K; the live path is
  hard-K=1 (M3), so lifting K is itself unbuilt work, not the config knob the survey models.

## Decision

**Top-level verdict: NEEDS-EXPERIMENT.** Two things are rankable now without new measurement; the
throughput question and the acceptance-quality question are both genuinely gated behind a defined,
cheap experiment. Rank by *measured leverage × cost-to-decide*.

### Rankable now (no experiment)

- **Live MTP correctness is settled — do not spend ranking effort on it.** Off by default (M1),
  greedy-only when enabled (M2), single-draft-token (M3), argmax-equivalent to plain greedy and
  regression-tested (M10). Nothing is ambiguous. The only correctness items worth tracking are the
  *dormant* probabilistic path (relevant only if E1 below turns favorable) and the audit-flagged
  KV-drift concern (#293/M16 — a "needs a runtime trace" item, not a ranking question).
- **Self-spec (`LATTICE_SELF_SPEC`) needs its own ADR + a first measurement (P-separate).** It is a
  structurally different mechanism (M14: no MTP head, GDN-only draft, its own greedy gate) with zero
  measured data and no design doc. It must **not** be silently folded into "MTP" in any priority
  document. Ranking it is premature until MTP's own experiment settles — see the closing gate.

### NEEDS-EXPERIMENT — gated, in sequence

1. **E1 — batch-GEMM verifier net throughput (P1 experiment, cheapest, zero new code).** The
   `verify_tokens_batch_gemm` path (M4/M5) is purpose-built for exactly the cost M6 measured and has
   never been run. Flip `LATTICE_MTP_BATCH=1` alongside `LATTICE_MTP=1` and re-run M6's benchmark
   using the engine's already-emitted counters (`LATTICE_MTP_VERBOSE=1`, M15). This either invalidates
   M6's net-negative verdict or confirms it *under the intended-fast verifier*, closing the loop #419
   and the survey both left open. **This is the single highest-value next step** — it decides whether
   MTP has any future on this architecture, at the cost of two env-var flips and a bench run.
2. **E2 — a trustworthy acceptance-rate measurement, on any workload (P1, prerequisite to ranking
   #419's headline).** M7 flags the only acceptance number (0%) as possibly a non-counter-rotated
   weight artifact, not a speculation-quality signal. Before "is MTP profitable on structured/code
   workloads" (#419's headline, the survey's top uncertainty) can be *ranked* at all, the shipped Q4
   checkpoint's MTP weights need a basis-correctness check. The method already exists (M9) and needs
   extending from a synthetic fixture to the actual shipped checkpoint — no new mechanism, an
   extension of a committed test.
3. **E3 — temperature-correct probabilistic acceptance (#388, the survey's headline concern).**
   Confirmed genuinely missing (M13), but it lives entirely in a dormant, production-unreachable path
   (M11) — there is **no live user-facing correctness gap today**. Sequence it **after** E1/E2: wiring
   probabilistic sampling into a feature nobody can profitably enable is wasted effort if MTP stays
   net-negative under the fast verifier; if E1 shows a net-positive path, E3 becomes a real
   precondition for ever enabling MTP at temperature>0.
4. **E4 — tree-structured verification (#591), deferred.** Matches the survey's own downstream
   framing: gated behind GDN-snapshot rollback correctness (#176, "highest-risk experiment; kill early
   if <0.72 accept") *and* behind E1-E3 settling whether the linear K=1 case is even profitable. Not a
   near-term experiment.

### Minimal experiment (the E1+E2 gate — before ranking anything else)

**Step 1 — weight correctness (~1 session).** Extend the counter-rotation equivalence test (M9) — or
an equivalent CPU-side check — against the *actual* shipped `qwen3.5-0.8b-q4` checkpoint's restored
MTP tensors, comparing draft logits with and without the ADR-051 counter-rotation shim. Resolves M7:
either the weights are basis-correct (and 0% acceptance needs a different explanation) or they are not
(and #418's loader fix needs a follow-up before any acceptance number can be trusted).

**Step 2 — the real experiment, no new code (env-var A/B).** Three legs — baseline (`LATTICE_MTP`
unset), default MTP (`LATTICE_MTP=1`), batch-GEMM MTP (`LATTICE_MTP=1 LATTICE_MTP_BATCH=1`) — at
n≥3 runs each (CV<20%, this repo's rigor bar) on two workload shapes: free-form (repeats M6's setup
for a clean baseline) and one structured/repetitive workload (code-completion or JSON, per #419's own
list). Use `make bench-compare`-style methodology and the engine's per-round counters (`metrics.mtp_ms`,
`metrics.verify_ms`, `metrics.verify_calls`). **Commit the result durably** to `docs/bench_results/`
or a PR body (M15 — #419's number was only ever an un-repeated comment). Run under the GPU flock
(`/tmp/lion-metal-gpu-test.lock`) — contended GPU corrupts both timing and numerics.

**Step 3 — decision gate.** If the batch-GEMM leg is net-positive tok/s on *any* workload at K=1, MTP
graduates from "measure" to "build": E3 (temperature-correctness), then #419's parked dynamic-depth
work. If it is net-negative everywhere *even with the fast verifier*, MTP is **closed by measurement**
on this GDN-hybrid family — a legitimate, cheap-to-reach outcome (mirrors ADR-073's honest-negative
framing), and `LATTICE_SELF_SPEC` (M14, structurally different, GDN-only) becomes the more promising
remaining speculative-decode lever, worth its own ADR + first measurement rather than MTP depth work.

**Ordering rationale (Apple-GPU framing).** Decode is memory-bandwidth-bound, so spec-decode's payoff
is `E[accepted] / (verify_cost / baseline)`. M6 shows the **denominator**, not the numerator, is the
problem — the default sequential verifier streams the full model's weights once per verified token
(M4), which is why a bandwidth-bound decoder can go net-negative *before acceptance quality even
enters*. E1 targets the denominator directly and is the single cheapest diagnostic; E2/E3 target the
numerator and only matter if E1 clears the denominator. Doing E1 first is the cheaper, more
diagnostic order.

**Not an MTP-quant lever — noted to prevent conflation.** `LATTICE_SELF_SPEC` (M14) is a distinct
mechanism, ranked separately above. #614 (consolidate 13+ decode loops) and #423 (mixed-precision
policy table) touch MTP as one line item each but are broader epics, out of this ADR's scope.

## Consequences

- **Positive.** The ADR refuses to rank MTP profitability on a self-flagged-unreliable number or on
  the survey's imported stale acceptance figure. It names the one flag flip (E1) that decides the
  whole feature's future at near-zero cost, sequences the acceptance-quality check (E2) as its
  prerequisite, and keeps the dormant temperature work (E3) and tree verification (E4) correctly
  behind that gate. It separates self-spec from MTP so the two mechanisms are ranked on their own
  evidence.
- **Cost.** E1 is two env-var flips + a bench run under GPU flock. E2 extends an existing equivalence
  test to the shipped checkpoint. Both are ≤1-session efforts with no new mechanism. E3/E4 are real
  work but gated — no cost incurred unless E1 clears.
- **Risk.** If E1 shows batch-GEMM net-negative everywhere, MTP is closed by measurement on this
  family. That is the outcome the experiment is *designed* to reach cheaply, not a failure — it
  redirects speculative-decode effort to `LATTICE_SELF_SPEC` with a clear evidentiary reason rather
  than leaving MTP as a perpetually-parked "maybe." The one open correctness risk on the live path
  (#293/M16, KV-drift) is independent of profitability and tracked as a runtime-trace item.

## Follow-ups

- #419 (MTP profitability / dynamic depth) → **E1 the gate**: run the batch-GEMM A/B under GPU flock, commit the artifact; its parked "lift `MTP_VERIFY_MAX_TOKENS`/dynamic depth" sub-tasks stay parked until E1 clears.
- #388 (temperature-correct probabilistic acceptance) → **E3**, sequenced after E1/E2; dormant path (M11), no live gap today. Seed half already fixed (#329).
- #591 (tree-structured verification) → **E4, deferred**, gated on #176 (GDN rollback) + E1-E3.
- #176 (GDN snapshot-per-window rollback) → precondition for E4; "kill early if <0.72 accept" — itself experiment-gated.
- #293 (MTP-local KV-drift after accept) → runtime-trace correctness item (M16), independent of profitability.
- `LATTICE_SELF_SPEC` (M14) → **P-separate**: needs its own ADR + first measurement; do not fold into MTP ranking. Becomes the leading spec-decode lever if E1 closes MTP by measurement.
- #598 (spec-decode usage docs) / M15 (no committed bench) → the artifact gap E1-Step-2 closes; a committed MTP bench is the substrate any future ranking needs beyond one issue comment.
- ADR-006 / ADR-050 / ADR-051 (speculative decoding, rejection sampling, QuaRot MTP rotation) → this ADR sequences their open follow-ups at the priority level; it supersedes none of them.
