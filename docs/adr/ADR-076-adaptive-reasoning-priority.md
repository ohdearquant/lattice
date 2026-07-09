# ADR-076: Adaptive Reasoning-Budget Priority (Metal + CPU)

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own shipped source and one
> merged, measured result (the **Measured / source-verified reality** table). Like the KV-quant ADR
> (ADR-073), this is a **split verdict** at two genuinely different maturity levels. The **static**
> s1-style reasoning-budget (force-inject `</think>` after N reasoning tokens) is fully shipped,
> wired on both backends, and grammar-fail-closed with a mutation-verified test — its only open
> sub-lever is **surface coverage** (missing from the public `lattice` CLI), a standard wiring task
> rankable today with zero new data. The **adaptive** lever (input-conditioned, entropy/confidence-
> gated budget) has **zero lattice code, zero committed eval data, and no instrumentation** — it is a
> well-specified, already-filed, unstarted 6-issue research family whose own first prerequisite (a
> margin/entropy-vs-correctness measurement) is the defined minimal experiment gating the rest. This
> ADR ranks **running that experiment** above **building the feature it gates**. No prior ADR is
> amended: this is the first ADR in the reasoning/thinking area. A parallel external prior-art survey
> is folded as the **[prior, unvalidated on our hardware]** section; it independently converged on the
> same research plan and is treated as corroboration, not new information. The ranking is fixed by the
> shipped source and the one measured result, not the survey.

## Context

"Adaptive reasoning" here means controlling *how much* reasoning/thinking compute the model spends
per query at inference time: reasoning-token budgets (s1-style budget forcing), thinking-mode on/off,
and any dynamic early-exit or confidence-gated stop. The question this ADR ranks: given what the
engine already ships, what is the highest-leverage next lever, and what must be **measured** before
the headline "adaptive" feature can even be prioritized.

What ships today (all confirmed against source, table below): a **static** s1 reasoning-budget —
`GenerateConfig.reasoning_budget: Option<usize>` force-injects `</think>` once the reasoning-token
count crosses the budget, on both the CPU decode loop and the Metal candidate-sampling path, with a
grammar-fail-closed interaction that is regression- and mutation-tested. It costs nothing when
disabled. It is exposed on the Studio-internal `chat_metal` binary and the standalone `lattice_serve`
daemon, but **not** on the public `lattice` CLI. The thinking on/off decision is not driven by the
engine's `enable_thinking` field (hardcoded `true` in every serving binary) but by ChatML prompt
priming; the Studio budget control is a **manual per-conversation stepper**, never derived from the
query. There is no dynamic, per-query, or confidence-based budget anywhere.

## Measured / source-verified reality (this engine)

Tags: **runtime-measured** (a number from running this hardware), **unit-test/gate-pinned** (a
committed test or lint script enforces it), **source-read** (a structural fact from merged code).
All pointers are on `origin/main @ 4cb006d32`.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| R1 | **Static s1 budget is a shipped engine field.** `GenerateConfig.reasoning_budget: Option<usize>` — "after this many reasoning tokens without a `</think>`, force-inject `</think>`." `None`/`Some(0)` = disabled, byte-identical to pre-feature behavior. | source-read | `crates/inference/src/model/qwen35_config.rs:690` (field), `:734` (`Default` = `None`) |
| R2 | **Decision + cap functions.** `force_close_think()` fires only when `enable_thinking && !thinking_closed && budget>0 && generated_so_far>=budget`; `decode_cap()` extends the loop bound to `budget + max_new_tokens + 1` so the answer budget survives a forced close. Both `pub(crate)`, not request-shaped. Full boundary + disabled-path test set. | source-read + unit-test-pinned | `qwen35_config.rs:747-761`, `:774-780`; tests `:1635-1743` (`decode_cap_*`, `force_close_think_*` incl. `_fires_at_budget_boundary`, `_does_not_fire_before_budget`) |
| R3 | **Wired into both CPU decode-loop variants** (fast + string-stop). `</think>` id resolved once only when a budget is set (zero extra lookup on the disabled path); prefill-seeded so `budget=1` is correct. | source-read | `crates/inference/src/model/qwen35/generation.rs:249-254` (resolve + seed), call sites `:611,:749,:935,:1063` |
| R4 | **Wired into the Metal candidate-sampling path** identically — `force_close_think` overrides `sampled_id` before grammar `advance()`, with an in-code note that grammar+budget is fail-closed. | source-read | `crates/inference/src/forward/metal_qwen35.rs:13131-13148` (call + interaction note), `:15671` (2nd site) |
| R5 | **Grammar × budget fail-closed is mutation-verified.** A test builds a model where `</think>` is out-of-grammar-vocab, sets `reasoning_budget=Some(1)`, and asserts generation stops via `StopReason::Grammar` **without** emitting the forbidden token; PR body documents the mutation flip that breaks it. | unit-test/gate-pinned | PR **#511** (MERGED 2026-07-01), test `grammar_budget_forced_close_fails_closed`; comment `metal_qwen35.rs:13141-13146` |
| R6 | **`enable_thinking` never reaches `false` in a live request path.** It is a real field that *does* gate `force_close_think`, but every serving binary hardcodes it `true`; no CLI flag, no request field sets it otherwise. Thinking on/off is done by prompt priming, not this field. | source-read (exhaustive grep) | `bin/chat_metal.rs:818,907`; `bin/lattice_serve.rs:869`; test `force_close_think_disabled_when_enable_thinking_false` (`qwen35_config.rs:1687`) |
| R7 | **The public `lattice` CLI exposes neither the budget nor any thinking control.** `lattice chat`/`lattice serve` construct `GenerateConfig { .. , ..Default::default() }`, so `reasoning_budget` is permanently `None`. This is gate-pinned prose, not free-floating. | source-read + doc-gate-pinned | `docs/capability-matrix.md` row 85 + `:36-40`; enforced by `scripts/check-capability-matrix.sh` in `make lint-docs` (`scripts/lint-docs.sh:9-10`) |
| R8 | **`lattice_serve` (OpenAI-compatible daemon) supports the budget** as a server-wide default **and** a per-request override, clamped to the real context window. | source-read | `bin/lattice_serve.rs:153,227` (fields), `:855-858` (`.or(d.reasoning_budget)` resolution), `:838-849` (clamp doc); `docs/capability-matrix.md` row 85 (right column) |
| R9 | **`chat_metal` supports `--reasoning-budget` + a per-request JSON field**, with `reasoning_budget: 0` normalized to "absent", unit-tested. | unit-test/gate-pinned | `bin/chat_metal.rs:529-538,603-608`; test `cm_chat_metal_serve_zero_reasoning_budget_falls_back_to_default` (`:1108`) |
| R10 | **Studio exposes only a manual, static, per-conversation control** — a numeric stepper (`String` default `"1024"`) plus a thinking toggle (`Bool` default `false`); the budget is `nil` unless thinking is on, and is set once before sending, never from the query. | source-read | `apps/macos/Sources/LatticeStudio/Store/AppStore.swift:76,79`; `Screens/ChatScreen.swift:450-451,1345-1349`; `:1308-1329` (`renderChatML`, ChatML priming = the real on/off) |
| R11 | **The only measurement of the feature is a throughput-neutrality check of the *disabled* path.** PR #435 (Metal e2e, qwen3.5-0.8B, greedy): HEAD-vs-BASE decode −0.53% (noise). Confirms it costs nothing off; it is **not** a quality-vs-budget or accuracy-vs-latency measurement of the feature doing anything. | runtime-measured (PR-body artifact, not committed) | PR **#435** (MERGED 2026-06-29), body §"bench-compare & decode A/B" |
| R12 | **No committed reasoning/budget eval or bench artifact exists**, and **no decision-grade margin/entropy telemetry exists in the decode loop.** `docs/bench_results/` holds none reasoning-shaped. `GenerateConfig.logprobs` records per-token logprobs/top-k **only when requested** (OpenAI parity, PR #620) — output-reporting, not a top1−top2 margin or exact-entropy signal, and no AUROC/ECE harness exists in-tree. This is exactly what the open experiment (#493) must build. | source-read (absence) | `docs/bench_results/` (listed, none match); `sampling.rs` (`record_logprob`, no margin/entropy fields); `crates/inference/src/` (no `entropy`/`margin`/`confidence`/`calibration` decode module) |

## Prior / unvalidated on our hardware

Folded from the external survey (`fleet_atlas_lat_adaptreason_001` harvest) as **data**, and used
only as corroboration — the ADR's proof rests on the repo-verifiable source and issue facts in the
table above, **not** on any packet-internal text. The packet artifact is internal and not
reproducible from this repository, so statements about what it read or cited are corroborating
attribution only. What **is** repo-verifiable and load-bearing: the survey's recommended plan —
instrument margin/entropy → calibrate as a gate → three-tier adaptive budget → local cascade, with
MTP-disagreement as an experimental side channel — is structurally the same as the already-filed
`#482` issue family, and its concrete repo references are checkable against the table above (its
`force_close_think`/`decode_cap` semantics and `0.8B` config values match; its invented `bench_decode`
binary [real: `bench_decode_ab`] and nonexistent `--confidence-trace` flag are grep-confirmable
divergences). The convergence is therefore treated as evidence the issue family's plan is a reasonable
one, not as independent confirmation beyond what the issues already record.

- **"Add top1/top2 margin + exact-entropy instrumentation" — CONFIRMED-BY-ANALYSIS.** Matches R12
  exactly: the gap is real and cheap (logits are already CPU-resident at sampling time, so the
  computation piggybacks the existing pass). This is the shared first step of the whole family.
- **"Calibrated token margin as the first gate" — NEEDS-EXPERIMENT.** Converges on #493's own
  AUROC-gated promote/kill framing; no data exists either way.
- **"Scratchpad improves small-model calibration" — NEEDS-EXPERIMENT.** Converges on #487, including
  the same caveat that chain-of-thought can make a small model **more confidently wrong** — which is
  why it must be measured, not assumed.
- **"MTP draft-head disagreement is a free uncertainty signal" — NEEDS-EXPERIMENT, low confidence.**
  Converges on #496 *and* carries the same saturation risk: ADR-074's MTP finding (~4.65% free-form
  acceptance) implies the signal may be uninformative on exactly the free-form reasoning text it would
  need to score.
- **Divergences (repo-verifiable, not trusted for ranking):** the packet's arithmetic and CLI shapes
  are approximate — the invented `bench_decode` binary and nonexistent `--confidence-trace` flag noted
  above are the grep-confirmable examples. Only the verdict-level convergence is used, never its
  concrete prescriptions.

## Decision

**Priority is split by maturity. Rank the measurement, not the unbuilt feature.**

**Rankable now (no experiment needed):**

- **P1 — run the margin/entropy-vs-correctness measurement (#493).** This is the pivot. It is
  G1-shaped: pure offline measurement, **zero serve-path change**, extending the already-resident
  full-vocab softmax at sampling time (R12) to also record top1−top2 margin and exact entropy at
  answer/label tokens, correlated against correctness on a held-out reasoning benchmark at
  Qwen3.5-0.8B. Its own promote/kill thresholds (AUROC ≥0.75 promote, <0.65 kill) decide whether any
  adaptive gating is worth building. It **blocks #500 by that issue's own text**.
- **P2 — run the scratchpad-vs-calibration measurement (#487).** Parallel G1 experiment gating the
  safety-triage framing (#482) specifically, distinct from #493's general-domain framing. Also
  unstarted, also cheap, also a measurement rather than a build.
- **P3 — expose the shipped static budget on the public `lattice` surface.** Certain-value, low-risk
  plumbing: wire the already-shipped `reasoning_budget` (and, if desired, the thinking toggle) into the
  public `lattice chat`/`lattice serve` path, copying `lattice_serve.rs`'s implementation (R8). No
  research, no new machinery. Ranked below P1/P2 because its magnitude is small and whether the public
  CLI should reach `lattice_serve` parity is a product-surface call; **no tracking issue is filed for
  it yet** (file one if pursued).

**Gated, not ranked:**

- **#500 (the adaptive-budget feature itself) is deliberately NOT given a build priority.** It
  graduates to a rankable priority **only if #493 clears its AUROC threshold** — building a G2
  (behind-a-flag, default-off) gate before proving the underlying signal predicts correctness risks
  shipping dead machinery. Its own kill clause is honest and adopted here: if the adaptive gate cannot
  beat the best fixed static budget, **#435's static budget stays the shipped default and nothing is
  lost**.
- **#496 (MTP-disagreement signal) and #497 (GDN-state observer signal) are conditional/secondary.**
  #497 literally states "only pursue if #493 misses its promote threshold"; #496 carries the
  saturation caveat above. Neither is worth prioritizing ahead of the primary margin/entropy signal.

**Ordering rationale.** #493 is the single unblocking measurement: until it runs, the priority of
#500 — the headline feature — is genuinely unknowable, and the two conditional signals (#496/#497)
are explicitly downstream of its result. The static half is already shipped and mutation-tested
(R2/R5), so the only static work left is surface plumbing (P3), which competes on certainty but not
on leverage. This is the experiment-gated-priority discipline established by ADR-073: when a lever
family has zero local-measured data, gate it behind a *defined* minimal experiment rather than
ranking it on literature confidence — here the experiment is already filed as #493 with its own
thresholds, so this ADR ratifies that gate and places running it above building what it gates.

**Not an adaptive-reasoning lever — noted to prevent conflation.** `GenerateConfig.logprobs` (R12) is
OpenAI output-reporting parity, not a decision-grade internal signal; it does not make the adaptive
gate cheaper beyond confirming logits are resident. Best-of-N / self-consistency / parallel-reasoning
sampling is neither filed nor built and is a different axis (sample multiple traces) from budget
control (spend on one trace); it is out of scope here.

## Consequences

- The reasoning area gets its first ADR and a clear split: static forcing is **done** (rank only its
  surface gap); adaptive gating is **measurement-gated**, not a feature to schedule yet.
- The next unit of work is an **experiment (#493)**, not a feature — and its result, either way,
  settles whether #500 is ever a real priority. A kill (AUROC <0.65) closes the adaptive family at
  minimal cost and leaves the shipped static budget as the correct default.
- The shared prerequisite for the whole family (margin/entropy instrumentation) is small and
  serve-path-neutral, so P1/P2 can proceed without destabilizing the decode loop.

## Follow-ups

- **Commit the experiment artifact.** R11/R12 show this feature's pattern-of-failure: #435's only
  measurement lived in a PR body, never a committed file. Whoever runs #493/#487 commits results to
  `docs/bench_results/` (or a `docs/eval_results/`-shaped location), not a PR body.
- **File a tracking issue for P3** (public-CLI surface exposure) if it is pursued — none exists today.
- **GPU flock on any measurement run.** Any Metal run for #493/#487 acquires
  `/tmp/lion-metal-gpu-test.lock` (machine-wide GPU test lock; contended GPU work corrupts both timing
  and numerics, #628/#629).
- **Sample-size gap.** #493 does not specify the minimum labeled-example count for a stable AUROC
  point estimate; whoever runs it fixes an `n` large enough for a reportable CI before quoting a
  verdict against the ≥0.75 / <0.65 thresholds.
- **If #493 clears its threshold**, record the result as a new row on this ADR (measured graduation of
  #500 to a ranked priority) rather than opening the adaptive-budget build silently — the same
  forward-instruction discipline the sibling ADRs carry.
