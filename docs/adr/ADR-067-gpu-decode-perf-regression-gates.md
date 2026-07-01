# ADR-067: GPU/Decode Performance Regression Gate System (Self-Hosted M2 Max)

**Status**: Proposed
**Date**: 2026-07-01
**Scope**: `crates/inference` Metal decode path; `scripts/bench_decode_slopefit.*`; a new nightly
`.github/workflows` job
**Depends on**: ADR-058 (CI-lower-bound gate method), ADR-061 (KernelTiming schema), ADR-064
(CI Gate Taxonomy and Promotion Policy)

> **Numbering note**: GitHub issue #167's body names this document
> `docs/adr/ADR-064-gpu-decode-perf-regression-gates.md`. By the time this ADR was written,
> `ADR-064` and `ADR-065` were already assigned to the CI-gate-taxonomy and feature-promotion-gate
> ADRs (both shipped, unrelated topics), and `ADR-066` (output-correctness gates, PR #517) merged
> to main while this branch was in flight. This document therefore uses the next free number,
> **ADR-067**. The content is unchanged by the renumber.

---

## Context

Issue #167 asks for a GPU/decode performance regression gate system that runs on a self-hosted
Apple M2 Max runner: it measures decode tok/s and the slope/intercept of per-token decode cost
against context length (issue #168, closed), applies a gate table (decode tok/s, slope, TTFT,
PPL, greedy/top-k agreement, contention/KV-layout), and fails only on a confirmed regression at
the CI lower bound, following the ADR-058 method.

The self-hosted M2 Max runner referenced by issue #167 **does not exist yet** — provisioning it is
a pending founder decision, tracked separately from this ADR. That means the nightly workflow
described below is buildable and correct today, but cannot execute until the runner is
provisioned. Scoping this ADR to "everything buildable now" keeps the harness, gate script, and
workflow definition ready to go live the moment the runner lands, without inventing speculative
runner behavior.

`scripts/bench_decode_slopefit.sh` / `.py` / the `bench_decode_slopefit` Rust binary already
implement most of the measurement loop (Theil-Sen fit, bootstrap CI, warmup/measure/repeat
protocol). What's missing is: (a) a JSON shape compatible with the ADR-061 KernelTiming /
perf-baselines conventions, (b) a script that turns two such JSON snapshots into a gate verdict,
and (c) a (currently inert) scheduled workflow that wires the two together.

---

## Decision

### Gate table (from issue #167, reproduced for traceability)

| Row | Metric(s) | Threshold (1-sided 95% CI lower bound) |
| --- | --- | --- |
| Decode tok/s | `decode/tok_s/{ctx}` | regression `> 7%` |
| Decode slope | `decode/slope_ms_per_ctx_tok` | regression `> 5%` |
| Decode intercept | `decode/intercept_ms` | regression `> 7%` |
| TTFT | `decode/ttft_ms/4096`, `decode/ttft_ms/16384` | regression `> 10%` |
| Dispatch/token | `decode/dispatches_per_token` | regression `> 5%` OR absolute `+10` |
| Command buffers/token | `decode/command_buffers_per_token` | absolute ceiling `<= 2` |
| Quality (PPL) | `quality/ppl_delta/{f16,bf16,q4_kv}` | absolute `<= 0.005 / 0.05 / 0.30` |
| Quality (agreement) | `quality/greedy_agreement`, `quality/topk_exact` | must equal `1.0` |
| Contention W=4 | `contention/loss_pp/w4` | absolute `+3pp` |
| Contention W=10 | `contention/loss_frac/w10` | absolute `<= 10%`, regression `> 5%` |
| KV layout | `runtime/kv_layout_assertion` | must equal `1.0` |

All comparisons use the ADR-058 method: for a lower-is-better metric,
`lb = (current.ci95_low - baseline.ci95_high) / baseline.ci95_high`; for a higher-is-better metric,
`lb = (baseline.ci95_low - current.ci95_high) / baseline.ci95_low`. A row FAILs only when `lb`
exceeds its threshold — noise inside the confidence interval cannot flake the gate. A row missing
required data reports `INCOMPLETE`, never a silent PASS.

### Artifacts landed in this change

1. **`scripts/adr064-gpu-decode-gate.py`** — takes `--current` and `--baseline` JSON (the harness
   output shape below), applies the table above row-by-row, writes a markdown report, and exits
   `0` (all PASS), `1` (any FAIL — a confirmed regression), or `2` (no FAIL, but some row
   `INCOMPLETE` because its inputs are still null placeholders).
2. **`tests/test_adr064_gpu_decode_gate.py`** — 18 unit tests: one synthetic fixture per gate-table
   row that trips only that row, a no-change fixture, a within-CI-noise fixture that must stay
   green, and missing-data fixtures proving `INCOMPLETE` never silently passes.
3. **`.github/workflows/adr064-gpu-decode-nightly.yml`** — schedule-only (`cron: '17 8 * * *'`),
   `runs-on: [self-hosted, m2max]`, no `pull_request`/`push`/`workflow_dispatch` trigger, guarded
   to `main` only, `permissions: contents: read`. Dormant by construction: GitHub Actions never
   schedules a job onto a runner-label set with zero registered runners.

### Harness JSON shape (input contract for the gate script)

Both `--current` and `--baseline` are JSON objects with `commit`, `date`, `arch`, and a `benches`
map from metric key (e.g. `decode/tok_s/4096`) to
`{value, ci95_low, ci95_high, unit, higher_is_better}`. Fields the harness has not implemented yet
(TTFT, PPL, greedy/top-k agreement, contention, KV-layout assertion) are emitted as
`{value: null, ci95_low: null, ci95_high: null, ...}` — an honest-nil placeholder, not a
fabricated pass. The gate script treats any row depending on a null field as `INCOMPLETE`.

### What is explicitly NOT decided here

- **The runner does not exist.** This ADR proposes the workflow shape; it does not request or
  imply approval to provision `[self-hosted, m2max]`. That is a separate founder decision.
- **No gate in this table is merge-blocking.** The nightly workflow is intentionally excluded from
  any required-status-check set. Promoting any row to merge-blocking requires a separate founder
  sign-off, per the ADR-064 taxonomy's promotion policy (ADR-064 §"gated vs. informational").
- **Baseline publishing.** This workflow reads from the `perf-baselines` branch but does not push
  to it; wiring an automatic baseline-update step is left for a later change once the runner is
  live and its first several nightly runs have been reviewed by a human.

---

## Implementation status (2026-07-01)

This ADR documents the GPU/decode perf-regression gate design and its currently-shippable
artifacts. The self-hosted M2 Max runner does not exist; everything that does not depend on it is
implemented, everything that does remains dormant and clearly marked as such.

**SHIPPED in this change:**

1. `scripts/adr064-gpu-decode-gate.py` — gate-evaluation script implementing the table above,
   using the ADR-058 CI-lower-bound method. Runs standalone: `python3
   scripts/adr064-gpu-decode-gate.py --current <json> --baseline <json> [--out report.md]`.
2. `tests/test_adr064_gpu_decode_gate.py` — 18 passing unit tests (`python3 -m pytest
   tests/test_adr064_gpu_decode_gate.py -v`), covering a no-change fixture, a within-noise
   fixture, one regression fixture per gate-table row (decode tok/s, slope, intercept, TTFT,
   dispatches/token, command-buffers/token, each PPL row, greedy/top-k agreement, both contention
   rows, KV-layout assertion), and missing-data → `INCOMPLETE` fixtures that confirm a null metric
   never silently passes and never masks a real `FAIL` elsewhere.
3. This ADR document.

**DORMANT — pending founder sign-off on the self-hosted runner:**

1. `.github/workflows/adr064-gpu-decode-nightly.yml` — schedule-only, non-required, no PR trigger,
   targets `runs-on: [self-hosted, m2max]`. Committed now so it activates the moment the runner is
   provisioned, but it cannot run before then; this is expected, not a defect.
2. Wiring the harness's currently-null fields (TTFT, PPL deltas, greedy/top-k agreement,
   contention loss, KV-layout assertion) into `scripts/bench_decode_slopefit.py` — out of scope
   for this change; the gate script already handles their absence honestly via `INCOMPLETE`.
3. Promoting any row in the gate table to a required, merge-blocking status check — explicitly not
   done here; requires founder sign-off per the ADR-064 taxonomy's promotion policy.

No existing gate (in this ADR or elsewhere in `.github/workflows`) was weakened, removed, or made
merge-blocking by this change. Everything above is additive.

---

## Consequences

- **Positive**: the gate table has a single, tested, deterministic implementation ready before the
  hardware exists to run it nightly. Reviewers get a concrete "here is exactly what will fire and
  why" reference instead of a hypothetical description.
- **Positive**: `INCOMPLETE` as a first-class verdict prevents the common perf-harness failure mode
  of a null/zero field being silently read as "passed."
- **Negative / accepted**: the nightly workflow is unverifiable in CI until the runner exists — its
  correctness rests on the unit tests and a manual `workflow_dispatch`-style dry run once the
  runner is available (not added here, to avoid a PR-triggerable path onto self-hosted hardware).
- **Follow-up**: once the runner is live, the first several nightly runs should be reviewed by a
  human before any row is proposed for required-status promotion.
