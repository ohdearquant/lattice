# ADR-087: bench-compare gate calibration, coverage, and admissible structural proof

**Status**: Proposed
**Date**: 2026-08-01
**Crate**: workspace (bench harness; governs PRs touching lattice-inference, lattice-embed, lattice-fann)

## Context

`CLAUDE.md` states, as a rule with no exceptions, that every PR touching `crates/inference/`,
`crates/embed/`, or `crates/fann/` must include `make bench-compare` output. That rule is
currently attributed to ADR-058.

Three problems were measured on 2026-08-01, at `origin/main` `63a50832e3`. They are
independent of each other, and the third is the one that matters most.

### Problem 1: the citation points at a superseded record

ADR-058's `Status` is `Superseded`. Its own 2026-06-24 status update states that the
CPU-bench-as-gate design "was superseded by the e2e-parity approach" and that Criterion
micro-benchmarks "are collected as trend data by `bench-update.yml` but are not a merge gate."
Its subject is a workflow, `bench-regression.yml`, that its own status update confirms was
never created.

The string `bench-compare` has never appeared in ADR-058, in any revision. Verified with
`git log -S` over the file's full history, with a must-match control in the same invocation
(the same pickaxe finds two revisions touching `perf-baselines` in that file, so the search
works).

The history explains how this arose, and it is not a mis-stapled citation:

- **2026-05-24**, `efdbe6657b` (#83) creates ADR-058 and, in the same commit, adds both the
  `make bench-compare` rule and its `(ADR-058)` attribution to `CLAUDE.md`. The tool was never
  described by the ADR it was filed under; the citation was a forward reference from the first
  day, to a document whose subject was a different mechanism.
- **2026-06-24**, `df809c43c3` marks ADR-058 `Superseded` while reconciling stale status
  fields. The rule in `CLAUDE.md` requiring exactly the gate that update disclaims is left
  standing.
- **Since then**, `CLAUDE.md` has been edited twelve times. Three of those modified lines
  containing `bench-compare` itself: a gate baseline-parse fix adding per-group overrides
  (2026-07-02), the quiet-window rule (2026-07-16), and the run-conditions disclosure
  (2026-07-21). Every one of them passed over the citation without reading it.

The generalisable point is the third bullet. Superseding a decision record does not propagate
to the instruction files that cite it, and a parenthetical citation is not re-read when the
prose around it is edited. Twelve opportunities to notice, three of them by an editor working
on the cited rule's own text, produced no notice, because each editor was changing the rule's
content and the citation was not part of what they were changing.

This is a provenance defect and not a validity defect: the operative rule lives in `CLAUDE.md`,
its authority comes from that file being binding, and it is unaffected. But a rule sourced to a
record that contradicts it cannot be reasoned about, and a reader who follows the citation is
misled about what the rule is for.

### Problem 2: the gate's threshold has never been calibrated against its own null

`scripts/perf-bench-gate.py` classifies on `ci_low`, the lower bound of Criterion's two-sided
95% CI on the change estimate, with `WARN_PCT = 3.0` and `FAIL_PCT = 7.0`.

Four A/A runs were performed, meaning both arms built from byte-identical source, so every
reported difference is by construction a null result. The proportion of groups the harness
labelled significant:

| run | harness arm ordering          | groups labelled significant |
| --- | ----------------------------- | --------------------------- |
| 1   | two-arm (base then head)      | 62%                         |
| 2   | two-arm (base then head)      | 53%                         |
| 3   | ABBA (base, head, head, base) | 63.4%                       |
| 4   | ABBA (base, head, head, base) | 75.8%                       |

A correctly calibrated instrument returns approximately 5% on a true null. None of the four
runs came close, under either arm ordering. Two of the four ran in windows later found to be
contaminated by concurrent load, which is why no magnitude band is derived from them here; the
p-inflation is what reproduces across all four, not any particular excursion size.

A threshold placed inside an instrument's own noise band has near-zero power and devalues
every verdict the instrument produces, including its correct ones.

### Problem 3: the gate triggers on changes its instrument cannot observe

This is the finding that subsumes the other two.

The workspace declares 31 bench targets. `bench-compare` runs exactly two:
`lattice-inference:elementwise_cpu_bench` and `lattice-embed:simd`. `BENCHES_INFERENCE` is
overridable by environment variable; `BENCHES_EMBED` is a bare assignment with no parameter
expansion and cannot be overridden at all.

`elementwise_cpu_bench` imports from exactly one module, `lattice_inference::forward::cpu`.
The embed `simd` bench imports `lattice_embed::simd`, `lattice_embed::service`, and
`EmbeddingModel`. The directly reachable source surface is therefore
`crates/inference/src/forward/cpu/` (14 files) and `crates/embed/src/{simd,service}/`
(14 files): 28 files.

The trigger set named by the rule is 452 files (`git ls-files 'crates/inference/*'
'crates/embed/*' 'crates/fann/*'`). The instrument reaches 28 of them, or 6.2%.

Measured against the open pull requests on 2026-08-01: of the 29 open PRs that trigger the
rule, **zero** touch any file in the reachable surface.

The exclusion is structural rather than accidental. The targets that would exercise the Metal
forward path are feature-gated: `mtp_decode`, `metal_decode_bench`, and
`cross_turn_prefix_cache_bench` all declare `required-features = ["metal-gpu", "f16"]`, and
`decode_attn_bench` declares `["f16"]`. The post-merge gate runs on `ubuntu-latest` and
`ubuntu-24.04-arm`, which cannot run Metal at all, so the default target set is the
Linux-portable one. The harness then keeps that same default on macOS, where Metal coverage
would be possible.

A further narrowing applies in the default resolution: the `lattice-embed:simd` target is
classified informational in `--quick` mode, so in the mode a developer runs by default the
only gating target is `elementwise_cpu_bench`.

## Decision

### D1: the bench-compare gate is ADVISORY until its threshold is re-derived

The gate is demoted to advisory for gating purposes. This is the application of an existing
standing requirement, that no blocking threshold ships without same-SHA variance calibration,
to a gate that has now been measured. It is not a waiver and not an exception.

**Expiry condition.** The demotion lifts when, and only when, the null distribution is measured
on an uncontaminated machine and the thresholds are re-derived against that measurement. Until
then a FAIL verdict from this gate is information, not a decision.

Advisory status is not a licence to merge performance-relevant changes unmeasured. A PR still
owes positive evidence proportionate to what it changes. The demotion buys nothing to any PR
that alters logic, algorithm, data layout, generic instantiation, or hot-path control flow.

### D2: admissible structural proof, and its conditions

A provable pure intra-crate move may satisfy the disposition requirement with a structural
proof instead of an A/B table. All four conditions are required.

1. **Intra-crate only.** Within a crate, module boundaries are not codegen boundaries: the
   compiler sees the whole crate, so a module move does not change inlining eligibility.
   Across a crate boundary it does, since cross-crate inlining requires `#[inline]`, generics,
   or LTO. A cross-crate move is not a pure move for performance purposes however
   byte-identical the body.
2. **Closed category set for permitted differences**: visibility widening within the crate,
   comment and doc text, rustfmt reflow, and import-path adjustment. That is the entire list.
   A changed signature, a changed generic bound or where-clause, a struct field reorder, an
   added or removed `#[inline]` / `#[cold]` / `#[repr]`, or reordered match arms voids the
   proof and returns the PR to measurement.
3. **Exhaustive enumeration, stated.** Report lines removed, lines added, lines byte-identical,
   and name and categorise every differing line. "Substantially identical" is not a proof, and
   a sampled diff is not a proof. Where a permitted difference causes another (a rustfmt reflow
   consequent on a visibility change, for instance), state the causal link, because a reflow
   with no named cause is indistinguishable from an unrelated edit that landed in the same
   commit.
4. **The proof names its own direction of error** in the PR body.

If an author finds themselves arguing that a PR is almost a pure move, it is not one.

### D3: coverage requirement

**A gate must not trigger on a change its instrument cannot reach. Where the trigger set
exceeds the instrument's reach, the excess is not a weak measurement but an absent one, and a
re-derived threshold over a structurally silent target set is a calibrated instrument pointed
at the wrong code.**

Operationally, for this gate:

- Before citing any bench-compare output as evidence about a diff, show that the diff is
  reachable from the bench targets that were run. The burden is on the citation, not on the
  reader. This holds whether the output is green or red, and whether the gate is advisory or
  enforcing.
- Re-deriving the D1 thresholds does not by itself discharge D3. Calibrating the current
  two-target set produces a correct threshold for code that no current PR touches.
- Widening the target set is itself an instrument change: any change to which targets or
  groups can contribute to a FAIL verdict requires an A/A null re-run and threshold
  re-derivation before the new member gates anything.

### D4: citation correction

`CLAUDE.md`'s Performance Workflow section is updated to cite this ADR rather than ADR-058.
ADR-058 remains superseded and is not modified, which is what a superseded record is for.

## Evidence and method

The coverage result is the load-bearing measurement, so its method is recorded rather than
just its conclusion.

Classification predicate: a PR is covered if any changed path begins with
`crates/inference/src/forward/cpu/`, `crates/embed/src/simd/`, or `crates/embed/src/service/`.

Controls, run in the same invocation as the result:

- **Must-match arm**: a synthetic PR touching `crates/inference/src/forward/cpu/elementwise.rs`
  is selected by the predicate. Without this, a zero is indistinguishable from a dead filter.
- **Must-not-match arm**: a synthetic PR touching `crates/inference/src/forward/cpu_f16.rs` is
  rejected by the predicate.

The must-not-match arm exists because the first version of this analysis was wrong. An earlier
predicate used the prefix `crates/inference/src/forward/cpu` without the trailing separator,
which also matches `forward/cpu_f16.rs` and `forward/cpu_q8.rs`. Those are sibling modules
(`forward::cpu_f16`) that the bench does not import, not members of the `forward::cpu` module.
That predicate reported one covered PR. The error was found by asking which file made the
single positive positive.

That error inflated coverage, so it ran against the conclusion being argued, which is why it
was noticed. An error in the other direction would have confirmed the result and would
probably have shipped. This is recorded because a coverage claim whose author does not say how
they nearly got it wrong is weaker evidence than one that does.

**Scope of the coverage claim.** It measures direct file-surface reachability. Functions within
`forward::cpu` may transitively call code that these PRs do touch; that is not established in
either direction here and would require call-graph analysis. The claim is therefore not "these
diffs cannot affect the benched code" but "no one has shown that they can, and the burden sits
with any citation that assumes it."

## Consequences

- Of the 29 open PRs that trigger the rule, at most 4 are candidates for D2 admission on a
  line-balance screen, and a line-balance screen is not a proof. Each still requires condition 3
  enumeration before it is called a pure move. A balanced diff at several thousand lines
  deserves more suspicion than one at six hundred, not less, because churn of that size hides a
  real change more easily.
- The remaining PRs do not merge on a structural proof, and they do not merge on an A/B either,
  because the A/B did not observe them. They wait on the instrument. This is a worse outcome
  than a clean gate and a better one than a fabricated clearance.
- The provenance record emitted by the harness is not currently reaching pull request bodies.
  Across the 32 open PRs, zero bodies contain either the `[state]` line or the
  `=== Run conditions ===` block, against a control of 25 bodies mentioning `bench-compare`.
  Both strings were verified present in `scripts/lib/bench-compare-impl.sh` before that
  negative was trusted. A provenance feature with no consumer records nothing, and correcting
  what it emits does not by itself change that.

## Direction of error

Every decision in this ADR reduces work. D1 relaxes a gate, D2 admits a cheaper proof, D3
declares existing evidence inadmissible rather than requiring more of it, and D4 removes a
citation. That is the direction in which errors are not investigated, because a pass invites no
scrutiny.

Two guards follow from that. The D2 conditions are deliberately tight and closed, so that
admission requires enumeration rather than judgement. And D3 is written to prevent the reading
this ADR most invites, which is that a gate found to be uncalibrated and structurally silent
was therefore never binding. It was binding, it remains binding, and what changed is that its
output is no longer admissible as evidence without a reachability showing.
