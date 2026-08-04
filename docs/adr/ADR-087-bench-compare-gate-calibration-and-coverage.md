# ADR-087: bench-compare gate calibration, coverage, and admissible structural proof

**Status**: Accepted
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

  This is not a missing requirement. `CLAUDE.md` already instructs authors to quote the
  `Run conditions` block alongside the numbers, giving the reason: a figure that does not
  record what produced it is indistinguishable from one produced on a quiet machine. The
  number of bodies that comply with it is zero, across every body examined. So the gap is
  between a requirement's existence and its enforcement, and the recommendation follows from
  which of those two is missing: nothing is gained by restating the instruction, and the
  emitter already produces the record, so the fix has to be a check that reads the body. Until
  such a check
  exists, a disposition citing bench-compare output should be read as unprovenanced by
  default, on the same footing as the D3 reachability burden — the citation carries the
  burden, not the reader.

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

## Amendment 1 (2026-08-04): reachable-surface correction and the granularity of the reach predicate

### Why the surface needed re-checking

D3 puts the reachability burden on any citation of bench-compare output. Interim practice now
goes further: a PR whose changes the default A/B cannot observe may state that fact, citing the
derivation in this record, in place of an A/B table. That promotes the reachable surface from a
supporting measurement into the predicate that decides whether a PR owes bench evidence at all.
A surface used that way has to be right, and this one was not.

### Correction 1: two directly imported files were missing from the surface

Problem 3 names three things the embed `simd` bench imports: `lattice_embed::simd`,
`lattice_embed::service`, and `EmbeddingModel`. The surface derived in the next sentence covers
the first two and drops the third.

`EmbeddingModel` is imported at `crates/embed/benches/simd.rs:9` and constructed at `:52`. It is
defined at `crates/embed/src/model.rs:70` and re-exported at `crates/embed/src/lib.rs:34`. Both
files are reached directly, in exactly the sense that admitted the two directories, and neither
matches the classification predicate as it was written.

The corrected surface has five entries, three directory prefixes and two exact paths:

    crates/inference/src/forward/cpu/
    crates/embed/src/simd/
    crates/embed/src/service/
    crates/embed/src/lib.rs
    crates/embed/src/model.rs

Measured at `origin/main` on 2026-08-04: 14 files under `forward/cpu/`, 11 under `embed/src/simd/`,
4 under `embed/src/service/`, plus the two exact paths, for 31 files.

This record states 28 for the first three prefixes, as 14 plus 14. The embed side is 15 today
rather than 14 because `crates/embed/src/simd/manhattan.rs` was added after this record was
written. The 14 was correct on its date. A hardcoded file count in a decision record went stale
in three days here, which is the argument for stating a derivation command beside any count
rather than only the number it produced.

### Correction 2: two numerals do not reproduce under the commands named

- **Bench target count.** This record states 31 declared bench targets. Measured 30 at
  `origin/main` on 2026-08-04, and 30 at the record's own date, under both available readings:
  `[[bench]]` sections declared across `crates/*/Cargo.toml`, and files matching
  `crates/*/benches/*.rs`. The 31 was already off by one when it was written.
- **Trigger set size.** This record states 452 files from `git ls-files 'crates/inference/*'
  'crates/embed/*' 'crates/fann/*'`. That command yields 473 on the current checkout, and the
  equivalent read at the record's own date yields 472. The 452 does not reproduce under the
  command it is attributed to. This amendment states that and does not assert what produced it.

### The result is unchanged, and now reproduces under both surfaces

Measured on 2026-08-04 across the 34 pull requests open at the time of measurement, which
excludes the one carrying this amendment because it did not yet exist and is in any case
documentation only: 26 trigger the rule. Under the surface
as originally written, zero are reached. Under the corrected surface, one is reached, PR #1289,
and it is reached through `crates/embed/src/lib.rs` alone.

So the conclusion of Problem 3 survives its own evidence being corrected. What changes is the
predicate that is about to decide 25 dispositions.

The omission understated reach, which is the direction that supports this record's conclusion.
The closing section of this record names that direction as the one in which errors are not
investigated. It was right about itself.

One further scope note, and it is not a correction. This record already declares that it
measures direct file-surface reachability and that transitive calls out of `forward::cpu` are
established in neither direction. That declaration holds. A concrete instance of what it leaves
open: `crates/inference/src/forward/cpu/softmax.rs:44` reaches
`crates/inference/src/attention/softmax_row.rs` through a function-scoped import inside
`softmax_attention_scalar`, which is production code on a directly benched path. That file is
absent from the surface because the surface is file-level and direct by construction. It is the
disclosed bound working as disclosed, not a second omission, and it is recorded here so that a
reader who finds it does not have to re-derive whether it was missed.

### D3-A: the reach predicate has two granularities, and file-level reach only opens the second

The surface is file-level. The judgement it feeds is not. Stating only one of those leaves the
next reader to pick, and the two picks disagree on real PRs. The predicate is therefore:

**Step 1, file-level, mechanical.** Does the diff touch any path in the corrected surface? This
is decidable from the surface list with no reading of the diff. If no, the PR may carry an
unreachability statement naming this record. If yes, the file-level answer is a trigger and not
a verdict, and step 2 is owed.

An unreachability statement names the pull request head it was evaluated at and the ref the
surface was read from, and is void if either moves. Neither operand is stable. A PR head moves
on every push, and the surface itself gained a file within three days of the record it is
derived from, which is the `manhattan.rs` case above. A statement that pins neither cannot be
checked later by anyone, including its author.

**Step 2, symbol-level, not mechanical.** Does the diff perturb anything the benched code path
executes? Discharged only by naming, for every changed hunk in every reached file, why the
benched path cannot observe it. Three forms are admissible:

- the hunk adds an item and modifies none, with the added item shown to be absent from the
  bench's import and call surface;
- the hunk is behind a `cfg` the bench build does not enable, with the bench build's resolved
  feature set stated rather than assumed;
- the hunk changes only text the compiler discards, shown by the diff itself.

Anything else fails step 2 and the PR runs the A/B. Not admissible at either step: an argument
that a change is small, cold, or obviously harmless. That is the reading D3 exists to prevent,
and it would re-enter here if this step were left to judgement without a closed list.

**Step 3, residual, always stated.** A discharged symbol-level claim still leaves a
compiled-artifact residual. Adding an item to a crate the bench links changes what the compiler
sees, and layout and inlining are not invariant under additions. The statement says that rather
than claiming the benched path is provably unchanged. Where the residual is load-bearing, run
the A/B instead of arguing about it.

### Worked example: PR #1289 under D3-A

Step 1: reached, through `crates/embed/src/lib.rs`, and no other file in the surface.

Step 2: the change to that file is two lines, `#[cfg(feature = "native")]` and `pub mod drift;`,
inserted between existing module declarations. No existing item is modified. The bench's only
reference into this file's surface is the `EmbeddingModel` re-export at `:34`, which the diff
does not touch, and `mod model;` is likewise untouched. This discharges under the first form.

Step 3: `native` is in `lattice-embed`'s default feature set, so the added module does compile
into the bench build. The claim is that no benched symbol changed. It is not a claim that the
emitted binary is identical.

Disposition: PR #1289 runs the A/B regardless. It is the only reached PR in the current
population, and running one comparison is cheaper than establishing that its step 3 residual
does not matter. That is an operational call about a population of one, and it is recorded
separately from the predicate so that a later reader does not mistake it for the predicate
producing a different answer than it does.

### D5: direction for the target and trigger sets

D3 identifies the mismatch. This sets the direction for closing it, and any implementation
remains subject to the D3 condition that a target change is an instrument change requiring an
A/A null re-run before the new member gates anything.

- **Widen where a bench target already exists.** `BENCHES_INFERENCE` is overridable by
  environment variable. `BENCHES_EMBED` is a bare assignment at
  `scripts/lib/bench-compare-impl.sh:413` with no parameter expansion, so widening the embed
  side is a code change and not a configuration change.
- **`lattice-fann` is a widen case, not a narrow case.** An earlier characterization of fann as
  code the instrument cannot compile is wrong. `crates/fann/Cargo.toml` declares a bench target
  `router_online` with `required-features = ["online-router"]`, and
  `crates/fann/benches/router_online.rs` imports `lattice_fann::training` and
  `lattice_fann::{Activation, Network, NetworkBuilder}`. The instrument can reach fann. The
  default target set does not include it, and the feature is not on by default. Removing fann
  from the trigger set would discard a bench that exists.
- **Narrow the trigger only where no bench target reaches the code at all.** That set is
  established per path by the same derivation used here, not by inspection.

### Evidence and method for this amendment

All reads are at `origin/main` at `e842a892d950f3d3b2687eaf6cc573e0587785a7` unless a ref is
named, and use `git ls-tree` rather than `git ls-files` where the question is about a ref rather
than about a checkout.

    # bench imports, both default targets
    grep -n '^use ' crates/inference/benches/elementwise_cpu_bench.rs crates/embed/benches/simd.rs

    # surface file counts
    git ls-tree -r --name-only origin/main -- crates/inference/src/forward/cpu/ | grep -c '\.rs$'
    git ls-tree -r --name-only origin/main -- crates/embed/src/simd/         | grep -c '\.rs$'
    git ls-tree -r --name-only origin/main -- crates/embed/src/service/      | grep -c '\.rs$'

    # declared bench targets, both readings
    git show origin/main:crates/<c>/Cargo.toml | grep -c '^\[\[bench\]\]'   # summed over crates
    git ls-tree -r --name-only origin/main | grep -c '^crates/.*/benches/.*\.rs$'

    # trigger set
    git ls-tree -r --name-only origin/main -- crates/inference/ crates/embed/ crates/fann/ | wc -l

    # open-PR join
    gh pr list --state open --limit 100 --json number
    gh api "repos/<slug>/pulls/<n>/files?per_page=300" --jq '.[].filename'

The classification predicate is an anchored alternation with a trailing separator on every
directory prefix and an end anchor on every exact path:

    ^(crates/inference/src/forward/cpu/|crates/embed/src/simd/|crates/embed/src/service/|crates/embed/src/lib\.rs$|crates/embed/src/model\.rs$)

Both arms were run in the same invocation before any count was taken. Must-match:
`crates/inference/src/forward/cpu/softmax.rs`, `crates/embed/src/simd/dot.rs`,
`crates/embed/src/lib.rs`, `crates/embed/src/model.rs`. Must-not-match:
`crates/inference/src/forward/cpu_f16.rs`, which is the prefix error this record already
discloses and which a missing trailing separator reintroduces;
`crates/embed/src/model_loader.rs`, which is the same error in the new exact-path entries;
`crates/inference/src/attention/softmax_row.rs`, which must stay out because the surface is
direct rather than transitive; and `crates/fann/src/lib.rs`. All eight arms returned their
expected values.

### Direction of error for this amendment

The surface correction increases reach, which increases work, and that is the direction that
does not get skipped for convenience. The symbol-level step gives work back, and it is the only
step here that rests on judgement rather than on a list. That step is therefore where a future
permissive error will live, which is why its admissible forms are closed and why step 3 requires
a residual statement even when step 2 succeeds.
