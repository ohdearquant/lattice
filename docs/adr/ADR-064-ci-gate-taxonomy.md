# ADR-064: CI Gate Taxonomy and Promotion Policy

**Status**: Proposed
**Date**: 2026-06-27
**Crate**: workspace CI (`.github/workflows`, `scripts/ci.sh`, `deny.toml`)
**Research**: Flow artifacts `../explorer/gates_inventory.md`, `../researcher/adr_format_and_gaps.md`, `../analyst/taxonomy_and_gaps.md`
**Issues**: N/A
**Depends on**: ADR-058 (CPU performance regression CI), ADR-063 (serving architecture release surface)

## Implementation status (2026-06-27)

This ADR systematizes the existing CI gates, records verified gaps, and defines the promotion rule
for new merge-blocking gates.

Implemented additive changes in this run:

1. This ADR (`docs/adr/ADR-064-ci-gate-taxonomy.md`) — documentation only.
2. `rustdoc-lint` observation job added to `.github/workflows/ci.yml` with `continue-on-error: true`
   at the job level. Not added to any required-status set.
3. `unwrap-in-lib-lint` observation job added to `.github/workflows/ci.yml` with
   `continue-on-error: true` at the job level. Not added to any required-status set.

All new merge-blocking gates listed below remain proposed until founder sign-off. No proposed gate
in this ADR should be treated as accepted branch-protection policy. The two observation jobs above
are explicitly non-blocking and will show existing debt on first run.

---

## Context

### The CI gate taxonomy problem

The repository's CI gates were added over time as specific risks appeared: default workspace checks,
feature-matrix checks, parity gates, supply-chain policy, performance baseline maintenance, and
packaging coverage for app-shipped binaries. The result is useful but hard to reason about because
workflow names do not consistently state the protected property.

This ADR classifies each substantive gate by the property it protects:

| property | meaning |
|---|---|
| correctness | Runtime behavior, parity, drift, panic-safety, or test result validity. |
| build-hygiene | Formatting, linting, compile coverage, feature coverage, toolchain compatibility, or documentation build health. |
| supply-chain | Dependency license, ban, source, or advisory policy. |
| performance | Benchmark buildability, baseline measurement, speed reporting, or future regression policy. |
| packaging | App-shipped binary completeness or package publishability. |

Checkout, cache, setup, toolchain install, and cleanup steps are support mechanics. Their failures
attach to the substantive gate they support and are not listed as independent protected properties.

### Scope and evidence

The taxonomy covers the five workflow files under `.github/workflows`:

- `ci.yml`
- `e2e-parity.yml`
- `bench-update.yml`
- `app-binaries.yml`
- `embed-parity-release.yml`

The evidence base is repository-local:

- `../explorer/gates_inventory.md` exhaustively inventoried workflow jobs and steps.
- `../researcher/adr_format_and_gaps.md` verified candidate gaps with targeted searches.
- `../analyst/taxonomy_and_gaps.md` expanded delegated `./scripts/ci.sh` behavior into substantive gates.

### Constraints

The project is a pure-Rust inference engine. CI design must not add new dependencies in this ADR and
must not weaken or remove any existing gate. Additive observation gates can be introduced separately
only when they are non-required or explicitly allowed to fail during their observation period.

Any change that would make a new gate merge-blocking is not decided here. It requires founder
sign-off because it changes branch-protection policy, migration expectations, and merge risk.

### Verified absence ledger

Only gaps verified as absent by the researcher ledger are listed. Rank order follows protected
property priority: correctness first, then build-hygiene, then packaging. No supply-chain or
performance gap was verified absent by the evidence artifacts.

| rank | verified absent gap | protected property | evidence | implication |
|---|---|---|---|---|
| 1 | Unwrap and expect lint for library code | correctness | No workflow, script, Cargo, or crate match for `clippy::unwrap_used` or `clippy::expect_used`; existing Clippy denies warnings generally but does not deny these lints. | Panic-safety debt in library surfaces is not separately surfaced by CI. |
| 2 | MSRV gate | build-hygiene | No `rust-version` in Cargo manifests; CI pins `dtolnay/rust-toolchain@1.94.1`; workspace lint policy allows `incompatible_msrv`. | The current fixed toolchain proves pinned-toolchain buildability, not minimum supported Rust compatibility. |
| 3 | Rustdoc and intra-doc-link lint gate | build-hygiene | No matches for `RUSTDOCFLAGS`, `cargo doc`, `rustdoc`, `broken_intra_doc_links`, `private_intra_doc_links`, `intra_doc`, or `intra-doc` across workflows, scripts, Cargo files, and crates. | Rust API docs and links can rot without CI signal. |
| 4 | Packaging or publish dry-run CI gate | packaging | No workflow invokes `cargo package`, `cargo publish`, `scripts/publish.sh`, or `--dry-run`; local `scripts/publish.sh --dry-run` exists but is not wired to CI. | Package metadata and publishability are only locally checkable today. |

---

## Decision

Adopt the following CI-gate taxonomy and promotion policy.

### D1: Classify every substantive CI gate by protected property

Each row below records the gate, the property it protects, its trigger and path-filter behavior, and
whether it is gated or informational today.

| gate | protected property | trigger/path-filter | gated/informational + why |
|---|---|---|---|
| `CI / rustfmt` (`cargo fmt --all -- --check`) | build-hygiene | `.github/workflows/ci.yml`; `push` to `main`, `pull_request` to `main`; no path filter; matrix `ubuntu-latest`, `macos-latest`, `ubuntu-24.04-arm`; delegated through `scripts/ci.sh` | gated - deterministic format drift should fail the required CI context. |
| `CI / default Clippy` (`cargo clippy --workspace -- -D warnings`) | build-hygiene | Same `CI` workflow trigger; no path filter; same OS matrix; delegated through `scripts/ci.sh` | gated - default-feature lint cleanliness is required before merge, but this does not cover feature-gated lint debt. |
| `CI / Markdown doc format lint` (`deno fmt --check **/*.md`) | build-hygiene | Same `CI` workflow trigger; no path filter; delegated through `scripts/ci.sh`; only runs when `deno` is available | gated - a doc-format failure fails CI when the tool is present; skip-on-missing makes this a weak hygiene gate, not an informational report. |
| `CI / workspace tests` (`cargo test --workspace`) | correctness | Same `CI` workflow trigger; no path filter; delegated through `scripts/ci.sh` | gated - workspace tests are the default functional correctness gate for every PR. |
| `CI / tokenizer parity tests plus anti-skip` | correctness | Same `CI` workflow trigger; no path filter; delegated through `scripts/ci.sh`; runs `audit_tokenizer_parity` and `tokenizer_parity_e2e`, then fails if skip markers appear | gated - tokenizer parity is correctness-critical and explicitly fail-closed against silent skips. |
| `CI / embedding parity vs HF, weights optional` | correctness | Same `CI` workflow trigger; no path filter; delegated through `scripts/ci.sh` | gated - the test command is in required CI, so command failure blocks; coverage can still be limited by optional model weights. |
| `CI / workspace release build` (`cargo build --workspace --release`) | build-hygiene | Same `CI` workflow trigger; no path filter; delegated through `scripts/ci.sh` | gated - release-mode workspace build breakage is a required build-health failure. |
| `feature-matrix / lattice-tune safetensors + inference-hook + serde compile tests` | build-hygiene | `.github/workflows/ci.yml`; `push` to `main`, `pull_request` to `main`; no path filter; matrix `ubuntu-latest`, `macos-latest` | gated - feature-gated code is invisible to default workspace CI, so compile rot must fail the feature-matrix context. |
| `feature-matrix / lattice-tune train-backward compile tests` | build-hygiene | Same `feature-matrix` trigger; no path filter; Linux and macOS matrix | gated - protects non-default training code from compile rot without executing tests. |
| `feature-matrix / lattice-inference f16 + train-backward build` | build-hygiene | Same `feature-matrix` trigger; no path filter; Linux and macOS matrix | gated - protects a non-default inference feature combination from build regressions. |
| `feature-matrix / lattice-inference wgpu-gpu build` | build-hygiene | Same `feature-matrix` trigger; no path filter; Linux and macOS matrix | gated - workflow comments identify prior module-resolution rot in this host-compilable surface. |
| `feature-matrix / lattice-embed local build` | build-hygiene | Same `feature-matrix` trigger; no path filter; Linux and macOS matrix | gated - protects local embedding feature code that default workspace builds do not cover. |
| `feature-matrix / lattice-fann no-default-features build` | build-hygiene | Same `feature-matrix` trigger; no path filter; Linux and macOS matrix | gated - explicit no-default-features coverage prevents minimal-feature compile rot. |
| `feature-matrix / lattice-inference metal-gpu build` | build-hygiene | Same `feature-matrix` trigger; no path filter; macOS-only step | gated - Metal code only exists on macOS, so the gate is platform-scoped but required for the macOS leg. |
| `feature-matrix / lattice-embed metal-gpu build` | build-hygiene | Same `feature-matrix` trigger; no path filter; macOS-only step | gated - protects the Metal embedding feature surface from macOS-only build regressions. |
| `feature-matrix / lattice-inference metal-gpu Clippy` | build-hygiene | Same `feature-matrix` trigger; no path filter; macOS-only step | gated - workflow comments identify prior accumulated Metal lint debt outside default Clippy coverage. |
| `feature-matrix / lattice-embed metal-gpu Clippy` | build-hygiene | Same `feature-matrix` trigger; no path filter; macOS-only step | gated - extends Clippy coverage to a platform feature omitted from default workspace linting. |
| `feature-matrix / lattice-inference metal-gpu Q4 tests` | correctness | Same `feature-matrix` trigger; no path filter; macOS-only step; `LATTICE_METAL_TEST_ENFORCE=1` | gated - this is behavioral coverage for Metal Q4 dispatch correctness, with enforced fail-closed behavior if the runner loses required capability. |
| `bench-compile / lattice-inference bench-internals compile` | performance | `.github/workflows/ci.yml`; `push` to `main`, `pull_request` to `main`; no path filter; `macos-latest` | gated - the protected surface is the performance harness; it blocks merge on benchmark compile rot, not on benchmark score changes. |
| `bench-compile / lattice-embed benches compile` | performance | Same `bench-compile` trigger; no path filter; `macos-latest` | gated - keeps embedding benchmark code buildable so performance baselines remain measurable. |
| `cargo-deny / licenses + bans + sources` | supply-chain | `.github/workflows/ci.yml`; `push` to `main`, `pull_request` to `main`; no path filter; `ubuntu-latest` | gated - these checks are deterministic over manifest and lockfile state, so policy violations fail the required supply-chain context. |
| `cargo-deny / advisories` | supply-chain | Same `cargo-deny` trigger; no path filter; `continue-on-error: true` | informational - fresh external advisories can appear without a repo change, so the workflow intentionally reports them without wedging merges. |
| `e2e-parity / detect-engine-changes` | correctness | `.github/workflows/e2e-parity.yml`; `pull_request` to `main` with no workflow-level PR path filter; `push` to `main` filtered to inference and embed source paths; weekly schedule; manual; internal regex also checks relevant Cargo files, `Cargo.lock`, script, and workflow | gated - change detection controls whether expensive correctness jobs must run and `parity-gate` fails closed if detection errors. |
| `e2e-parity / greedy token agreement` | correctness | Same e2e workflow trigger; expensive job runs only when `detect-engine-changes` emits `engine=true` | gated - workflow comments state this gates greedy token agreement; engine changes cannot merge if parity fails. |
| `e2e-parity / speed report` | performance | Same e2e workflow trigger; produced by the parity report | informational - workflow comments explicitly say speed is reported informationally while token agreement is the gate. |
| `e2e-parity / embedding drift baseline` | correctness | Same e2e workflow trigger; expensive job runs only when `engine=true`; `LATTICE_DRIFT_GATE_ENFORCE=1` | gated - re-embeds frozen corpus texts and fails closed on drift or missing weights, so this is correctness rather than performance. |
| `e2e-parity / parity-gate required aggregator` | correctness | Same e2e workflow trigger; `if: always()`; always reports; mirrors parity, embed-drift, and q4-composed when `engine=true`; passes when no engine surface changed | gated - this is the required context shape that avoids path-filtered required checks getting stuck while still blocking changed engine PRs on failed parity, drift, or Q4 golden divergence. |
| `bench-update / build and run Criterion baselines` | performance | `.github/workflows/bench-update.yml`; `push` to `main` with CPU, attention, embed SIMD, bench, Cargo, workflow, and perf script paths; weekly schedule; manual | informational - no `pull_request` trigger; it maintains performance baselines on main rather than blocking merges. |
| `bench-update / publish perf-baselines branch` | performance | Same `bench-update` trigger; needs bench artifacts; writes orphan `perf-baselines` branch | informational - publishing baseline artifacts is a reporting and history-maintenance action, not a merge gate. |
| `app-binaries / build app-shipped binaries` | packaging | `.github/workflows/app-binaries.yml`; `pull_request` to `main` filtered to `crates/**`, `scripts/build-app-bins.sh`, and workflow file; `push` to `main` filtered to `crates/**`; manual | gated - it compiles the binaries the macOS app ships with their required features, so the protected property is packaging completeness. |
| `embed-parity-release / production-model embedding parity` | correctness | `.github/workflows/embed-parity-release.yml`; `push` tags `v*`; `push` branches `release-prep/**`; manual | gated - it fails the release-prep or tag workflow on production-model embedding parity regressions, but it is intentionally not a per-PR merge gate. |

### D2: Preserve the current gated vs informational split

The current split is intentional enough to preserve:

- Deterministic build, lint, test, parity, supply-chain policy, benchmark compile, and app-binary
  build failures remain gated when their workflows trigger.
- External advisory reporting remains informational because it can change independently of a repo
  diff.
- Speed reporting and baseline publishing remain informational because the repository has not
  defined variance, retry, hardware, or threshold policy for merge-blocking performance decisions.
- Release-cadence embedding parity remains gated for release-prep and tag workflows, but is not a
  per-PR required gate.

### D3: Implemented additive changes in this run

| implemented additive change | property | enforcement effect | rationale |
|---|---|---|---|
| Add this ADR as `docs/adr/ADR-064-ci-gate-taxonomy.md`. | documentation | No CI or branch-protection change. | Establishes a coherent taxonomy, verified gap ledger, and promotion rule before any gate expansion. |
| `rustdoc-lint` observation job in `.github/workflows/ci.yml` with `continue-on-error: true`. | build-hygiene | Non-required; cannot block merges. | Covers a verified absent docs gate using only the existing Rust toolchain; no new dependencies. |
| `unwrap-in-lib-lint` observation job in `.github/workflows/ci.yml` with `continue-on-error: true`. | correctness | Non-required; cannot block merges. | Covers a verified absent panic-safety lint using existing Clippy; expected to surface existing debt on first run. |

Neither observation job is added to any required-status set. Both jobs carry `continue-on-error: true`
at the job level so a failure is visible in the workflow UI but cannot block a merge. No Rust code,
dependency, script, or branch-protection rule is changed.

### D4: Skipped candidate: MSRV check

The MSRV check candidate was evaluated and skipped for this run. No workspace crate defines
`rust-version` in its `Cargo.toml`, so any MSRV job must either:

- Run `cargo check --workspace` under the same pinned toolchain (trivially passes, no new signal), or
- Pick an arbitrary older toolchain version that has no policy backing and could produce persistent
  noise if the codebase uses syntax unavailable on that version.

Without a founder-approved minimum Rust version, a non-trivial MSRV job cannot be made cleanly
non-blocking in spirit — it would be guessing at a policy rather than measuring against one. The
absence of `rust-version` remains a verified gap (rank 2 in the verified absence ledger). It is
recorded below as a founder-gated proposal where the required policy decision can be made explicitly.

### D5: Remaining non-blocking additive candidates (not yet implemented)

| candidate | protected property | non-blocking shape | blocker to implementation |
|---|---|---|---|
| `publish-dry-run` | packaging | New non-required job invoking existing `scripts/publish.sh --dry-run`, path-filtered to Cargo and crate/package surfaces, allowed to fail initially. | Requires review of the publish script's environment expectations before CI wiring; deferred to a separate change. |

### D6: Founder-gated proposals are not accepted changes

The following proposals would create or promote a merge-blocking gate. They are explicitly proposed
for founder sign-off and are not decided by this ADR.

| proposal | protected property | founder decision needed |
|---|---|---|
| Make `unwrap-in-lib-lint` required, or encode unwrap and expect denial in workspace lint policy. | correctness | Blocks merges on existing or future panic-safety violations; requires agreement on tolerated exceptions and migration path. |
| Define official MSRV, add `rust-version`, and make an MSRV compile check required. | build-hygiene | Establishes a compatibility contract that can constrain dependency and language-feature choices. |
| Make `rustdoc-lint` required. | build-hygiene | Blocks merges on public/internal doc link rot; requires agreement that rustdoc completeness is branch-protection material. |
| Make `publish-dry-run` required. | packaging | Blocks merges on package metadata or packaging issues; requires workspace publication policy decisions. |
| Promote `cargo-deny advisories` from informational to blocking. | supply-chain | Changes the existing deliberate split where external advisory publication should not wedge unrelated merges. |
| Add a PR performance-regression threshold gate. | performance | Current performance CI is compile-gated or informational baseline maintenance; enforcing thresholds requires threshold, variance, hardware, and retry policy decisions. |

### D7: Promotion rule for future gates

A CI check may become merge-blocking only when all of the following are true:

1. The protected property is named.
2. The trigger and path-filter behavior are explicit.
3. The failure mode is documented.
4. Existing failures are either fixed or covered by an accepted migration plan.
5. Branch-protection impact is approved by the founder.

This keeps new gates additive while preventing surprise merge blockage.

---

## Alternatives considered

1. Leave CI gates as ad-hoc workflow names.

   Rejected. The current workflows are useful, but without a protected-property taxonomy it is hard
   to tell which risks are intentionally covered, informational, or missing.

2. Make every verified gap a required gate immediately.

   Rejected. The verified gaps are real, but making them merge-blocking changes project policy and
   could block unrelated work before exception rules and migration plans exist.

3. Treat all advisory and performance signals as merge-blocking.

   Rejected. Advisory and performance signals have external or statistical failure modes. They need
   explicit retry, threshold, and ownership policy before branch-protection promotion.

4. Add a new third-party CI tool for each missing property.

   Rejected for this ADR. The current requirement is no new dependencies; the immediate candidates
   can be observed with the existing Rust toolchain, Clippy, and repository scripts.

---

## Consequences

### Positive

- Every substantive CI gate now maps to one protected property.
- Verified gaps are recorded without overstating them as accepted branch-protection changes.
- Required and informational signals are separated by policy, not by accident.
- Future CI additions have a promotion checklist that prevents accidental merge-blocking behavior.

### Negative

- The ADR does not itself increase CI coverage.
- Non-required observation jobs, if added later, may produce noise before the repo has cleanup
  budgets and ownership assigned.
- MSRV remains undefined until a separate policy decision sets a minimum supported Rust version.

### Risks

- Branch protection is configured outside workflow YAML, so "gated" status must be verified against
  repository settings when enforcement changes are proposed.
- Path-filtered workflows can produce missing required contexts if not paired with an aggregator or
  always-reporting check.
- Strict panic-safety and documentation gates may expose existing debt; promotion must include an
  exception and migration policy.

---

## Implementation plan

### Phase 1: Documentation taxonomy

Land this ADR with no workflow changes.

### Phase 2: Optional observation jobs

If accepted separately, add non-required jobs for rustdoc linting, unwrap/expect lint observation,
publish dry-run visibility, or MSRV telemetry. These jobs must not be added to branch protection in
the same change.

### Phase 3: Founder sign-off

For any proposed merge-blocking gate, decide the exact protected property, failure mode, migration
plan, and branch-protection context.

### Phase 4: Enforcement

Promote only signed-off gates to required status. Keep informational checks informational unless a
new ADR or explicit approval changes their failure policy.

---

## References

- `.github/workflows/ci.yml`
- `.github/workflows/e2e-parity.yml`
- `.github/workflows/bench-update.yml`
- `.github/workflows/app-binaries.yml`
- `.github/workflows/embed-parity-release.yml`
- `scripts/ci.sh`
- `scripts/publish.sh`
- `deny.toml`
- `Cargo.toml`
- `../explorer/gates_inventory.md`
- `../researcher/adr_format_and_gaps.md`
- `../analyst/taxonomy_and_gaps.md`
