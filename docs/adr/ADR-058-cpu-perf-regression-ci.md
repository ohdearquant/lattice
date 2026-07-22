# ADR-058: CPU Performance Regression Tracking in CI

**Status**: Superseded
**Date**: 2026-05-24
**Crate**: workspace (CI infrastructure, touches lattice-inference + lattice-embed bench surfaces)

## Status update (2026-06-24)

The proposed `bench-regression.yml` workflow was never created (confirmed: `ls .github/workflows/`
shows `app-binaries.yml`, `bench-update.yml`, `ci.yml`, `e2e-parity.yml` — no
`bench-regression.yml`). The actual regression gate shipped as `.github/workflows/e2e-parity.yml`
(PR #192), which runs HF transformers vs lattice on the same macOS runner and requires first-3-token
greedy parity. Criterion micro-benchmarks are collected as trend data by `bench-update.yml` but are
not a merge gate. This ADR's CPU-bench-as-gate design was superseded by the e2e-parity approach.

## Context

The decode hot path on Apple Silicon depends on tight CPU SIMD kernels (NEON elementwise/norm ops in `crates/inference/src/forward/cpu/`, int8 dot products in `crates/embed/src/simd/quantized.rs`, decode-attention QK/softmax/V in `crates/inference/src/attention/decode.rs`). These kernels are easy to break in non-obvious ways — recent examples from PR #76 alone:

1. Fused `E[x²]-E[x]²` variance in `layer_norm` introduced catastrophic cancellation on large-offset inputs; reverting to two-pass cost ~3% layer_norm throughput but was correctness-required.
2. Restoring the `-128` validation scan on the public `dot_product_i8` entry point cost a measurable percentage of int8 throughput before being moved behind a `_dispatch` indirection.
3. Tightening the polynomial `exp` upper clamp from 88.72 → 88.0 — no measurable impact, but easy to imagine a similar change costing 5% of softmax throughput.

The end-to-end decode bench (`scripts/bench_decode_harness.py run --profile apples_to_apples_q8`) catches macro regressions but is too noisy to attribute them: on a busy M2 Max with 5–10 concurrent GPU-bench processes, lattice decode throughput swings by 25% (issue #77). We cannot use noisy end-to-end numbers as a merge gate.

The GPU path cannot run on GitHub Actions runners (no Metal). But the CPU paths can — and they account for elementwise activations, normalization, the int8 embed path, decode attention reductions, and the entire CPU fallback for systems without GPU acceleration.

**Currently absent**: any automated mechanism to catch CPU regressions per commit. Criterion benches exist (`crates/inference/benches/elementwise_bench.rs`, `crates/embed/benches/quantized_bench.rs`, etc.) but run on-demand only, with no baseline retention or comparison gate.

### Constraints

- **No paid services.** Bencher.dev and similar are out. Self-hosted only.
- **Free GitHub-hosted runners only.** No self-hosted M-series runner for now (separate decision).
- **GH Actions noise.** Shared runners exhibit run-to-run variance; threshold must be loose enough to not flake, tight enough to catch real regressions.
- **Cross-architecture coverage.** Both NEON (aarch64 Linux via `ubuntu-24.04-arm`) and AVX2 (x86_64 Linux via `ubuntu-latest`) must be gated separately — a regression in only one path is still a regression.
- **Trend visibility.** The design requires "very granular" tracking — not just "did this PR regress" but "how has performance evolved over time."

## Decision

Introduce a baseline-comparing CI workflow with a version-tracked baseline branch.

### D1: Baseline storage — orphan `perf-baselines` branch

A long-lived orphan branch holds Criterion baseline data. Layout:

```text
perf-baselines/
  README.md                    # human-readable trend summary, regenerated each update
  aarch64-linux/
    elementwise_bench/
      base/                    # Criterion's --save-baseline=base output
        new/estimates.json     # per-bench medians + CIs
        ...
    quantized_bench/
      base/
    decode_bench/
      base/
  x86_64-linux/
    elementwise_bench/
      base/
    ...
  history/
    2026-05-24_<sha>_aarch64.json    # rolled-up summary per main commit, per arch
    2026-05-24_<sha>_x86_64.json
```

The orphan branch contains no source code, only baseline data — never merged into main, never affected by main rebases.

**Rationale**: explicit history via `git log perf-baselines`. Cheap (Criterion JSON is ~10KB per bench × ~20 benches × 2 archs = ~400KB total). Reproducible — anyone can `git checkout perf-baselines` and inspect a 30-day-old baseline. No bot tokens beyond `GITHUB_TOKEN` (built-in).

### D2: Two workflows — update + gate

**Workflow A: `bench-update.yml`** (runs on push to `main`)

Triggers: `push` to `main`, paths-filter on CPU-relevant directories (D5 below). Plus a weekly schedule (`cron: '0 6 * * 1'`) to refresh baselines even when no relevant file changed — guards against environmental drift on the runner side.

Matrix: `{ os: ubuntu-latest, arch: x86_64 }` × `{ os: ubuntu-24.04-arm, arch: aarch64 }`.

Steps per matrix job:

1. Checkout `main` (current SHA).
2. Build benches in release with `RUSTFLAGS="-C target-cpu=native"` (matches `bench_apples_*.sh` settings).
3. `cargo bench --bench <name> -- --save-baseline base --noplot`.
4. Append summary JSON to `perf-baselines/history/<date>_<sha>_<arch>.json`.
5. Commit + push `perf-baselines` branch (force-with-lease against itself, since it's orphan).

**Workflow B: `bench-regression.yml`** (runs on PRs)

Triggers: `pull_request` to `main`, paths-filter (D5).

Matrix: same `{x86_64, aarch64}`.

Steps per matrix job:

1. Checkout PR HEAD into `./pr`.
2. Checkout `perf-baselines` branch into `./baselines` (sparse, only the matching `<arch>-linux/` subdir).
3. Restore `./pr/target/criterion/<bench>/base/` from `./baselines/<arch>-linux/<bench>/base/`.
4. Build PR benches: `cargo bench --bench <name> -- --baseline base --noplot`.
5. Parse `target/criterion/<bench>/<test>/change/estimates.json` for each bench/test pair.
6. Apply gate logic (D3) and emit GH check + PR comment.

### D3: Gate logic

For each `(arch, bench, test)` triple, Criterion reports `change.estimates.mean.point_estimate` (relative change, e.g. `-0.05` = 5% faster, `+0.12` = 12% slower) along with a 95% confidence interval (`change.estimates.mean.confidence_interval.{lower,upper}_bound`).

**The threshold is applied to the _lower bound_ of the 95% CI of the change, not the point estimate.** A regression "counts" only when Criterion is statistically confident that the change exceeds the threshold — eliminating most noise-driven false positives.

| CI-lower-bound of change               | Action                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| ≤ +3%                                  | Pass silently                                                                      |
| +3% to +7%                             | Pass with PR-comment warning ("⚠ minor regression confirmed: ..."), no merge block |
| > +7%                                  | Fail the check, block merge                                                        |
| Point estimate < −3% AND CI-upper < 0% | Pass with celebratory PR comment ("🚀 confirmed improvement: ...")                 |

PRs may bypass the >7% fail by applying the label `bench-allow-regression`. The PR comment must then quote the rationale (e.g. "intentional: layer_norm two-pass revert for numerical stability, see ADR-058 context"). The label is intentionally awkward to apply — protects against accidental bypass.

**Why 7% on CI-lower-bound**: a naive 7% threshold on the raw point estimate would produce unacceptable flake — shared-runner variance puts the P95 of single-bench point-estimate noise around ±7%, so with ~25 measurements per workflow run the per-PR false-positive rate would approach 70%. Using the _lower bound of Criterion's own 95% CI_ requires the run to be statistically confident the regression is real before it can cross the threshold, which collapses noise-driven flakes back to <2% per PR while preserving 7% sensitivity to genuine kernel slowdowns. If observed false-positive rate is still >5% after the 7-day shadow period (D2/Rollout step 3), tighten the CI-confidence threshold from 95% → 99% rather than weakening the 7% sensitivity.

**Anti-game**: the lower-bound rule means a single noisy run with a wide CI passes — but that single run also doesn't update the baseline (only push-to-main does). The PR author cannot escape a real regression by re-running until they get lucky variance; the regression remains, and a subsequent confident measurement will catch it.

### D4: Bench scope (Phase 1)

Phase 1 includes every bench whose result is _directly_ informative for per-token decode CPU work AND whose target code exists on `main` today. The decode-attention NEON kernels live in `crates/inference/src/attention/decode.rs` which is currently unmerged (PR #76); they enter the suite via Phase 2.

| Crate     | Bench file              | Status          | Group                                                                                                                                                 |
| --------- | ----------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| inference | `elementwise_cpu_bench` | NEW in this ADR | `rms_norm/4096`, `layer_norm/4096`, `silu_inplace/4096`, `gelu/4096`, `add_bias_gelu/4096`, `softmax_attention/{seq=128,512}`, `elementwise_mul/4096` |
| embed     | `simd` (existing)       | EXISTING        | `simd_dot_product`, `simd_normalize`, `int8_quantization`, `int8_vs_float32_cosine`, `int8_batch_cosine`                                              |

**Implementation note for `elementwise_cpu_bench`**: targets the public functions exported from `lattice_inference::forward::cpu::*` — `rms_norm`, `layer_norm`, `silu_inplace`, `gelu`, `add_bias_gelu`, `softmax_attention`, `elementwise_mul`. Deterministic LCG-seeded inputs. Two hidden sizes (4096 for the Qwen3.5-0.8B FFN dimension, 896 for the model dimension) where size variation is informative. ~120 LOC.

Phase 2 (deferred until PR #76 merges): add `decode_attn_cpu_bench` covering decode-step QK dot, softmax, V accumulation kernels from `crates/inference/src/attention/decode.rs`. The bench list is config-driven (`.github/perf-benches.json`) so adding it after PR #76 lands is a one-line PR.

Phase 3 (deferred): synthetic "CPU portion of one decode step" bench — sequence of `rms_norm → qkv_projection → attention → ffn → residual` on Qwen3.5-0.8B-sized buffers. Closer to end-to-end signal. Deferred because realistic buffer construction without the full model loader requires fixture work that doesn't earn its keep until Phase 1 + 2 demonstrate the workflow value.

### D5: Path filters

Both workflows trigger only when relevant paths change:

```yaml
paths:
  - "crates/inference/src/forward/cpu/**"
  - "crates/inference/src/attention/decode.rs"
  - "crates/inference/src/attention/standard.rs"
  - "crates/inference/benches/**"
  - "crates/embed/src/simd/**"
  - "crates/embed/benches/**"
  - "Cargo.toml"
  - "Cargo.lock"
  - ".github/workflows/bench-*.yml"
```

Plus `workflow_dispatch` for manual runs. Plus the weekly cron on the update workflow (D2).

### D6: Trend visibility — auto-generated `perf-baselines/README.md`

The `bench-update.yml` workflow regenerates `perf-baselines/README.md` at the end of every run on main. The regenerator (`scripts/perf-baselines-readme.py`, lives in main and is invoked by the workflow against the checked-out `perf-baselines` branch) produces:

**Section 1 — Headline (per arch)**

```
## aarch64-linux (last update: 2026-05-24 14:00 UTC, commit a1b2c3d)
- 30-day worst regression:  +11% rms_norm/4096 (commit f9e8d7c, "perf: revert layer_norm fusion")
- 30-day best improvement:  -23% silu_inplace/4096 (commit 4842197, "perf: 4x unroll silu NEON")
- All-time best (current):  silu_inplace/4096 = 1.98 µs (commit 4842197, 2026-05-22)
```

**Section 2 — Sparkline table (per arch, per bench)**

A markdown table where each row is a bench, each cell is one of the last 20 measurements on main. Cells render as relative throughput vs the row's all-time best, with a tiny ASCII sparkline using only block characters (▁▂▃▄▅▆▇█). Example row:

```
| silu_inplace/4096 | ▇▇▆▆▇▇▇▇▆▇▇▇▇▇▆▇▇▇█▇ | best: 1.98µs | latest: 2.04µs (+3%) |
```

**Section 3 — Per-bench drill-down (one collapsible `<details>` per bench)**

Inside each, the last 30 commits with: SHA (linked to commit), date, point estimate, 95% CI bounds, change% from previous, change% from all-time best. Sortable by clicking column headers? — no, that's JS. Just sorted reverse-chronologically.

**Why this shape**: renders natively on GitHub when browsing the branch. No external dependencies — only Python stdlib in the regenerator. Sparklines use Unicode block characters that render in any monospace terminal or GitHub's markdown viewer. The file is human-readable raw (in `cat`) and rich on GitHub.

The regenerator is deterministic — given identical `history/` contents it produces byte-identical output, so the README only changes when the underlying data changes.

### D7: Local reproduction

A `make bench-ci` target builds and runs the exact same matrix locally on the developer's host (without the baseline comparison — that requires `perf-baselines` checkout). A `make bench-gate` target checks out `perf-baselines` into `./.cache/perf-baselines/`, restores the baseline for the local arch, runs benches, and prints the same change% table the CI would compute. Lets developers preview the gate before pushing.

## Consequences

### Positive

- **Catches micro-regressions before merge.** Every SIMD kernel change is now version-tracked at the bench level.
- **Cross-arch coverage at zero marginal cost.** GH Actions ARM runners are free; we get NEON validation that local x86 dev boxes can't provide and vice versa.
- **Trend history is forever-queryable.** `git log perf-baselines -- aarch64-linux/elementwise_bench` shows every regression and recovery on that bench, with commit messages.
- **No third-party dependency.** Entire stack is GH Actions + Criterion + git. Survives indefinitely without any external service.
- **Force-pushable orphan branch never affects main.** No risk of accidentally polluting main history with bench data.

### Negative

- **2x bench wall time per PR** (build baseline + build current). Mitigation: `Swatinem/rust-cache` for deps; Criterion bench builds themselves are seconds, not minutes.
- **Noise tolerance is real.** Even with a 10% threshold, ~5% of bench runs may flake-warn. The `bench-allow-regression` label and the warning band (5-10%) give an honest escape valve without weakening the hard gate.
- **No Apple Silicon coverage in CI.** Metal GPU regressions and M-series-specific NEON behavior remain caught only by local `scripts/bench_decode_harness.py run --profile apples_precise` runs. This ADR explicitly does not solve the Apple Silicon CI problem — see Alternatives Considered §A3.
- **Bench shape is now part of the contract.** Renaming a Criterion bench group requires updating `perf-baselines` (it'll appear as a "new bench, no baseline" rather than a regression). Document this in `perf-baselines/README.md`.

### Risks

- **`perf-baselines` branch corruption.** If a force-push lands wrong, baselines for all benches reset. Mitigation: keep daily history snapshots under `history/`; can restore by checking out an old commit on the perf-baselines branch.
- **Path-filter false negatives.** A change to, say, `crates/inference/src/lib.rs` that affects inlining in a hot path won't trigger the bench. Mitigation: weekly cron baseline-refresh on main catches drift; consider expanding paths filter in Phase 2.
- **Criterion's change-detection statistical model.** Criterion uses a Welch's t-test on samples — its `regressed` flag is more conservative than our 10% threshold. We use the raw `change.point_estimate` directly to enforce our own threshold, but we surface Criterion's verdict alongside in PR comments for cases where the t-test disagrees (e.g. very high variance in one run).

## Alternatives Considered

### A1: Criterion-internal regression detection only (no baseline branch)

Just use `cargo bench --baseline previous` and rely on Criterion's built-in `regressed` field. Rejected because:

- No trend history across commits.
- Baseline must be re-established on every PR, doubling cost.
- Criterion's statistical test isn't tunable per-bench — some benches need stricter thresholds than others.

### A2: Bencher.dev or Codspeed (paid hosted services)

Rejected: building in-house avoids external cost and dependency.

### A3: Self-hosted Apple Silicon runner

Would solve the Metal GPU CI gap and give M-series-accurate NEON numbers. Rejected for now because:

- Operational cost: a Mac mini M2 idle in a closet is not free (electricity, security patches, network).
- Single point of failure: if the runner dies, all perf CI dies.
- Security review: self-hosted runners need careful isolation from secrets.

Revisit if Phase 3 (synthetic decode CPU bench) proves insufficient to catch M-series-specific issues, or if Metal GPU regressions become a recurring problem.

### A4: Run benches inside the existing `ci.yml` workflow

Bench wall time (~5-10 min on shared runners) would double the CI duration for every PR, even ones touching only docs or unrelated crates. The path-filter + separate workflow approach keeps perf CI off the critical path for non-CPU PRs.

## Rollout

1. **D-day**: merge this ADR (Draft → Accepted).
2. **+1 day**: add `attention_bench.rs` (D4 Implementation note), implement `scripts/perf-baselines-readme.py`, implement `bench-update.yml`. Trigger `workflow_dispatch` manually to seed `perf-baselines` branch with current main as baseline for both archs.
3. **+2 days**: implement `bench-regression.yml` in advisory mode (`continue-on-error: true`) for 7 days. Collect false-positive rate from real PRs.
4. **+9 days**: enable gating (remove `continue-on-error`).
5. **+30 days**: review observed flake rate:
   - <2% flake: keep 7% threshold + 95% CI confirmation (the chosen configuration).
   - 2-5% flake: tighten CI confirmation from 95% → 99%, keep 7% threshold.
   - 5% flake: raise threshold to 10%, keep 95% CI. (Acknowledges hardware variance is genuinely higher than expected.)

## References

- Criterion JSON output schema: <https://bheisler.github.io/criterion.rs/book/user_guide/csv_output.html>
- GitHub-hosted runner specs: <https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners>
- ARM runner availability: <https://github.blog/2024-09-03-arm64-on-github-actions-powering-faster-more-efficient-build-systems/>
- Issue #77 — GPU contention variance (M2 Max local bench noise)
- PR #76 — context for catastrophic-cancellation revert (layer_norm two-pass)

## Post-acceptance note (2026-07-12): the 95% → 99% escalation is unsound — do not arm it

A subsequent adversarial statistical review of this design found the confidence-escalation
path (D3 "Why 7% on CI-lower-bound" and Rollout step 5: on persistent false positives,
tighten the CI confidence from 95% to 99% "rather than weakening the 7% sensitivity")
to be backwards. At a fixed threshold, requiring higher confidence widens the margin the
lower bound must clear before tripping, which *reduces* power against genuine regressions;
it does not preserve sensitivity. If observed flake is high, the honest levers are more
samples per run, better runner isolation, or an explicitly raised threshold.

Two further findings compound the problem as specified:

- Criterion's change CI is bootstrapped from the observed baseline and candidate benchmark
  sessions; it does not model between-session or runner-pool variance. On shared CI
  hardware, run-to-run variance can be substantially larger than what one observed pair of
  sessions shows, so a "statistically confident" bound from that pair is not confidence
  about the distribution the gate actually faces; a bimodal runner pool can hand the
  bootstrap two internally stable sessions from different modes, yielding a narrow,
  confidently wrong comparison and false failures at zero true regression.
- Refreshing the baseline on each qualifying main update (path-filtered pushes, plus
  scheduled and manual refreshes) can ratchet a real regression into the baseline before
  any gate observes it.

Status: the D2/D3 hard merge gate described here was never armed (an advisory-mode
workflow was implemented, then removed). Merge gating shipped instead as same-run
end-to-end greedy-token parity against a reference implementation (a within-run
comparison), plus an absolute Q4 perplexity golden captured on the CI runner class and
enforced with a fixed tolerance (a cross-run comparison by construction, carrying its own
environment-drift risk). Criterion micro-benchmarks are collected as trend data only
(`bench-update.yml` / `perf-baselines`). Any future revival of a hard Criterion gate needs
a cross-run variance model measured on the actual runner pool, not the escalation path in
this document.
