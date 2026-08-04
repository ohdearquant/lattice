# Lattice — Claude Code Instructions

Read `AGENTS.md` first for coding conventions, crate structure, and design principles.

## Development Process

### Measure First, Code Second

Every PR that touches `crates/inference/`, `crates/embed/`, or `crates/fann/` must include `make bench-compare` output. No exceptions. A PR without before/after numbers is incomplete regardless of what the code looks like. One honesty note on `crates/fann/`: the default bench build does not enable the optional `mixture` feature, so a fann-only diff is usually compiled out of the two default bench targets — but default-target compilation status alone is not sufficient to claim the waiver. `crates/fann/` also declares its own feature-gated `router_online` bench target (requires the `online-router` feature; `crates/fann/Cargo.toml:45-48`), which directly exercises FANN training and network APIs. A fann diff takes the same all-declared-target reachability search as any other crate: search every declared target, including `router_online`, and if a reachable one exists, run it instead of claiming the waiver.

```bash
make bench-compare                         # origin/main vs HEAD (--quick, the default; not separately measured)
make bench-compare BASE=main HEAD=pr/x     # explicit refs
scripts/bench-compare.sh --full main       # tight CIs; see the timing note below before booking a window
```

Paste the output in the PR description. If nothing changed, say "bench-compare showed no change (p > 0.05 on all groups)." If something regressed, explain why it's acceptable or revert it.

This process caught a decode throughput regression that had been attributed to "GPU contention noise" for days: an A/B comparison against the parent commit identified the f32 dot_product unrolling as the cause. The original figures are not quoted here because that run's conditions were not recorded.

### Keep the Machine Quiet During A/B Runs — Quiet Means Zero Disk Activity, Not Just Zero Builds

`scripts/bench-compare.sh` now enforces the machine side of this itself, so do not wrap it in an external bench-window helper. It takes both machine-wide advisory locks (the bench window and the Metal GPU lock) unconditionally for the whole run. At each of the three boundaries — before base, between phases, and after head — macOS runs hold a mandatory 30-second cooldown, then require AC power, nominal thermal state, at least 30 seconds of HID idle, and the portable ambient-CPU idle floor. Linux CI records that the macOS-only probes are unavailable and still applies the CPU-idle gate. The CPU floor is settable with `BENCH_IDLE_FLOOR`; if you move it, say so wherever the numbers are quoted. Every run prints a `Run conditions` block recording the refs, effective bench targets and features, resolution, lock dispositions, machine-state checkpoints, and measured idle samples — quote that block along with the numbers, because a figure that does not record what produced it is indistinguishable from one produced on a quiet machine.

Both locks are taken regardless of what the run benches. That is deliberate: deciding whether a target is GPU-driving would mean maintaining an enumeration of bench names, feature combinations, and transitive dependencies that pull Metal in without saying so, and every miss would pass the check while the GPU spins. Serializing a CPU-only bench against GPU work is correct rather than merely tolerable, since GPU work during a CPU bench is exactly the ambient load the idle floor exists to exclude.

What the locks and the probe cannot see is disk activity, so the rest of this section still applies to you rather than to the script. A bench window means no concurrent builds AND no git checkouts, worktree adds, large file copies, or downloads. Filesystem indexing churn (Spotlight `mdworker`, `fseventsd`) from a repository checkout lands asymmetrically in whichever measurement phase it overlaps, and base-then-head runs make that asymmetry read as a code regression. A/B runs have been corrupted this way: worktree checkouts overlapping the head phase produce large apparent swings in groups the diff could not reach.

Before re-running a corrupted A/B, check structural reachability first, and run that check before booking a window rather than after spending one. The question is not whether the changed code is compiled into the bench binaries. It is whether any bench group ever executes it. Two different situations both answer no, and both are equally fatal to the measurement. A diff confined to a `cfg`-gated module (e.g. `#[cfg(all(target_os = "macos", feature = "metal-gpu"))]`), or to a bench target excluded by its own `required-features` list under default features, is compiled out of a default-feature bench build entirely, so base and head binaries are built from identical effective source. A diff that is compiled in but sits on a call path no bench group reaches is just as unmeasurable: the binaries genuinely differ, and every group still times identical executed code. Compilation is the wrong predicate. Reachability is the right one, and a change can pass the compilation test while failing the one that matters.

To be precise about how this interacts with the "no exceptions" rule above: the bench-compare disposition section of the PR is still mandatory for every `crates/inference/`, `crates/embed/`, or `crates/fann/` PR. An unreachability proof is the one narrow case where that section may contain a structural argument instead of an A/B table, and it carries a heavier burden than the old compiled-out wording did.

The proof must name the population it searched, because the two targets `bench-compare` runs by default are not the population. `crates/inference` declares 23 bench targets and the default disposition path runs one of them, so "the default A/B showed nothing" is a statement about coverage, not about the change. Search every declared bench target for one that reaches the changed code — Cargo selects bench targets by `required-features` as well as by `cfg`, so search both. If a reachable target exists, run it (`--bench <name>`, plus whatever features its `cfg` gate or `required-features` entry require) instead of claiming a waiver. Only when no declared target reaches the change does the waiver apply, and then the proof states which targets were searched, and either names the `cfg` gate or `required-features` entry together with the bench build's feature set, or names the call path that does not exist. State the residual risk in the same paragraph: an unreachable change is not a safe change, it is an unmeasured one, and saying so is the point of the disposition.

"Run it" assumes the reachable target sits inside `bench-compare`'s paired machinery. It doesn't always: `scripts/lib/bench-compare-impl.sh` drives exactly two packages, `lattice-inference` and `lattice-embed`, each through `cargo bench`'s Criterion `--save-baseline`/`--baseline` pair, and `Makefile`'s `bench-compare` target reaches only that script. A reachable target outside those two packages, or one that never calls into Criterion in the first place (a plain `fn main()` binary, not `criterion_group!`/`criterion_main!`, so there is no baseline to save or diff against), has no route through that pair — `crates/fann`'s `router_online` is both at once: it lives in `lattice-fann`, not `lattice-inference`/`lattice-embed`, and its body is a plain `fn main()` (`crates/fann/benches/router_online.rs:412`) with no Criterion dependency anywhere in `lattice-fann`'s manifest. For that target, the disposition still requires a before/after comparison; it's just not `bench-compare`'s. Check out base, run the target once under `scripts/bench-command.sh --label <name> --durable -- <command>` (`--durable` is required for this: it takes the same two machine-wide locks `bench-compare.sh` uses, plus a CPU-idle floor check before the command; the after-check runs only if the command exits zero — a nonzero exit returns that status immediately and skips the after-sample, so a failed run is gated only on entry — omitting `--durable` still takes both locks but runs no idle check at all. Even with `--durable`, this is a narrower gate than `bench-compare.sh`'s — it has no macOS cooldown, AC-power, thermal, or HID-idle checkpoint, so treat it as lock-serialized and CPU-idle-gated, not as the full quiet-machine discipline), record its output, then repeat at head and diff the two outputs by hand. A single run of such a target — head only, or base only — is supplemental: it shows the change executes, not how it moved anything, and it does not substitute for the paired before/after comparison the disposition requires.

This reachability search covers runtime source changes: it answers whether a call path reaches an edited line, which presupposes the edit is a line a call path could reach. It does not cover the change's build inputs. A manifest change (including a `[[bench]]` table or a `required-features` list), a dependency or lockfile bump — `crates/inference/Cargo.toml`'s `criterion` entry is a direct input to the locked bench invocation (`cargo bench --locked`, `scripts/lib/bench-compare-impl.sh:509-510`) — a feature or default-feature change, a `[profile.*]` change, a build script, generated source, or a change to a bench target's own definition or to the bench harness has no absent call path to name and no `cfg` or `required-features` gate to name closed: the change alters what gets compiled or how the runner invokes it before any function body executes. No `cfg`-gate or absent-call-path proof is available for these, and their absence from a call graph is not evidence that they don't move the numbers. The structural waiver applies only to runtime source changes whose performance-relevant build inputs — manifest, dependency and lockfile versions, feature set, profile, build script and generated output, bench-target definitions, and harness — are identical between base and head; run a measured target whenever one of those inputs differs and can affect a bench.

A reachability verdict costs minutes of reading and decides whether a bench window is worth booking at all. Run it first. A `--full` A/B on the two default targets measured 41 minutes for the base arm alone (264 groups, Apple silicon laptop, 2026-08-02), and the embed `simd` target was 97% of that; the head arm repeats the same work. Treat 41 minutes as a slow-side bound rather than a clean figure: that run's idle probe certified before the base arm and then failed at the following phase boundary, so the arm's tail was measured on a machine that had gone loud. Booking that against a change no group executes spends a window sized off the measured base-arm bound above — doubled for the head arm, which repeats the same work — to produce a table that cannot say anything; the doubled figure is a derivation, not a measurement.

### Bench by Group, Not All at Once

Never run the full Criterion suite; see the measured slow-side bound above. Filter to the groups your PR touches:

```bash
scripts/bench-command.sh --label embed-simd -- cargo bench -p lattice-embed --bench simd -- "simd_dot_product"
scripts/bench-command.sh --label embed-simd -- cargo bench -p lattice-embed --bench simd -- "int8_raw|normalize"
scripts/bench-command.sh --label inference-cpu -- cargo bench -p lattice-inference --bench elementwise_cpu_bench
```

For the A/B workflow, pass the same Criterion filter through `make bench-compare`:

```bash
make bench-compare BENCH_GROUPS_INFERENCE="rms_norm|gelu"
make bench-compare BENCH_GROUPS_EMBED="simd_dot_product|int8_raw"
```

Leaving these variables unset keeps the default `elementwise_cpu_bench` and `simd` bench targets.

The local script paths classified in `scripts/bench-measurements.toml` enter a
cooperative wrapper on ordinary direct invocation. This prevents accidental
unlocked runs but is not a same-user authentication boundary. Add new local
measurement entry points to that inventory; the CI contract rejects an
unclassified `scripts/bench*` entry. Source-pattern discovery is advisory: a
lexical no-match does not prove that a script never measures. The Rust
inventory in the same manifest covers only its declared path grammar; other
Rust examples, binaries, and tests require manual classification. Use
`scripts/bench-command.sh --label <name> -- <command>` for an ad-hoc raw CPU
Criterion command. `make bench-ci` and `make bench-gate` also refuse below the
ambient-idle floor because their baseline or result outlives the process that
produced it.

Quick mode (`--quick`) is sufficient for direction + magnitude. Full mode only when you need tight CIs for a PR description or ADR evidence.

### Differential Test First (Cross-Framework Bugs)

When lattice produces different output than a reference framework (MLX, HF transformers, llama.cpp), write a self-contained Python script that runs the same primitive in both frameworks and compares max-diff **before** reading lattice code or spawning investigation agents. A 20-line script gives a definitive answer in seconds; code-reading and agent analysis take hours and can converge on wrong conclusions.

```python
# Template: /tmp/test_<primitive>_conv.py
import numpy as np, mlx.core as mx, mlx.nn as nn
# 1. Construct minimal input
# 2. Run via MLX (reference)
# 3. Run via each candidate lattice convention (as numpy)
# 4. Compare: which candidate has max-diff < 1e-4?
```

The template above constructs no input and prints no max-diff; the figures below are a recorded result from one past investigation that followed it, not an output the template itself reproduces. That investigation closed a 0.77 PPL gap on Qwen3.5-0.8B (16.62 pre-fix vs. 15.86 MLX gold) that had been misdiagnosed as "f32-vs-bf16 precision drift" for days. The actual bug was a RoPE pairing convention mismatch — interleaved `(2i, 2i+1)` vs stride-half `(i, half+i)`. Verified in 5 seconds: stride-half max-diff `8e-6`, interleaved `67.5`. PPL dropped from 16.62 → 15.89 (MLX gold 15.86).

**Quantitative bounds reject hypotheses cheaply.** Before chasing "FP precision drift" or other plausible-sounding causes, check the literature for typical magnitude:

- f16 vs f32 PPL delta: `~0.00x` (llama.cpp community)
- bf16 vs f32 PPL delta: `<0.05` (arxiv:2510.26788)
- Q4 quantization PPL delta: `0.1-0.3` (llama.cpp #406)

If the gap you're investigating exceeds these bounds, the cause is structural (algorithm, layout, convention), not numerical. Reject the precision hypothesis on quantitative grounds and look for a real bug.

**Be skeptical of comments that paraphrase config fields.** A comment that says "X uses field=true" without explaining what the field actually controls in the reference implementation is a footgun. The lattice RoPE comment said "Qwen3.5 uses mrope_interleaved=true" — technically matched config, but `mrope_interleaved` controls multimodal M-RoPE section interleaving (video/image tokens), not 1-D text RoPE pairing. The bug existed for months because nobody verified the comment against HF's `rotate_half` or MLX's `nn.RoPE`.

### Triage Flaky vs Deterministic Before Filing

When a test fails intermittently or fails alongside unrelated work, run the discriminating experiment before writing the issue: ONE test, solo, `--test-threads=1`, idle GPU, exact main SHA. Concurrent GPU load corrupts both timing and numerics, so a "pre-existing failure on main" verified while other GPU work runs is not verified at all.

This split a two-test failure report cleanly: one test failed deterministically at every prompt length under solo idle-GPU conditions (real chunk-boundary accumulation drift, its own issue), while the other passed solo (a load flake, a different issue). Filing them as one regression would have sent the fix to the wrong place.

### Machine-Wide GPU Test Lock

Metal lock coverage is enforced by `crates/inference/tests/metal_measurement_lock_contract.rs`: discovered `MetalQwen35State` construction sites must acquire the shared lock before construction or name an explicit exemption, while raw measurement markers use an exact inventory. This is not a claim that every Metal-touching target is locked; long-running processes and explicitly listed legacy targets are exempt. Locking callers serialize through the single `gpu_test_lock()` implementation in `crates/inference/src/measurement.rs`. The module is a `#[doc(hidden)]`, Metal-only export because Cargo builds integration tests, benches, examples, and binaries as crates separate from `lattice-inference`; it is not production API. The guard holds two locks: an in-process mutex (thread serialization within one test binary) and an exclusive advisory flock on `/tmp/lion-metal-gpu-test.lock` (cross-process serialization, machine-wide convention). Any harness on this machine that drives the GPU for measurements — other repos' test suites, bench runners, one-off scripts — should acquire the same flock before touching Metal. Concurrent GPU work corrupts both timing and numerics: contended confirmation batches inflated top-k logit margins enough to produce false failure reports (#628, #629).

The lock blocks for up to 30 minutes, then panics with an `lsof /tmp/lion-metal-gpu-test.lock` hint rather than hanging silently. If a run appears stuck at test start, another process is holding the GPU; check who with `lsof` before killing anything.

### Regression Tests Must Be Mutation-Sensitive

A regression test that passes with the fix reverted is decoration. Before claiming a test guards a fix: revert the fix (reverse-apply the diff — never `git checkout` over uncommitted work), `touch` the source file so cargo actually rebuilds, and watch the test fail. Then restore the fix and watch it pass.

### Grep for Sibling Invocation Paths

When a harness or guard fix lands in one invocation path, grep the same file for sibling paths that construct the same operation independently: a second subprocess command builder, a second workflow step calling the same script, a reimplementation of a guarded method. Any fix expressible as "add flag X to the call" has an unguarded copy-paste sibling until proven otherwise, and the fix's own description ("mirror the CPU path") is the grep query.

This class recurs. It became a rule after a greedy-decoding sampler flag was added to the CPU parity harness path while the Metal-path command builder in the same file went without it, surfacing only on that leg's first live CI run. No count is given deliberately: a recurrence tally in guidance is a claim that no reader ever re-measures, so it can only go stale, and the invariant carries the rule without it.

### E2E Parity Gate

PRs touching `crates/inference/src/` or `crates/embed/src/` trigger `e2e-parity.yml`. It runs HF transformers (reference) and lattice on the same macOS runner, comparing greedy generation output. The reference runs first to warm the machine.

- **Token parity**: first 3 greedy tokens must match HF exactly (2 for the long-prefill prompt)
- **Speed**: reported informationally, not gated
- **Baseline tracking**: `bench-update.yml` still collects Criterion micro-benchmarks on merge to main (trend data, not a gate)

## Session Protocol

- Run `cargo clippy --workspace -- -D warnings` before reporting any task complete.
- Use `make ci` for full validation (fmt + clippy + doc lint + test + build).
- Feature branches + PRs for all changes. Never push directly to main.
- Conventional commits with crate scope: `feat(inference): add Qwen3.5 MoE support`.

### Merge Gate

A PR merges once its required checks are green. Branch protection no longer requires the
branch to be up to date with main first (`required_status_checks.strict = false` as of
2026-07-12). Note the precise semantics: PR workflow checks still run on GitHub's synthetic
merge ref (the PR merged into main as of when the run started), but with strict off a
stale head may merge after main has since advanced, so a green run is not guaranteed to
cover the eventual merge with current main.

That gap is exactly what broke main for two hours after #634: it merged green-on-a-stale-base,
but main had gained call sites of the API it changed, and the merged combination was never
compiled anywhere before landing (four more stale-base PRs — #636, #638, #639, #642 — then
auto-merged onto the red main). With strict checks off, closing that gap is a review-side job:
when a PR changes a function signature, trait, or other public surface, check it against
current main during review, not just against the PR's own diff.

- After every merge to main, watch the main-push CI run.
- If main goes red, revert fast rather than fix-forward — root-cause after main is green again.

## Agent Spawning

- Use `subagent_type` from: `implementer`, `tester`, `critic`, `architect`, `researcher`, `analyst`, `reviewer`.
- Critic agents run AFTER implementers, never in parallel.
- Max 5 agents per batch.
- Implementers for code changes, critics for review, analysts for investigation.

## What Not To Do

- Do not guess the public API of any crate — read `src/lib.rs` first.
- Do not add CUDA support or ONNX dependencies.
- Do not use `unwrap()` in library code.
- Do not add comments explaining WHAT the code does.
- Do not create new crates without explicit approval.
- Do not modify the dependency direction (leaf crates must stay leaf).
- Do not claim "X% faster" without a bench measurement from this session.
- Do not run the full Criterion suite when you can filter to relevant groups.
- Do not submit a perf PR without `make bench-compare` output.

## Performance Workflow (ADR-087)

- **Every perf PR must include before/after numbers.** No exception. Run `make bench-compare` (or `scripts/bench-compare.sh <base> <head>`) to get an A/B table. Paste the output in the PR description.
- Default to `--quick` (not separately measured; see the measured `--full` slow-side bound above before booking a window). Use `--full` only when CIs are too wide to tell.
- After merging to main, `bench-update.yml` auto-updates the `perf-baselines` branch (trend data, not a gate). PRs are gated by `e2e-parity.yml` (greedy token agreement vs HF).
- For local baseline tracking: `make bench-ci` saves a local baseline, `make bench-gate` compares against the `perf-baselines` branch.
- Do not claim "X% faster" without a measurement from this session. Stale numbers from prior sessions are not evidence.

## Crate Ownership

Changes to `inference` affect `embed` in an ordinary build, and affect `tune` only under `inference-hook` or `train-backward` — neither is in tune's default set, so a default `cargo build -p lattice-tune` does not compile `lattice-inference` at all. Changes to `fann` affect `tune` unconditionally and `inference` only under the optional `mixture` feature, which is likewise off by default. Only `fann` and `transport` are leaf crates; `transport` has no internal dependents (`embed` uses it in dev-tests only). When sizing the blast radius of a change, read the consuming crate's default feature set rather than the arrow: two of these four edges are absent from a default build.

## Publishing

Publish order follows the internal dependency DAG (deps before dependents): fann, transport (leaves) → (wait 30s) → inference (depends on fann via the `mixture` feature) → (wait 30s) → embed, tune. Use `make publish`. Internal path deps' `version =` field must match the current workspace version (bump them in lockstep when bumping `[workspace.package].version`). When a feature adds a new internal dep (e.g. `mixture` made inference depend on fann), the publish order changes — re-derive it from `crates/*/Cargo.toml` path deps, do not assume the old order.

**Shipped-bug recovery (bump-and-yank).** crates.io versions are immutable. When a published release has a correctness bug:

1. Bump workspace + path-dep versions to the next patch
2. Update release notes file (rename if needed); add a "Note on v<broken>" section explaining the yank
3. Tag + GH release + `make publish`
4. `for c in lattice-inference lattice-fann lattice-transport lattice-embed lattice-tune; do cargo yank --version <broken> "$c"; done`
5. Verify: `curl -s https://crates.io/api/v1/crates/<crate>` should show `latest_unyanked=<new>`, `yanked=[<broken>]`

Done in v0.2.3 (yanked broken 0.2.2 which shipped with the RoPE bug). New `cargo add` users get the fix; existing pinned users get a yank warning on next `cargo update`.
