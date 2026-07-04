# Lattice — Claude Code Instructions

Read `AGENTS.md` first for coding conventions, crate structure, and design principles.

## Development Process

### Measure First, Code Second

Every PR that touches `crates/inference/` or `crates/embed/` must include `make bench-compare` output. No exceptions. A PR without before/after numbers is incomplete regardless of what the code looks like.

```bash
make bench-compare                         # origin/main vs HEAD (~2 min, default)
make bench-compare BASE=main HEAD=pr/x     # explicit refs
scripts/bench-compare.sh --full main       # tight CIs (~15 min)
```

Paste the output in the PR description. If nothing changed, say "bench-compare showed no change (p > 0.05 on all groups)." If something regressed, explain why it's acceptable or revert it.

This process caught a 15% decode throughput regression (157 → 130 tok/s) that had been attributed to "GPU contention noise" for days. The f32 dot_product unrolling that caused it was identified in under 2 minutes via A/B comparison.

### Bench by Group, Not All at Once

The full Criterion suite takes 15-30 min. Never run it all. Filter to the groups your PR touches:

```bash
cargo bench -p lattice-embed --bench simd -- "simd_dot_product"     # one group
cargo bench -p lattice-embed --bench simd -- "int8_raw|normalize"   # multiple groups
cargo bench -p lattice-inference --bench elementwise_cpu_bench      # inference CPU ops
```

For the A/B workflow, pass the same Criterion filter through `make bench-compare`:

```bash
make bench-compare BENCH_GROUPS_INFERENCE="rms_norm|gelu"
make bench-compare BENCH_GROUPS_EMBED="simd_dot_product|int8_raw"
```

Leaving these variables unset keeps the default `elementwise_cpu_bench` and `simd` bench targets.

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

This process closed a 0.77 PPL gap on Qwen3.5-0.8B that had been misdiagnosed as "f32-vs-bf16 precision drift" for days. The actual bug was a RoPE pairing convention mismatch — interleaved `(2i, 2i+1)` vs stride-half `(i, half+i)`. Verified in 5 seconds: stride-half max-diff `8e-6`, interleaved `67.5`. PPL dropped from 16.62 → 15.89 (MLX gold 15.86).

**Quantitative bounds reject hypotheses cheaply.** Before chasing "FP precision drift" or other plausible-sounding causes, check the literature for typical magnitude:

- f16 vs f32 PPL delta: `~0.00x` (llama.cpp community)
- bf16 vs f32 PPL delta: `<0.05` (arxiv:2510.26788)
- Q4 quantization PPL delta: `0.1-0.3` (llama.cpp #406)

If the gap you're investigating exceeds these bounds, the cause is structural (algorithm, layout, convention), not numerical. Reject the precision hypothesis on quantitative grounds and look for a real bug.

**Be skeptical of comments that paraphrase config fields.** A comment that says "X uses field=true" without explaining what the field actually controls in the reference implementation is a footgun. The lattice RoPE comment said "Qwen3.5 uses mrope_interleaved=true" — technically matched config, but `mrope_interleaved` controls multimodal M-RoPE section interleaving (video/image tokens), not 1-D text RoPE pairing. The bug existed for months because nobody verified the comment against HF's `rotate_half` or MLX's `nn.RoPE`.

### Triage Flaky vs Deterministic Before Filing

When a test fails intermittently or fails alongside unrelated work, run the discriminating experiment before writing the issue: ONE test, solo, `--test-threads=1`, idle GPU, exact main SHA. Concurrent GPU load corrupts both timing and numerics, so a "pre-existing failure on main" verified while other GPU work runs is not verified at all.

This split a two-test failure report cleanly: one test failed deterministically at every prompt length under solo idle-GPU conditions (real chunk-boundary accumulation drift, its own issue), while the other passed solo in 65 seconds (a load flake, a different issue). Filing them as one regression would have sent the fix to the wrong place.

### Machine-Wide GPU Test Lock

All Metal-touching tests in this repo serialize through `gpu_test_lock()` (crates/inference/src/forward/metal_qwen35.rs), which holds two locks: an in-process mutex (thread serialization within one test binary) and an exclusive advisory flock on `/tmp/lion-metal-gpu-test.lock` (cross-process serialization, fleet-wide convention). Any harness on this machine that drives the GPU for measurements — other repos' test suites, bench runners, one-off scripts — should acquire the same flock before touching Metal. Concurrent GPU work corrupts both timing and numerics: contended confirmation batches inflated top-k logit margins ~3x and produced false failure reports (#628, #629).

The lock blocks for up to 30 minutes, then panics with an `lsof /tmp/lion-metal-gpu-test.lock` hint rather than hanging silently. If a run appears stuck at test start, another process is holding the GPU; check who with `lsof` before killing anything.

### Regression Tests Must Be Mutation-Sensitive

A regression test that passes with the fix reverted is decoration. Before claiming a test guards a fix: revert the fix (reverse-apply the diff — never `git checkout` over uncommitted work), `touch` the source file so cargo actually rebuilds, and watch the test fail. Then restore the fix and watch it pass.

### Grep for Sibling Invocation Paths

When a harness or guard fix lands in one invocation path, grep the same file for sibling paths that construct the same operation independently: a second subprocess command builder, a second workflow step calling the same script, a reimplementation of a guarded method. Any fix expressible as "add flag X to the call" has an unguarded copy-paste sibling until proven otherwise, and the fix's own description ("mirror the CPU path") is the grep query.

This class recurred three times in one week before becoming a rule — most recently a greedy-decoding sampler flag added to the CPU parity harness path while the Metal-path command builder in the same file went without it, surfacing only on that leg's first live CI run.

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

### Merge Gate (Ocean directive 2026-07-04)

A PR merges only when its branch is **up to date with main** AND **green at its actual head**.
Branch protection enforces this (`required_status_checks.strict = true`), added after #634
merged green-on-a-stale-base and broke main for two hours: main had gained call sites of the
API it changed, and the merged combination was never compiled anywhere before landing. Two
more stale-base PRs then auto-merged onto the red main.

- Never arm auto-merge on a PR whose branch is behind main; `gh pr update-branch <N>` first.
- When main goes red, treat every armed auto-merge as suspect: disarm or hold until main is
  green, then refresh branches and let CI re-run at the true merge state.
- After each merge from a queue of armed PRs, the survivors are out of date again by
  definition; refresh them one at a time rather than arming a stale stack.

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

## Performance Workflow (ADR-058)

- **Every perf PR must include before/after numbers.** No exception. Run `make bench-compare` (or `scripts/bench-compare.sh <base> <head>`) to get an A/B table. Paste the output in the PR description.
- Default to `--quick` (~2 min). Use `--full` only when CIs are too wide to tell.
- After merging to main, `bench-update.yml` auto-updates the `perf-baselines` branch (trend data, not a gate). PRs are gated by `e2e-parity.yml` (greedy token agreement vs HF).
- For local baseline tracking: `make bench-ci` saves a local baseline, `make bench-gate` compares against the `perf-baselines` branch.
- Do not claim "X% faster" without a measurement from this session. Stale numbers from prior sessions are not evidence.

## Crate Ownership

Changes to `inference` affect `embed` and `tune`. Changes to `fann` affect `inference` (via the optional `mixture` feature) and `tune`. Only `fann` and `transport` are leaf crates; `transport` has no internal dependents (`embed` uses it in dev-tests only).

## Publishing

Publish order follows the internal dependency DAG (deps before dependents): fann, transport (leaves) → (wait 30s) → inference (depends on fann via the `mixture` feature) → (wait 30s) → embed, tune. Use `make publish`. Internal path deps' `version =` field must match the current workspace version (bump them in lockstep when bumping `[workspace.package].version`). When a feature adds a new internal dep (e.g. `mixture` made inference depend on fann), the publish order changes — re-derive it from `crates/*/Cargo.toml` path deps, do not assume the old order.

**Shipped-bug recovery (bump-and-yank).** crates.io versions are immutable. When a published release has a correctness bug:

1. Bump workspace + path-dep versions to the next patch
2. Update release notes file (rename if needed); add a "Note on v<broken>" section explaining the yank
3. Tag + GH release + `make publish`
4. `for c in lattice-inference lattice-fann lattice-transport lattice-embed lattice-tune; do cargo yank --version <broken> "$c"; done`
5. Verify: `curl -s https://crates.io/api/v1/crates/<crate>` should show `latest_unyanked=<new>`, `yanked=[<broken>]`

Done in v0.2.3 (yanked broken 0.2.2 which shipped with the RoPE bug). New `cargo add` users get the fix; existing pinned users get a yank warning on next `cargo update`.
