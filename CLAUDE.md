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

Quick mode (`--quick`) is sufficient for direction + magnitude. Full mode only when you need tight CIs for a PR description or ADR evidence.

### Regression Gate (ADR-058)

PRs touching CPU kernel paths trigger `bench-regression.yml` in CI. It runs on both `x86_64-linux` (AVX2) and `aarch64-linux` (NEON) against baselines stored on the orphan `perf-baselines` branch.

- **>7% regression** (95% CI lower bound): blocks merge
- **3-7%**: warning comment, doesn't block
- **Override**: PR label `bench-allow-regression` with rationale

The `perf-baselines` branch auto-updates on every push to main via `bench-update.yml`.

## Session Protocol

- Run `cargo clippy --workspace -- -D warnings` before reporting any task complete.
- Use `make ci` for full validation (fmt + clippy + doc lint + test + build).
- Feature branches + PRs for all changes. Never push directly to main.
- Conventional commits with crate scope: `feat(inference): add Qwen3.5 MoE support`.

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
- After merging to main, `bench-update.yml` auto-updates the `perf-baselines` branch. PRs touching CPU paths are gated by `bench-regression.yml` (>7% CI-lower = fail).
- For local baseline tracking: `make bench-ci` saves a local baseline, `make bench-gate` compares against the `perf-baselines` branch.
- Do not claim "X% faster" without a measurement from this session. Stale numbers from prior sessions are not evidence.

## Crate Ownership

Changes to `inference` affect `embed` and `tune`. Changes to `fann` affect `tune`. Changes to `transport`, `fann`, or `inference` alone are safe — they're leaf crates.

## Publishing

Leaf crates publish first: inference → fann → transport → (wait 30s) → embed → (wait 30s) → tune. Use `make publish`. Internal path deps must have `version = "0.1.0"`.
