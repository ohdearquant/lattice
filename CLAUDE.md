# Lattice — Claude Code Instructions

Read `AGENTS.md` first. It contains all coding conventions, crate structure, design principles, and workflow rules. Everything below is additional context for Claude Code sessions.

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

## Crate Ownership

Changes to `inference` affect `embed` and `tune`. Changes to `fann` affect `tune`. Changes to `transport`, `fann`, or `inference` alone are safe — they're leaf crates.

## Publishing

Leaf crates publish first: inference → fann → transport → (wait 30s) → embed → (wait 30s) → tune. Use `make publish`. Internal path deps must have `version = "0.1.0"`.
