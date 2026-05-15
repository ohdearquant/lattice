<!--
Attribution: if you are an AI agent authoring this PR description via someone's
gh CLI session, START the body with a blockquote attribution line per AGENTS.md
§AI-Assisted Contribution Policy. Example:

> _PR description authored by Claude (Anthropic agent) on behalf of @<handle>._

Human-authored PRs: delete this comment block.
-->

## Summary

<!-- 1-3 bullets: what changed and why. Match the actual diff. -->

## Test plan

<!-- Concrete commands and outcomes. Tick boxes as they pass.
     For performance-sensitive changes, include cargo bench output. -->

- [ ] `cargo test --workspace`
- [ ] `cargo clippy --workspace -- -D warnings`
- [ ] `cargo fmt --all -- --check`

## ADR

<!-- If this PR introduces a significant feature or architectural change,
     link to the ADR (docs/adr/ADR-NNN-*.md). Update docs/adr/INDEX.md in
     the same PR. See docs/_templates/ADR_TEMPLATE.md.
     For small fixes, write "n/a — small fix, no ADR." -->

## AI-assisted contribution checklist

<!-- Required if any part of this diff was AI-generated (code, tests, docs,
     or this PR body). Delete the section if fully human-authored. -->

- [ ] Every claim in this PR description matches the actual diff
- [ ] SIMD intrinsics, if any, were manually verified — not just reviewed by an AI
- [ ] Any agent-authored comment / PR body / issue starts with the attribution line from AGENTS.md
- [ ] `cargo test` output included for any behavior-changing code; `cargo bench` output for perf-sensitive changes

## Out of scope

<!-- What this PR intentionally does NOT do, so reviewers can stop asking.
     If part of a multi-PR series, note which step this is. -->
