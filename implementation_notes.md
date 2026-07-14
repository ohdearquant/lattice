# Implementation Notes

- `crates/inference/src/sampling.rs`: added the canonical `Sampler` NaN/non-finite pre-scan already used by the shared Qwen full-logit sampler, with the issue's discriminating logits and seed as a regression test.
- `crates/inference/src/batch/worker.rs`: finish zero-token-budget requests at admission without enqueueing prefill/decode work, so they emit no token event.
- `crates/inference/src/speculative.rs`: made `generate_with_speculation` return `Result`, reject empty prompts with `InferenceError::Inference("empty prompt")`, and updated every repository caller.
- `docs/generation-entrypoint-matrix.md`: removed the obsolete empty-prompt exception for speculative generation.

Sibling audit: the shared Qwen full-logit sampler already has the NaN scan; generic, Qwen, and batch-prefill generators already short-circuit zero token budgets; the two `BatchWorker` sampling calls cannot receive a zero-budget sequence after admission; and no non-test repository caller of `generate_with_speculation` exists.

Verification: focused regressions were observed failing before the fixes and passing after them; `cargo test -p lattice-inference`, `cargo clippy -p lattice-inference -- -D warnings`, `cargo fmt --all --check`, `deno fmt --check docs/generation-entrypoint-matrix.md`, and `make lint-docs` pass with `CARGO_TARGET_DIR=/private/tmp/cargo-target-inf-decode` for Cargo builds.

Performance: the canonical sampler now performs the same full-logit pre-scan as the live Qwen path. `make bench-compare` is required before merge; no GPU or benchmark command was run in this leg.

domain_utility: medium — the composed Rust API/performance guidance reinforced boundary validation and benchmark discipline; repository code, issue evidence, and regression behavior determined the implementation.
