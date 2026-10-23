//! Cross-path stop-token contract sweep (#613).
//!
//! Before this fix, the crate had two incompatible contracts for what happens
//! to the token that triggers a stop condition (`eos_token_id` or
//! `stop_token_ids`):
//!
//! - **INCLUDE**: the stop token is pushed into `token_ids`/`text` and then
//!   generation halts (the Metal MTP/self-speculative decode loops in
//!   [`crate::forward::metal_qwen35`]).
//! - **EXCLUDE**: the stop token is detected *before* it is pushed, so
//!   `token_ids`/`text` never contain it (the Qwen3.5 CPU family in
//!   [`crate::model::qwen35::generation`], [`crate::forward::cpu_f16`],
//!   [`crate::forward::cpu_q8`], [`crate::forward::neon_forward`], and the
//!   Metal plain/streaming/prefix-cache paths in
//!   [`crate::forward::metal_qwen35`]).
//!
//! **Chosen contract: EXCLUDE.** It was already the majority behaviour (9 of
//! 11 surveyed call sites — the current entry-point manifest below, minus the
//! two that needed a production change), it matches the OpenAI/HF `stop`
//! semantics most callers expect (the stop marker is a control signal, not
//! content), and it
//! composes correctly with string-level `stop_strings` truncation, which by
//! construction already drops the matched suffix. [`crate::generation::GenerateOutput`]
//! records this explicitly. The MTP and self-spec loops were the two sites
//! that needed a production change; see `crates/inference/src/forward/metal_qwen35.rs`.
//!
//! ## Entry-point manifest
//!
//! | # | Entry point | Coverage |
//! |---|---|---|
//! | 1 | `Qwen35Model::generate` (f32) | this file |
//! | 2 | `Qwen35Model::generate_streaming` (f32) | this file |
//! | 3 | `forward::cpu_f16::generate_f16` | this file |
//! | 4 | `forward::cpu_q8::generate_q8` | this file |
//! | 5 | `forward::neon_forward::generate_q8_neon` | this file |
//! | 6 | Metal `generate` (plain, no MTP/self-spec) | `forward::metal_qwen35::inner::tests::stop_token_sweep` |
//! | 7 | Metal `generate` with `LATTICE_MTP=1` | `forward::metal_qwen35::inner::tests::stop_token_sweep` + `mtp_greedy_round_tests` |
//! | 8 | Metal `generate` with `LATTICE_SELF_SPEC=1` | `forward::metal_qwen35::inner::tests::stop_token_sweep` + `self_spec_eos_tests` |
//! | 9 | Metal `generate_streaming` | `forward::metal_qwen35::inner::tests::stop_token_sweep` |
//! | 10 | Metal `generate_streaming_with_prefix_cache` | `forward::metal_qwen35::inner::tests::stop_token_sweep` |
//! | 11 | Metal `generate_multimodal` | `forward::metal_qwen35::inner::tests::stop_token_sweep` |
//!
//! Entry 11 (`generate_multimodal`) has its own independent sampling loop —
//! it does not call `generate`/`generate_streaming` internally — and was
//! missing from this manifest until PR #632 added it as
//! an unguarded sibling-invocation path (#613's own warning pattern). Its
//! loop already used the EXCLUDE contract (checks `is_stop` before pushing
//! at every termination point: first-token sample, KV-full break, and the
//! autoregressive decode loop), so no production change was needed here —
//! only manifest + test coverage.
//!
//! The Metal-family tests (6-11) live inside `metal_qwen35.rs`'s existing
//! `mod tests` rather than here because its GPU test fixtures
//! (`tiny_hybrid_fixture`, `minimal_bpe_tokenizer`, `with_self_spec_env`) are
//! private to that module and already proven correct by dozens of existing
//! tests; duplicating or relocating them would add risk (see the module's
//! `MetalQwen35State` construction path, which needs a live `MTLDevice`) for
//! no contract-coverage benefit. Grep `stop_token_contract` or `#613` in
//! `metal_qwen35.rs` to find them.
//!
//! ## What "sweep" means here
//!
//! Every test below builds the smallest possible zero-layer model (no
//! attention/FFN, just `embed -> final_norm -> lm_head`) with all-zero
//! weights, so greedy sampling deterministically always picks token 0.
//! `eos_token_id` is set to a different value (5) and `stop_token_ids =
//! [0]`, so hitting the stop condition on the very first generated token is
//! unambiguous and reproducible. Each test then asserts the EXCLUDE contract:
//! `generated_tokens == 0`, `token_ids` and `text` empty, `stopped == true`,
//! `stop_reason == Some(StopReason::Eos)`.
//!
//! Most of the paths tested here (1-5) were already EXCLUDE before this
//! session's fix — they regression-lock behaviour that was already correct.
//! The two paths whose logic changed (MTP, self-spec) are unit-tested at the
//! decision-function level in `metal_qwen35`'s `mtp_greedy_round_tests` and
//! `self_spec_eos_tests` modules, where the same all-zero-weights style
//! before/after vectors were hand-verified to flip from INCLUDE to EXCLUDE
//! when the production fix is reverted.

#[cfg(test)]
mod tests;
