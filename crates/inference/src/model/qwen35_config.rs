//! Qwen3.5 configuration facade over `lattice-model-config`.
//!
//! Every public item keeps the path it had before the configuration moved into its own crate. The
//! re-exports are named rather than globbed so the facade never widens the supported API. The
//! deprecated generation aliases and the reasoning-budget helpers below are generation code and stay
//! in this crate.

pub(crate) use lattice_model_config::qwen35_config::MAX_VISION_TENSOR_BYTES;
#[cfg(any(test, feature = "test-utils"))]
pub(crate) use lattice_model_config::qwen35_config::compute_layer_types;
#[doc(inline)]
pub use lattice_model_config::qwen35_config::{
    LayerType, MAX_HIDDEN_LAYERS, QWEN_CHAT_IM_END_TOKEN_ID, QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_NO_THINK_PREFIX, QWEN3_THINK_CLOSE_TOKEN_ID, QWEN3_THINK_OPEN_TOKEN_ID, Qwen35Config,
    RopeParams, ValidatedQwen35Config, VisionModelConfig,
};

/// Neutral generation types, moved to [`crate::generation`] under ADR-092.
///
/// These aliases keep every `model::qwen35_config::` path resolving. They are deprecated
/// aliases rather than deprecated re-exports on purpose: `#[deprecated]` on a `pub use` is
/// accepted by the compiler and emits nothing, so it would read as a live deprecation and
/// warn no one. A deprecated type alias does fire.
#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::GenerateConfig`"
)]
pub type GenerateConfig = crate::generation::GenerateConfig;

#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::GenerateOutput`"
)]
pub type GenerateOutput = crate::generation::GenerateOutput;

#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::TokenLogprob`"
)]
pub type TokenLogprob = crate::generation::TokenLogprob;

#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::TopLogprob`"
)]
pub type TopLogprob = crate::generation::TopLogprob;

/// Number of additional tokens a forced reasoning close may wait for a line boundary.
pub(crate) const REASONING_CLOSE_GRACE_TOKENS: usize = 32;

/// Decide whether to force-close the thinking block this step (s1 budget forcing).
///
/// Returns `Some(close_id)` to override the sampled token with `</think>`, else `None`.
/// Once the budget is reached, the close waits for the previous emitted token to
/// contain a newline, and fires without one after
/// [`REASONING_CLOSE_GRACE_TOKENS`] more tokens. All of thinking enabled, a
/// non-zero budget, an open block, and the budget being reached must also hold.
/// Returns `None` immediately if any guard fails so the common disabled path
/// costs a single `Option::None` check per step.
///
/// `enable_thinking: false` makes `reasoning_budget` inert here regardless of its value --
/// no reasoning block is produced in that state (the caller is contracted to have primed
/// the prompt accordingly, see [`crate::generation::GenerateConfig::enable_thinking`]), so there is nothing to
/// force-close. `resolve_reasoning_close_token` (`qwen35/generation.rs`), which resolves
/// `close_id` for this function's `close_id` parameter, applies the identical
/// short-circuit before validating the tokenizer, so the two functions agree about what the
/// combination means instead of one rejecting it and the other silently no-opping.
#[inline]
pub(crate) fn force_close_think(
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    thinking_closed: bool,
    generated_so_far: usize,
    close_id: Option<u32>,
    previous_token_is_boundary: bool,
) -> Option<u32> {
    let budget = reasoning_budget?;
    let close = close_id?;
    if !enable_thinking || thinking_closed || budget == 0 || generated_so_far < budget {
        return None;
    }
    let grace_end = budget.saturating_add(REASONING_CLOSE_GRACE_TOKENS);
    (generated_so_far >= grace_end || previous_token_is_boundary).then_some(close)
}

/// Decode-loop iteration cap.
///
/// When a reasoning budget is active, reasoning tokens get their OWN budget (`rb`) ON TOP
/// of the answer budget (`max_new_tokens`), plus **1** for the forced `</think>` delimiter
/// itself: worst case `rb + max_new_tokens + 1` total tokens. The +1 is necessary because
/// the forced `</think>` is an extra token that is not part of either the reasoning content
/// or the answer — omitting it leaves the answer one token short (off-by-one).
///
/// The close may wait up to [`REASONING_CLOSE_GRACE_TOKENS`] past `rb` for a line
/// boundary. Those grace tokens are part of the same cap and come out of the answer
/// budget; they never extend it.
///
/// Without a budget (`None` or `Some(0)`) the cap is unchanged (`max_new_tokens`), so the
/// disabled path is byte-identical to the pre-budget behaviour (parity-safe).
#[inline]
pub(crate) fn decode_cap(reasoning_budget: Option<usize>, max_new_tokens: usize) -> usize {
    match reasoning_budget {
        // rb reasoning-content tokens + 1 forced </think> delimiter + max_new_tokens answer tokens.
        Some(rb) if rb > 0 => rb.saturating_add(max_new_tokens).saturating_add(1),
        _ => max_new_tokens,
    }
}

#[cfg(test)]
mod tests;
