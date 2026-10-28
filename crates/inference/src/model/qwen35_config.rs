//! Qwen3.5 configuration facade over `lattice-arch`.
//!
//! Every public item keeps the path it had before the configuration moved into its own crate. The
//! re-exports are named rather than globbed so the facade never widens the supported API. The
//! deprecated generation aliases below stay in this crate. The reasoning-budget helpers moved to
//! `lattice-arch` with the generation code.
//!
//! **Compatibility re-exports.** The items this module re-exports from `lattice-arch` keep code
//! written against the pre-split `lattice-inference` compiling. Those low-level re-exports are not
//! a stable API and will be removed at 1.0; import them from the owning crate directly.

#[doc(inline)]
pub use lattice_arch::qwen35_config::{
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
