//! Unsupported, version-coupled interface for `lattice-serve` (ADR-103
//! Amendment 4).
//!
//! Not public API and not covered by semver: `lattice-serve` depends on this
//! crate with an exact version and the two release together. The item list is
//! closed; an item not here stays crate-private, and adding one is an
//! amendment. Each item is a free wrapper over the crate-private original and
//! changes nothing about it.

use std::path::Path;

use crate::decoder::driver::DriverTrace;
use crate::error::InferenceError;
use crate::generation::{GenerateConfig, GenerateOutput};
#[cfg(feature = "qwen35")]
use crate::model::Qwen35Model;
#[cfg(feature = "gemma4")]
use crate::model::gemma4_model::Gemma4Model;

/// The shared driver's counters for one request: sampling steps opened and
/// consumed. All zero when the request never reached the driver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TraceObservation {
    /// Sampling steps the driver opened.
    pub opened: usize,
    /// Sampling steps the driver consumed.
    pub consumed: usize,
}

impl From<DriverTrace> for TraceObservation {
    fn from(trace: DriverTrace) -> Self {
        Self {
            opened: trace.opened,
            consumed: trace.consumed,
        }
    }
}

/// `Qwen35Model::generate` with the driver's counters kept.
///
/// # Errors
/// Propagates tokenizer and generation failures.
#[cfg(feature = "qwen35")]
pub fn qwen_generate_traced(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, TraceObservation), InferenceError> {
    lattice_model_qwen35::__internal::generate_traced(model, prompt, gen_cfg)
        .map(|(output, trace)| (output, trace.into()))
}

/// `Qwen35Model::generate_streaming_with_cancel` with the driver's counters
/// kept, run with a no-op raw-event observer as that entry runs it.
///
/// # Errors
/// Propagates tokenizer and generation failures.
#[cfg(feature = "qwen35")]
pub fn qwen_generate_streaming_traced<F, C>(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<(GenerateOutput, TraceObservation), InferenceError>
where
    F: FnMut(&str) -> bool,
    C: FnMut() -> bool,
{
    lattice_model_qwen35::__internal::generate_streaming_traced(
        model,
        prompt,
        gen_cfg,
        on_token,
        should_cancel,
    )
    .map(|(output, trace)| (output, trace.into()))
}

/// Gemma generation over prepared prompt ids with the driver's counters kept.
///
/// # Errors
/// Propagates generation failures.
#[cfg(feature = "gemma4")]
pub fn gemma_generate_traced(
    model: &Gemma4Model,
    prompt_ids: &[u32],
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, TraceObservation), InferenceError> {
    lattice_model_gemma4::__internal::generate_traced(model, prompt_ids, gen_cfg)
        .map(|(output, trace)| (output, trace.into()))
}

/// Gemma streaming generation over prepared prompt ids with the driver's
/// counters kept.
///
/// # Errors
/// Propagates generation failures.
#[cfg(feature = "gemma4")]
pub fn gemma_generate_streaming_traced<F, C>(
    model: &Gemma4Model,
    prompt_ids: &[u32],
    gen_cfg: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<(GenerateOutput, TraceObservation), InferenceError>
where
    F: FnMut(&str) -> bool,
    C: FnMut() -> bool,
{
    lattice_model_gemma4::__internal::generate_streaming_traced(
        model,
        prompt_ids,
        gen_cfg,
        on_token,
        should_cancel,
    )
    .map(|(output, trace)| (output, trace.into()))
}

/// Metal prefix-cache streaming generation with the driver's counters kept.
///
/// # Errors
/// Propagates tokenizer, context and generation failures.
#[cfg(all(feature = "qwen35", target_os = "macos", feature = "metal-gpu"))]
pub fn metal_generate_streaming_with_prefix_cache_traced<F, C>(
    state: &mut crate::forward::metal_qwen35::MetalQwen35State,
    slot_id: crate::kv_cache::CrossTurnSlotId,
    prompt: &str,
    tokenizer: &crate::tokenizer::bpe::BpeTokenizer,
    gen_cfg: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<
    (
        crate::forward::metal_qwen35::CachedGenerateOutput,
        TraceObservation,
    ),
    InferenceError,
>
where
    F: FnMut(&str, u32) -> bool,
    C: FnMut() -> bool,
{
    lattice_model_qwen35::__internal::generate_streaming_with_prefix_cache_traced(
        state,
        slot_id,
        prompt,
        tokenizer,
        gen_cfg,
        on_token,
        should_cancel,
    )
    .map(|(output, trace)| (output, trace.into()))
}

/// Gemma stop-token ids: the text config's EOS plus any top-level EOS ids in
/// the checkpoint's generation and model configs.
pub fn resolve_stop_token_ids(model_dir: &Path, text_config_eos: u32) -> Vec<u32> {
    crate::model::gemma4_config::resolve_stop_token_ids(model_dir, text_config_eos)
}

/// The reasoning budget a configuration actually applies: none unless
/// thinking is enabled.
pub fn effective_reasoning_budget(gen_cfg: &GenerateConfig) -> Option<usize> {
    gen_cfg.effective_reasoning_budget()
}

/// The router refit helpers: a field-for-field copy of a router for refitting,
/// and a refit source for the serving router's recorded artifact and policy.
#[cfg(feature = "router-learning")]
#[doc(inline)]
pub use lattice_router::serving_support::{refit_snapshot, refit_source_from_serving};

#[cfg(all(feature = "qwen35", target_os = "macos", feature = "metal-gpu"))]
#[doc(inline)]
pub use lattice_model_qwen35::serving_support::metal_engine_config;
#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::serving_support::{
    VitObservation, chat_role_name, format_chat_template_parts,
    load_qwen35_vision_weights_with_cancel, push_chat_generation_open, push_chat_turn_close,
    push_chat_turn_open, qwen35_merger_forward_with_cancel, qwen35_vit_forward_metal_with_cancel,
};
