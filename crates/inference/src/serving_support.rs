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
use crate::model::Qwen35Model;
use crate::model::gemma4_model::Gemma4Model;
use crate::model::qwen35_config::VisionModelConfig;
use crate::vision::VisionError;
use crate::vision::checkpoint::{Qwen35VisionWeights, VisualMergerWeights};
use crate::vision::qwen35_vit::GridThw;

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
pub fn qwen_generate_traced(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, TraceObservation), InferenceError> {
    model
        .generate_with_trace(prompt, gen_cfg)
        .map(|(output, trace)| (output, trace.into()))
}

/// `Qwen35Model::generate_streaming_with_cancel` with the driver's counters
/// kept, run with a no-op raw-event observer as that entry runs it.
///
/// # Errors
/// Propagates tokenizer and generation failures.
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
    model
        .generate_streaming_with_trace(prompt, gen_cfg, on_token, should_cancel, |_| {})
        .map(|(output, trace)| (output, trace.into()))
}

/// Gemma generation over prepared prompt ids with the driver's counters kept.
///
/// # Errors
/// Propagates generation failures.
pub fn gemma_generate_traced(
    model: &Gemma4Model,
    prompt_ids: &[u32],
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, TraceObservation), InferenceError> {
    model
        .generate_with_trace(prompt_ids, gen_cfg)
        .map(|(output, trace)| (output, trace.into()))
}

/// Gemma streaming generation over prepared prompt ids with the driver's
/// counters kept.
///
/// # Errors
/// Propagates generation failures.
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
    model
        .generate_streaming_via_driver(prompt_ids, gen_cfg, on_token, should_cancel)
        .map(|(output, trace)| (output, trace.into()))
}

/// Metal prefix-cache streaming generation with the driver's counters kept.
///
/// # Errors
/// Propagates tokenizer, context and generation failures.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
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
    state
        .generate_streaming_with_prefix_cache_with_trace(
            slot_id,
            prompt,
            tokenizer,
            gen_cfg,
            on_token,
            should_cancel,
        )
        .map(|(output, trace)| (output, trace.into()))
}

/// The model configuration a Metal state's engine was built with, by value.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn metal_engine_config(
    state: &crate::forward::metal_qwen35::MetalQwen35State,
) -> crate::model::qwen35_config::Qwen35Config {
    state.engine.config.clone()
}

/// ChatML rendering of `(role, content)` turns followed by the open
/// generation turn, as the canonical text renderer emits it.
pub fn format_chat_template_parts<'a>(
    messages: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    crate::forward::metal_qwen35::format_chat_template_parts(messages)
}

/// The role name a ChatML turn opener renders for `role`.
pub fn chat_role_name(role: &crate::forward::metal_qwen35::ChatRole) -> &str {
    role.as_str()
}

/// Append a ChatML turn opener for `role`.
pub fn push_chat_turn_open(prompt: &mut String, role: &str) {
    crate::forward::metal_qwen35::push_chat_turn_open(prompt, role);
}

/// Append a ChatML turn close.
pub fn push_chat_turn_close(prompt: &mut String) {
    crate::forward::metal_qwen35::push_chat_turn_close(prompt);
}

/// Append the open assistant turn left unclosed for generation.
pub fn push_chat_generation_open(prompt: &mut String) {
    crate::forward::metal_qwen35::push_chat_generation_open(prompt);
}

/// Gemma stop-token ids: the text config's EOS plus any top-level EOS ids in
/// the checkpoint's generation and model configs.
pub fn resolve_stop_token_ids(model_dir: &Path, text_config_eos: u32) -> Vec<u32> {
    crate::model::gemma4_config::resolve_stop_token_ids(model_dir, text_config_eos)
}

/// Load Qwen3.5 vision weights, polling `should_cancel`. `Ok(None)` means
/// cancelled.
///
/// # Errors
/// Propagates checkpoint and validation failures.
pub fn load_qwen35_vision_weights_with_cancel(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Qwen35VisionWeights>, InferenceError> {
    crate::vision::checkpoint::load_qwen35_vision_weights_with_cancel(
        model_dir,
        vision_cfg,
        should_cancel,
    )
}

/// Output of the Metal ViT forward.
#[derive(Debug, Clone)]
pub struct VitObservation {
    /// Pre-merger hidden states.
    pub hidden_states: Vec<f32>,
    /// Metal GEMM dispatches the forward issued.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub metal_dispatches: usize,
    /// GEMM calls the forward made.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub gemm_calls: usize,
}

/// Qwen3.5 ViT forward on Metal, polling `should_cancel`. `Ok(None)` means
/// cancelled.
///
/// # Errors
/// Propagates vision failures, including a build without Metal.
pub fn qwen35_vit_forward_metal_with_cancel(
    weights: &Qwen35VisionWeights,
    cfg: &VisionModelConfig,
    pixel_values: &[f32],
    grid: GridThw,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<VitObservation>, VisionError> {
    Ok(
        crate::vision::qwen35_vit_metal::qwen35_vit_forward_metal_with_cancel(
            weights,
            cfg,
            pixel_values,
            grid,
            should_cancel,
        )?
        .map(|output| VitObservation {
            hidden_states: output.hidden_states,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            metal_dispatches: output.metal_dispatches,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            gemm_calls: output.gemm_calls,
        }),
    )
}

/// Qwen3.5 visual merger forward, polling `should_cancel`. `Ok(None)` means
/// cancelled.
///
/// # Errors
/// Propagates vision failures.
pub fn qwen35_merger_forward_with_cancel(
    weights: &VisualMergerWeights,
    cfg: &VisionModelConfig,
    pre_merger_hidden: &[f32],
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Vec<f32>>, VisionError> {
    crate::vision::qwen35_merger::qwen35_merger_forward_with_cancel(
        weights,
        cfg,
        pre_merger_hidden,
        should_cancel,
    )
}

/// The reasoning budget a configuration actually applies: none unless
/// thinking is enabled.
pub fn effective_reasoning_budget(gen_cfg: &GenerateConfig) -> Option<usize> {
    gen_cfg.effective_reasoning_budget()
}

/// A field-for-field copy of `router` for refitting.
#[cfg(feature = "router-learning")]
pub fn refit_snapshot(router: &crate::mixture::AdapterRouter) -> crate::mixture::AdapterRouter {
    router.refit_snapshot()
}

/// A refit source for the serving router's recorded artifact and policy.
///
/// # Errors
/// Refuses an ineligible source: learning disabled, an artifact that is not
/// learner eligible, or a recorded policy or lineage the refit cannot use.
#[cfg(feature = "router-learning")]
pub fn refit_source_from_serving(
    artifact: crate::router_state::RouterArtifact,
    router: crate::mixture::AdapterRouter,
    learning_enabled: bool,
    exploration_scale: Option<f64>,
) -> Result<crate::router_refit::RefitSource, crate::router_refit::RefitError> {
    crate::router_refit::RefitSource::from_serving(
        artifact,
        router,
        learning_enabled,
        exploration_scale,
    )
}
