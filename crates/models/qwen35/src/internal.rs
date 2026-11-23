//! Version-coupled private-method bridges for the inference facade.

use crate::model::qwen35::Qwen35Model;
use lattice_arch::decoder::driver::DriverTrace;
use lattice_arch::generation::{GenerateConfig, GenerateOutput};
use lattice_inference_core::InferenceError;

/// Retains driver observations without exposing an inherent method.
pub fn generate_traced(
    model: &Qwen35Model,
    prompt: &str,
    config: &GenerateConfig,
) -> Result<(GenerateOutput, DriverTrace), InferenceError> {
    model.generate_with_trace(prompt, config)
}

/// Retains streaming driver observations without exposing an inherent method.
pub fn generate_streaming_traced<F, C>(
    model: &Qwen35Model,
    prompt: &str,
    config: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<(GenerateOutput, DriverTrace), InferenceError>
where
    F: FnMut(&str) -> bool,
    C: FnMut() -> bool,
{
    model.generate_streaming_with_trace(prompt, config, on_token, should_cancel, |_| {})
}

/// [`generate_traced`] for prompt ids the caller tokenized.
pub fn generate_ids_traced(
    model: &Qwen35Model,
    prompt_ids: Vec<u32>,
    config: &GenerateConfig,
) -> Result<(GenerateOutput, DriverTrace), InferenceError> {
    model.generate_ids_via_driver(prompt_ids, config)
}

/// [`generate_streaming_traced`] for prompt ids the caller tokenized.
pub fn generate_streaming_ids_traced<F, C>(
    model: &Qwen35Model,
    prompt_ids: Vec<u32>,
    config: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<(GenerateOutput, DriverTrace), InferenceError>
where
    F: FnMut(&str) -> bool,
    C: FnMut() -> bool,
{
    model.generate_streaming_ids_via_driver(prompt_ids, config, on_token, should_cancel, |_| {})
}

/// [`generate_streaming_with_prefix_cache_traced`] for prompt ids the caller
/// tokenized.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn generate_streaming_with_prefix_cache_ids_traced<F, C>(
    state: &mut crate::forward::metal_qwen35::MetalQwen35State,
    slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
    prompt_ids: Vec<u32>,
    tokenizer: &lattice_tokenizer::bpe::BpeTokenizer,
    config: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<
    (
        crate::forward::metal_qwen35::CachedGenerateOutput,
        DriverTrace,
    ),
    InferenceError,
>
where
    F: FnMut(&str, u32) -> bool,
    C: FnMut() -> bool,
{
    state.generate_streaming_with_prefix_cache_ids_with_trace(
        slot_id,
        prompt_ids,
        tokenizer,
        config,
        on_token,
        should_cancel,
    )
}

/// Retains prefix-cache driver observations at the state owner.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn generate_streaming_with_prefix_cache_traced<F, C>(
    state: &mut crate::forward::metal_qwen35::MetalQwen35State,
    slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
    prompt: &str,
    tokenizer: &lattice_tokenizer::bpe::BpeTokenizer,
    config: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<
    (
        crate::forward::metal_qwen35::CachedGenerateOutput,
        DriverTrace,
    ),
    InferenceError,
>
where
    F: FnMut(&str, u32) -> bool,
    C: FnMut() -> bool,
{
    state.generate_streaming_with_prefix_cache_with_trace(
        slot_id,
        prompt,
        tokenizer,
        config,
        on_token,
        should_cancel,
    )
}

/// Borrow the existing engine configuration without cloning or widening a field.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn engine_config(
    state: &crate::forward::metal_qwen35::MetalQwen35State,
) -> &lattice_arch::qwen35_config::Qwen35Config {
    &state.engine.config
}

/// The existing base rotation seed, copied with the same scalar semantics.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn base_quarot_seed(state: &crate::forward::metal_qwen35::MetalQwen35State) -> Option<u64> {
    state.engine.base_quarot_seed
}
