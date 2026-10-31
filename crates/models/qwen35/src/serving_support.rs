//! Version-coupled Qwen3.5 serving observations and rendering operations.

use crate::vision::{
    VisionError,
    checkpoint::{Qwen35VisionWeights, VisualMergerWeights},
    qwen35_vit::GridThw,
};
use lattice_arch::qwen35_config::VisionModelConfig;
use lattice_inference_core::InferenceError;
use std::path::Path;

/// The model configuration a Metal state's engine was built with, by value.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn metal_engine_config(
    state: &crate::forward::metal_qwen35::MetalQwen35State,
) -> lattice_arch::qwen35_config::Qwen35Config {
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
