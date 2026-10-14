//! EmbeddingGemma 2 text encoder, f32 forward pass on Metal.
//!
//! An explicit backend: it requires macOS and the `metal-gpu` feature, and the CPU path in
//! [`crate::model::embeddinggemma2`] stays the default. Weights are uploaded once as f32 (the
//! checkpoint is bf16 on disk and is widened when the model loads) and activations are f32
//! throughout; f16 activations overflow on this model.
//!
//! Build a state with [`MetalEmbeddingGemma2State::new`] from a loaded
//! [`EmbeddingGemma2Model`](crate::model::embeddinggemma2::EmbeddingGemma2Model) and run it with
//! `token_states_metal` or `encode_ids_at_widths_metal` on that model.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod state;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use state::MetalEmbeddingGemma2State;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use state::VisionTimings;

/// EmbeddingGemma 2 Metal state; unavailable without macOS and `metal-gpu`.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalEmbeddingGemma2State;

/// Per-segment wall time of one vision-tower pass.
///
/// Not a stable API: `#[doc(hidden)]` is a convention for the `dump_embeddinggemma2_vision`
/// example, which reports the segments of a measured run.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisionTimings {
    /// Sum of the attention command buffers' GPU spans.
    pub attention_gpu_us: u64,
    /// Sum of the MLP command buffers' GPU spans.
    pub mlp_gpu_us: u64,
    /// Host time of the per-layer work before attention.
    pub cpu_pre_attention_us: u64,
    /// Host time of the per-layer work after attention.
    pub cpu_post_attention_us: u64,
    /// Wall time of the whole tower pass, handoffs included.
    pub tower_us: u64,
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
impl MetalEmbeddingGemma2State {
    /// Upload a loaded model's text tower to the GPU.
    ///
    /// # Errors
    /// Returns an availability error on this build.
    pub fn new(
        _model: &crate::model::embeddinggemma2::EmbeddingGemma2Model,
    ) -> Result<Self, crate::InferenceError> {
        Err(unavailable())
    }

    pub(crate) fn check_model(
        &self,
        _model: &crate::model::embeddinggemma2::EmbeddingGemma2Model,
    ) -> Result<(), crate::InferenceError> {
        Err(unavailable())
    }

    pub(crate) fn forward(
        &mut self,
        _embeddings: &[f32],
        _seq_len: usize,
    ) -> Result<Vec<f32>, crate::InferenceError> {
        Err(unavailable())
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn unavailable() -> crate::InferenceError {
    crate::InferenceError::Inference(
        "EmbeddingGemma 2 Metal forward requires macOS and the metal-gpu feature".into(),
    )
}
