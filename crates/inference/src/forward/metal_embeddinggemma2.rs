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

/// EmbeddingGemma 2 Metal state; unavailable without macOS and `metal-gpu`.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalEmbeddingGemma2State;

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
