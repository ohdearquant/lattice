//! Multimodal input compatibility exports.

/// A multimodal prompt: merged visual patch embeddings prepended to text tokens.
///
/// Constructed by callers who have merged visual patch embeddings and a tokenized text prompt:
/// ```no_run
/// use lattice_inference::vision::MultimodalInput;
///
/// fn build_multimodal(
///     patch_embeddings: Vec<f32>,
///     raw_patches: usize,
///     visual_tokens: usize,
///     d_model: usize,
///     text_ids: Vec<u32>,
/// ) -> MultimodalInput {
///     MultimodalInput {
///         patch_embeddings,
///         raw_patches,
///         visual_tokens,
///         d_model,
///         text_tokens: text_ids,
///     }
/// }
/// ```
#[doc(inline)]
pub use lattice_model_io::multimodal::MultimodalInput;

#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::vision::multimodal::Qwen35VisionRequest;
