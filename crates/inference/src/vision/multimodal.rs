//! Multimodal input compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-io` and
//! `lattice-model-qwen35` so that code written against the pre-split `lattice-inference` keeps
//! compiling. These low-level re-exports are not a stable API and will be removed at 1.0; import
//! from `lattice-model-io` and `lattice-model-qwen35` directly.

/// A multimodal prompt: merged visual patch embeddings prepended to text tokens.
///
/// Constructed by callers who have a `VisionOutput` and a tokenized text prompt:
/// ```no_run
/// use lattice_inference::vision::{VisionOutput, MultimodalInput};
///
/// fn build_multimodal(vision_out: VisionOutput, text_ids: Vec<u32>) -> MultimodalInput {
///     MultimodalInput {
///         patch_embeddings: vision_out.patch_embeddings,
///         raw_patches: vision_out.raw_patches,
///         visual_tokens: vision_out.visual_tokens,
///         d_model: vision_out.d_model,
///         text_tokens: text_ids,
///     }
/// }
/// ```
#[doc(inline)]
pub use lattice_model_io::multimodal::MultimodalInput;

#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::vision::multimodal::Qwen35VisionRequest;
