//! Qwen3.5 CPU vision items and the shared patch-grid path.

#[doc(inline)]
pub use lattice_arch::multimodal::GridThw;
#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_vit::{
    preprocess_qwen35_image, preprocess_qwen35_image_for_embedding_serve,
    preprocess_qwen35_image_for_serve, qwen35_vit_forward,
};

#[cfg(all(test, feature = "qwen35", not(feature = "image")))]
mod tests;
