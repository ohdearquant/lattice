//! Qwen3.5 vision weights, preprocessing and encoder composition.

pub mod checkpoint;
pub mod multimodal;
pub mod pooled_embed;
pub mod qwen35_merger;
pub mod qwen35_mrope;
pub mod qwen35_vit;
pub mod qwen35_vit_metal;

/// Synthetic single-image Qwen3-VL vision experiment.
#[cfg(feature = "synthetic-qwen3-vl")]
pub mod qwen3_vl_synthetic;

pub use lattice_model_io::image::VisionError;
pub use multimodal::MultimodalInput;
