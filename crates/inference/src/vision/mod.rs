//! Vision encoder module. Two vision paths live here: the real Qwen3.5 path
//! and the PaddleOCR-VL path. PaddleOCR-VL has
//! its own weights and config and uses the shared arithmetic helpers in
//! `lattice_compute::vision`.
//!
//! ## The real path: Qwen3.5-0.8B (ADR-069, S1-S3b shipped)
//!
//! [`checkpoint::load_qwen35_vision_weights`](crate::vision::checkpoint::load_qwen35_vision_weights) loads the real
//! `model.visual.*` tensors (S1/S2); [`qwen35_vit::qwen35_vit_forward`](crate::vision::qwen35_vit::qwen35_vit_forward) runs
//! the depth-12/hidden-768 forward pass over them, producing pre-merger
//! hidden states `[num_patches, hidden_size]` (S3a — CPU reference, gated at
//! cosine > 0.999 vs a committed HF differential golden in
//! `tests/vision_s3_vit_forward_test.rs`, wired as a required CI job).
//! [`qwen35_vit_metal::qwen35_vit_forward_metal`](crate::vision::qwen35_vit_metal::qwen35_vit_forward_metal) is the Metal port (S3b),
//! gated against the S3a CPU reference under the machine GPU lock in
//! `tests/vision_s3b_vit_metal_gate_test.rs`.
//! [`qwen35_merger::qwen35_merger_forward`](crate::vision::qwen35_merger::qwen35_merger_forward) runs the spatial-merge +
//! projection MLP over the pre-merger hidden states, producing the
//! post-merger visual embeddings `[num_patches / spatial_merge_size^2,
//! out_hidden_size]` that the decoder consumes (S4 — gated at cosine >
//! 0.999 vs a committed HF differential golden in
//! `tests/vision_s4_merger_test.rs`). S5 (decoder injection + M-RoPE) is
//! unimplemented — **this is the path future vision work should build on.**
//!
//! ```text
//! image bytes → preprocess_qwen35_image → qwen35_vit_forward
//!   → [num_patches, hidden_size] pre-merger hidden states
//!   → qwen35_merger_forward
//!   → [num_patches / spatial_merge_size^2, out_hidden_size] post-merger visual embeddings
//!   (S5 not yet implemented: no decoder injection)
//! ```

#[cfg(feature = "qwen35")]
pub mod checkpoint;
pub mod multimodal;
#[cfg(feature = "paddleocr-vl")]
pub mod paddleocr_preprocess;
#[cfg(feature = "paddleocr-vl")]
pub mod paddleocr_vit;
#[cfg(feature = "qwen35")]
pub mod pooled_embed;
#[cfg(feature = "qwen35")]
pub mod qwen35_merger;
#[cfg(feature = "qwen35")]
pub mod qwen35_mrope;
pub mod qwen35_vit;
#[cfg(feature = "qwen35")]
pub mod qwen35_vit_metal;

pub use multimodal::MultimodalInput;
#[cfg(feature = "qwen35")]
pub use pooled_embed::{embed_image_from_bytes_f16, embed_image_from_bytes_f16_metal};

#[doc(inline)]
pub use lattice_model_io::image::VisionError;
