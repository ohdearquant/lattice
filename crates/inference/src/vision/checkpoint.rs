//! Qwen3.5 vision checkpoint items, owned by the Qwen3.5 family.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::vision::checkpoint::{
    Qwen35VisionWeights, VisualBlockWeights, VisualMergerWeights, load_qwen35_vision_weights,
    load_qwen35_vision_weights_from_safetensors, open_qwen35_single_decoder_safetensors,
    resolve_qwen35_single_decoder_safetensors, validate_qwen35_vision_weight_inventory,
};
