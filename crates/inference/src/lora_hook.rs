//! LoRA adapter hooks and model-specific projection geometry.
//!
//! Named exports retain the existing public surface.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute`,
//! `lattice-model-bert` and `lattice-model-qwen35` so that code written against the pre-split
//! `lattice-inference` keeps compiling. These low-level re-exports are not a stable API and will be
//! removed at 1.0; import from `lattice-compute`, `lattice-model-bert` and `lattice-model-qwen35`
//! directly.

#[doc(inline)]
pub use lattice_compute::lora_hook::{
    GDN_LORA_MODULES, GDN_LORA_MODULES_SERVABLE, LoraHook, LoraProjectionShape, LoraTargetGeometry,
    NoopLoraHook, gdn_lora_module_is_servable,
};

#[cfg(feature = "bert")]
#[doc(inline)]
pub use lattice_model_bert::lora::BertLoraGeometry;

#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::lora_hook::{Qwen35LoraGeometry, qwen35_projection_shape};
