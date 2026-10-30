//! Qwen3 embedding configuration and optional family implementation.
//!
//! **Compatibility re-exports.** The items this module re-exports from `lattice-arch` and
//! `lattice-model-qwen3-embedding` keep code written against the pre-split
//! `lattice-inference` compiling. Those low-level re-exports are not a stable API and will be
//! removed at 1.0; import them from the owning crate directly.

#[doc(inline)]
pub use lattice_arch::qwen_config::QwenConfig;
#[cfg(feature = "qwen3")]
#[doc(inline)]
pub use lattice_model_qwen3_embedding::{
    LayerTimings, ModelInferenceConfig, ProfileTimings, QwenModel,
};

/// Qwen3 tensor assembly from model-neutral storage reads.
#[cfg(feature = "qwen3")]
pub mod weight_loading {
    #[doc(inline)]
    pub use lattice_model_qwen3_embedding::weight_loading::{
        OwnedQwenWeights, load_owned_weights, load_weights,
    };
}
