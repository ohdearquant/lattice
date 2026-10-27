//! Qwen3 embedding configuration and optional family implementation.

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
