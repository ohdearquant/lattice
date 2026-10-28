//! Weight storage paths for the Qwen3.5 forward implementations.
//!
//! Each module joins the storage types from `lattice-weights` and the kernels from
//! `lattice-compute` with the Qwen3.5 conversion and loading code, under the paths
//! `lattice_inference::weights` exports.

pub mod f16_weights {
    pub use crate::model::qwen35::f16_loading::load_f16_weights;
    #[doc(inline)]
    pub use lattice_compute::forward::cpu::f16_matmul::matmul_bt_f16;
    #[doc(inline)]
    pub use lattice_weights::f16_weights::{
        F16, F16AttentionWeights, F16CommonLayerWeights, F16FeedForwardWeights,
        F16FullAttentionLayerWeights, F16GatedDeltaNetWeights, F16ModelWeights, F16MoeLayerWeights,
        F16MoeRouter, F16RoutedExperts, F16SharedExpert, f16_to_f32_slice, f32_to_f16_slice,
    };
}

pub mod q8_weights {
    pub use crate::forward::q8_matmul::matmul_bt_q8;
    pub(crate) use crate::forward::q8_matmul::{Q8MatmulScratch, matmul_bt_q8_with_scratch};
    pub use crate::model::qwen35::q8_conversion::{
        Q8AttentionWeights, Q8CommonLayerWeights, Q8FullAttentionLayerWeights,
        Q8GatedDeltaNetWeights, Q8ModelWeights, memory_report, quantize_gdn_weights,
    };
    pub(crate) use crate::model::qwen35::q8_conversion::{
        quantize_model_weights, validate_cfg_len, validate_gdn_shapes,
    };
    #[doc(inline)]
    pub use lattice_weights::q8_weights::{Q8Matrix, quantize_matrix};
}
