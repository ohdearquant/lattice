//! Weight storage facade over `lattice-weights`.
//!
//! Every public item keeps the path it had before weight storage moved into its own crate. The
//! re-exports are named rather than globbed so the facade never widens the supported API.

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

pub mod f32_weights {
    #[doc(inline)]
    pub use lattice_weights::f32_weights::{
        BertWeights, CrossEncoderWeights, QwenLayerWeights, QwenWeights, SafetensorsFile,
        SafetensorsIndex, ShardedQwenBacking, ShardedSafetensors, Tensor, Tensor1D, Tensor2D,
        TensorSource, TransformerLayerWeights, contained_shard_path, load_sharded, parse_index,
        resolve_shard,
    };
}

pub mod q3_weights {
    #[doc(inline)]
    pub use lattice_weights::q3_weights::{
        Q3_BLOCK_BYTES, Q3Block, Q3FileHeader, Q3Tensor, dequantize_q3_to_f32, dequantize_row_q3_0,
        gemm_q3_reference, gemv_q3_reference, load_q3_file, quantize_bf16_to_q3,
        quantize_f32_to_q3, quantize_f32_to_q3_mode, quantize_row_q3_0, quantize_tensor_q3_0,
        read_q3_header, save_q3_file, validate_q3_mlp_role,
    };
}

pub mod q4_weights {
    #[doc(inline)]
    pub use lattice_weights::q4_weights::{
        F16LoadError, Q4_BLOCK_BYTES, Q4Block, Q4FileHeader, Q4Tensor, dequantize_q4_to_f32,
        dequantize_row_q4_0, load_f16_tensor_file, load_f16_tensor_file_expecting, load_q4_file,
        quantize_bf16_to_q4, quantize_f32_to_q4, quantize_f64_to_q4, quantize_f64_to_q4_mode,
        quantize_row_q4_0, quantize_tensor_q4_0, read_q4_header, save_q4_file,
        stream_quantize_shard,
    };
}

pub mod q8_weights {
    pub use crate::forward::cpu::q8_matmul::matmul_bt_q8;
    pub(crate) use crate::forward::cpu::q8_matmul::{Q8MatmulScratch, matmul_bt_q8_with_scratch};
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

// Re-export from f32_weights (the primary/base weight types)
#[doc(no_inline)]
pub use self::f32_weights::*;

/// Bench-only, `bench-internals`-gated access to the production
/// scalar f16/bf16 decoders in `lattice_weights::half_bits` (lattice#799): `f16_convert_bench`
/// previously carried its own copied scalar decoder instead of calling the
/// migrated production path, so its Criterion numbers said nothing about
/// this crate's actual conversion call sites. This module keeps the crate's
/// default public API unchanged — outside
/// `bench-internals` builds, the `half_bits` decoders are not part of
/// this crate's public API, the
/// same visibility discipline `forward::metal_qwen35::bench_support` uses
/// for its own bench-only surface.
#[cfg(feature = "bench-internals")]
pub mod bench_support {
    /// Bench-only re-export of [`lattice_weights::half_bits::f16_bits_to_f32`].
    #[inline]
    pub fn f16_bits_to_f32(bits: u16) -> f32 {
        lattice_weights::half_bits::f16_bits_to_f32(bits)
    }

    /// Bench-only re-export of [`lattice_weights::half_bits::f32_to_f16_bits`].
    #[inline]
    pub fn f32_to_f16_bits(v: f32) -> u16 {
        lattice_weights::half_bits::f32_to_f16_bits(v)
    }

    /// Bench-only re-export of [`lattice_weights::half_bits::bf16_bits_to_f32`].
    #[inline]
    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        lattice_weights::half_bits::bf16_bits_to_f32(bits)
    }
}
