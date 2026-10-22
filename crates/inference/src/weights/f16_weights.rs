//! Mixed-precision f16 weight storage and matmul utilities for Qwen3.5-2B.
//!
//! The forward pass in this model is primarily limited by memory bandwidth, not raw FLOPs.
//! Storing large weight matrices as IEEE-754 half precision cuts weight traffic in half while
//! keeping activations, accumulators, norms, and outputs in `f32`.
//!
//! The core pattern implemented here is:
//! - store large matrices as packed `u16` half values,
//! - widen weight elements to `f32` on demand during matmul,
//! - accumulate in `f32` for numerical stability,
//! - keep small vectors and scalar parameters in `f32`.
//!
//! The scalar paths are fully IEEE-754 aware and handle signed zero, subnormals, infinities,
//! and NaNs. SIMD paths accelerate bulk conversion and the mixed-precision dot product, while
//! deliberately preserving scalar accumulation order so normal-value results remain consistent
//! with the scalar fallback.

use crate::error::InferenceError;
#[cfg(test)]
use crate::weights::SafetensorsFile;
use crate::weights::half_bits::{f16_bits_to_f32, f32_to_f16_bits};

pub use crate::forward::cpu::f16_matmul::matmul_bt_f16;
pub use crate::model::qwen35::f16_loading::load_f16_weights;

/// **Unstable**: IEEE 754 half-precision float wrapper; used for bandwidth-efficient weight storage.
///
/// IEEE 754 half-precision float, stored as raw `u16` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    pub const ZERO: Self = Self(0x0000);
    pub const ONE: Self = Self(0x3c00);
    pub const NEG_ONE: Self = Self(0xbc00);
    pub const INFINITY: Self = Self(0x7c00);
    pub const NEG_INFINITY: Self = Self(0xfc00);
    pub const NAN: Self = Self(0x7e00);

    /// **Unstable**: convert f32 to f16 with round-to-nearest-even.
    ///
    /// Convert `f32` to IEEE-754 half precision using round-to-nearest-even.
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Self(f32_to_f16_bits(v))
    }

    /// **Unstable**: widen f16 to f32 exactly.
    ///
    /// Widen IEEE-754 half precision to `f32` exactly.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }
}

/// **Unstable**: bulk f16-to-f32 conversion; SIMD path selection may change.
///
/// Convert a slice of packed f16 values to `f32`.
///
/// `src.len()` must equal `dst.len()`.
pub fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "f16_to_f32_slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: AVX + F16C are verified at runtime; src and dst have
            // equal length (asserted above), and the AVX helper uses unaligned
            // loads/stores while guarding each 8-lane chunk.
            unsafe {
                f16_to_f32_slice_avx(src, dst);
            }
            return;
        }
    }

    // Note: aarch64 NEON fp16 intrinsics (vcvt_f32_f16, vreinterpret_f16_u16) require the
    // unstable `stdarch_neon_f16` feature as of Rust 1.87. The scalar path below is used
    // on all non-x86_64 targets until the NEON fp16 API stabilises.
    for (s, d) in src.iter().copied().zip(dst.iter_mut()) {
        *d = F16(s).to_f32();
    }
}

/// **Unstable**: bulk f32-to-f16 conversion; SIMD path selection may change.
///
/// Convert a slice of `f32` values to packed f16 values using round-to-nearest-even.
///
/// `src.len()` must equal `dst.len()`.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len(), "f32_to_f16_slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: AVX + F16C are verified at runtime; src and dst have
            // equal length (asserted above), and the AVX helper uses unaligned
            // loads/stores while guarding each 8-lane chunk.
            unsafe {
                f32_to_f16_slice_avx(src, dst);
            }
            return;
        }
    }

    // See f16_to_f32_slice — scalar fallback until NEON fp16 stabilises.
    for (s, d) in src.iter().copied().zip(dst.iter_mut()) {
        *d = F16::from_f32(s).0;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f16_to_f32_slice_avx(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};

    const LANES: usize = 8;
    debug_assert_eq!(src.len(), dst.len());
    let mut i = 0usize;
    while i + LANES <= src.len() {
        debug_assert!(i + LANES <= src.len());
        debug_assert!(i + LANES <= dst.len());
        let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), f);
        i += LANES;
    }

    while i < src.len() {
        dst[i] = F16(src[i]).to_f32();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32_to_f16_slice_avx(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::{__m128i, _mm_storeu_si128, _mm256_cvtps_ph, _mm256_loadu_ps};

    const LANES: usize = 8;
    debug_assert_eq!(src.len(), dst.len());
    let mut i = 0usize;
    while i + LANES <= src.len() {
        debug_assert!(i + LANES <= src.len());
        debug_assert!(i + LANES <= dst.len());
        let f = _mm256_loadu_ps(src.as_ptr().add(i));
        let h = _mm256_cvtps_ph(f, 0);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        i += LANES;
    }

    while i < src.len() {
        dst[i] = F16::from_f32(src[i]).0;
        i += 1;
    }
}

/// **Unstable**: f16 weights for a GatedDeltaNet layer; field layout may change with new projections.
///
/// F16 weights for a single GatedDeltaNet layer.
#[derive(Debug, Clone)]
pub struct F16GatedDeltaNetWeights {
    pub in_proj_qkv: Vec<u16>,
    pub in_proj_qkv_rows: usize,
    pub in_proj_qkv_cols: usize,

    pub in_proj_z: Vec<u16>,
    pub in_proj_z_rows: usize,
    pub in_proj_z_cols: usize,

    pub in_proj_b: Vec<u16>,
    pub in_proj_b_rows: usize,
    pub in_proj_b_cols: usize,

    pub in_proj_a: Vec<u16>,
    pub in_proj_a_rows: usize,
    pub in_proj_a_cols: usize,

    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,

    pub conv1d_weight: Vec<f32>,
    pub conv_dim: usize,
    pub kernel_size: usize,

    pub norm_weight: Vec<f32>,

    pub out_proj: Vec<u16>,
    pub out_proj_rows: usize,
    pub out_proj_cols: usize,
}

/// **Unstable**: f16 weights for a full GQA layer; field set mirrors float weights.
///
/// F16 weights for a full-attention (GQA) layer.
#[derive(Debug, Clone)]
pub struct F16FullAttentionLayerWeights {
    pub q_proj: Vec<u16>,
    pub k_proj: Vec<u16>,
    pub v_proj: Vec<u16>,
    pub o_proj: Vec<u16>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// **Unstable**: f16 common layer weights for norms plus FFN; field layout may change.
///
/// F16 common layer weights (norms + dense-or-MoE FFN).
#[derive(Debug, Clone)]
pub struct F16CommonLayerWeights {
    /// Input RMSNorm weights, shape `[hidden]`, kept in f32.
    pub input_layernorm: Vec<f32>,
    /// Post-attention RMSNorm weights, shape `[hidden]`, kept in f32.
    pub post_attention_layernorm: Vec<f32>,
    /// Per-layer FFN weights, dense or MoE depending on config.
    pub ffn: F16FeedForwardWeights,
}

/// Top-k router weights for f16 Mixture-of-Experts layers.
#[derive(Debug, Clone)]
pub struct F16MoeRouter {
    /// Router gate matrix, shape `[num_experts, hidden_size]`, stored as packed f16 bits.
    pub gate: Vec<u16>,
    /// Number of routed experts in the layer.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub num_experts_per_tok: usize,
    /// Input hidden width for the router.
    pub hidden_size: usize,
}

/// Routed sparse expert weights for f16 MoE layers.
#[derive(Debug, Clone)]
pub struct F16RoutedExperts {
    /// Fused gate/up projection weights, shape `[num_experts, 2 * intermediate_size, hidden_size]`.
    pub gate_up_proj: Vec<u16>,
    /// Down projection weights, shape `[num_experts, hidden_size, intermediate_size]`.
    pub down_proj: Vec<u16>,
    /// Number of routed experts.
    pub num_experts: usize,
    /// Input and output hidden width for routed experts.
    pub hidden_size: usize,
    /// Routed expert intermediate width.
    pub intermediate_size: usize,
}

/// Shared always-active expert weights for f16 MoE layers.
#[derive(Debug, Clone)]
pub struct F16SharedExpert {
    /// Shared expert gate projection, shape `[intermediate_size, hidden_size]`.
    pub gate_proj: Vec<u16>,
    /// Shared expert up projection, shape `[intermediate_size, hidden_size]`.
    pub up_proj: Vec<u16>,
    /// Shared expert down projection, shape `[hidden_size, intermediate_size]`.
    pub down_proj: Vec<u16>,
    /// Shared expert scalar gate row, shape `[1, hidden_size]`.
    pub shared_expert_gate: Vec<u16>,
    /// Input and output hidden width for the shared expert.
    pub hidden_size: usize,
    /// Shared expert intermediate width.
    pub intermediate_size: usize,
}

/// Complete f16 MoE FFN weights for one transformer layer.
#[derive(Debug, Clone)]
pub struct F16MoeLayerWeights {
    /// Router weights and top-k metadata.
    pub router: F16MoeRouter,
    /// Routed expert weights.
    pub experts: F16RoutedExperts,
    /// Shared always-active expert weights.
    pub shared_expert: F16SharedExpert,
}

/// Per-layer f16 FFN weights: dense SwiGLU or Mixture-of-Experts.
#[derive(Debug, Clone)]
pub enum F16FeedForwardWeights {
    /// Dense SwiGLU FFN weights.
    Dense {
        /// Dense gate projection, shape `[intermediate, hidden]`.
        gate_proj: Vec<u16>,
        /// Dense up projection, shape `[intermediate, hidden]`.
        up_proj: Vec<u16>,
        /// Dense down projection, shape `[hidden, intermediate]`.
        down_proj: Vec<u16>,
    },
    /// Mixture-of-Experts FFN weights.
    Moe(F16MoeLayerWeights),
}

impl F16MoeRouter {
    pub fn new(
        gate: Vec<u16>,
        num_experts: usize,
        num_experts_per_tok: usize,
        hidden_size: usize,
    ) -> Result<Self, InferenceError> {
        let expected = num_experts
            .checked_mul(hidden_size)
            .ok_or_else(|| InferenceError::Inference("F16 MoE router shape overflow".into()))?;
        if gate.len() != expected {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.gate.weight".to_string(),
                expected: vec![num_experts, hidden_size],
                actual: vec![gate.len()],
            });
        }
        if num_experts_per_tok == 0 || num_experts_per_tok > num_experts {
            return Err(InferenceError::UnsupportedModel(format!(
                "invalid f16 MoE top_k={num_experts_per_tok} for num_experts={num_experts}"
            )));
        }
        Ok(Self {
            gate,
            num_experts,
            num_experts_per_tok,
            hidden_size,
        })
    }
}

impl F16RoutedExperts {
    pub fn new(
        gate_up_proj: Vec<u16>,
        down_proj: Vec<u16>,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, InferenceError> {
        let two_intermediate = intermediate_size.checked_mul(2).ok_or_else(|| {
            InferenceError::Inference("F16 RoutedExperts gate_up_proj shape overflow".into())
        })?;
        let expected_gate_up = num_experts
            .checked_mul(two_intermediate)
            .and_then(|x| x.checked_mul(hidden_size))
            .ok_or_else(|| {
                InferenceError::Inference("F16 RoutedExperts gate_up_proj shape overflow".into())
            })?;
        let expected_down = num_experts
            .checked_mul(hidden_size)
            .and_then(|x| x.checked_mul(intermediate_size))
            .ok_or_else(|| {
                InferenceError::Inference("F16 RoutedExperts down_proj shape overflow".into())
            })?;
        if gate_up_proj.len() != expected_gate_up {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.experts.gate_up_proj".to_string(),
                expected: vec![num_experts, two_intermediate, hidden_size],
                actual: vec![gate_up_proj.len()],
            });
        }
        if down_proj.len() != expected_down {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.experts.down_proj".to_string(),
                expected: vec![num_experts, hidden_size, intermediate_size],
                actual: vec![down_proj.len()],
            });
        }
        Ok(Self {
            gate_up_proj,
            down_proj,
            num_experts,
            hidden_size,
            intermediate_size,
        })
    }
}

impl F16SharedExpert {
    pub fn new(
        gate_proj: Vec<u16>,
        up_proj: Vec<u16>,
        down_proj: Vec<u16>,
        shared_expert_gate: Vec<u16>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, InferenceError> {
        let up_shape = intermediate_size.checked_mul(hidden_size).ok_or_else(|| {
            InferenceError::Inference("F16 shared expert up/gate shape overflow".into())
        })?;
        let down_shape = hidden_size.checked_mul(intermediate_size).ok_or_else(|| {
            InferenceError::Inference("F16 shared expert down shape overflow".into())
        })?;
        if gate_proj.len() != up_shape {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert.gate_proj.weight".to_string(),
                expected: vec![intermediate_size, hidden_size],
                actual: vec![gate_proj.len()],
            });
        }
        if up_proj.len() != up_shape {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert.up_proj.weight".to_string(),
                expected: vec![intermediate_size, hidden_size],
                actual: vec![up_proj.len()],
            });
        }
        if down_proj.len() != down_shape {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert.down_proj.weight".to_string(),
                expected: vec![hidden_size, intermediate_size],
                actual: vec![down_proj.len()],
            });
        }
        if shared_expert_gate.len() != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert_gate.weight".to_string(),
                expected: vec![1, hidden_size],
                actual: vec![shared_expert_gate.len()],
            });
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            shared_expert_gate,
            hidden_size,
            intermediate_size,
        })
    }
}

/// **Unstable**: per-layer f16 attention weights; variant set tied to hybrid architecture.
///
/// Per-layer f16 attention weights.
#[derive(Debug, Clone)]
pub enum F16AttentionWeights {
    Linear(F16GatedDeltaNetWeights),
    Full(F16FullAttentionLayerWeights),
}

/// **Unstable**: all model weights in f16 storage; layer layout mirrors float weights.
///
/// All model weights using f16 storage for large matrices.
#[derive(Debug, Clone)]
pub struct F16ModelWeights {
    pub embed_tokens: Vec<u16>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<(F16AttentionWeights, F16CommonLayerWeights)>,
}

impl SafetensorsFile {
    /// **Unstable**: load safetensors tensor as packed f16; double conversion will be eliminated.
    ///
    /// Load a tensor as packed f16 values.
    ///
    /// **Warning**: This currently performs a double conversion for on-disk F16 tensors:
    /// the underlying `get_f32_tensor()` widens F16 to F32, and then this method narrows
    /// back to F16. This is correct but wasteful. When raw dtype/byte access is exposed
    /// from `weights.rs`, this can be upgraded to keep on-disk F16 tensors zero-copy and
    /// to convert BF16 directly without changing call sites.
    pub fn get_f16_tensor(&self, name: &str) -> Result<(Vec<u16>, &[usize]), InferenceError> {
        let (data, shape) = self.get_f32_tensor(name)?;
        let mut out = vec![0u16; data.len()];
        f32_to_f16_slice(data, &mut out);
        Ok((out, shape))
    }
}

#[cfg(test)]
mod tests;
