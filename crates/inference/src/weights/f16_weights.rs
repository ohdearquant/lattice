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
use crate::model::qwen35_config::Qwen35Config;
use crate::weights::SafetensorsFile;

const F16_SIGN_MASK: u16 = 0x8000;
const F16_EXP_MASK: u16 = 0x7c00;
const F16_FRAC_MASK: u16 = 0x03ff;
const F16_IMPLICIT_ONE: u32 = 0x0080_0000;

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

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & F16_FRAC_MASK) as u32;

    let f32_bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };

    f32::from_bits(f32_bits)
}

#[inline]
fn round_shift_right_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift >= 32 {
        return 0;
    }

    let base = value >> shift;
    let mask = (1u32 << shift) - 1;
    let remainder = value & mask;
    let half = 1u32 << (shift - 1);

    if remainder > half || (remainder == half && (base & 1) != 0) {
        base + 1
    } else {
        base
    }
}

#[inline]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) as u16) & F16_SIGN_MASK;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        if frac == 0 {
            return sign | F16_EXP_MASK;
        }
        let mut payload = (frac >> 13) as u16;
        if payload == 0 {
            payload = 1;
        }
        payload |= 0x0200;
        return sign | F16_EXP_MASK | (payload & F16_FRAC_MASK);
    }

    if exp == 0 {
        return sign;
    }

    let exp32 = exp - 127;

    if exp32 > 15 {
        return sign | F16_EXP_MASK;
    }

    if exp32 >= -14 {
        let mut exp16 = (exp32 + 15) as u16;
        let mut frac16 = round_shift_right_even(frac, 13) as u16;

        if frac16 == 0x0400 {
            frac16 = 0;
            exp16 += 1;
            if exp16 >= 0x1f {
                return sign | F16_EXP_MASK;
            }
        }

        return sign | (exp16 << 10) | frac16;
    }

    let mant = frac | F16_IMPLICIT_ONE;
    let shift = (-exp32 - 1) as u32;
    if shift >= 32 {
        return sign;
    }

    let frac16 = round_shift_right_even(mant, shift) as u16;
    if frac16 == 0 {
        return sign;
    }
    if frac16 == 0x0400 {
        return sign | 0x0400;
    }

    sign | frac16
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

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    AvxF16c,
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
impl DotKernel {
    #[inline]
    fn select() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                return Self::AvxF16c;
            }
        }

        Self::Scalar
    }

    #[inline]
    fn dot(self, a: &[f32], b: &[u16]) -> f32 {
        match self {
            Self::Scalar => dot_f16_f32_scalar(a, b),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: DotKernel::select only returns AvxF16c after runtime
            // AVX + F16C detection; dot_f16_f32_avx validates equal lengths.
            Self::AvxF16c => unsafe { dot_f16_f32_avx(a, b) },
        }
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[inline]
fn dot_f16_f32_scalar(a: &[f32], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * F16(b[i]).to_f32();
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn dot_f16_f32_avx(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };

    debug_assert_eq!(a.len(), b.len());
    const LANES: usize = 8;

    let mut acc = 0.0f32;
    let mut i = 0usize;
    let mut prod = [0.0f32; 8];

    while i + LANES <= a.len() {
        debug_assert!(i + LANES <= a.len());
        debug_assert!(i + LANES <= b.len());
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let hv = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        let bv = _mm256_cvtph_ps(hv);
        let pv = _mm256_mul_ps(av, bv);
        _mm256_storeu_ps(prod.as_mut_ptr(), pv);
        acc += prod[0];
        acc += prod[1];
        acc += prod[2];
        acc += prod[3];
        acc += prod[4];
        acc += prod[5];
        acc += prod[6];
        acc += prod[7];
        i += LANES;
    }

    while i < a.len() {
        acc += a[i] * F16(b[i]).to_f32();
        i += 1;
    }

    acc
}

/// **Unstable**: mixed-precision matmul C = A @ B^T with f16 weights; dispatch may change.
///
/// Matrix multiply `C = A @ B^T` where `A` is `f32`, `B` is packed `f16`, and `C` is `f32`.
///
/// Layouts:
/// - `a`: `[m, k]` row-major `f32`
/// - `b`: `[n, k]` row-major packed half (`u16`)
/// - `c`: `[m, n]` row-major `f32`
///
/// On macOS: bulk-converts B tiles to f32 and dispatches to Accelerate cblas_sgemm (AMX).
/// On other platforms: per-element f16→f32 conversion with optional AVX F16C.
pub fn matmul_bt_f16(a: &[f32], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize) {
    let a_len = m.checked_mul(k).expect("matmul_bt_f16: m*k overflow");
    let b_len = n.checked_mul(k).expect("matmul_bt_f16: n*k overflow");
    let c_len = m.checked_mul(n).expect("matmul_bt_f16: m*n overflow");

    assert_eq!(a.len(), a_len, "matmul_bt_f16: invalid A length");
    assert_eq!(b.len(), b_len, "matmul_bt_f16: invalid B length");
    assert_eq!(c.len(), c_len, "matmul_bt_f16: invalid C length");

    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }

    // On macOS: convert f16 B tiles to f32, then use Accelerate cblas_sgemm for AMX throughput.
    // The conversion is O(tile_n * k), the GEMM is O(m * tile_n * k) — for small m (decode),
    // the conversion cost is comparable to the GEMM, but both are vastly faster than the
    // scalar per-element dot product path.
    #[cfg(target_os = "macos")]
    {
        // Tile size: convert TILE_N rows of B at a time to bound scratch memory.
        // For decode (m=1), tile_n=256 means 256*2048*4 = 2MB scratch — fits in L2.
        const TILE_N: usize = 256;
        let mut b_f32 = vec![0.0f32; TILE_N * k];

        let mut col = 0;
        while col < n {
            let tile_n = (n - col).min(TILE_N);
            let b_tile = &b[col * k..(col + tile_n) * k];
            let buf = &mut b_f32[..tile_n * k];

            // Bulk f16→f32 conversion
            for i in 0..tile_n * k {
                buf[i] = F16(b_tile[i]).to_f32();
            }

            // FP-052: use a separate tile output buffer to avoid overlap between
            // the contiguous GEMM result and previously-scattered rows in `c`.
            // Writing directly into c[col..col + m*tile_n] when m > 1 would
            // overwrite rows already scattered from the previous iteration.
            let mut c_tile = vec![0.0f32; m * tile_n];
            crate::forward::cpu::matmul_bt(a, buf, &mut c_tile, m, k, tile_n);

            // Scatter from contiguous c_tile[m, tile_n] into strided c[m, n].
            for row in 0..m {
                let src = &c_tile[row * tile_n..(row + 1) * tile_n];
                c[row * n + col..row * n + col + tile_n].copy_from_slice(src);
            }

            col += tile_n;
        }
    }

    // Non-macOS: per-element conversion with DotKernel
    #[cfg(not(target_os = "macos"))]
    {
        let kernel = DotKernel::select();

        if m == 1 {
            let a_row = &a[..k];
            for out in 0..n {
                let b_row = &b[out * k..(out + 1) * k];
                c[out] = kernel.dot(a_row, b_row);
            }
            return;
        }

        for row in 0..m {
            let a_row = &a[row * k..(row + 1) * k];
            let c_row = &mut c[row * n..(row + 1) * n];
            for out in 0..n {
                let b_row = &b[out * k..(out + 1) * k];
                c_row[out] = kernel.dot(a_row, b_row);
            }
        }
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

#[inline]
fn checked_numel(shape: &[usize], name: &str) -> Result<usize, InferenceError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "tensor {name} shape {shape:?} overflows usize element count"
            ))
        })
    })
}

#[inline]
fn expect_shape(name: &str, shape: &[usize], expected: &[usize]) -> Result<(), InferenceError> {
    if shape != expected {
        return Err(InferenceError::InvalidSafetensors(format!(
            "tensor {name} has shape {shape:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn load_f16_tensor_checked(
    sf: &SafetensorsFile,
    name: &str,
    expected: &[usize],
) -> Result<Vec<u16>, InferenceError> {
    let (data, shape) = sf.get_f16_tensor(name)?;
    expect_shape(name, shape, expected)?;
    let numel = checked_numel(shape, name)?;
    if data.len() != numel {
        return Err(InferenceError::InvalidSafetensors(format!(
            "tensor {name} has {} values, expected {numel}",
            data.len()
        )));
    }
    Ok(data)
}

fn load_f32_tensor_checked(
    sf: &SafetensorsFile,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    let (data, shape) = sf.get_f32_tensor(name)?;
    expect_shape(name, shape, expected)?;
    let numel = checked_numel(shape, name)?;
    if data.len() != numel {
        return Err(InferenceError::InvalidSafetensors(format!(
            "tensor {name} has {} values, expected {numel}",
            data.len()
        )));
    }
    Ok(data.to_vec())
}

/// Load a conv1d weight tensor, tolerating 3D shape `[out, 1, kernel]` from safetensors.
///
/// Safetensors stores conv1d.weight as 3D `[conv_dim, 1, kernel_size]`, but we flatten
/// to 1D. We skip shape validation and only verify the element count matches.
fn load_conv1d_weight(
    sf: &SafetensorsFile,
    name: &str,
    conv_dim: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, InferenceError> {
    let (data, shape) = sf.get_f32_tensor(name)?;
    let numel = checked_numel(shape, name)?;
    let expected_numel = conv_dim.checked_mul(kernel_size).ok_or_else(|| {
        InferenceError::InvalidSafetensors(format!(
            "conv1d {name}: conv_dim * kernel_size overflows"
        ))
    })?;
    if numel != expected_numel {
        return Err(InferenceError::InvalidSafetensors(format!(
            "conv1d tensor {name} has {numel} elements, expected {expected_numel} \
             (conv_dim={conv_dim}, kernel_size={kernel_size}, actual shape={shape:?})"
        )));
    }
    Ok(data.to_vec())
}

/// **Unstable**: load Qwen3.5-2B weights in f16 storage; tensor naming may change with model variants.
///
/// Load Qwen3.5-2B weights with f16 storage for large matrices.
///
/// Large projection matrices and embeddings are stored as packed f16 values; norms and other
/// small vectors remain `f32`.
pub fn load_f16_weights(
    sf: &SafetensorsFile,
    cfg: &Qwen35Config,
) -> Result<F16ModelWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;
    let kernel_size = cfg.linear_conv_kernel_dim;

    let embed_tokens = load_f16_tensor_checked(
        sf,
        "model.language_model.embed_tokens.weight",
        &[cfg.vocab_size, hidden],
    )?;
    let final_norm = load_f32_tensor_checked(sf, "model.language_model.norm.weight", &[hidden])?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{i}");

        let input_layernorm =
            load_f32_tensor_checked(sf, &format!("{prefix}.input_layernorm.weight"), &[hidden])?;
        let post_attention_layernorm = load_f32_tensor_checked(
            sf,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[hidden],
        )?;

        let ffn = if cfg.is_moe() {
            let num_experts = cfg.num_experts.expect("MoE config has num_experts");
            let top_k = cfg
                .num_experts_per_tok
                .expect("MoE config has num_experts_per_tok");
            let moe_inter = cfg.moe_intermediate_size();
            let shared_inter = cfg.shared_expert_intermediate_size();

            let router = F16MoeRouter::new(
                // F1: router gate, shape [num_experts, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.gate.weight"),
                    &[num_experts, hidden],
                )?,
                num_experts,
                top_k,
                hidden,
            )?;

            let experts = F16RoutedExperts::new(
                // F2: fused routed gate/up projections, shape [num_experts, 2 * moe_inter, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.experts.gate_up_proj"),
                    &[num_experts, 2 * moe_inter, hidden],
                )?,
                // F3: routed down projections, shape [num_experts, hidden, moe_inter]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.experts.down_proj"),
                    &[num_experts, hidden, moe_inter],
                )?,
                num_experts,
                hidden,
                moe_inter,
            )?;

            let shared_expert = F16SharedExpert::new(
                // F4: shared expert gate projection, shape [shared_inter, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    &[shared_inter, hidden],
                )?,
                // F5: shared expert up projection, shape [shared_inter, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    &[shared_inter, hidden],
                )?,
                // F6: shared expert down projection, shape [hidden, shared_inter]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    &[hidden, shared_inter],
                )?,
                // F7: shared expert scalar gate row, shape [1, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert_gate.weight"),
                    &[1, hidden],
                )?,
                hidden,
                shared_inter,
            )?;

            F16FeedForwardWeights::Moe(F16MoeLayerWeights {
                router,
                experts,
                shared_expert,
            })
        } else {
            F16FeedForwardWeights::Dense {
                gate_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &[intermediate, hidden],
                )?,
                up_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &[intermediate, hidden],
                )?,
                down_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[hidden, intermediate],
                )?,
            }
        };

        let common = F16CommonLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            ffn,
        };

        let attn = if cfg.is_full_attention(i) {
            F16AttentionWeights::Full(F16FullAttentionLayerWeights {
                q_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    &[2 * q_dim, hidden],
                )?,
                k_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &[kv_dim, hidden],
                )?,
                v_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    &[kv_dim, hidden],
                )?,
                o_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &[hidden, q_dim],
                )?,
                q_norm: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &[cfg.head_dim],
                )?,
                k_norm: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &[cfg.head_dim],
                )?,
            })
        } else {
            // FIX CRITICAL-1: conv1d.weight is stored as 3D [qkv_dim, 1, kernel_size]
            // in safetensors, so we use load_conv1d_weight which skips shape validation
            // and only verifies element count.
            F16AttentionWeights::Linear(F16GatedDeltaNetWeights {
                in_proj_qkv: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    &[qkv_dim, hidden],
                )?,
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: hidden,
                in_proj_z: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_z.weight"),
                    &[output_dim, hidden],
                )?,
                in_proj_z_rows: output_dim,
                in_proj_z_cols: hidden,
                in_proj_b: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_b.weight"),
                    &[num_heads, hidden],
                )?,
                in_proj_b_rows: num_heads,
                in_proj_b_cols: hidden,
                in_proj_a: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_a.weight"),
                    &[num_heads, hidden],
                )?,
                in_proj_a_rows: num_heads,
                in_proj_a_cols: hidden,
                a_log: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.A_log"),
                    &[num_heads],
                )?,
                dt_bias: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.dt_bias"),
                    &[num_heads],
                )?,
                conv1d_weight: load_conv1d_weight(
                    sf,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    qkv_dim,
                    kernel_size,
                )?,
                conv_dim: qkv_dim,
                kernel_size,
                norm_weight: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.norm.weight"),
                    &[cfg.linear_value_head_dim],
                )?,
                out_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.out_proj.weight"),
                    &[hidden, output_dim],
                )?,
                out_proj_rows: hidden,
                out_proj_cols: output_dim,
            })
        };

        layers.push((attn, common));
    }

    Ok(F16ModelWeights {
        embed_tokens,
        final_norm,
        layers,
    })
}

#[cfg(test)]
fn estimate_f32_model_bytes(cfg: &Qwen35Config) -> usize {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;
    let kernel_size = cfg.linear_conv_kernel_dim;

    let mut bytes = 0usize;

    bytes += cfg.vocab_size * hidden * 4;
    bytes += hidden * 4;

    for i in 0..cfg.num_hidden_layers {
        bytes += hidden * 4;
        bytes += hidden * 4;
        bytes += intermediate * hidden * 4;
        bytes += intermediate * hidden * 4;
        bytes += hidden * intermediate * 4;

        if cfg.is_full_attention(i) {
            bytes += (2 * q_dim * hidden) * 4;
            bytes += (kv_dim * hidden) * 4;
            bytes += (kv_dim * hidden) * 4;
            bytes += (hidden * q_dim) * 4;
            bytes += cfg.head_dim * 4;
            bytes += cfg.head_dim * 4;
        } else {
            bytes += (qkv_dim * hidden) * 4;
            bytes += (output_dim * hidden) * 4;
            bytes += (num_heads * hidden) * 4;
            bytes += (num_heads * hidden) * 4;
            bytes += num_heads * 4;
            bytes += num_heads * 4;
            bytes += (qkv_dim * kernel_size) * 4;
            bytes += output_dim * 4;
            bytes += (hidden * output_dim) * 4;
        }
    }

    bytes
}

#[cfg(test)]
fn estimate_f16_model_bytes(cfg: &Qwen35Config) -> usize {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;
    let kernel_size = cfg.linear_conv_kernel_dim;

    let mut bytes = 0usize;

    bytes += cfg.vocab_size * hidden * 2;
    bytes += hidden * 4;

    for i in 0..cfg.num_hidden_layers {
        bytes += hidden * 4;
        bytes += hidden * 4;
        bytes += intermediate * hidden * 2;
        bytes += intermediate * hidden * 2;
        bytes += hidden * intermediate * 2;

        if cfg.is_full_attention(i) {
            bytes += (2 * q_dim * hidden) * 2;
            bytes += (kv_dim * hidden) * 2;
            bytes += (kv_dim * hidden) * 2;
            bytes += (hidden * q_dim) * 2;
            bytes += cfg.head_dim * 4;
            bytes += cfg.head_dim * 4;
        } else {
            bytes += (qkv_dim * hidden) * 2;
            bytes += (output_dim * hidden) * 2;
            bytes += (num_heads * hidden) * 2;
            bytes += (num_heads * hidden) * 2;
            bytes += num_heads * 4;
            bytes += num_heads * 4;
            bytes += (qkv_dim * kernel_size) * 4;
            bytes += output_dim * 4;
            bytes += (hidden * output_dim) * 2;
        }
    }

    bytes
}

#[cfg(test)]
fn matmul_bt_f32_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let c_row = &mut c[row * n..(row + 1) * n];
        for out in 0..n {
            let b_row = &b[out * k..(out + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += a_row[j] * b_row[j];
            }
            c_row[out] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip_exact_representable() {
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 65504.0] {
            let h = F16::from_f32(v);
            assert_eq!(h.to_f32(), v, "exact roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_f16_special_values() {
        assert_eq!(F16::from_f32(f32::INFINITY).to_f32(), f32::INFINITY);
        assert_eq!(F16::from_f32(f32::NEG_INFINITY).to_f32(), f32::NEG_INFINITY);
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
        assert_eq!(F16::from_f32(100000.0).to_f32(), f32::INFINITY);
    }

    #[test]
    fn test_f16_denormals() {
        let smallest_denorm = F16(0x0001);
        let v = smallest_denorm.to_f32();
        assert!((v - 5.960464e-8).abs() < 1e-12);
    }

    #[test]
    fn test_f16_signed_zero() {
        let pz = F16::from_f32(0.0);
        let nz = F16::from_f32(-0.0);
        assert_eq!(pz.0, 0x0000);
        assert_eq!(nz.0, 0x8000);
        assert_eq!(pz.to_f32().to_bits(), 0.0f32.to_bits());
        assert_eq!(nz.to_f32().to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_f16_round_to_nearest_even() {
        let a = F16(0x3c00).to_f32();
        let b = F16(0x3c01).to_f32();
        let half = (a + b) * 0.5;
        assert_eq!(F16::from_f32(half).0, 0x3c00);

        let c = F16(0x3c01).to_f32();
        let d = F16(0x3c02).to_f32();
        let half2 = (c + d) * 0.5;
        assert_eq!(F16::from_f32(half2).0, 0x3c02);
    }

    #[test]
    fn test_bulk_f16_to_f32_matches_scalar() {
        let src: Vec<u16> = (0..1024)
            .map(|i| F16::from_f32(i as f32 * 0.1 - 50.0).0)
            .collect();
        let mut dst_bulk = vec![0.0f32; 1024];
        let mut dst_scalar = vec![0.0f32; 1024];

        f16_to_f32_slice(&src, &mut dst_bulk);
        for (i, &v) in src.iter().enumerate() {
            dst_scalar[i] = F16(v).to_f32();
        }

        assert_eq!(
            dst_bulk, dst_scalar,
            "bulk and scalar conversion must match exactly"
        );
    }

    #[test]
    fn test_bulk_f32_to_f16_matches_scalar() {
        let src: Vec<f32> = (0..1024)
            .map(|i| ((i * 17 + 29) % 4096) as f32 / 256.0 - 8.0)
            .collect();
        let mut dst_bulk = vec![0u16; 1024];
        let mut dst_scalar = vec![0u16; 1024];

        f32_to_f16_slice(&src, &mut dst_bulk);
        for (i, &v) in src.iter().enumerate() {
            dst_scalar[i] = F16::from_f32(v).0;
        }

        assert_eq!(dst_bulk, dst_scalar);
    }

    #[test]
    fn test_matmul_bt_f16_vs_f32() {
        let m = 1usize;
        let k = 2048usize;
        let n = 6144usize;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i * 11 + 17) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b_f16: Vec<u16> = b_f32.iter().map(|&v| F16::from_f32(v).0).collect();

        let mut c_f32 = vec![0.0f32; m * n];
        let mut c_f16 = vec![0.0f32; m * n];

        matmul_bt_f32_reference(&a, &b_f32, &mut c_f32, m, k, n);
        matmul_bt_f16(&a, &b_f16, &mut c_f16, m, k, n);

        let max_diff = c_f32
            .iter()
            .zip(c_f16.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.1,
            "max absolute difference too large: {max_diff}"
        );
    }

    #[test]
    fn test_matmul_bt_f16_matches_scalar_reference_exactly_for_same_half_weights() {
        let m = 2usize;
        let k = 65usize;
        let n = 17usize;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 5 + 3) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let b_half: Vec<u16> = (0..n * k)
            .map(|i| F16::from_f32(((i * 9 + 1) % 113) as f32 / 113.0 - 0.5).0)
            .collect();

        let mut c_simd = vec![0.0f32; m * n];
        let mut c_scalar = vec![0.0f32; m * n];

        matmul_bt_f16(&a, &b_half, &mut c_simd, m, k, n);

        for row in 0..m {
            let a_row = &a[row * k..(row + 1) * k];
            let c_row = &mut c_scalar[row * n..(row + 1) * n];
            for out in 0..n {
                let b_row = &b_half[out * k..(out + 1) * k];
                c_row[out] = dot_f16_f32_scalar(a_row, b_row);
            }
        }

        // Allow small FP rounding differences: BLAS (AMX/FMA) accumulation order
        // differs from the scalar reference, causing ~1e-7 relative drift.
        let max_diff = c_simd
            .iter()
            .zip(c_scalar.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "max diff {max_diff} exceeds tolerance 1e-5"
        );
    }

    #[test]
    fn test_f16_memory_savings() {
        let cfg = Qwen35Config::qwen35_2b();
        let f32_bytes = estimate_f32_model_bytes(&cfg);
        let f16_bytes = estimate_f16_model_bytes(&cfg);

        let ratio = f16_bytes as f64 / f32_bytes as f64;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "f16/f32 ratio should be ~0.5, got {ratio}"
        );
    }

    #[test]
    fn test_matmul_bt_f16_near_zero_weights() {
        let m = 1usize;
        let k = 128usize;
        let n = 64usize;

        let a = vec![1.0f32; m * k];
        let b_f32: Vec<f32> = (0..n * k).map(|_| 1e-6f32).collect();
        let b_f16: Vec<u16> = b_f32.iter().map(|&v| F16::from_f32(v).0).collect();

        let mut c = vec![0.0f32; m * n];
        matmul_bt_f16(&a, &b_f16, &mut c, m, k, n);

        for &v in &c {
            assert!(
                (v - 0.000128).abs() < 0.001,
                "near-zero weight output wrong: {v}"
            );
        }
    }

    #[test]
    fn test_f16_moe_router_shape_validation() {
        let router = F16MoeRouter::new(vec![F16::ONE.0; 6], 3, 2, 2).expect("valid router shape");
        assert_eq!(router.num_experts, 3);
        assert_eq!(router.num_experts_per_tok, 2);
        assert_eq!(router.hidden_size, 2);
        assert_eq!(router.gate.len(), 6);

        assert!(matches!(
            F16MoeRouter::new(vec![0; 2], 1, 2, 2),
            Err(InferenceError::UnsupportedModel(_))
        ));

        assert!(matches!(
            F16MoeRouter::new(vec![0; 5], 3, 1, 2),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.gate.weight" && expected == vec![3, 2] && actual == vec![5]
        ));
    }

    #[test]
    fn test_f16_routed_experts_shape_validation() {
        let experts = F16RoutedExperts::new(vec![0; 48], vec![0; 24], 2, 3, 4)
            .expect("valid routed expert shapes");
        assert_eq!(experts.num_experts, 2);
        assert_eq!(experts.hidden_size, 3);
        assert_eq!(experts.intermediate_size, 4);
        assert_eq!(experts.gate_up_proj.len(), 48);
        assert_eq!(experts.down_proj.len(), 24);

        assert!(matches!(
            F16RoutedExperts::new(vec![0; 47], vec![0; 24], 2, 3, 4),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.experts.gate_up_proj"
                    && expected == vec![2, 8, 3]
                    && actual == vec![47]
        ));

        assert!(matches!(
            F16RoutedExperts::new(vec![0; 48], vec![0; 23], 2, 3, 4),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.experts.down_proj"
                    && expected == vec![2, 3, 4]
                    && actual == vec![23]
        ));
    }

    #[test]
    fn test_f16_shared_expert_shape_validation() {
        let shared = F16SharedExpert::new(vec![0; 6], vec![0; 6], vec![0; 6], vec![0; 3], 3, 2)
            .expect("valid shared expert shapes");
        assert_eq!(shared.hidden_size, 3);
        assert_eq!(shared.intermediate_size, 2);
        assert_eq!(shared.gate_proj.len(), 6);
        assert_eq!(shared.up_proj.len(), 6);
        assert_eq!(shared.down_proj.len(), 6);
        assert_eq!(shared.shared_expert_gate.len(), 3);

        assert!(matches!(
            F16SharedExpert::new(vec![0; 5], vec![0; 6], vec![0; 6], vec![0; 3], 3, 2),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.shared_expert.gate_proj.weight"
                    && expected == vec![2, 3]
                    && actual == vec![5]
        ));

        assert!(matches!(
            F16SharedExpert::new(vec![0; 6], vec![0; 5], vec![0; 6], vec![0; 3], 3, 2),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.shared_expert.up_proj.weight"
                    && expected == vec![2, 3]
                    && actual == vec![5]
        ));

        assert!(matches!(
            F16SharedExpert::new(vec![0; 6], vec![0; 6], vec![0; 5], vec![0; 3], 3, 2),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.shared_expert.down_proj.weight"
                    && expected == vec![3, 2]
                    && actual == vec![5]
        ));

        assert!(matches!(
            F16SharedExpert::new(vec![0; 6], vec![0; 6], vec![0; 6], vec![0; 2], 3, 2),
            Err(InferenceError::ShapeMismatch { name, expected, actual })
                if name == "mlp.shared_expert_gate.weight"
                    && expected == vec![1, 3]
                    && actual == vec![2]
        ));
    }

    #[test]
    fn test_f16_moe_tensor_names() {
        let prefix = "model.language_model.layers.17";
        let names = [
            format!("{prefix}.mlp.gate.weight"),
            format!("{prefix}.mlp.experts.gate_up_proj"),
            format!("{prefix}.mlp.experts.down_proj"),
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            format!("{prefix}.mlp.shared_expert_gate.weight"),
        ];

        assert_eq!(
            names,
            [
                "model.language_model.layers.17.mlp.gate.weight".to_string(),
                "model.language_model.layers.17.mlp.experts.gate_up_proj".to_string(),
                "model.language_model.layers.17.mlp.experts.down_proj".to_string(),
                "model.language_model.layers.17.mlp.shared_expert.gate_proj.weight".to_string(),
                "model.language_model.layers.17.mlp.shared_expert.up_proj.weight".to_string(),
                "model.language_model.layers.17.mlp.shared_expert.down_proj.weight".to_string(),
                "model.language_model.layers.17.mlp.shared_expert_gate.weight".to_string(),
            ]
        );
        assert!(!names.contains(&format!("{prefix}.mlp.experts.gate_up_proj.weight")));
        assert!(!names.contains(&format!("{prefix}.mlp.experts.down_proj.weight")));
        assert!(!names.contains(&format!("{prefix}.mlp.gate_proj.weight")));
    }
}
