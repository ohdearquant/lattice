//! INT8 quantization for efficient embedding storage and similarity computation.
//!
//! Quantized vectors provide ~3x speedup and 4x memory reduction with 99%+ accuracy.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::sync::OnceLock;

use super::simd_config;

/// **Unstable**: INT8 quantization parameters; scale/bias scheme may change.
///
/// Quantization parameters for int8 conversion.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationParams {
    /// **Unstable**: scale factor; formula may change with scheme update.
    pub scale: f32,
    /// **Unstable**: zero point offset; may be removed for symmetric-only quantization.
    pub zero_point: i8,
    /// **Unstable**: min float value; may be removed.
    pub min_val: f32,
    /// **Unstable**: max float value; may be removed.
    pub max_val: f32,
}

impl QuantizationParams {
    /// **Unstable**: parameter computation; may be folded into `QuantizedVector::from_f32`.
    ///
    /// Handles edge cases: empty vectors, NaN, Inf, near-zero vectors.
    pub fn from_vector(vector: &[f32]) -> Self {
        // Single pass over finite values to handle NaN/Inf gracefully.
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &v in vector {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        // Handle edge case: empty or all non-finite.
        if !min_val.is_finite() || !max_val.is_finite() {
            min_val = 0.0;
            max_val = 0.0;
        }

        // Symmetric quantization: map [-max_abs, max_abs] to [-127, 127]
        let max_abs = min_val.abs().max(max_val.abs());

        // Epsilon guard to avoid division by near-zero
        let scale = if max_abs > 1e-10 {
            127.0 / max_abs
        } else {
            1.0 // All zeros or near-zero case
        };

        Self {
            scale,
            zero_point: 0,
            min_val,
            max_val,
        }
    }
}

/// **Unstable**: INT8 quantized vector; struct layout and invariants may change.
///
/// Quantized int8 vector with its parameters.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// **Unstable**: raw quantized data; invariant (`[-127, 127]`) enforced by constructor.
    ///
    /// # Invariant
    /// All values must be in the range `[-127, 127]`. The value `-128` causes
    /// incorrect results in AVX-512 VNNI and AVX2 SIMD paths due to `vpabsb`
    /// saturation behavior. The `from_f32` constructor enforces this via clamping.
    pub data: Vec<i8>,
    /// **Unstable**: quantization parameters; may be separated from the vector.
    pub params: QuantizationParams,
    /// **Unstable**: L2 norm; may be removed or moved.
    pub norm: f32,
}

impl QuantizedVector {
    /// **Unstable**: quantization constructor; clamping behavior may change.
    pub fn from_f32(vector: &[f32]) -> Self {
        let mut params = QuantizationParams::from_vector(vector);

        // Defensive guard: avoid NaN/Inf/zero scale.
        if !params.scale.is_finite() || params.scale == 0.0 {
            params.scale = 1.0;
        }

        // Compute L2 norm of finite values (NaN/Inf are treated as 0.0).
        let mut norm_sq = 0.0f32;
        for &v in vector {
            if v.is_finite() {
                norm_sq += v * v;
            }
        }
        let norm = norm_sq.sqrt();

        let data: Vec<i8> = vector
            .iter()
            .map(|&v| {
                if !v.is_finite() {
                    0
                } else {
                    (v * params.scale).round().clamp(-127.0, 127.0) as i8
                }
            })
            .collect();

        Self { data, params, norm }
    }

    /// **Unstable**: dequantization; output precision may change with scheme update.
    ///
    /// # Precision
    ///
    /// INT8 symmetric quantization maps `[-max_abs, max_abs]` to `[-127, 127]`,
    /// so the quantization step size is `max_abs / 127`. The maximum per-element
    /// round-trip error is bounded by half a quantization step: `max_abs / 254`.
    ///
    /// For a 384-dim unit-norm embedding (`max_abs` ≈ 1.0), expect element-wise
    /// absolute error ≤ 0.004 and cosine-similarity error ≤ 0.5%.
    pub fn to_f32(&self) -> Vec<f32> {
        let scale = if self.params.scale.is_finite() && self.params.scale != 0.0 {
            self.params.scale
        } else {
            1.0
        };

        self.data.iter().map(|&v| v as f32 / scale).collect()
    }

    /// **Unstable**: delegates to `dot_product_i8`; SIMD dispatch may change.
    #[inline]
    pub fn dot_product(&self, other: &QuantizedVector) -> f32 {
        dot_product_i8(self, other)
    }

    /// **Unstable**: delegates to `cosine_similarity_i8`; SIMD dispatch may change.
    #[inline]
    pub fn cosine_similarity(&self, other: &QuantizedVector) -> f32 {
        cosine_similarity_i8(self, other)
    }
}

/// **Unstable**: SIMD INT8 dot product; VNNI/AVX2/NEON dispatch may change.
///
/// Returns the approximate float dot product.
/// Returns 0.0 if vectors have different lengths.
///
/// # Feature gate asymmetry
///
/// The float32 AVX-512F path (dot_product, cosine, normalize, distance) activates
/// unconditionally via runtime `is_x86_feature_detected!("avx512f")` -- no Cargo
/// feature gate is needed because `_mm512_loadu_ps` / `_mm512_fmadd_ps` etc. are
/// part of the base AVX-512F ISA and Rust's `#[target_feature(enable = "avx512f")]`
/// annotation is sufficient.
///
/// The integer VNNI path below requires `--features avx512` at compile time because
/// it uses `_mm512_dpbusd_epi32` (AVX-512 VNNI) and `_mm512_cmplt_epi8_mask`
/// (AVX-512BW), which are behind nightly-gated intrinsics that need an explicit
/// Cargo feature to opt in to the extended instruction sets at compile time.
#[inline]
pub fn dot_product_i8(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    // FP-033: enforce at call time (not just debug) — -128 causes incorrect results
    // in AVX-512 VNNI via vpabsb saturation; from_f32 clamps to [-127, 127] but
    // the data field is pub so callers can bypass the constructor.
    assert!(
        a.data.iter().all(|&v| v != -128i8),
        "QuantizedVector a contains -128, which violates the [-127, 127] VNNI invariant"
    );
    assert!(
        b.data.iter().all(|&v| v != -128i8),
        "QuantizedVector b contains -128, which violates the [-127, 127] VNNI invariant"
    );

    // Runtime length check to prevent UB in release builds
    if a.data.len() != b.data.len() {
        return 0.0;
    }
    debug_assert_eq!(a.data.len(), b.data.len());

    let denom = a.params.scale * b.params.scale;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }

    dot_product_i8_raw(&a.data, &b.data) / denom
}

/// Trusted INT8 dot product for constructor-owned vectors in prepared-query paths.
///
/// Uses `debug_assert!` instead of `assert!`; callers must guarantee vectors
/// were produced by `QuantizedVector::from_f32` or equivalent (clamped to [-127,127]).
#[inline]
pub(crate) fn dot_product_i8_trusted(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    if a.data.len() != b.data.len() {
        return 0.0;
    }
    let denom = a.params.scale * b.params.scale;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    debug_assert!(a.data.iter().all(|&v| v != i8::MIN));
    debug_assert!(b.data.iter().all(|&v| v != i8::MIN));
    dot_product_i8_raw(&a.data, &b.data) / denom
}

/// **Unstable**: SIMD INT8 cosine similarity; norm storage approach may change.
///
/// Uses pre-computed norms for efficiency.
#[inline]
pub fn cosine_similarity_i8(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    let denom = a.norm * b.norm;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    dot_product_i8(a, b) / denom
}

/// Trusted INT8 cosine similarity for constructor-owned vectors in prepared-query paths.
///
/// Uses `dot_product_i8_trusted` instead of `dot_product_i8` to skip release-mode
/// O(N) invariant scans. Callers must guarantee vectors were produced by
/// `QuantizedVector::from_f32` or equivalent (clamped to [-127, 127]).
#[inline]
pub(crate) fn cosine_similarity_i8_trusted(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    let denom = a.norm * b.norm;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    dot_product_i8_trusted(a, b) / denom
}

/// NEON int8 dot product using vmull/vpadal with 4x unrolling.
///
/// Processes 64 int8s per iteration with 4 accumulators.
///
/// # Safety
///
/// Caller must ensure:
/// - Running on aarch64 (NEON is mandatory, always available)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `vld1q_s8` for loads (handles any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder handled via safe slice iteration
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_product_i8_neon_unrolled(a: &[i8], b: &[i8]) -> f32 {
    const SIMD_WIDTH: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    // 4 independent int32 accumulators
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    let mut sum2 = vdupq_n_s32(0);
    let mut sum3 = vdupq_n_s32(0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        // Unroll 0: Load 16 int8s, split, widening multiply, pairwise add
        let a0 = vld1q_s8(a.as_ptr().add(base));
        let b0 = vld1q_s8(b.as_ptr().add(base));
        let a0_lo = vget_low_s8(a0);
        let a0_hi = vget_high_s8(a0);
        let b0_lo = vget_low_s8(b0);
        let b0_hi = vget_high_s8(b0);
        let prod0_lo = vmull_s8(a0_lo, b0_lo);
        let prod0_hi = vmull_s8(a0_hi, b0_hi);
        sum0 = vpadalq_s16(sum0, prod0_lo);
        sum0 = vpadalq_s16(sum0, prod0_hi);

        // Unroll 1
        let a1 = vld1q_s8(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = vld1q_s8(b.as_ptr().add(base + SIMD_WIDTH));
        let a1_lo = vget_low_s8(a1);
        let a1_hi = vget_high_s8(a1);
        let b1_lo = vget_low_s8(b1);
        let b1_hi = vget_high_s8(b1);
        let prod1_lo = vmull_s8(a1_lo, b1_lo);
        let prod1_hi = vmull_s8(a1_hi, b1_hi);
        sum1 = vpadalq_s16(sum1, prod1_lo);
        sum1 = vpadalq_s16(sum1, prod1_hi);

        // Unroll 2
        let a2 = vld1q_s8(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = vld1q_s8(b.as_ptr().add(base + SIMD_WIDTH * 2));
        let a2_lo = vget_low_s8(a2);
        let a2_hi = vget_high_s8(a2);
        let b2_lo = vget_low_s8(b2);
        let b2_hi = vget_high_s8(b2);
        let prod2_lo = vmull_s8(a2_lo, b2_lo);
        let prod2_hi = vmull_s8(a2_hi, b2_hi);
        sum2 = vpadalq_s16(sum2, prod2_lo);
        sum2 = vpadalq_s16(sum2, prod2_hi);

        // Unroll 3
        let a3 = vld1q_s8(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = vld1q_s8(b.as_ptr().add(base + SIMD_WIDTH * 3));
        let a3_lo = vget_low_s8(a3);
        let a3_hi = vget_high_s8(a3);
        let b3_lo = vget_low_s8(b3);
        let b3_hi = vget_high_s8(b3);
        let prod3_lo = vmull_s8(a3_lo, b3_lo);
        let prod3_hi = vmull_s8(a3_hi, b3_hi);
        sum3 = vpadalq_s16(sum3, prod3_lo);
        sum3 = vpadalq_s16(sum3, prod3_hi);
    }

    // Combine accumulators
    let sum01 = vaddq_s32(sum0, sum1);
    let sum23 = vaddq_s32(sum2, sum3);
    let mut sum_vec = vaddq_s32(sum01, sum23);

    // Tail SIMD chunks: process remaining full 16-byte vectors before scalar tail.
    // Helps dimensions like 127 (3 tail chunks) or 129 (0 tail chunks, 1 scalar byte).
    let tail_start = chunks * CHUNK_SIZE;
    let tail_chunks = (n - tail_start) / SIMD_WIDTH;
    for j in 0..tail_chunks {
        let base = tail_start + j * SIMD_WIDTH;
        let at = vld1q_s8(a.as_ptr().add(base));
        let bt = vld1q_s8(b.as_ptr().add(base));
        let at_lo = vget_low_s8(at);
        let at_hi = vget_high_s8(at);
        let bt_lo = vget_low_s8(bt);
        let bt_hi = vget_high_s8(bt);
        let pt_lo = vmull_s8(at_lo, bt_lo);
        let pt_hi = vmull_s8(at_hi, bt_hi);
        sum_vec = vpadalq_s16(sum_vec, pt_lo);
        sum_vec = vpadalq_s16(sum_vec, pt_hi);
    }

    // Horizontal sum
    let sum = vgetq_lane_s32(sum_vec, 0)
        + vgetq_lane_s32(sum_vec, 1)
        + vgetq_lane_s32(sum_vec, 2)
        + vgetq_lane_s32(sum_vec, 3);

    // Scalar tail: only the final < SIMD_WIDTH elements
    let remainder_start = tail_start + tail_chunks * SIMD_WIDTH;
    let remainder: i32 = a[remainder_start..]
        .iter()
        .zip(b[remainder_start..].iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum();

    (sum + remainder) as f32
}

/// Emulate `mm512_sign_epi8(b, a)` which doesn't exist in AVX-512.
///
/// Returns: b[i] if a[i] > 0, -b[i] if a[i] < 0, 0 if a[i] == 0.
///
/// # Safety
/// Requires AVX-512BW.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
unsafe fn mm512_sign_epi8(b: __m512i, a: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let neg_b = _mm512_sub_epi8(zero, b);
    // mask where a < 0
    let mask_neg = _mm512_cmplt_epi8_mask(a, zero);
    // mask where a == 0
    let mask_zero = _mm512_cmpeq_epi8_mask(a, zero);
    // Start with b, replace with -b where a < 0
    let result = _mm512_mask_blend_epi8(mask_neg, b, neg_b);
    // Replace with 0 where a == 0
    _mm512_mask_blend_epi8(mask_zero, result, zero)
}

/// AVX-512 VNNI int8 dot product using _mm512_dpbusd_epi32.
///
/// Processes 256 int8s per iteration (4x64 with 4 accumulators).
/// Note: VNNI expects unsigned x signed, so we handle signs carefully.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F, AVX-512VNNI, and AVX-512BW (verified via `simd_config()`)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm512_loadu_si512` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder handled via safe slice iteration
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
unsafe fn dot_product_i8_avx512vnni(a: &[i8], b: &[i8]) -> f32 {
    const SIMD_WIDTH: usize = 64; // 64 int8s per 512-bit register
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(a.iter().all(|&v| v != i8::MIN));
    debug_assert!(b.iter().all(|&v| v != i8::MIN));
    let chunks = n / CHUNK_SIZE;

    // 4 independent int32 accumulators (16 int32s each)
    let mut sum0 = _mm512_setzero_si512();
    let mut sum1 = _mm512_setzero_si512();
    let mut sum2 = _mm512_setzero_si512();
    let mut sum3 = _mm512_setzero_si512();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        // VNNI: dpbusd computes sum += a[unsigned] * b[signed]
        // For signed * signed, we use: abs(a) * sign(b, a)
        let a0 = _mm512_loadu_si512(a.as_ptr().add(base) as *const __m512i);
        let b0 = _mm512_loadu_si512(b.as_ptr().add(base) as *const __m512i);
        let a0_abs = _mm512_abs_epi8(a0);
        let b0_signed = mm512_sign_epi8(b0, a0);
        sum0 = _mm512_dpbusd_epi32(sum0, a0_abs, b0_signed);

        let a1 = _mm512_loadu_si512(a.as_ptr().add(base + SIMD_WIDTH) as *const __m512i);
        let b1 = _mm512_loadu_si512(b.as_ptr().add(base + SIMD_WIDTH) as *const __m512i);
        let a1_abs = _mm512_abs_epi8(a1);
        let b1_signed = mm512_sign_epi8(b1, a1);
        sum1 = _mm512_dpbusd_epi32(sum1, a1_abs, b1_signed);

        let a2 = _mm512_loadu_si512(a.as_ptr().add(base + SIMD_WIDTH * 2) as *const __m512i);
        let b2 = _mm512_loadu_si512(b.as_ptr().add(base + SIMD_WIDTH * 2) as *const __m512i);
        let a2_abs = _mm512_abs_epi8(a2);
        let b2_signed = mm512_sign_epi8(b2, a2);
        sum2 = _mm512_dpbusd_epi32(sum2, a2_abs, b2_signed);

        let a3 = _mm512_loadu_si512(a.as_ptr().add(base + SIMD_WIDTH * 3) as *const __m512i);
        let b3 = _mm512_loadu_si512(b.as_ptr().add(base + SIMD_WIDTH * 3) as *const __m512i);
        let a3_abs = _mm512_abs_epi8(a3);
        let b3_signed = mm512_sign_epi8(b3, a3);
        sum3 = _mm512_dpbusd_epi32(sum3, a3_abs, b3_signed);
    }

    // Combine accumulators
    let sum01 = _mm512_add_epi32(sum0, sum1);
    let sum23 = _mm512_add_epi32(sum2, sum3);
    let sum_vec = _mm512_add_epi32(sum01, sum23);

    // Horizontal sum of 16 int32s
    let sum = _mm512_reduce_add_epi32(sum_vec);

    // Handle remainder with scalar
    let remainder_start = chunks * CHUNK_SIZE;
    let remainder: i32 = a[remainder_start..]
        .iter()
        .zip(b[remainder_start..].iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum();

    (sum + remainder) as f32
}

/// AVX2 int8 dot product with 4x unrolling.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 (verified via `simd_config()`)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm256_loadu_si256` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder handled via safe slice iteration
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_i8_avx2_unrolled(a: &[i8], b: &[i8]) -> f32 {
    const SIMD_WIDTH: usize = 32;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(a.iter().all(|&v| v != i8::MIN));
    debug_assert!(b.iter().all(|&v| v != i8::MIN));
    let chunks = n / CHUNK_SIZE;

    // 4 independent int32 accumulators
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();

    let ones = _mm256_set1_epi16(1);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        // Unroll 0
        let a0 = _mm256_loadu_si256(a.as_ptr().add(base) as *const __m256i);
        let b0 = _mm256_loadu_si256(b.as_ptr().add(base) as *const __m256i);
        let prod0 = _mm256_maddubs_epi16(_mm256_abs_epi8(a0), _mm256_sign_epi8(b0, a0));
        let prod0_32 = _mm256_madd_epi16(prod0, ones);
        sum0 = _mm256_add_epi32(sum0, prod0_32);

        // Unroll 1
        let a1 = _mm256_loadu_si256(a.as_ptr().add(base + SIMD_WIDTH) as *const __m256i);
        let b1 = _mm256_loadu_si256(b.as_ptr().add(base + SIMD_WIDTH) as *const __m256i);
        let prod1 = _mm256_maddubs_epi16(_mm256_abs_epi8(a1), _mm256_sign_epi8(b1, a1));
        let prod1_32 = _mm256_madd_epi16(prod1, ones);
        sum1 = _mm256_add_epi32(sum1, prod1_32);

        // Unroll 2
        let a2 = _mm256_loadu_si256(a.as_ptr().add(base + SIMD_WIDTH * 2) as *const __m256i);
        let b2 = _mm256_loadu_si256(b.as_ptr().add(base + SIMD_WIDTH * 2) as *const __m256i);
        let prod2 = _mm256_maddubs_epi16(_mm256_abs_epi8(a2), _mm256_sign_epi8(b2, a2));
        let prod2_32 = _mm256_madd_epi16(prod2, ones);
        sum2 = _mm256_add_epi32(sum2, prod2_32);

        // Unroll 3
        let a3 = _mm256_loadu_si256(a.as_ptr().add(base + SIMD_WIDTH * 3) as *const __m256i);
        let b3 = _mm256_loadu_si256(b.as_ptr().add(base + SIMD_WIDTH * 3) as *const __m256i);
        let prod3 = _mm256_maddubs_epi16(_mm256_abs_epi8(a3), _mm256_sign_epi8(b3, a3));
        let prod3_32 = _mm256_madd_epi16(prod3, ones);
        sum3 = _mm256_add_epi32(sum3, prod3_32);
    }

    // Combine accumulators
    let sum01 = _mm256_add_epi32(sum0, sum1);
    let sum23 = _mm256_add_epi32(sum2, sum3);
    let sum_vec = _mm256_add_epi32(sum01, sum23);

    // Horizontal sum
    let sum128_lo = _mm256_castsi256_si128(sum_vec);
    let sum128_hi = _mm256_extracti128_si256(sum_vec, 1);
    let sum128 = _mm_add_epi32(sum128_lo, sum128_hi);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let sum = _mm_cvtsi128_si32(sum32);

    // Handle remainder
    let remainder_start = chunks * CHUNK_SIZE;
    let remainder: i32 = a[remainder_start..]
        .iter()
        .zip(b[remainder_start..].iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum();

    (sum + remainder) as f32
}

// ============================================================================
// INT8 kernel dispatch cache (mirrors f32 DotKernel pattern in dot_product.rs)
// ============================================================================

/// INT8 dot-product kernel function pointer type.
pub type I8DotKernel = fn(&[i8], &[i8]) -> f32;

static I8_DOT_KERNEL: OnceLock<I8DotKernel> = OnceLock::new();

/// Return the cached INT8 dot-product kernel.
///
/// Callers that invoke INT8 dot product in a tight loop can hoist this call
/// outside the loop so the OnceLock check runs once, not per-iteration.
#[inline]
pub fn resolved_i8_dot_kernel() -> I8DotKernel {
    *I8_DOT_KERNEL.get_or_init(resolve_i8_dot_kernel)
}

fn resolve_i8_dot_kernel() -> I8DotKernel {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            return dot_product_i8_neon_kernel;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if config.avx512vnni_enabled {
                return dot_product_i8_avx512vnni_kernel;
            }
        }
        if config.avx2_enabled {
            return dot_product_i8_avx2_kernel;
        }
    }

    dot_product_i8_scalar_kernel
}

#[cfg(target_arch = "aarch64")]
fn dot_product_i8_neon_kernel(a: &[i8], b: &[i8]) -> f32 {
    // SAFETY: stored only when NEON was detected at init time (always true on aarch64).
    unsafe { dot_product_i8_neon_unrolled(a, b) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn dot_product_i8_avx512vnni_kernel(a: &[i8], b: &[i8]) -> f32 {
    debug_assert!(a.iter().all(|&v| v != i8::MIN));
    debug_assert!(b.iter().all(|&v| v != i8::MIN));
    // SAFETY: stored only when AVX-512F+VNNI+BW were detected at init time.
    unsafe { dot_product_i8_avx512vnni(a, b) }
}

#[cfg(target_arch = "x86_64")]
fn dot_product_i8_avx2_kernel(a: &[i8], b: &[i8]) -> f32 {
    debug_assert!(a.iter().all(|&v| v != i8::MIN));
    debug_assert!(b.iter().all(|&v| v != i8::MIN));
    // SAFETY: stored only when AVX2 was detected at init time.
    unsafe { dot_product_i8_avx2_unrolled(a, b) }
}

fn dot_product_i8_scalar_kernel(a: &[i8], b: &[i8]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum::<i32>() as f32
}

/// **Unstable**: raw SIMD INT8 hot path; signature and scaling semantics may change.
///
/// This is the hot-path function for HNSW quantized search. Unlike `dot_product_i8`,
/// it takes raw `&[i8]` slices and does NOT divide by scale factors -- the caller
/// handles scaling. This avoids allocating `QuantizedVector` wrappers.
///
/// Returns 0.0 if slices have different lengths.
///
/// # Performance
///
/// Uses the same SIMD paths as `dot_product_i8`:
/// - aarch64: NEON with 4x unrolling + tail SIMD chunks
/// - x86_64: AVX-512 VNNI > AVX2 > scalar
///
/// The key difference is zero allocation overhead: no `Vec<i8>`, no
/// `QuantizedVector`, no `QuantizationParams`. Just raw slices in, f32 out.
#[inline]
pub fn dot_product_i8_raw(a: &[i8], b: &[i8]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    debug_assert_eq!(a.len(), b.len());
    resolved_i8_dot_kernel()(a, b)
}

#[cfg(test)]
mod simd_parity_tests {
    use super::*;

    fn gen_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed ^ ((dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        (0..dim)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    .wrapping_add(i as u64);
                let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
                unit * 2.0 - 1.0
            })
            .collect()
    }

    // FP-034: NEON vs scalar parity for INT8 dot product.
    #[test]
    fn test_i8_neon_scalar_parity() {
        #[cfg(target_arch = "aarch64")]
        for dim in [7usize, 16, 64, 128, 384, 768] {
            let a_q = QuantizedVector::from_f32(&gen_vec(dim, 200 + dim as u64));
            let b_q = QuantizedVector::from_f32(&gen_vec(dim, 300 + dim as u64));

            // SAFETY: NEON is mandatory on aarch64; slices have equal length from from_f32.
            let neon = unsafe { dot_product_i8_neon_unrolled(&a_q.data, &b_q.data) };
            let scalar: f32 = a_q
                .data
                .iter()
                .zip(b_q.data.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>() as f32;

            let diff = (neon - scalar).abs();
            assert!(
                diff <= 1.0,
                "NEON vs scalar i8 dot product dim={dim}: neon={neon} scalar={scalar} diff={diff}"
            );
        }
    }

    // FP-034: AVX2 vs scalar parity for INT8 dot product.
    #[test]
    fn test_i8_avx2_scalar_parity() {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            for dim in [7usize, 16, 64, 128, 384, 768] {
                let a_q = QuantizedVector::from_f32(&gen_vec(dim, 400 + dim as u64));
                let b_q = QuantizedVector::from_f32(&gen_vec(dim, 500 + dim as u64));

                // SAFETY: AVX2 verified by is_x86_feature_detected! above; slices have equal length.
                let avx2 = unsafe { dot_product_i8_avx2_unrolled(&a_q.data, &b_q.data) };
                let scalar: f32 = a_q
                    .data
                    .iter()
                    .zip(b_q.data.iter())
                    .map(|(&x, &y)| x as i32 * y as i32)
                    .sum::<i32>() as f32;

                let diff = (avx2 - scalar).abs();
                assert!(
                    diff <= 1.0,
                    "AVX2 vs scalar i8 dot product dim={dim}: avx2={avx2} scalar={scalar} diff={diff}"
                );
            }
        }
    }
}
