//! INT8 vector quantization and approximate similarity kernels.
//!
//! Constructor-owned values preserve the SIMD range invariant.
//!
//! See docs/simd.md for the encoding, error model, and dispatch strategy.

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
        let (mut min_val, mut max_val) = minmax_finite(vector);

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

/// Minimum and maximum over the finite lanes of `v`, in one pass.
///
/// Non-finite lanes are skipped. An empty or all-non-finite input yields
/// `(+inf, -inf)` so the caller's reset branch fires.
///
/// Explicit NEON kernel with a scalar fallback rather than a plain guarded
/// loop: the auto-vectorized form of this reduction was demoted to scalar
/// code by an unrelated codegen-unit reshuffle (+48% on per-call int8
/// quantization at 1024 dims), so the hot path must not depend on the
/// auto-vectorizer's mood.
fn minmax_finite(v: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            // SAFETY: NEON confirmed available at runtime.
            return unsafe { minmax_finite_neon(v) };
        }
    }
    minmax_finite_scalar(v)
}

fn minmax_finite_scalar(v: &[f32]) -> (f32, f32) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &x in v {
        if x.is_finite() {
            min_val = min_val.min(x);
            max_val = max_val.max(x);
        }
    }
    (min_val, max_val)
}

#[cfg(target_arch = "aarch64")]
unsafe fn minmax_finite_neon(v: &[f32]) -> (f32, f32) {
    let chunks = v.len() / 4;
    let inf = unsafe { vdupq_n_f32(f32::INFINITY) };
    let neg_inf = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    let mut vmin = inf;
    let mut vmax = neg_inf;
    for i in 0..chunks {
        // SAFETY: `i * 4 + 3 < v.len()` by the chunk bound.
        let x = unsafe { vld1q_f32(v.as_ptr().add(i * 4)) };
        unsafe {
            // `|x| < +inf` holds exactly for finite lanes (false for NaN, ±inf),
            // so masked lanes contribute identity elements, same as skipping.
            let finite = vcaltq_f32(x, inf);
            vmin = vminq_f32(vmin, vbslq_f32(finite, x, inf));
            vmax = vmaxq_f32(vmax, vbslq_f32(finite, x, neg_inf));
        }
    }
    let (mut min_val, mut max_val) = unsafe { (vminvq_f32(vmin), vmaxvq_f32(vmax)) };
    for &x in &v[chunks * 4..] {
        if x.is_finite() {
            min_val = min_val.min(x);
            max_val = max_val.max(x);
        }
    }
    (min_val, max_val)
}

/// **Unstable**: INT8 quantized vector; struct layout and invariants may change.
///
/// Quantized int8 vector with its parameters.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// Invariant: all values in `[-127, 127]`. Enforced by `from_f32` clamping.
    /// Private — the invariant makes release-mode assert scans unnecessary.
    data: Vec<i8>,
    /// **Unstable**: quantization parameters; may be separated from the vector.
    pub params: QuantizationParams,
    /// **Unstable**: L2 norm; may be removed or moved.
    pub norm: f32,
}

impl QuantizedVector {
    /// Returns the quantized data as a slice. All values are in `[-127, 127]`.
    #[inline]
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Returns the number of quantized elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the quantized vector has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
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

    /// **Unstable**: dequantizes this vector using its stored scale.
    ///
    /// See [`docs/simd.md`](../../docs/simd.md#int8-vectors) for the encoding and error bounds.
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

/// **Unstable**: computes an approximate float dot product, returning `0.0` for a mismatch.
///
/// Inputs satisfy the constructor-owned `[-127, 127]` invariant.
/// See [`docs/simd.md`](../../docs/simd.md#raw-int8-input-invariant) for its SIMD requirement.
#[inline]
pub fn dot_product_i8(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    debug_assert!(a.data.iter().all(|&v| v != -128i8));
    debug_assert!(b.data.iter().all(|&v| v != -128i8));

    if a.data.len() != b.data.len() {
        return 0.0;
    }

    let denom = a.params.scale * b.params.scale;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }

    dot_product_i8_dispatch(&a.data, &b.data) / denom
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
    dot_product_i8_dispatch(&a.data, &b.data) / denom
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

/// Computes INT8 cosine similarity for constructor-owned vectors without a release scan.
///
/// See [`docs/simd.md`](../../docs/simd.md#raw-int8-input-invariant) for the trusted-path precondition.
#[inline]
pub(crate) fn cosine_similarity_i8_trusted(a: &QuantizedVector, b: &QuantizedVector) -> f32 {
    let denom = a.norm * b.norm;
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    dot_product_i8_trusted(a, b) / denom
}

/// Computes an INT8 dot product with FEAT_DotProd and guarded prefetch.
///
/// # Safety
/// Caller must provide FEAT_DotProd, equal `[-127, 127]` slices, and bounded prefetches.
/// See [`docs/simd.md`](../../docs/simd.md#int8-vectors) for dispatch and implementation details.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_product_i8_neon_unrolled(a: &[i8], b: &[i8]) -> f32 {
    const SIMD_WIDTH: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    const PREFETCH_DISTANCE: usize = CHUNK_SIZE;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    let mut sum2 = vdupq_n_s32(0);
    let mut sum3 = vdupq_n_s32(0);

    // SDOT is selected only after runtime FEAT_DotProd detection — see docs/simd.md.
    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let next_base = base + PREFETCH_DISTANCE;
        if next_base + CHUNK_SIZE <= n {
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) a.as_ptr().add(next_base),
                options(nostack, readonly, preserves_flags)
            );
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) b.as_ptr().add(next_base),
                options(nostack, readonly, preserves_flags)
            );
        }

        let a0 = vld1q_s8(a.as_ptr().add(base));
        let b0 = vld1q_s8(b.as_ptr().add(base));
        let a1 = vld1q_s8(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = vld1q_s8(b.as_ptr().add(base + SIMD_WIDTH));
        let a2 = vld1q_s8(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = vld1q_s8(b.as_ptr().add(base + SIMD_WIDTH * 2));
        let a3 = vld1q_s8(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = vld1q_s8(b.as_ptr().add(base + SIMD_WIDTH * 3));

        core::arch::asm!(
            "sdot {s0:v}.4s, {a0:v}.16b, {b0:v}.16b",
            "sdot {s1:v}.4s, {a1:v}.16b, {b1:v}.16b",
            "sdot {s2:v}.4s, {a2:v}.16b, {b2:v}.16b",
            "sdot {s3:v}.4s, {a3:v}.16b, {b3:v}.16b",
            s0 = inout(vreg) sum0,
            a0 = in(vreg) a0,
            b0 = in(vreg) b0,
            s1 = inout(vreg) sum1,
            a1 = in(vreg) a1,
            b1 = in(vreg) b1,
            s2 = inout(vreg) sum2,
            a2 = in(vreg) a2,
            b2 = in(vreg) b2,
            s3 = inout(vreg) sum3,
            a3 = in(vreg) a3,
            b3 = in(vreg) b3,
            options(nomem, nostack, preserves_flags)
        );
    }

    let sum01 = vaddq_s32(sum0, sum1);
    let sum23 = vaddq_s32(sum2, sum3);
    let mut sum_vec = vaddq_s32(sum01, sum23);

    // Tail: remaining full 16-byte vectors using sdot
    let tail_start = chunks * CHUNK_SIZE;
    let tail_chunks = (n - tail_start) / SIMD_WIDTH;
    for j in 0..tail_chunks {
        let base = tail_start + j * SIMD_WIDTH;
        let at = vld1q_s8(a.as_ptr().add(base));
        let bt = vld1q_s8(b.as_ptr().add(base));
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) sum_vec,
            a = in(vreg) at,
            b = in(vreg) bt,
            options(nomem, nostack, preserves_flags)
        );
    }

    let sum = vaddvq_s32(sum_vec);

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

/// Computes an INT8 dot product with AVX-512 VNNI.
///
/// # Safety
/// Caller must provide AVX-512F/VNNI/BW and equal `[-127, 127]` slices; bounds are chunked.
/// See [`docs/simd.md`](../../docs/simd.md#int8-vectors) for signed-product transformation details.
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

/// Computes an INT8 dot product with AVX2 and guarded prefetch.
///
/// # Safety
/// Caller must provide AVX2 and equal `[-127, 127]` slices; bounds and prefetch are guarded.
/// See [`docs/simd.md`](../../docs/simd.md#int8-vectors) for signed-product transformation details.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_i8_avx2_unrolled(a: &[i8], b: &[i8]) -> f32 {
    const SIMD_WIDTH: usize = 32;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    // Prefetch one full chunk ahead for both input arrays.
    const PREFETCH_DISTANCE: usize = CHUNK_SIZE;
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

        // Software prefetch for the next chunk.
        let next_base = base + PREFETCH_DISTANCE;
        if next_base + CHUNK_SIZE <= n {
            _mm_prefetch(a.as_ptr().add(next_base), _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(next_base), _MM_HINT_T0);
        }

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

/// Return the cached INT8 dot-product kernel for tight loops.
#[inline]
pub fn resolved_i8_dot_kernel() -> I8DotKernel {
    *I8_DOT_KERNEL.get_or_init(resolve_i8_dot_kernel)
}

fn resolve_i8_dot_kernel() -> I8DotKernel {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        // The NEON kernel uses SDOT (ARMv8.2 FEAT_DotProd), which is optional on
        // Armv8.2/v8.3. Only dispatch when dotprod_enabled is confirmed at runtime.
        if config.neon_enabled && config.dotprod_enabled {
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
    // SAFETY: stored only when NEON+dotprod detected at init time.
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

/// Dispatch a validated raw INT8 dot product.
#[inline]
fn dot_product_i8_dispatch(a: &[i8], b: &[i8]) -> f32 {
    resolved_i8_dot_kernel()(a, b)
}

/// **Unstable**: computes an unscaled raw INT8 dot product, returning `0.0` for a mismatch.
///
/// Every value must lie in `[-127, 127]`; `i8::MIN` is numerically invalid.
/// See [`docs/simd.md`](../../docs/simd.md#raw-int8-input-invariant) for the release-mode precondition.
#[inline]
pub fn dot_product_i8_raw(a: &[i8], b: &[i8]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    debug_assert!(
        a.iter().all(|&v| v != -128i8),
        "dot_product_i8_raw: slice a contains -128, violating the [-127, 127] SIMD invariant"
    );
    debug_assert!(
        b.iter().all(|&v| v != -128i8),
        "dot_product_i8_raw: slice b contains -128, violating the [-127, 127] SIMD invariant"
    );
    dot_product_i8_dispatch(a, b)
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

    // FP-034: NEON SDOT vs scalar parity for INT8 dot product.
    // Gated on dotprod: SDOT is FEAT_DotProd, not baseline NEON.
    #[test]
    fn test_i8_neon_scalar_parity() {
        #[cfg(target_arch = "aarch64")]
        {
            if !super::super::SimdConfig::detect().dotprod_enabled {
                eprintln!("skipping SDOT parity test: dotprod not available");
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        for dim in [7usize, 16, 64, 128, 384, 768] {
            let a_q = QuantizedVector::from_f32(&gen_vec(dim, 200 + dim as u64));
            let b_q = QuantizedVector::from_f32(&gen_vec(dim, 300 + dim as u64));

            // SAFETY: dotprod confirmed above; slices have equal length from from_f32.
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
