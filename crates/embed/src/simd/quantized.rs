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
    #[cfg(target_arch = "x86_64")]
    {
        if simd_config().avx2_enabled {
            // SAFETY: AVX2 was detected at runtime; the kernel bounds every load.
            return unsafe { minmax_finite_avx2(v) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            // SAFETY: NEON confirmed available at runtime.
            return unsafe { minmax_finite_neon(v) };
        }
    }
    minmax_finite_scalar(v)
}

/// Resolves the sign of a zero-valued min/max so every kernel agrees bit-for-bit.
///
/// `f32::min`/`f32::max`, NEON's `vminvq_f32`/`vmaxvq_f32`, and AVX2's
/// `_mm256_min_ps`/`_mm256_max_ps` each return an unspecified one of their operands
/// when the operands compare equal, so a `-0.0`/`+0.0` tie is broken differently per
/// kernel and per codegen. Applying IEEE 754-2019 `minimum`/`maximum` ordering
/// (`-0.0 < +0.0`) to the finished pair makes the choice a stated contract instead of
/// an artifact: the min takes `-0.0` and the max takes `+0.0` whenever that sign is
/// present in the input.
///
/// The sign scan runs only when a bound is exactly zero, so the cost on a typical
/// vector is the two comparisons in the guard.
#[inline]
fn pin_zero_signs(v: &[f32], min_val: f32, max_val: f32) -> (f32, f32) {
    if min_val != 0.0 && max_val != 0.0 {
        return (min_val, max_val);
    }
    let mut has_negative_zero = false;
    let mut has_positive_zero = false;
    for &value in v {
        has_negative_zero |= value.to_bits() == (-0.0f32).to_bits();
        has_positive_zero |= value.to_bits() == 0.0f32.to_bits();
    }
    let min_val = if min_val == 0.0 {
        if has_negative_zero { -0.0 } else { 0.0 }
    } else {
        min_val
    };
    let max_val = if max_val == 0.0 {
        if has_positive_zero { 0.0 } else { -0.0 }
    } else {
        max_val
    };
    (min_val, max_val)
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
    pin_zero_signs(v, min_val, max_val)
}

#[cfg(test)]
thread_local! {
    static I8_MINMAX_SIMD_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn minmax_finite_avx2(v: &[f32]) -> (f32, f32) {
    #[cfg(test)]
    I8_MINMAX_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let chunks = v.len() / 8;
    let inf = _mm256_set1_ps(f32::INFINITY);
    let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
    let sign = _mm256_set1_ps(-0.0);
    let mut vmin = inf;
    let mut vmax = neg_inf;

    for i in 0..chunks {
        let x = _mm256_loadu_ps(v.as_ptr().add(i * 8));
        let abs = _mm256_andnot_ps(sign, x);
        let finite = _mm256_cmp_ps(abs, inf, _CMP_LT_OQ);
        vmin = _mm256_min_ps(vmin, _mm256_blendv_ps(inf, x, finite));
        vmax = _mm256_max_ps(vmax, _mm256_blendv_ps(neg_inf, x, finite));
    }

    let mut min_lanes = [0.0f32; 8];
    let mut max_lanes = [0.0f32; 8];
    _mm256_storeu_ps(min_lanes.as_mut_ptr(), vmin);
    _mm256_storeu_ps(max_lanes.as_mut_ptr(), vmax);

    let mut min_val = min_lanes.into_iter().fold(f32::INFINITY, f32::min);
    let mut max_val = max_lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &x in &v[chunks * 8..] {
        if x.is_finite() {
            min_val = min_val.min(x);
            max_val = max_val.max(x);
        }
    }
    pin_zero_signs(v, min_val, max_val)
}

#[cfg(target_arch = "aarch64")]
unsafe fn minmax_finite_neon(v: &[f32]) -> (f32, f32) {
    #[cfg(test)]
    I8_MINMAX_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

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
    pin_zero_signs(v, min_val, max_val)
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

        let data = quantize_i8(vector, params.scale);

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

fn quantize_i8(vector: &[f32], scale: f32) -> Vec<i8> {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_config().avx2_enabled {
            // SAFETY: AVX2 was detected at runtime; the kernel bounds every load and store.
            return unsafe { quantize_i8_avx2(vector, scale) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            // SAFETY: NEON was detected at runtime; the kernel bounds every load and store.
            return unsafe { quantize_i8_neon(vector, scale) };
        }
    }
    quantize_i8_scalar(vector, scale)
}

fn quantize_i8_scalar(vector: &[f32], scale: f32) -> Vec<i8> {
    vector
        .iter()
        .map(|&v| quantize_i8_value(v, scale))
        .collect()
}

#[inline]
fn quantize_i8_value(value: f32, scale: f32) -> i8 {
    if value.is_finite() {
        (value * scale).round().clamp(-127.0, 127.0) as i8
    } else {
        0
    }
}

#[cfg(test)]
thread_local! {
    static I8_QUANTIZE_SIMD_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_i8_avx2(vector: &[f32], scale: f32) -> Vec<i8> {
    #[cfg(test)]
    I8_QUANTIZE_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let mut data = vec![0i8; vector.len()];
    let chunks = vector.len() / 8;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let sign = _mm256_set1_ps(-0.0);
    let low = _mm256_set1_ps(-127.0);
    let high = _mm256_set1_ps(127.0);
    let half = _mm256_set1_ps(0.5);
    let negative_half = _mm256_set1_ps(-0.5);
    let one = _mm256_set1_epi32(1);
    let negative_one = _mm256_set1_epi32(-1);

    for i in 0..chunks {
        let base = i * 8;
        let input = _mm256_loadu_ps(vector.as_ptr().add(base));
        let abs = _mm256_andnot_ps(sign, input);
        let finite = _mm256_cmp_ps(abs, inf, _CMP_LT_OQ);
        let values = _mm256_and_ps(input, finite);
        let scaled = _mm256_mul_ps(values, scale);
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, low), high);
        let truncated = _mm256_cvttps_epi32(clamped);
        let fraction = _mm256_sub_ps(clamped, _mm256_cvtepi32_ps(truncated));
        let round_up = _mm256_castps_si256(_mm256_cmp_ps(fraction, half, _CMP_GE_OQ));
        let round_down = _mm256_castps_si256(_mm256_cmp_ps(fraction, negative_half, _CMP_LE_OQ));
        let rounded = _mm256_add_epi32(
            _mm256_add_epi32(truncated, _mm256_and_si256(round_up, one)),
            _mm256_and_si256(round_down, negative_one),
        );
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), rounded);
        for (offset, lane) in lanes.into_iter().enumerate() {
            data[base + offset] = lane as i8;
        }
    }

    for i in chunks * 8..vector.len() {
        data[i] = quantize_i8_value(vector[i], scale_scalar);
    }
    data
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_i8_neon(vector: &[f32], scale: f32) -> Vec<i8> {
    #[cfg(test)]
    I8_QUANTIZE_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let mut data = vec![0i8; vector.len()];
    let chunks = vector.len() / 4;
    let scale_vector = vdupq_n_f32(scale);
    let inf = vdupq_n_f32(f32::INFINITY);
    let zero = vdupq_n_f32(0.0);
    let low = vdupq_n_f32(-127.0);
    let high = vdupq_n_f32(127.0);

    for i in 0..chunks {
        let base = i * 4;
        let input = vld1q_f32(vector.as_ptr().add(base));
        let finite = vcaltq_f32(input, inf);
        let values = vbslq_f32(finite, input, zero);
        let scaled = vmulq_f32(values, scale_vector);
        let clamped = vminq_f32(vmaxq_f32(scaled, low), high);
        let rounded = vcvtaq_s32_f32(clamped);
        let mut lanes = [0i32; 4];
        vst1q_s32(lanes.as_mut_ptr(), rounded);
        for (offset, lane) in lanes.into_iter().enumerate() {
            data[base + offset] = lane as i8;
        }
    }

    for i in chunks * 4..vector.len() {
        data[i] = quantize_i8_value(vector[i], scale);
    }
    data
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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
        if config.avx512vnni_enabled {
            return dot_product_i8_avx512vnni_kernel;
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

#[cfg(target_arch = "x86_64")]
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

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_i8_quantize_explicit_simd_matches_scalar_and_is_dispatched() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
            let mut input = gen_vec(dim, 900 + dim as u64);
            if dim > 0 {
                input[0] = f32::NAN;
            }
            if dim > 1 {
                input[1] = f32::INFINITY;
            }
            if dim > 2 {
                input[2] = f32::NEG_INFINITY;
            }
            if dim > 3 {
                input[3] = 0.25;
            }
            if dim > 4 {
                input[4] = -0.25;
            }
            if dim > 5 {
                input[5] = f32::from_bits(0.25f32.to_bits() - 1);
            }
            if dim > 6 {
                input[6] = f32::from_bits(0.25f32.to_bits() + 1);
            }

            let scalar = quantize_i8_scalar(&input, 2.0);
            #[cfg(target_arch = "aarch64")]
            // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
            let simd = unsafe { quantize_i8_neon(&input, 2.0) };
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 was detected above; the kernel bounds every access.
            let simd = unsafe { quantize_i8_avx2(&input, 2.0) };
            assert_eq!(simd, scalar, "explicit SIMD mismatch at dim={dim}");
        }

        let input = gen_vec(385, 1_063);
        let before = I8_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
        let quantized = QuantizedVector::from_f32(&input);
        let after = I8_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
        assert_eq!(
            after,
            before + 1,
            "QuantizedVector::from_f32 did not execute its explicit SIMD quantizer"
        );
        assert_eq!(
            quantized.data,
            quantize_i8_scalar(&input, quantized.params.scale)
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_finite_minmax_explicit_simd_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
            let mut input = gen_vec(dim, 1_100 + dim as u64);
            if dim > 0 {
                input[0] = f32::NAN;
            }
            if dim > 1 {
                input[1] = f32::INFINITY;
            }
            if dim > 2 {
                input[2] = f32::NEG_INFINITY;
            }
            if dim > 3 {
                input[3] = -0.0;
            }
            if dim > 4 {
                input[4] = 0.0;
            }

            let scalar = minmax_finite_scalar(&input);
            #[cfg(target_arch = "aarch64")]
            // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
            let simd = unsafe { minmax_finite_neon(&input) };
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 was detected above; the kernel bounds every access.
            let simd = unsafe { minmax_finite_avx2(&input) };
            assert_eq!(
                (simd.0.to_bits(), simd.1.to_bits()),
                (scalar.0.to_bits(), scalar.1.to_bits()),
                "finite min/max mismatch at dim={dim}"
            );
        }

        let mut min_zero_input = vec![1.0f32; 16];
        min_zero_input[0] = -0.0;
        min_zero_input[8] = 0.0;
        let mut max_zero_input = vec![-1.0f32; 16];
        max_zero_input[0] = 0.0;
        max_zero_input[8] = -0.0;
        for input in [&min_zero_input, &max_zero_input] {
            let scalar = minmax_finite_scalar(input);
            #[cfg(target_arch = "aarch64")]
            // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
            let simd = unsafe { minmax_finite_neon(input) };
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 was detected above; the kernel bounds every access.
            let simd = unsafe { minmax_finite_avx2(input) };
            assert_eq!(
                (simd.0.to_bits(), simd.1.to_bits()),
                (scalar.0.to_bits(), scalar.1.to_bits()),
                "finite min/max signed-zero mismatch"
            );
        }

        let input = gen_vec(385, 1_063);
        let before = I8_MINMAX_SIMD_HITS.with(std::cell::Cell::get);
        let params = QuantizationParams::from_vector(&input);
        let after = I8_MINMAX_SIMD_HITS.with(std::cell::Cell::get);
        assert_eq!(
            after,
            before + 1,
            "QuantizationParams::from_vector did not execute its explicit SIMD reducer"
        );
        let scalar = minmax_finite_scalar(&input);
        assert_eq!(
            (params.min_val.to_bits(), params.max_val.to_bits()),
            (scalar.0.to_bits(), scalar.1.to_bits())
        );
    }

    /// The signed-zero tie-break is a contract, not whatever the reduction happened to
    /// pick. Parity tests can only compare kernels against each other, so they pass on
    /// any architecture whose kernels agree by accident; this pins the value itself and
    /// runs everywhere, including targets with no explicit SIMD path at all.
    ///
    /// The first block drives `pin_zero_signs` directly with bounds carrying the wrong
    /// sign. That matters: on aarch64 the scalar fold already returns the contracted
    /// signs, so assertions routed through `minmax_finite_scalar` alone still pass with
    /// the sign pass deleted. Only x86-64 breaks the tie the other way, and a guard that
    /// can be removed without any local test noticing is not a guard.
    #[test]
    fn test_minmax_finite_pins_zero_signs() {
        let both_signs = [-0.0f32, 0.0, -1.0];
        assert_eq!(
            pin_zero_signs(&both_signs, -1.0, -0.0).1.to_bits(),
            0.0f32.to_bits(),
            "a max of -0.0 must be rewritten to +0.0 when +0.0 is present"
        );
        assert_eq!(
            pin_zero_signs(&both_signs, 0.0, -1.0).0.to_bits(),
            (-0.0f32).to_bits(),
            "a min of +0.0 must be rewritten to -0.0 when -0.0 is present"
        );

        let max_ties = [-0.0f32, 0.0, -1.0];
        let (_, max_val) = minmax_finite_scalar(&max_ties);
        assert_eq!(
            max_val.to_bits(),
            0.0f32.to_bits(),
            "max must take +0.0 when both zero signs are present"
        );

        let min_ties = [0.0f32, -0.0, 1.0];
        let (min_val, _) = minmax_finite_scalar(&min_ties);
        assert_eq!(
            min_val.to_bits(),
            (-0.0f32).to_bits(),
            "min must take -0.0 when both zero signs are present"
        );

        // A bound that is zero with only one sign available keeps that sign.
        let (only_neg_min, only_neg_max) = minmax_finite_scalar(&[-0.0f32, -1.0]);
        assert_eq!(only_neg_max.to_bits(), (-0.0f32).to_bits());
        assert_eq!(only_neg_min.to_bits(), (-1.0f32).to_bits());
        let (only_pos_min, only_pos_max) = minmax_finite_scalar(&[0.0f32, 1.0]);
        assert_eq!(only_pos_min.to_bits(), 0.0f32.to_bits());
        assert_eq!(only_pos_max.to_bits(), 1.0f32.to_bits());

        // Non-zero bounds are returned untouched, and an all-nonfinite input keeps the
        // identity pair rather than being rewritten by the sign pass.
        assert_eq!(
            minmax_finite_scalar(&[]),
            (f32::INFINITY, f32::NEG_INFINITY)
        );
        assert_eq!(
            minmax_finite_scalar(&[f32::NAN, f32::INFINITY]),
            (f32::INFINITY, f32::NEG_INFINITY)
        );
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

    // FP-034: AVX-512 VNNI vs scalar parity for INT8 dot product.
    // Compiled unconditionally on x86_64 (the kernel is no longer feature-gated);
    // runs only when the host advertises AVX-512F+BW+VNNI, matching the kernel's
    // safety contract and SimdConfig::avx512vnni_enabled.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_i8_avx512vnni_scalar_parity() {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            for dim in [7usize, 16, 64, 128, 384, 768] {
                let a_q = QuantizedVector::from_f32(&gen_vec(dim, 600 + dim as u64));
                let b_q = QuantizedVector::from_f32(&gen_vec(dim, 700 + dim as u64));

                // SAFETY: AVX-512F+BW+VNNI verified above; slices have equal length from from_f32.
                let vnni = unsafe { dot_product_i8_avx512vnni(&a_q.data, &b_q.data) };
                let scalar: f32 = a_q
                    .data
                    .iter()
                    .zip(b_q.data.iter())
                    .map(|(&x, &y)| x as i32 * y as i32)
                    .sum::<i32>() as f32;

                let diff = (vnni - scalar).abs();
                assert!(
                    diff <= 1.0,
                    "VNNI vs scalar i8 dot product dim={dim}: vnni={vnni} scalar={scalar} diff={diff}"
                );
            }
        }
    }
}
