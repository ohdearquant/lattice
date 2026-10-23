//! INT4 packed-vector quantization and approximate dot products.
//!
//! Nibble layout and offset correction are shared by scalar and NEON paths.
//!
//! See docs/simd.md for the packed format and corrected dot-product derivation.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::simd_config;

/// **Unstable**: INT4 quantization internals; scale/bias scheme may change.
///
/// Quantization parameters for INT4 conversion.
///
/// Uses symmetric unsigned quantization: the float range [-max_abs, max_abs]
/// is mapped to the integer range [0, 15].
#[derive(Debug, Clone, Copy)]
pub struct Int4Params {
    /// **Unstable**: scale factor; formula may change with quantization scheme update.
    pub scale: f32,
    /// **Unstable**: maximum absolute value; field may be removed.
    pub max_abs: f32,
}

impl Int4Params {
    /// **Unstable**: quantization parameter computation; may be folded into `Int4Vector::from_f32`.
    pub fn from_vector(vector: &[f32]) -> Self {
        let max_abs = max_abs_finite(vector);

        // Epsilon guard: avoid division by near-zero
        let scale = if max_abs > 1e-10 {
            15.0 / (2.0 * max_abs)
        } else {
            1.0
        };

        Self { scale, max_abs }
    }
}

fn max_abs_finite(vector: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_config().avx2_enabled {
            // SAFETY: AVX2 was detected at runtime; the kernel bounds every load.
            return unsafe { max_abs_finite_avx2(vector) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            // SAFETY: NEON was detected at runtime; the kernel bounds every load.
            return unsafe { max_abs_finite_neon(vector) };
        }
    }
    max_abs_finite_scalar(vector)
}

fn max_abs_finite_scalar(vector: &[f32]) -> f32 {
    vector
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .fold(0.0, f32::max)
}

#[cfg(test)]
thread_local! {
    static INT4_MAX_ABS_SIMD_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_abs_finite_avx2(vector: &[f32]) -> f32 {
    #[cfg(test)]
    INT4_MAX_ABS_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let chunks = vector.len() / 8;
    let sign = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let mut maximum = _mm256_setzero_ps();

    for i in 0..chunks {
        let input = _mm256_loadu_ps(vector.as_ptr().add(i * 8));
        let abs = _mm256_andnot_ps(sign, input);
        let finite = _mm256_cmp_ps(abs, inf, _CMP_LT_OQ);
        maximum = _mm256_max_ps(maximum, _mm256_and_ps(abs, finite));
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), maximum);
    let mut max_abs = lanes.into_iter().fold(0.0f32, f32::max);
    for &value in &vector[chunks * 8..] {
        if value.is_finite() {
            max_abs = max_abs.max(value.abs());
        }
    }
    max_abs
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn max_abs_finite_neon(vector: &[f32]) -> f32 {
    #[cfg(test)]
    INT4_MAX_ABS_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let chunks = vector.len() / 4;
    let inf = vdupq_n_f32(f32::INFINITY);
    let zero = vdupq_n_f32(0.0);
    let mut maximum = zero;

    for i in 0..chunks {
        let input = vld1q_f32(vector.as_ptr().add(i * 4));
        let abs = vabsq_f32(input);
        let finite = vcltq_f32(abs, inf);
        maximum = vmaxq_f32(maximum, vbslq_f32(finite, abs, zero));
    }

    let mut max_abs = vmaxvq_f32(maximum);
    for &value in &vector[chunks * 4..] {
        if value.is_finite() {
            max_abs = max_abs.max(value.abs());
        }
    }
    max_abs
}

/// **Unstable**: INT4 quantization format is under active design; struct layout may change.
///
/// Quantized INT4 vector with packed nibble storage.
#[derive(Debug, Clone)]
pub struct Int4Vector {
    /// **Unstable**: packed nibble data; bit packing scheme may change.
    pub data: Vec<u8>,
    /// **Unstable**: number of original dimensions.
    pub dims: usize,
    /// **Unstable**: quantization parameters; may be separated from the vector.
    pub params: Int4Params,
    /// **Unstable**: L2 norm; may be removed or moved.
    pub norm: f32,
}

impl Int4Vector {
    /// **Unstable**: quantization format; nibble packing may change.
    ///
    /// Each pair of consecutive dimensions is packed into one byte:
    /// - High nibble (bits 7..4) = even-indexed value
    /// - Low nibble (bits 3..0) = odd-indexed value
    pub fn from_f32(vector: &[f32]) -> Self {
        let params = Int4Params::from_vector(vector);
        let dims = vector.len();

        // Compute L2 norm
        let mut norm_sq = 0.0f32;
        for &v in vector {
            if v.is_finite() {
                norm_sq += v * v;
            }
        }
        let norm = norm_sq.sqrt();

        let data = quantize_int4(vector, params);

        Self {
            data,
            dims,
            params,
            norm,
        }
    }

    /// **Unstable**: dequantizes packed INT4 data, or returns empty for a malformed buffer.
    ///
    /// See [`docs/simd.md`](../../docs/simd.md#int4-vectors) for format and precision bounds.
    pub fn to_f32(&self) -> Vec<f32> {
        let required_bytes = self.dims.div_ceil(2);
        if self.data.len() < required_bytes {
            return Vec::new();
        }

        let scale = if self.params.scale.is_finite() && self.params.scale != 0.0 {
            self.params.scale
        } else {
            1.0
        };

        let mut result = Vec::with_capacity(self.dims);
        for i in 0..self.dims {
            let byte_idx = i / 2;
            let q = if i % 2 == 0 {
                (self.data[byte_idx] >> 4) & 0x0F
            } else {
                self.data[byte_idx] & 0x0F
            };
            result.push(q as f32 / scale - self.params.max_abs);
        }
        result
    }

    /// **Unstable**: INT4 dot product approximation; formula may change.
    ///
    /// Returns the dequantized dot product suitable for cosine distance computation.
    #[inline]
    pub fn dot_product(&self, other: &Int4Vector) -> f32 {
        dot_product_int4(self, other)
    }

    /// **Unstable**: INT4 cosine similarity approximation; delegates to `dot_product`.
    #[inline]
    pub fn cosine_similarity(&self, other: &Int4Vector) -> f32 {
        let denom = self.norm * other.norm;
        if denom == 0.0 || !denom.is_finite() {
            return 0.0;
        }
        self.dot_product(other) / denom
    }

    /// **Unstable**: complement of `cosine_similarity`; definition may evolve.
    #[inline]
    pub fn cosine_distance(&self, other: &Int4Vector) -> f32 {
        1.0 - self.cosine_similarity(other)
    }
}

fn quantize_int4(vector: &[f32], params: Int4Params) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_config().avx2_enabled {
            // SAFETY: AVX2 was detected at runtime; the kernel bounds every load and store.
            return unsafe { quantize_int4_avx2(vector, params) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            // SAFETY: NEON was detected at runtime; the kernel bounds every load and store.
            return unsafe { quantize_int4_neon(vector, params) };
        }
    }
    quantize_int4_scalar(vector, params)
}

fn quantize_int4_scalar(vector: &[f32], params: Int4Params) -> Vec<u8> {
    let mut data = vec![0u8; vector.len().div_ceil(2)];
    quantize_int4_scalar_tail(vector, params, &mut data, 0);
    data
}

fn quantize_int4_scalar_tail(vector: &[f32], params: Int4Params, data: &mut [u8], start: usize) {
    for (i, &value) in vector.iter().enumerate().skip(start) {
        let quantized = quantize_int4_value(value, params);
        if i % 2 == 0 {
            data[i / 2] |= quantized << 4;
        } else {
            data[i / 2] |= quantized;
        }
    }
}

#[inline]
fn quantize_int4_value(value: f32, params: Int4Params) -> u8 {
    let finite_value = if value.is_finite() { value } else { 0.0 };
    ((finite_value + params.max_abs) * params.scale)
        .round()
        .clamp(0.0, 15.0) as u8
}

#[cfg(test)]
thread_local! {
    static INT4_QUANTIZE_SIMD_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_int4_avx2(vector: &[f32], params: Int4Params) -> Vec<u8> {
    #[cfg(test)]
    INT4_QUANTIZE_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let mut data = vec![0u8; vector.len().div_ceil(2)];
    let chunks = vector.len() / 8;
    let sign = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let max_abs = _mm256_set1_ps(params.max_abs);
    let scale = _mm256_set1_ps(params.scale);
    let zero = _mm256_setzero_ps();
    let high = _mm256_set1_ps(15.0);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_epi32(1);

    for i in 0..chunks {
        let base = i * 8;
        let input = _mm256_loadu_ps(vector.as_ptr().add(base));
        let abs = _mm256_andnot_ps(sign, input);
        let finite = _mm256_cmp_ps(abs, inf, _CMP_LT_OQ);
        let values = _mm256_and_ps(input, finite);
        let scaled = _mm256_mul_ps(_mm256_add_ps(values, max_abs), scale);
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero), high);
        let truncated = _mm256_cvttps_epi32(clamped);
        let fraction = _mm256_sub_ps(clamped, _mm256_cvtepi32_ps(truncated));
        let round_up = _mm256_castps_si256(_mm256_cmp_ps(fraction, half, _CMP_GE_OQ));
        let rounded = _mm256_add_epi32(truncated, _mm256_and_si256(round_up, one));
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), rounded);
        for pair in 0..4 {
            data[base / 2 + pair] = ((lanes[pair * 2] as u8) << 4) | lanes[pair * 2 + 1] as u8;
        }
    }

    quantize_int4_scalar_tail(vector, params, &mut data, chunks * 8);
    data
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_int4_neon(vector: &[f32], params: Int4Params) -> Vec<u8> {
    #[cfg(test)]
    INT4_QUANTIZE_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let mut data = vec![0u8; vector.len().div_ceil(2)];
    let chunks = vector.len() / 4;
    let inf = vdupq_n_f32(f32::INFINITY);
    let zero = vdupq_n_f32(0.0);
    let max_abs = vdupq_n_f32(params.max_abs);
    let scale = vdupq_n_f32(params.scale);
    let high = vdupq_n_f32(15.0);

    for i in 0..chunks {
        let base = i * 4;
        let input = vld1q_f32(vector.as_ptr().add(base));
        let finite = vcaltq_f32(input, inf);
        let values = vbslq_f32(finite, input, zero);
        let scaled = vmulq_f32(vaddq_f32(values, max_abs), scale);
        let clamped = vminq_f32(vmaxq_f32(scaled, zero), high);
        let rounded = vcvtaq_s32_f32(clamped);
        let mut lanes = [0i32; 4];
        vst1q_s32(lanes.as_mut_ptr(), rounded);
        data[base / 2] = ((lanes[0] as u8) << 4) | lanes[1] as u8;
        data[base / 2 + 1] = ((lanes[2] as u8) << 4) | lanes[3] as u8;
    }

    quantize_int4_scalar_tail(vector, params, &mut data, chunks * 4);
    data
}

/// **Unstable**: dequantized INT4 dot product; dispatch may change.
#[inline]
pub fn dot_product_int4(a: &Int4Vector, b: &Int4Vector) -> f32 {
    if a.dims != b.dims {
        return 0.0;
    }

    let scale_a = a.params.scale;
    let scale_b = b.params.scale;
    if scale_a == 0.0 || scale_b == 0.0 || !scale_a.is_finite() || !scale_b.is_finite() {
        return 0.0;
    }

    let packed_len = a.dims.div_ceil(2);
    if a.data.len() < packed_len || b.data.len() < packed_len {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let config = simd_config();
        if config.neon_enabled {
            // SAFETY: aarch64 NEON is available by config, the packed data length guard
            // above prevents out-of-bounds loads, and the callee handles odd dimensions
            // without reading the padding nibble as a real dimension.
            let (raw_dot, sum_a, sum_b) =
                unsafe { dot_product_int4_neon_unrolled(&a.data, &b.data, a.dims) };
            return finish_int4_dot(raw_dot, sum_a, sum_b, a, b);
        }
    }

    let (raw_dot, sum_a, sum_b) = dot_product_int4_packed_scalar(&a.data, &b.data, a.dims);
    finish_int4_dot(raw_dot, sum_a, sum_b, a, b)
}

#[inline]
fn finish_int4_dot(raw_dot: i32, sum_a: i32, sum_b: i32, a: &Int4Vector, b: &Int4Vector) -> f32 {
    let raw_dot = raw_dot as f32;
    let sum_a = sum_a as f32;
    let sum_b = sum_b as f32;
    let scale_a = a.params.scale;
    let scale_b = b.params.scale;

    raw_dot / (scale_a * scale_b)
        - (b.params.max_abs * sum_a / scale_a)
        - (a.params.max_abs * sum_b / scale_b)
        + (a.dims as f32 * a.params.max_abs * b.params.max_abs)
}

#[inline]
fn dot_product_int4_packed_scalar(a: &[u8], b: &[u8], dims: usize) -> (i32, i32, i32) {
    let full_bytes = dims / 2;
    let mut raw_dot = 0i32;
    let mut sum_a = 0i32;
    let mut sum_b = 0i32;

    for i in 0..full_bytes {
        let av = a[i];
        let bv = b[i];
        let ah = ((av >> 4) & 0x0f) as i32;
        let al = (av & 0x0f) as i32;
        let bh = ((bv >> 4) & 0x0f) as i32;
        let bl = (bv & 0x0f) as i32;
        raw_dot += ah * bh + al * bl;
        sum_a += ah + al;
        sum_b += bh + bl;
    }

    if dims % 2 == 1 {
        let av = a[full_bytes];
        let bv = b[full_bytes];
        let ah = ((av >> 4) & 0x0f) as i32;
        let bh = ((bv >> 4) & 0x0f) as i32;
        raw_dot += ah * bh;
        sum_a += ah;
        sum_b += bh;
    }

    (raw_dot, sum_a, sum_b)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_product_int4_neon_unrolled(a: &[u8], b: &[u8], dims: usize) -> (i32, i32, i32) {
    debug_assert!(a.len() >= dims.div_ceil(2));
    debug_assert!(b.len() >= dims.div_ceil(2));

    const BLOCK_BYTES: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_BYTES: usize = BLOCK_BYTES * UNROLL;

    // Only bytes containing two valid dimensions are processed in SIMD.
    // If dims is odd, the final high nibble is handled separately and the low
    // padding nibble is ignored to preserve current to_f32 semantics.
    let full_bytes = dims / 2;
    let chunks = full_bytes / CHUNK_BYTES;

    let mut raw0 = vdupq_n_u32(0);
    let mut raw1 = vdupq_n_u32(0);
    let mut raw2 = vdupq_n_u32(0);
    let mut raw3 = vdupq_n_u32(0);
    let mut sum_a = vdupq_n_u32(0);
    let mut sum_b = vdupq_n_u32(0);
    let mask = vdupq_n_u8(0x0f);

    macro_rules! accumulate_block {
        ($base:expr, $raw:ident) => {{
            let a_bytes = vld1q_u8(a.as_ptr().add($base));
            let b_bytes = vld1q_u8(b.as_ptr().add($base));

            let a_hi = vshrq_n_u8::<4>(a_bytes);
            let b_hi = vshrq_n_u8::<4>(b_bytes);
            let a_lo = vandq_u8(a_bytes, mask);
            let b_lo = vandq_u8(b_bytes, mask);

            $raw = vpadalq_u16($raw, vmull_u8(vget_low_u8(a_hi), vget_low_u8(b_hi)));
            $raw = vpadalq_u16($raw, vmull_u8(vget_high_u8(a_hi), vget_high_u8(b_hi)));
            $raw = vpadalq_u16($raw, vmull_u8(vget_low_u8(a_lo), vget_low_u8(b_lo)));
            $raw = vpadalq_u16($raw, vmull_u8(vget_high_u8(a_lo), vget_high_u8(b_lo)));

            sum_a = vpadalq_u16(sum_a, vpaddlq_u8(a_hi));
            sum_a = vpadalq_u16(sum_a, vpaddlq_u8(a_lo));
            sum_b = vpadalq_u16(sum_b, vpaddlq_u8(b_hi));
            sum_b = vpadalq_u16(sum_b, vpaddlq_u8(b_lo));
        }};
    }

    for i in 0..chunks {
        let base = i * CHUNK_BYTES;
        accumulate_block!(base, raw0);
        accumulate_block!(base + BLOCK_BYTES, raw1);
        accumulate_block!(base + BLOCK_BYTES * 2, raw2);
        accumulate_block!(base + BLOCK_BYTES * 3, raw3);
    }

    let raw_vec = vaddq_u32(vaddq_u32(raw0, raw1), vaddq_u32(raw2, raw3));
    let mut raw_total = (vgetq_lane_u32::<0>(raw_vec)
        + vgetq_lane_u32::<1>(raw_vec)
        + vgetq_lane_u32::<2>(raw_vec)
        + vgetq_lane_u32::<3>(raw_vec)) as i32;
    let mut sum_a_total = (vgetq_lane_u32::<0>(sum_a)
        + vgetq_lane_u32::<1>(sum_a)
        + vgetq_lane_u32::<2>(sum_a)
        + vgetq_lane_u32::<3>(sum_a)) as i32;
    let mut sum_b_total = (vgetq_lane_u32::<0>(sum_b)
        + vgetq_lane_u32::<1>(sum_b)
        + vgetq_lane_u32::<2>(sum_b)
        + vgetq_lane_u32::<3>(sum_b)) as i32;

    let remainder_start = chunks * CHUNK_BYTES;
    for byte_idx in remainder_start..full_bytes {
        let av = *a.get_unchecked(byte_idx);
        let bv = *b.get_unchecked(byte_idx);
        let ah = ((av >> 4) & 0x0f) as i32;
        let al = (av & 0x0f) as i32;
        let bh = ((bv >> 4) & 0x0f) as i32;
        let bl = (bv & 0x0f) as i32;

        raw_total += ah * bh + al * bl;
        sum_a_total += ah + al;
        sum_b_total += bh + bl;
    }

    if dims % 2 == 1 {
        let av = *a.get_unchecked(full_bytes);
        let bv = *b.get_unchecked(full_bytes);
        let ah = ((av >> 4) & 0x0f) as i32;
        let bh = ((bv >> 4) & 0x0f) as i32;

        raw_total += ah * bh;
        sum_a_total += ah;
        sum_b_total += bh;
    }

    (raw_total, sum_a_total, sum_b_total)
}

#[cfg(test)]
mod tests;
