//! INT4 quantization for ultra-compact embedding storage.
//!
//! Two 4-bit values packed per byte (8x compression vs f32).
//! Uses symmetric unsigned quantization: maps [-max_abs, max_abs] to [0, 15].
//!
//! ## Packing format
//!
//! High nibble = even index, low nibble = odd index.
//! For D dimensions, storage is `ceil(D / 2)` bytes.
//!
//! ## Dot product
//!
//! Dot products dequantize before accumulation so the unsigned INT4 offset is
//! handled identically on every target.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
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
        let mut max_abs: f32 = 0.0;
        for &v in vector {
            if v.is_finite() {
                max_abs = max_abs.max(v.abs());
            }
        }

        // Epsilon guard: avoid division by near-zero
        let scale = if max_abs > 1e-10 {
            15.0 / (2.0 * max_abs)
        } else {
            1.0
        };

        Self { scale, max_abs }
    }
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

        // Quantize each value to [0, 15] and pack pairs into bytes
        let packed_len = dims.div_ceil(2);
        let mut data = vec![0u8; packed_len];

        for (i, &elem) in vector[..dims].iter().enumerate() {
            let v = if elem.is_finite() { elem } else { 0.0 };
            // Map [-max_abs, max_abs] -> [0, 15]
            let q = ((v + params.max_abs) * params.scale)
                .round()
                .clamp(0.0, 15.0) as u8;

            let byte_idx = i / 2;
            if i % 2 == 0 {
                // Even index -> high nibble
                data[byte_idx] |= q << 4;
            } else {
                // Odd index -> low nibble
                data[byte_idx] |= q;
            }
        }

        Self {
            data,
            dims,
            params,
            norm,
        }
    }

    /// **Unstable**: dequantization output semantics may change.
    ///
    /// Reverses the quantization: `v[i] = q[i] / scale - max_abs`
    ///
    /// # Precision
    ///
    /// INT4 unsigned symmetric quantization maps `[-max_abs, max_abs]` to `[0, 15]`
    /// (16 levels), so the quantization step size is `2 * max_abs / 15`. The maximum
    /// per-element round-trip error is bounded by half a step: `max_abs / 15`.
    ///
    /// For a 384-dim unit-norm embedding (`max_abs` ≈ 1.0), expect element-wise
    /// absolute error ≤ 0.067 and relative dot-product error ≤ 15% (see
    /// `test_int4_dot_product_vs_f32` and `test_int4_roundtrip_accuracy`).
    /// Use `Int8` tier when higher fidelity is required.
    pub fn to_f32(&self) -> Vec<f32> {
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

/// **Unstable**: SIMD INT4 dot product; NEON/scalar dispatch may change.
///
/// Unpacks nibbles, computes dot product of quantized values, then applies
/// dequantization scaling: `result = (raw_dot / (scale_a * scale_b)) - correction`
///
/// The correction accounts for the unsigned offset in the quantization formula.
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

    let a_deq = a.to_f32();
    let b_deq = b.to_f32();
    a_deq.iter().zip(b_deq.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
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
mod tests {
    use super::*;

    fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
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

    #[test]
    fn test_int4_roundtrip_accuracy() {
        let original = generate_vector(384, 42);
        let quantized = Int4Vector::from_f32(&original);
        let dequantized = quantized.to_f32();

        assert_eq!(dequantized.len(), original.len());

        // INT4 has only 16 levels, so error is larger than INT8.
        // Max error should be within 1/15 of the range.
        let max_abs = original
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let expected_max_error = 2.0 * max_abs / 15.0;

        for (i, (orig, deq)) in original.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            assert!(
                error <= expected_max_error + 1e-5,
                "INT4 roundtrip error too large at index {i}: orig={orig}, deq={deq}, error={error}, max_allowed={expected_max_error}"
            );
        }
    }

    #[test]
    fn test_int4_packing_correctness() {
        // Verify nibble packing: even index -> high nibble, odd -> low
        let v = vec![0.5, -0.5, 0.0, 1.0]; // 4 values -> 2 packed bytes
        let q = Int4Vector::from_f32(&v);
        assert_eq!(q.data.len(), 2);
        assert_eq!(q.dims, 4);

        // Verify roundtrip preserves approximate values
        let deq = q.to_f32();
        assert_eq!(deq.len(), 4);
        // 0.5 should map to roughly the right region
        assert!((deq[0] - 0.5).abs() < 0.15, "deq[0]={}", deq[0]);
        assert!((deq[1] - (-0.5)).abs() < 0.15, "deq[1]={}", deq[1]);
    }

    #[test]
    fn test_int4_odd_dimensions() {
        // Odd number of dimensions: last nibble has a padding zero
        let v = generate_vector(383, 77);
        let q = Int4Vector::from_f32(&v);
        assert_eq!(q.data.len(), 192); // ceil(383/2) = 192
        assert_eq!(q.dims, 383);

        let deq = q.to_f32();
        assert_eq!(deq.len(), 383);
    }

    #[test]
    fn test_int4_zero_vector() {
        let v = vec![0.0; 384];
        let q = Int4Vector::from_f32(&v);
        let deq = q.to_f32();
        for &val in &deq {
            assert!(
                val.abs() < 1e-5,
                "Zero vector should dequantize to near-zero"
            );
        }
    }

    #[test]
    fn test_int4_dot_product_vs_f32() {
        // Use correlated vectors so the true dot product is large relative to noise.
        // For uncorrelated random vectors, the expected dot product is ~0 while
        // quantization noise is O(dims * step^2), so relative error is unbounded.
        let a = generate_vector(384, 101);
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.2 * (i as f32 * 0.3).sin())
            .collect();

        // f32 reference
        let f32_dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        let qa = Int4Vector::from_f32(&a);
        let qb = Int4Vector::from_f32(&b);
        let int4_dot = qa.dot_product(&qb);

        // INT4 has 16 levels; for correlated vectors the relative error should be
        // within ~15% (quantization step = 2*max_abs/15 per component).
        let rel_error = (f32_dot - int4_dot).abs() / f32_dot.abs().max(1.0);
        assert!(
            rel_error < 0.15,
            "INT4 dot product relative error too large: f32={f32_dot}, int4={int4_dot}, rel_error={rel_error}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_int4_neon_matches_dequantized_scalar() {
        for dim in [1, 2, 31, 64, 127, 384, 768] {
            let a = generate_vector(dim, 501);
            let b = generate_vector(dim, 777);
            let qa = Int4Vector::from_f32(&a);
            let qb = Int4Vector::from_f32(&b);

            let a_deq = qa.to_f32();
            let b_deq = qb.to_f32();
            let expected: f32 = a_deq.iter().zip(b_deq.iter()).map(|(&x, &y)| x * y).sum();
            let got = qa.dot_product(&qb);

            assert!(
                (expected - got).abs() < 1e-4,
                "INT4 NEON mismatch for dim={dim}: expected={expected}, got={got}"
            );
        }
    }

    #[test]
    fn test_int4_cosine_similarity() {
        let a = generate_vector(384, 301);
        let b = generate_vector(384, 302);

        let qa = Int4Vector::from_f32(&a);
        let qb = Int4Vector::from_f32(&b);
        let int4_cos = qa.cosine_similarity(&qb);

        // Compute f32 reference cosine
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let f32_cos = dot / (norm_a * norm_b);

        assert!(
            (f32_cos - int4_cos).abs() < 0.1,
            "INT4 cosine too far from f32: f32={f32_cos}, int4={int4_cos}"
        );
    }

    #[test]
    fn test_int4_memory_savings() {
        let v = generate_vector(384, 999);
        let q = Int4Vector::from_f32(&v);

        // f32: 384 * 4 = 1536 bytes
        // INT4: ceil(384/2) = 192 bytes = 8x compression
        assert_eq!(q.data.len(), 192);
        assert_eq!(v.len() * 4, 1536);
    }

    #[test]
    fn test_int4_nan_inf_handling() {
        let v = vec![
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.0,
            0.5,
            0.0,
            -0.3,
        ];
        let q = Int4Vector::from_f32(&v);
        let deq = q.to_f32();
        assert_eq!(deq.len(), 8);
        // NaN and Inf should be treated as 0
        // The dequantized value for the "0" slot should be near -max_abs + something,
        // but the key invariant is no panics and finite output.
        for &val in &deq {
            assert!(val.is_finite(), "Dequantized value should be finite");
        }
    }
}
