//! Binary quantization for ultra-fast pre-filtering.
//!
//! Sign-bit quantization: 1 bit per dimension (32x compression vs f32).
//! Distance via Hamming distance using hardware popcount.
//!
//! ## Format
//!
//! `b[i] = 1 if v[i] >= 0 else 0`. Pack 8 dimensions into one byte
//! (bit 7 = dimension 0, bit 6 = dimension 1, etc.).
//! For D dimensions, storage is `ceil(D / 8)` bytes.
//!
//! ## Distance
//!
//! Hamming distance counts differing bits between two binary vectors.
//! Approximate cosine distance: `1.0 - (1.0 - 2.0 * hamming / dims)`
//! i.e. `2.0 * hamming / dims`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::simd_config;

/// **Unstable**: binary quantization format and struct layout are under active design.
///
/// Binary quantized vector with packed bit storage.
#[derive(Debug, Clone)]
pub struct BinaryVector {
    /// **Unstable**: packed bit data; bit layout may change with format revision.
    pub data: Vec<u8>,
    /// **Unstable**: number of original dimensions.
    pub dims: usize,
    /// **Unstable**: L2 norm of the original float vector; field may be removed.
    pub norm: f32,
}

impl BinaryVector {
    /// **Unstable**: quantization API; threshold default may change.
    ///
    /// Values >= threshold map to 1, values < threshold map to 0.
    /// Default threshold is 0.0 (sign bit).
    pub fn from_f32(vector: &[f32]) -> Self {
        Self::from_f32_with_threshold(vector, 0.0)
    }

    /// **Unstable**: custom-threshold variant; may be merged into a config struct.
    pub fn from_f32_with_threshold(vector: &[f32], threshold: f32) -> Self {
        let dims = vector.len();

        // Compute norm
        let mut norm_sq = 0.0f32;
        for &v in vector {
            if v.is_finite() {
                norm_sq += v * v;
            }
        }
        let norm = norm_sq.sqrt();

        let packed_len = dims.div_ceil(8);
        let mut data = vec![0u8; packed_len];

        for (i, &v) in vector.iter().enumerate() {
            let val = if v.is_finite() { v } else { 0.0 };
            if val >= threshold {
                let byte_idx = i / 8;
                let bit_idx = 7 - (i % 8); // bit 7 = first dimension in byte
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        Self { data, dims, norm }
    }

    /// **Unstable**: dequantize to float32; output semantics may change.
    ///
    /// Binary quantization is lossy: 1 -> +1.0, 0 -> -1.0.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dims);
        for i in 0..self.dims {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            let bit = (self.data[byte_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        result
    }

    /// **Unstable**: Hamming dispatch; delegates to NEON or scalar based on runtime detection.
    ///
    /// Returns the number of differing bits (dimensions with different signs).
    #[inline]
    pub fn hamming_distance(&self, other: &BinaryVector) -> u32 {
        hamming_distance_binary(self, other)
    }

    /// **Unstable**: approximation formula may be revised; do not use in latency-sensitive production paths.
    ///
    /// The relationship between Hamming distance and angular distance:
    /// `cos_approx = 1.0 - 2.0 * hamming / dims`
    /// `cosine_distance_approx = 2.0 * hamming / dims`
    #[inline]
    pub fn cosine_distance_approx(&self, other: &BinaryVector) -> f32 {
        if self.dims == 0 {
            return 0.0;
        }
        let hamming = self.hamming_distance(other) as f32;
        2.0 * hamming / self.dims as f32
    }

    /// **Unstable**: approximation formula may be revised; complement of `cosine_distance_approx`.
    #[inline]
    pub fn cosine_similarity_approx(&self, other: &BinaryVector) -> f32 {
        1.0 - self.cosine_distance_approx(other)
    }
}

/// **Unstable**: free function dispatches to NEON or scalar; may become private in a future cleanup.
///
/// Uses popcount on XOR of the packed bytes.
#[inline]
pub fn hamming_distance_binary(a: &BinaryVector, b: &BinaryVector) -> u32 {
    if a.dims != b.dims {
        return u32::MAX;
    }

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            debug_assert_eq!(a.data.len(), b.data.len());
            // SAFETY: NEON is available on aarch64. Matching dimensions require
            // matching packed-byte lengths for vectors built by the constructor;
            // the debug guard catches public-field invariant violations. The callee
            // uses unaligned loads and chunk/remainder bounds within those slices.
            return unsafe { hamming_distance_neon(&a.data, &b.data) };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = config;
    }

    hamming_distance_scalar(&a.data, &b.data)
}

/// Scalar Hamming distance using u64 chunks and count_ones().
fn hamming_distance_scalar(a: &[u8], b: &[u8]) -> u32 {
    let mut total: u32 = 0;

    // Process 8 bytes at a time as u64
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let offset = c * 8;
        let a_u64 = u64::from_ne_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let b_u64 = u64::from_ne_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);
        total += (a_u64 ^ b_u64).count_ones();
    }

    // Handle remaining bytes
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        total += (a[i] ^ b[i]).count_ones();
    }

    total
}

/// NEON Hamming distance using `vcnt` (per-byte population count).
///
/// `vcnt` counts set bits in each byte of a NEON register.
/// We XOR the two vectors, apply vcnt, then horizontally sum.
///
/// # Safety
///
/// Caller must ensure running on aarch64 (NEON is mandatory).
/// `a` and `b` must have equal length.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hamming_distance_neon(a: &[u8], b: &[u8]) -> u32 {
    // SA-163/164: verify equal-length backing slices before the SIMD loop.
    debug_assert_eq!(
        a.len(),
        b.len(),
        "hamming_distance_neon: slice lengths differ ({} vs {})",
        a.len(),
        b.len()
    );
    let len = a.len();
    const SIMD_WIDTH: usize = 16;
    let chunks = len / SIMD_WIDTH;

    // Accumulate popcount bytes into u16 to avoid overflow
    // (max 8 bits per byte, 16 bytes per register = 128 per chunk, fits u8 for ~1 chunk)
    // Use vpaddlq to widen: u8 -> u16 -> u32 -> u64
    let mut sum_u64 = vdupq_n_u64(0);

    for c in 0..chunks {
        let base = c * SIMD_WIDTH;
        let va = vld1q_u8(a.as_ptr().add(base));
        let vb = vld1q_u8(b.as_ptr().add(base));

        // XOR to find differing bits
        let xor = veorq_u8(va, vb);

        // Population count per byte
        let popcnt = vcntq_u8(xor);

        // Widen and accumulate: u8 -> u16 -> u32 -> u64
        let sum_u16 = vpaddlq_u8(popcnt);
        let sum_u32 = vpaddlq_u16(sum_u16);
        sum_u64 = vaddq_u64(sum_u64, vpaddlq_u32(sum_u32));
    }

    // Extract final sum
    let total = vgetq_lane_u64(sum_u64, 0) + vgetq_lane_u64(sum_u64, 1);
    let mut result = total as u32;

    // Handle remainder
    let remainder_start = chunks * SIMD_WIDTH;
    for i in remainder_start..len {
        result += (a[i] ^ b[i]).count_ones();
    }

    result
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
    fn test_binary_quantize_basic() {
        let v = vec![0.5, -0.3, 0.0, -1.0, 1.0, 0.1, -0.1, 0.9];
        let bv = BinaryVector::from_f32(&v);
        assert_eq!(bv.data.len(), 1); // 8 dims -> 1 byte
        assert_eq!(bv.dims, 8);

        // Expected bits (MSB first): 1, 0, 1, 0, 1, 1, 0, 1 = 0b10101101 = 0xAD
        assert_eq!(bv.data[0], 0xAD, "packed bits: {:08b}", bv.data[0]);
    }

    #[test]
    fn test_binary_roundtrip() {
        let v = vec![0.5, -0.3, 0.0, -1.0, 1.0, 0.1, -0.1, 0.9];
        let bv = BinaryVector::from_f32(&v);
        let deq = bv.to_f32();

        // Binary: positive -> +1.0, negative -> -1.0
        assert_eq!(deq, vec![1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_binary_hamming_distance() {
        // Same vector should have 0 Hamming distance
        let v = generate_vector(384, 42);
        let bv = BinaryVector::from_f32(&v);
        assert_eq!(bv.hamming_distance(&bv), 0);

        // Opposite sign vector should have max Hamming distance
        let neg_v: Vec<f32> = v.iter().map(|x| -x).collect();
        let neg_bv = BinaryVector::from_f32(&neg_v);
        // Some values might be exactly 0.0, which maps to +1 in both cases
        // So Hamming may not be exactly 384
        let hamming = bv.hamming_distance(&neg_bv);
        // But it should be close to 384 for random vectors
        assert!(hamming > 350, "hamming={hamming}, expected close to 384");
    }

    #[test]
    fn test_binary_cosine_approx_identical() {
        let v = generate_vector(384, 55);
        let bv = BinaryVector::from_f32(&v);
        let cos_dist = bv.cosine_distance_approx(&bv);
        assert!(
            cos_dist.abs() < 1e-5,
            "Identical binary vectors should have 0 cosine distance, got {cos_dist}"
        );
    }

    #[test]
    fn test_binary_cosine_approx_quality() {
        let a = generate_vector(384, 101);
        let b = generate_vector(384, 202);

        // f32 reference cosine
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let f32_cos = dot / (norm_a * norm_b);

        let ba = BinaryVector::from_f32(&a);
        let bb = BinaryVector::from_f32(&b);
        let bin_cos = ba.cosine_similarity_approx(&bb);

        // Binary is a rough approximation -- within 0.3 is acceptable for pre-filtering
        assert!(
            (f32_cos - bin_cos).abs() < 0.35,
            "Binary cosine too far from f32: f32={f32_cos}, binary={bin_cos}"
        );
    }

    #[test]
    fn test_binary_memory_savings() {
        let v = generate_vector(384, 999);
        let bv = BinaryVector::from_f32(&v);

        // f32: 384 * 4 = 1536 bytes
        // Binary: ceil(384/8) = 48 bytes = 32x compression
        assert_eq!(bv.data.len(), 48);
    }

    #[test]
    fn test_binary_non_multiple_of_8_dims() {
        // 385 dims -> ceil(385/8) = 49 bytes
        let v = generate_vector(385, 77);
        let bv = BinaryVector::from_f32(&v);
        assert_eq!(bv.data.len(), 49);
        assert_eq!(bv.dims, 385);

        // Roundtrip should preserve all 385 values
        let deq = bv.to_f32();
        assert_eq!(deq.len(), 385);
    }

    #[test]
    fn test_binary_with_threshold() {
        let v = vec![0.5, 0.3, 0.1, -0.1, -0.3, -0.5, 0.7, 0.2];
        // With threshold 0.25, only values >= 0.25 map to 1
        let bv = BinaryVector::from_f32_with_threshold(&v, 0.25);
        let deq = bv.to_f32();
        // Expected: 1, 1, -1, -1, -1, -1, 1, -1
        assert_eq!(deq, vec![1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_binary_nan_inf_handling() {
        let v = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            0.0,
            0.5,
            -0.5,
        ];
        let bv = BinaryVector::from_f32(&v);
        let deq = bv.to_f32();
        assert_eq!(deq.len(), 8);
        for &val in &deq {
            assert!(val == 1.0 || val == -1.0, "Binary should produce +/-1.0");
        }
    }

    #[test]
    fn test_hamming_scalar_vs_neon_parity() {
        // Generate two distinct binary vectors and verify both paths give same result
        let a = generate_vector(384, 111);
        let b = generate_vector(384, 222);
        let ba = BinaryVector::from_f32(&a);
        let bb = BinaryVector::from_f32(&b);

        let scalar_result = hamming_distance_scalar(&ba.data, &bb.data);
        let dispatch_result = ba.hamming_distance(&bb);

        assert_eq!(
            scalar_result, dispatch_result,
            "Scalar and dispatched Hamming should match"
        );
    }
}
