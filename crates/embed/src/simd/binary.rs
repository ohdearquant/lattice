//! Binary sign quantization and Hamming-distance operations.
//!
//! Packed format and partial-byte handling remain part of the API contract.
//!
//! See docs/simd.md for the format and cosine approximation.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
        let data = quantize_binary(vector, threshold, packed_len);

        Self { data, dims, norm }
    }

    /// **Unstable**: dequantize to float32; output semantics may change.
    ///
    /// Binary quantization is lossy: 1 -> +1.0, 0 -> -1.0.
    ///
    /// Returns an empty `Vec` if the packed buffer is shorter than `dims.div_ceil(8)`
    /// bytes — i.e. the vector was constructed with mismatched fields.
    pub fn to_f32(&self) -> Vec<f32> {
        let required_bytes = self.dims.div_ceil(8);
        if self.data.len() < required_bytes {
            return Vec::new();
        }
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

fn quantize_binary(vector: &[f32], threshold: f32, packed_len: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_config().avx2_enabled {
            // SAFETY: AVX2 was detected at runtime; the kernel bounds every load and store.
            return unsafe { quantize_binary_avx2(vector, threshold, packed_len) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            // SAFETY: NEON was detected at runtime; the kernel bounds every load and store.
            return unsafe { quantize_binary_neon(vector, threshold, packed_len) };
        }
    }
    quantize_binary_scalar(vector, threshold, packed_len)
}

fn quantize_binary_scalar(vector: &[f32], threshold: f32, packed_len: usize) -> Vec<u8> {
    let mut data = vec![0u8; packed_len];
    quantize_binary_scalar_tail(vector, threshold, &mut data, 0);
    data
}

fn quantize_binary_scalar_tail(vector: &[f32], threshold: f32, data: &mut [u8], start: usize) {
    for (i, &value) in vector.iter().enumerate().skip(start) {
        let finite_value = if value.is_finite() { value } else { 0.0 };
        if finite_value >= threshold {
            data[i / 8] |= 1 << (7 - i % 8);
        }
    }
}

#[cfg(test)]
thread_local! {
    static BINARY_QUANTIZE_SIMD_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_binary_avx2(vector: &[f32], threshold: f32, packed_len: usize) -> Vec<u8> {
    #[cfg(test)]
    BINARY_QUANTIZE_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let mut data = vec![0u8; packed_len];
    let chunks = vector.len() / 8;
    let threshold_scalar = threshold;
    let sign = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let threshold = _mm256_set1_ps(threshold_scalar);

    for i in 0..chunks {
        let input = _mm256_loadu_ps(vector.as_ptr().add(i * 8));
        let abs = _mm256_andnot_ps(sign, input);
        let finite = _mm256_cmp_ps(abs, inf, _CMP_LT_OQ);
        let values = _mm256_and_ps(input, finite);
        let above_threshold = _mm256_cmp_ps(values, threshold, _CMP_GE_OQ);
        data[i] = (_mm256_movemask_ps(above_threshold) as u8).reverse_bits();
    }

    quantize_binary_scalar_tail(vector, threshold_scalar, &mut data, chunks * 8);
    data
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_binary_neon(vector: &[f32], threshold: f32, packed_len: usize) -> Vec<u8> {
    #[cfg(test)]
    BINARY_QUANTIZE_SIMD_HITS.with(|hits| hits.set(hits.get() + 1));

    let mut data = vec![0u8; packed_len];
    let chunks = vector.len() / 8;
    let inf = vdupq_n_f32(f32::INFINITY);
    let zero = vdupq_n_f32(0.0);
    let threshold_vector = vdupq_n_f32(threshold);

    for i in 0..chunks {
        let base = i * 8;
        let first = vld1q_f32(vector.as_ptr().add(base));
        let second = vld1q_f32(vector.as_ptr().add(base + 4));
        let first_values = vbslq_f32(vcaltq_f32(first, inf), first, zero);
        let second_values = vbslq_f32(vcaltq_f32(second, inf), second, zero);
        let first_mask = vcgeq_f32(first_values, threshold_vector);
        let second_mask = vcgeq_f32(second_values, threshold_vector);
        let mut first_lanes = [0u32; 4];
        let mut second_lanes = [0u32; 4];
        vst1q_u32(first_lanes.as_mut_ptr(), first_mask);
        vst1q_u32(second_lanes.as_mut_ptr(), second_mask);

        let mut packed = 0u8;
        for (lane, mask) in first_lanes.into_iter().chain(second_lanes).enumerate() {
            if mask != 0 {
                packed |= 1 << (7 - lane);
            }
        }
        data[i] = packed;
    }

    quantize_binary_scalar_tail(vector, threshold, &mut data, chunks * 8);
    data
}

/// **Unstable**: returns packed-bit Hamming distance or `u32::MAX` for invalid inputs.
///
/// See [`docs/simd.md`](../../docs/simd.md#binary-vectors) for packing, masking, and approximation semantics.
#[inline]
pub fn hamming_distance_binary(a: &BinaryVector, b: &BinaryVector) -> u32 {
    if a.dims != b.dims {
        return u32::MAX;
    }

    let required_bytes = a.dims.div_ceil(8);
    if a.data.len() < required_bytes || b.data.len() < required_bytes {
        return u32::MAX;
    }

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64. Both slices have been verified above
            // to contain at least `required_bytes` elements, which is the exact packed
            // length for `dims` bits. The callee uses unaligned loads and chunk/remainder
            // bounds strictly within those slices; no out-of-bounds read is possible.
            return unsafe {
                hamming_distance_neon(&a.data[..required_bytes], &b.data[..required_bytes], a.dims)
            };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = config;
    }

    hamming_distance_scalar(&a.data[..required_bytes], &b.data[..required_bytes], a.dims)
}

/// Computes Hamming distance with scalar popcount, masking a partial final byte.
///
/// See [`docs/simd.md`](../../docs/simd.md#binary-vectors) for the MSB-first padding invariant.
fn hamming_distance_scalar(a: &[u8], b: &[u8], dims: usize) -> u32 {
    let mut total: u32 = 0;

    // Number of fully-populated 8-byte chunks (all bits valid).
    let full_bytes = dims / 8; // bytes where every bit is a real dimension
    let chunks = full_bytes / 8;

    // Process 8 bytes at a time as u64
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

    // Handle the remaining full bytes (between the last u64 chunk and the partial byte)
    let remainder_start = chunks * 8;
    for i in remainder_start..full_bytes {
        total += (a[i] ^ b[i]).count_ones();
    }

    // Handle the single partial byte (if dims is not a multiple of 8).
    // The `r` valid bits occupy the top `r` bits of the byte (MSB = dim 0 within byte).
    let r = dims % 8;
    if r != 0 {
        let mask = 0xFFu8 << (8 - r); // top `r` bits set, bottom `(8-r)` clear
        total += ((a[full_bytes] ^ b[full_bytes]) & mask).count_ones();
    }

    total
}

/// Computes Hamming distance with NEON `vcnt`, masking a partial final byte.
///
/// # Safety
/// Caller must run on aarch64 with equal packed slices for `dims` dimensions.
/// See [`docs/simd.md`](../../docs/simd.md#binary-vectors) for the MSB-first padding invariant.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hamming_distance_neon(a: &[u8], b: &[u8], dims: usize) -> u32 {
    // SA-163/164: verify equal-length backing slices before the SIMD loop.
    debug_assert_eq!(
        a.len(),
        b.len(),
        "hamming_distance_neon: slice lengths differ ({} vs {})",
        a.len(),
        b.len()
    );

    // Only full bytes (where every bit is a real dimension) go through the SIMD loop.
    let full_bytes = dims / 8;
    const SIMD_WIDTH: usize = 16;
    let chunks = full_bytes / SIMD_WIDTH;

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

    // Handle remaining full bytes (between last SIMD chunk and the partial byte)
    let remainder_start = chunks * SIMD_WIDTH;
    for i in remainder_start..full_bytes {
        result += (a[i] ^ b[i]).count_ones();
    }

    // Handle the single partial byte (if dims is not a multiple of 8).
    // Top `r` bits are real dimensions; bottom `(8 - r)` bits are padding.
    let r = dims % 8;
    if r != 0 {
        let mask = 0xFFu8 << (8 - r); // top `r` bits set, bottom `(8-r)` clear
        result += ((a[full_bytes] ^ b[full_bytes]) & mask).count_ones();
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

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_binary_quantize_explicit_simd_matches_scalar_and_is_dispatched() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for threshold in [0.0, 0.25, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
                let mut input = generate_vector(dim, 900 + dim as u64);
                if dim > 0 {
                    input[0] = f32::NAN;
                }
                if dim > 1 {
                    input[1] = f32::INFINITY;
                }
                if dim > 2 {
                    input[2] = f32::NEG_INFINITY;
                }

                let packed_len = dim.div_ceil(8);
                let scalar = quantize_binary_scalar(&input, threshold, packed_len);
                #[cfg(target_arch = "aarch64")]
                // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
                let simd = unsafe { quantize_binary_neon(&input, threshold, packed_len) };
                #[cfg(target_arch = "x86_64")]
                // SAFETY: AVX2 was detected above; the kernel bounds every access.
                let simd = unsafe { quantize_binary_avx2(&input, threshold, packed_len) };
                assert_eq!(
                    simd, scalar,
                    "explicit SIMD mismatch at dim={dim}, threshold={threshold}"
                );
            }
        }

        let input = generate_vector(385, 1_063);
        let before = BINARY_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
        let quantized = BinaryVector::from_f32(&input);
        let after = BINARY_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
        assert_eq!(
            after,
            before + 1,
            "BinaryVector::from_f32 did not execute its explicit SIMD quantizer"
        );
        assert_eq!(
            quantized.data,
            quantize_binary_scalar(&input, 0.0, input.len().div_ceil(8))
        );
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

        let scalar_result = hamming_distance_scalar(&ba.data, &bb.data, ba.dims);
        let dispatch_result = ba.hamming_distance(&bb);

        assert_eq!(
            scalar_result, dispatch_result,
            "Scalar and dispatched Hamming should match"
        );
    }

    // --- Issue #211 regression tests -------------------------------------------------

    #[test]
    fn test_hamming_short_data_returns_max() {
        // dims=128 requires 16 bytes; supply only 4 — must return u32::MAX, not OOB.
        let a = BinaryVector {
            dims: 128,
            data: vec![0xFFu8; 4],
            norm: 1.0,
        };
        let b = BinaryVector {
            dims: 128,
            data: vec![0x00u8; 4],
            norm: 1.0,
        };
        assert_eq!(
            hamming_distance_binary(&a, &b),
            u32::MAX,
            "Short data must yield u32::MAX, not an OOB read"
        );
    }

    #[test]
    fn test_hamming_one_side_short_returns_max() {
        // a is correct length, b is too short.
        let a = BinaryVector {
            dims: 128,
            data: vec![0xFFu8; 16],
            norm: 1.0,
        };
        let b = BinaryVector {
            dims: 128,
            data: vec![0x00u8; 8],
            norm: 1.0,
        };
        assert_eq!(hamming_distance_binary(&a, &b), u32::MAX);
    }

    #[test]
    fn test_hamming_correct_data_still_works() {
        // Verify the guard does not break the normal path.
        let v = generate_vector(128, 42);
        let bv = BinaryVector::from_f32(&v);
        assert_eq!(bv.hamming_distance(&bv), 0);
    }

    #[test]
    fn test_binary_to_f32_short_data_returns_empty() {
        // dims=128 requires 16 bytes; supply only 4.
        let bv = BinaryVector {
            dims: 128,
            data: vec![0xFFu8; 4],
            norm: 1.0,
        };
        let result = bv.to_f32();
        assert!(
            result.is_empty(),
            "to_f32 on malformed BinaryVector must return empty Vec"
        );
    }

    #[test]
    fn test_binary_to_f32_exact_length_works() {
        // Exactly the right number of bytes — must succeed.
        let v = generate_vector(128, 7);
        let bv = BinaryVector::from_f32(&v);
        let deq = bv.to_f32();
        assert_eq!(deq.len(), 128);
    }

    // --- Issue #249 regression: padding-bit masking in Hamming distance -------------

    /// Vectors with 12 dimensions use 2 bytes, with 4 padding bits in byte 1 (the low
    /// nibble). Two vectors that agree on all 12 real dimensions but differ in their
    /// padding bits must report Hamming distance 0, not 4.
    ///
    /// Before the fix, XOR-popcount over the raw final byte counted those 4 spurious
    /// differing padding bits. After the fix, the mask `0xFF << (8 - 4) = 0xF0` zeroes
    /// the low nibble before counting, giving the correct answer.
    #[test]
    fn test_hamming_ignores_padding_bits() {
        // 12 dims → 2 bytes; the last 4 bits of byte 1 are padding.
        // Build via pub fields so we can inject arbitrary padding.
        let clean = BinaryVector {
            dims: 12,
            // byte 1: top nibble = 4 valid dims, low nibble = 0 (zero padding)
            data: vec![0b10101010u8, 0b11110000u8],
            norm: 1.0,
        };
        let dirty = BinaryVector {
            dims: 12,
            // same valid bits in top nibble of byte 1, non-zero garbage in low-nibble padding
            data: vec![0b10101010u8, 0b11111111u8],
            norm: 1.0,
        };

        // Both vectors agree on all 12 real dimensions; Hamming distance must be 0.
        assert_eq!(
            hamming_distance_scalar(&clean.data, &dirty.data, 12),
            0,
            "scalar: padding bits must not be counted"
        );
        assert_eq!(
            clean.hamming_distance(&dirty),
            0,
            "dispatch: padding bits must not be counted"
        );

        // Cosine distance approximation must also be 0.0 for identical valid bits.
        assert_eq!(
            clean.cosine_distance_approx(&dirty),
            0.0,
            "cosine_distance_approx: padding bits must not be counted"
        );
    }

    /// Verifies that the partial-byte mask counts real differing bits correctly.
    ///
    /// byte 0: 0b10101010 ^ 0b01010101 = 0b11111111 → 8 differing bits.
    /// byte 1 (masked with 0xF0): (0b11110000 ^ 0b00000000) & 0xF0 = 0b11110000 → 4 bits.
    /// Total: 12, which equals `dims` (every real dimension differs).
    #[test]
    fn test_hamming_partial_byte_count() {
        let a = BinaryVector {
            dims: 12,
            data: vec![0b10101010u8, 0b11110000u8],
            norm: 1.0,
        };
        let b = BinaryVector {
            dims: 12,
            data: vec![0b01010101u8, 0b00000000u8],
            norm: 1.0,
        };

        assert_eq!(hamming_distance_scalar(&a.data, &b.data, 12), 12);
        assert_eq!(a.hamming_distance(&b), 12);
    }
}
