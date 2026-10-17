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
mod tests;
