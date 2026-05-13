// ===================================================================
// Softmax over attention scores — with SIMD fast path
// ===================================================================

#[cfg(target_arch = "x86_64")]
use super::arch_kernels::hsum_m256;
use super::simd::simd_config;

/// **Unstable**: softmax over attention scores; numerics and SIMD dispatch may change.
///
/// Softmax over attention scores.
pub fn softmax_attention(x: &mut [f32], seq_len: usize, num_heads: usize) {
    debug_assert_eq!(x.len(), num_heads * seq_len * seq_len);

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                softmax_attention_neon(x, seq_len, num_heads);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled {
            // SAFETY: The runtime feature check guarantees AVX2 support.
            unsafe {
                softmax_attention_avx2(x, seq_len, num_heads);
                return;
            }
        }
    }

    softmax_attention_scalar(x, seq_len, num_heads);
}

pub fn softmax_attention_scalar(x: &mut [f32], seq_len: usize, num_heads: usize) {
    for h in 0..num_heads {
        for s in 0..seq_len {
            let start = (h * seq_len + s) * seq_len;
            let row = &mut x[start..start + seq_len];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                *val = fast_exp(*val - max_val);
                sum += *val;
            }
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for val in row.iter_mut() {
                    *val *= inv_sum;
                }
            }
        }
    }
}

/// Fast exp approximation using the integer bit trick.
/// Based on Schraudolph's method: exp(x) ~ 2^(x/ln2) via float bit manipulation.
/// Accuracy: ~5-6% relative error. Systematic bias cancels in softmax normalization.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow in the integer conversion.
    let x = x.clamp(-87.0, 88.0);
    // 2^23 / ln(2) = 12102203.16..., bias = 127 * 2^23 = 1065353216
    let val = (12_102_203.0f32 * x + 1_065_353_216.0f32) as i32;
    f32::from_bits(val as u32)
}

/// NEON fast exp approximation for 4 lanes using the Schraudolph bit trick.
/// exp(x) ~ reinterpret_float(int(x * 2^23/ln2 + 127*2^23))
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fast_exp_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // Clamp to [-87, 88] to avoid overflow/underflow
    let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
    let x = vminq_f32(x, vdupq_n_f32(88.0));
    // t = x * (2^23 / ln2) + 127 * 2^23
    let t = vfmaq_f32(vdupq_n_f32(1_065_353_216.0), x, vdupq_n_f32(12_102_203.0));
    // Reinterpret float bits as int, then back to float
    vreinterpretq_f32_s32(vcvtq_s32_f32(t))
}

/// AVX2 fast exp approximation for 8 lanes using the Schraudolph bit trick.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // Clamp to [-87, 88]
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));
    // t = x * (2^23 / ln2) + 127 * 2^23
    let scale = _mm256_set1_ps(12_102_203.0);
    let bias = _mm256_set1_ps(1_065_353_216.0);
    let t = _mm256_add_ps(_mm256_mul_ps(x, scale), bias);
    // Convert to int then reinterpret as float
    _mm256_castsi256_ps(_mm256_cvtps_epi32(t))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_attention_neon(x: &mut [f32], seq_len: usize, num_heads: usize) {
    use std::arch::aarch64::*;

    for h in 0..num_heads {
        for s in 0..seq_len {
            let start = (h * seq_len + s) * seq_len;
            let row = &mut x[start..start + seq_len];
            let ptr = row.as_mut_ptr();
            let n = row.len();
            let chunks = n / 4;

            // --- Step 1: Find row max using SIMD ---
            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            for c in 0..chunks {
                // SAFETY: c * 4 + 3 < chunks * 4 <= n, within row bounds.
                vmax = vmaxq_f32(vmax, vld1q_f32(ptr.add(c * 4) as *const f32));
            }
            let mut max_val = vmaxvq_f32(vmax);
            for i in (chunks * 4)..n {
                max_val = max_val.max(*ptr.add(i));
            }

            // --- Step 2: Subtract max + fast exp + accumulate sum (all SIMD) ---
            let vmax_val = vdupq_n_f32(max_val);
            let mut vsum = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let off = c * 4;
                let v = vsubq_f32(vld1q_f32(ptr.add(off) as *const f32), vmax_val);
                let e = fast_exp_neon(v);
                vst1q_f32(ptr.add(off), e);
                vsum = vaddq_f32(vsum, e);
            }
            let mut sum = vaddvq_f32(vsum);
            for i in (chunks * 4)..n {
                let e = fast_exp(*ptr.add(i) - max_val);
                *ptr.add(i) = e;
                sum += e;
            }

            // --- Step 3: Divide by sum using SIMD ---
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                let vinv = vdupq_n_f32(inv_sum);
                for c in 0..chunks {
                    let off = c * 4;
                    let v = vmulq_f32(vld1q_f32(ptr.add(off) as *const f32), vinv);
                    vst1q_f32(ptr.add(off), v);
                }
                for i in (chunks * 4)..n {
                    *ptr.add(i) *= inv_sum;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_attention_avx2(x: &mut [f32], seq_len: usize, num_heads: usize) {
    use std::arch::x86_64::*;

    for h in 0..num_heads {
        for s in 0..seq_len {
            let start = (h * seq_len + s) * seq_len;
            let row = &mut x[start..start + seq_len];
            let ptr = row.as_mut_ptr();
            let n = row.len();
            let chunks = n / 8;

            // --- Step 1: Find row max using SIMD ---
            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            for c in 0..chunks {
                // SAFETY: c * 8 + 7 < chunks * 8 <= n, within row bounds.
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(ptr.add(c * 8) as *const f32));
            }
            // Horizontal max reduction
            let hi128 = _mm256_extractf128_ps(vmax, 1);
            let lo128 = _mm256_castps256_ps128(vmax);
            let max128 = _mm_max_ps(lo128, hi128);
            let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
            let max32 = _mm_max_ps(max64, _mm_movehdup_ps(max64));
            let mut max_val = _mm_cvtss_f32(max32);
            for i in (chunks * 8)..n {
                max_val = max_val.max(*ptr.add(i));
            }

            // --- Step 2: Subtract max + fast exp + accumulate sum (all SIMD) ---
            let vmax_val = _mm256_set1_ps(max_val);
            let mut vsum = _mm256_setzero_ps();
            for c in 0..chunks {
                let off = c * 8;
                let v = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off) as *const f32), vmax_val);
                let e = fast_exp_avx2(v);
                _mm256_storeu_ps(ptr.add(off), e);
                vsum = _mm256_add_ps(vsum, e);
            }
            let mut sum = hsum_m256(vsum);
            for i in (chunks * 8)..n {
                let e = fast_exp(*ptr.add(i) - max_val);
                *ptr.add(i) = e;
                sum += e;
            }

            // --- Step 3: Divide by sum using SIMD ---
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                let vinv = _mm256_set1_ps(inv_sum);
                for c in 0..chunks {
                    let off = c * 8;
                    let v = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(off) as *const f32), vinv);
                    _mm256_storeu_ps(ptr.add(off), v);
                }
                for i in (chunks * 8)..n {
                    *ptr.add(i) *= inv_sum;
                }
            }
        }
    }
}
