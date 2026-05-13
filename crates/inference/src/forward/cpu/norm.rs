// ===================================================================
// Layer normalization (in-place) — with SIMD fast path
// ===================================================================

#[cfg(target_arch = "x86_64")]
use super::arch_kernels::hsum_m256;
use super::simd::simd_config;

/// **Unstable**: layer normalization in-place; dispatch between SIMD and scalar paths may change.
///
/// Layer normalization (in-place).
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], hidden: usize, eps: f32) {
    debug_assert_eq!(gamma.len(), hidden);
    debug_assert_eq!(beta.len(), hidden);
    debug_assert_eq!(x.len() % hidden, 0);

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                layer_norm_neon(x, gamma, beta, hidden, eps);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
            unsafe {
                layer_norm_avx2(x, gamma, beta, hidden, eps);
                return;
            }
        }
    }

    layer_norm_scalar(x, gamma, beta, hidden, eps);
}

#[inline]
pub fn layer_norm_scalar(x: &mut [f32], gamma: &[f32], beta: &[f32], hidden: usize, eps: f32) {
    for row in x.chunks_exact_mut(hidden) {
        let mean = row.iter().copied().sum::<f32>() / hidden as f32;
        let variance = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / hidden as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        for i in 0..hidden {
            row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_neon(x: &mut [f32], gamma: &[f32], beta: &[f32], hidden: usize, eps: f32) {
    use std::arch::aarch64::*;

    let inv_n = 1.0 / hidden as f32;

    for row in x.chunks_exact_mut(hidden) {
        // --- Pass 1: compute mean using SIMD sum ---
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let chunks = hidden / 16;
        let ptr = row.as_ptr();
        for c in 0..chunks {
            let off = c * 16;
            sum0 = vaddq_f32(sum0, vld1q_f32(ptr.add(off)));
            sum1 = vaddq_f32(sum1, vld1q_f32(ptr.add(off + 4)));
            sum2 = vaddq_f32(sum2, vld1q_f32(ptr.add(off + 8)));
            sum3 = vaddq_f32(sum3, vld1q_f32(ptr.add(off + 12)));
        }
        let combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let mut total = vaddvq_f32(combined);
        for i in (chunks * 16)..hidden {
            total += *ptr.add(i);
        }
        let mean = total * inv_n;

        // --- Pass 2: compute variance using SIMD ---
        let vmean = vdupq_n_f32(mean);
        let mut var0 = vdupq_n_f32(0.0);
        let mut var1 = vdupq_n_f32(0.0);
        let mut var2 = vdupq_n_f32(0.0);
        let mut var3 = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let off = c * 16;
            let d0 = vsubq_f32(vld1q_f32(ptr.add(off)), vmean);
            var0 = vfmaq_f32(var0, d0, d0);
            let d1 = vsubq_f32(vld1q_f32(ptr.add(off + 4)), vmean);
            var1 = vfmaq_f32(var1, d1, d1);
            let d2 = vsubq_f32(vld1q_f32(ptr.add(off + 8)), vmean);
            var2 = vfmaq_f32(var2, d2, d2);
            let d3 = vsubq_f32(vld1q_f32(ptr.add(off + 12)), vmean);
            var3 = vfmaq_f32(var3, d3, d3);
        }
        let var_combined = vaddq_f32(vaddq_f32(var0, var1), vaddq_f32(var2, var3));
        let mut var_total = vaddvq_f32(var_combined);
        for i in (chunks * 16)..hidden {
            let d = *ptr.add(i) - mean;
            var_total += d * d;
        }
        let inv_std = 1.0 / (var_total * inv_n + eps).sqrt();

        // --- Pass 3: normalize using SIMD: (x - mean) * inv_std * gamma + beta ---
        let vinv_std = vdupq_n_f32(inv_std);
        let out_ptr = row.as_mut_ptr();
        let g_ptr = gamma.as_ptr();
        let b_ptr = beta.as_ptr();

        for c in 0..chunks {
            let off = c * 16;
            let d0 = vsubq_f32(vld1q_f32(out_ptr.add(off) as *const f32), vmean);
            let n0 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off)),
                vmulq_f32(d0, vinv_std),
                vld1q_f32(g_ptr.add(off)),
            );
            vst1q_f32(out_ptr.add(off), n0);

            let d1 = vsubq_f32(vld1q_f32(out_ptr.add(off + 4) as *const f32), vmean);
            let n1 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 4)),
                vmulq_f32(d1, vinv_std),
                vld1q_f32(g_ptr.add(off + 4)),
            );
            vst1q_f32(out_ptr.add(off + 4), n1);

            let d2 = vsubq_f32(vld1q_f32(out_ptr.add(off + 8) as *const f32), vmean);
            let n2 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 8)),
                vmulq_f32(d2, vinv_std),
                vld1q_f32(g_ptr.add(off + 8)),
            );
            vst1q_f32(out_ptr.add(off + 8), n2);

            let d3 = vsubq_f32(vld1q_f32(out_ptr.add(off + 12) as *const f32), vmean);
            let n3 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 12)),
                vmulq_f32(d3, vinv_std),
                vld1q_f32(g_ptr.add(off + 12)),
            );
            vst1q_f32(out_ptr.add(off + 12), n3);
        }
        for i in (chunks * 16)..hidden {
            *out_ptr.add(i) = (*out_ptr.add(i) - mean) * inv_std * *g_ptr.add(i) + *b_ptr.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn layer_norm_avx2(x: &mut [f32], gamma: &[f32], beta: &[f32], hidden: usize, eps: f32) {
    use std::arch::x86_64::*;

    let inv_n = 1.0 / hidden as f32;

    for row in x.chunks_exact_mut(hidden) {
        // --- Pass 1: compute mean ---
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let chunks = hidden / 32;
        let ptr = row.as_ptr();
        for c in 0..chunks {
            let off = c * 32;
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(ptr.add(off)));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(ptr.add(off + 8)));
            sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(ptr.add(off + 16)));
            sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(ptr.add(off + 24)));
        }
        let combined = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        let mut total = hsum_m256(combined);
        for i in (chunks * 32)..hidden {
            total += *ptr.add(i);
        }
        let mean = total * inv_n;

        // --- Pass 2: compute variance ---
        let vmean = _mm256_set1_ps(mean);
        let mut var0 = _mm256_setzero_ps();
        let mut var1 = _mm256_setzero_ps();
        let mut var2 = _mm256_setzero_ps();
        let mut var3 = _mm256_setzero_ps();

        for c in 0..chunks {
            let off = c * 32;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off)), vmean);
            var0 = _mm256_fmadd_ps(d0, d0, var0);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 8)), vmean);
            var1 = _mm256_fmadd_ps(d1, d1, var1);
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 16)), vmean);
            var2 = _mm256_fmadd_ps(d2, d2, var2);
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 24)), vmean);
            var3 = _mm256_fmadd_ps(d3, d3, var3);
        }
        let var_combined = _mm256_add_ps(_mm256_add_ps(var0, var1), _mm256_add_ps(var2, var3));
        let mut var_total = hsum_m256(var_combined);
        for i in (chunks * 32)..hidden {
            let d = *ptr.add(i) - mean;
            var_total += d * d;
        }
        let inv_std = 1.0 / (var_total * inv_n + eps).sqrt();

        // --- Pass 3: normalize ---
        let vinv_std = _mm256_set1_ps(inv_std);
        let out_ptr = row.as_mut_ptr();
        let g_ptr = gamma.as_ptr();
        let b_ptr = beta.as_ptr();

        for c in 0..chunks {
            let off = c * 32;
            for lane_off in (0..32).step_by(8) {
                let p = off + lane_off;
                let d = _mm256_sub_ps(_mm256_loadu_ps(out_ptr.add(p) as *const f32), vmean);
                let scaled = _mm256_mul_ps(d, vinv_std);
                // result = scaled * gamma + beta
                let result = _mm256_fmadd_ps(
                    scaled,
                    _mm256_loadu_ps(g_ptr.add(p)),
                    _mm256_loadu_ps(b_ptr.add(p)),
                );
                _mm256_storeu_ps(out_ptr.add(p), result);
            }
        }
        for i in (chunks * 32)..hidden {
            *out_ptr.add(i) = (*out_ptr.add(i) - mean) * inv_std * *g_ptr.add(i) + *b_ptr.add(i);
        }
    }
}
