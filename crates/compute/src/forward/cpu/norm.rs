//! Layer-norm dispatch and scalar layer-norm implementation.
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
    assert_eq!(
        gamma.len(),
        hidden,
        "layer_norm: gamma length must equal hidden before the SIMD kernels index it"
    );
    assert_eq!(
        beta.len(),
        hidden,
        "layer_norm: beta length must equal hidden before the SIMD kernels index it"
    );
    assert_eq!(
        x.len() % hidden,
        0,
        "layer_norm: x length must be a multiple of hidden"
    );

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
        let chunks = hidden / 16;
        let ptr = row.as_ptr();

        // --- Pass 1: compute sum → mean with 4x unrolled accumulators ---
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let off = c * 16;
            sum0 = vaddq_f32(sum0, vld1q_f32(ptr.add(off)));
            sum1 = vaddq_f32(sum1, vld1q_f32(ptr.add(off + 4)));
            sum2 = vaddq_f32(sum2, vld1q_f32(ptr.add(off + 8)));
            sum3 = vaddq_f32(sum3, vld1q_f32(ptr.add(off + 12)));
        }
        let combined_sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let mut total_sum = vaddvq_f32(combined_sum);
        for i in (chunks * 16)..hidden {
            total_sum += *ptr.add(i);
        }
        let mean = total_sum * inv_n;

        // --- Pass 2: compute sum of (x - mean)² → variance with 4x unrolled accumulators ---
        let vmean = vdupq_n_f32(mean);
        let mut sq0 = vdupq_n_f32(0.0);
        let mut sq1 = vdupq_n_f32(0.0);
        let mut sq2 = vdupq_n_f32(0.0);
        let mut sq3 = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let off = c * 16;
            let d0 = vsubq_f32(vld1q_f32(ptr.add(off)), vmean);
            let d1 = vsubq_f32(vld1q_f32(ptr.add(off + 4)), vmean);
            let d2 = vsubq_f32(vld1q_f32(ptr.add(off + 8)), vmean);
            let d3 = vsubq_f32(vld1q_f32(ptr.add(off + 12)), vmean);
            sq0 = vfmaq_f32(sq0, d0, d0);
            sq1 = vfmaq_f32(sq1, d1, d1);
            sq2 = vfmaq_f32(sq2, d2, d2);
            sq3 = vfmaq_f32(sq3, d3, d3);
        }
        let combined_sq = vaddq_f32(vaddq_f32(sq0, sq1), vaddq_f32(sq2, sq3));
        let mut total_sq = vaddvq_f32(combined_sq);
        for i in (chunks * 16)..hidden {
            let d = *ptr.add(i) - mean;
            total_sq += d * d;
        }
        let variance = total_sq * inv_n;
        let inv_std = 1.0 / (variance + eps).sqrt();

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
        let chunks = hidden / 32;
        let ptr = row.as_ptr();

        // --- Pass 1: compute sum → mean with 4x unrolled accumulators ---
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        for c in 0..chunks {
            let off = c * 32;
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(ptr.add(off)));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(ptr.add(off + 8)));
            sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(ptr.add(off + 16)));
            sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(ptr.add(off + 24)));
        }
        let combined_sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        let mut total_sum = hsum_m256(combined_sum);
        for i in (chunks * 32)..hidden {
            total_sum += *ptr.add(i);
        }
        let mean = total_sum * inv_n;

        // --- Pass 2: compute sum of (x - mean)² → variance with 4x unrolled accumulators ---
        let vmean = _mm256_set1_ps(mean);
        let mut sq0 = _mm256_setzero_ps();
        let mut sq1 = _mm256_setzero_ps();
        let mut sq2 = _mm256_setzero_ps();
        let mut sq3 = _mm256_setzero_ps();
        for c in 0..chunks {
            let off = c * 32;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off)), vmean);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 8)), vmean);
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 16)), vmean);
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 24)), vmean);
            sq0 = _mm256_fmadd_ps(d0, d0, sq0);
            sq1 = _mm256_fmadd_ps(d1, d1, sq1);
            sq2 = _mm256_fmadd_ps(d2, d2, sq2);
            sq3 = _mm256_fmadd_ps(d3, d3, sq3);
        }
        let combined_sq = _mm256_add_ps(_mm256_add_ps(sq0, sq1), _mm256_add_ps(sq2, sq3));
        let mut total_sq = hsum_m256(combined_sq);
        for i in (chunks * 32)..hidden {
            let d = *ptr.add(i) - mean;
            total_sq += d * d;
        }
        let variance = total_sq * inv_n;
        let inv_std = 1.0 / (variance + eps).sqrt();

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

/// **Unstable**: fused residual-add + layer normalization, in place.
///
/// Computes `out = layer_norm(out + dense)` where `out` enters holding the
/// residual (skip connection) and `dense` holds this sublayer's dense output.
/// Bit-identical to computing `out[i] += dense[i]` then `layer_norm(out, ...)`:
/// the per-element add feeds the same reduction order layer_norm uses, so the
/// mean/variance passes accumulate the exact same values in the exact same
/// order as the two-step version. This eliminates the separate scalar
/// residual-add loop and the buffer copy that surround `layer_norm` at every
/// BERT sublayer epilogue.
pub fn residual_add_layer_norm(
    out: &mut [f32],
    dense: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden: usize,
    eps: f32,
) {
    assert_eq!(
        gamma.len(),
        hidden,
        "residual_add_layer_norm: gamma length must equal hidden before the SIMD kernels index it"
    );
    assert_eq!(
        beta.len(),
        hidden,
        "residual_add_layer_norm: beta length must equal hidden before the SIMD kernels index it"
    );
    assert_eq!(
        dense.len(),
        out.len(),
        "residual_add_layer_norm: dense length must equal out length"
    );
    assert_eq!(
        out.len() % hidden,
        0,
        "residual_add_layer_norm: out length must be a multiple of hidden"
    );

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                residual_add_layer_norm_neon(out, dense, gamma, beta, hidden, eps);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
            unsafe {
                residual_add_layer_norm_avx2(out, dense, gamma, beta, hidden, eps);
                return;
            }
        }
    }

    residual_add_layer_norm_scalar(out, dense, gamma, beta, hidden, eps);
}

#[inline]
pub fn residual_add_layer_norm_scalar(
    out: &mut [f32],
    dense: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden: usize,
    eps: f32,
) {
    for (row, drow) in out.chunks_exact_mut(hidden).zip(dense.chunks_exact(hidden)) {
        for i in 0..hidden {
            row[i] += drow[i];
        }

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
unsafe fn residual_add_layer_norm_neon(
    out: &mut [f32],
    dense: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;

    let inv_n = 1.0 / hidden as f32;

    for (row, drow) in out.chunks_exact_mut(hidden).zip(dense.chunks_exact(hidden)) {
        let chunks = hidden / 16;
        let out_ptr = row.as_mut_ptr();
        let d_ptr = drow.as_ptr();

        // --- Pass 1: fused residual add + sum -> mean, 4x unrolled accumulators.
        // Each lane loads `out` and `dense`, adds them, stores the sum back into
        // `out` (so passes 2/3 read the summed value), and folds the same sum
        // into the mean accumulator -- identical accumulator order to
        // `layer_norm_neon`'s pass 1, just with a fused source value.
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let off = c * 16;
            let x0 = vaddq_f32(vld1q_f32(out_ptr.add(off)), vld1q_f32(d_ptr.add(off)));
            let x1 = vaddq_f32(
                vld1q_f32(out_ptr.add(off + 4)),
                vld1q_f32(d_ptr.add(off + 4)),
            );
            let x2 = vaddq_f32(
                vld1q_f32(out_ptr.add(off + 8)),
                vld1q_f32(d_ptr.add(off + 8)),
            );
            let x3 = vaddq_f32(
                vld1q_f32(out_ptr.add(off + 12)),
                vld1q_f32(d_ptr.add(off + 12)),
            );
            vst1q_f32(out_ptr.add(off), x0);
            vst1q_f32(out_ptr.add(off + 4), x1);
            vst1q_f32(out_ptr.add(off + 8), x2);
            vst1q_f32(out_ptr.add(off + 12), x3);
            sum0 = vaddq_f32(sum0, x0);
            sum1 = vaddq_f32(sum1, x1);
            sum2 = vaddq_f32(sum2, x2);
            sum3 = vaddq_f32(sum3, x3);
        }
        let combined_sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let mut total_sum = vaddvq_f32(combined_sum);
        for i in (chunks * 16)..hidden {
            let x = *out_ptr.add(i) + *d_ptr.add(i);
            *out_ptr.add(i) = x;
            total_sum += x;
        }
        let mean = total_sum * inv_n;

        // --- Pass 2: compute sum of (x - mean)^2 -> variance. Byte-for-byte the
        // same as `layer_norm_neon`'s pass 2, reading the fused sum `out` now
        // holds in place of the plain input it read there.
        let ptr = row.as_ptr();
        let vmean = vdupq_n_f32(mean);
        let mut sq0 = vdupq_n_f32(0.0);
        let mut sq1 = vdupq_n_f32(0.0);
        let mut sq2 = vdupq_n_f32(0.0);
        let mut sq3 = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let off = c * 16;
            let d0 = vsubq_f32(vld1q_f32(ptr.add(off)), vmean);
            let d1 = vsubq_f32(vld1q_f32(ptr.add(off + 4)), vmean);
            let d2 = vsubq_f32(vld1q_f32(ptr.add(off + 8)), vmean);
            let d3 = vsubq_f32(vld1q_f32(ptr.add(off + 12)), vmean);
            sq0 = vfmaq_f32(sq0, d0, d0);
            sq1 = vfmaq_f32(sq1, d1, d1);
            sq2 = vfmaq_f32(sq2, d2, d2);
            sq3 = vfmaq_f32(sq3, d3, d3);
        }
        let combined_sq = vaddq_f32(vaddq_f32(sq0, sq1), vaddq_f32(sq2, sq3));
        let mut total_sq = vaddvq_f32(combined_sq);
        for i in (chunks * 16)..hidden {
            let d = *ptr.add(i) - mean;
            total_sq += d * d;
        }
        let variance = total_sq * inv_n;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // --- Pass 3: normalize using SIMD: (x - mean) * inv_std * gamma + beta.
        // Byte-for-byte the same as `layer_norm_neon`'s pass 3.
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
unsafe fn residual_add_layer_norm_avx2(
    out: &mut [f32],
    dense: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;

    let inv_n = 1.0 / hidden as f32;

    for (row, drow) in out.chunks_exact_mut(hidden).zip(dense.chunks_exact(hidden)) {
        let chunks = hidden / 32;
        let out_ptr = row.as_mut_ptr();
        let d_ptr = drow.as_ptr();

        // --- Pass 1: fused residual add + sum -> mean, 4x unrolled accumulators.
        // Same accumulator order as `layer_norm_avx2`'s pass 1, fused source value.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        for c in 0..chunks {
            let off = c * 32;
            let x0 = _mm256_add_ps(
                _mm256_loadu_ps(out_ptr.add(off)),
                _mm256_loadu_ps(d_ptr.add(off)),
            );
            let x1 = _mm256_add_ps(
                _mm256_loadu_ps(out_ptr.add(off + 8)),
                _mm256_loadu_ps(d_ptr.add(off + 8)),
            );
            let x2 = _mm256_add_ps(
                _mm256_loadu_ps(out_ptr.add(off + 16)),
                _mm256_loadu_ps(d_ptr.add(off + 16)),
            );
            let x3 = _mm256_add_ps(
                _mm256_loadu_ps(out_ptr.add(off + 24)),
                _mm256_loadu_ps(d_ptr.add(off + 24)),
            );
            _mm256_storeu_ps(out_ptr.add(off), x0);
            _mm256_storeu_ps(out_ptr.add(off + 8), x1);
            _mm256_storeu_ps(out_ptr.add(off + 16), x2);
            _mm256_storeu_ps(out_ptr.add(off + 24), x3);
            sum0 = _mm256_add_ps(sum0, x0);
            sum1 = _mm256_add_ps(sum1, x1);
            sum2 = _mm256_add_ps(sum2, x2);
            sum3 = _mm256_add_ps(sum3, x3);
        }
        let combined_sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        let mut total_sum = hsum_m256(combined_sum);
        for i in (chunks * 32)..hidden {
            let x = *out_ptr.add(i) + *d_ptr.add(i);
            *out_ptr.add(i) = x;
            total_sum += x;
        }
        let mean = total_sum * inv_n;

        // --- Pass 2: compute sum of (x - mean)^2 -> variance. Byte-for-byte the
        // same as `layer_norm_avx2`'s pass 2.
        let ptr = row.as_ptr();
        let vmean = _mm256_set1_ps(mean);
        let mut sq0 = _mm256_setzero_ps();
        let mut sq1 = _mm256_setzero_ps();
        let mut sq2 = _mm256_setzero_ps();
        let mut sq3 = _mm256_setzero_ps();
        for c in 0..chunks {
            let off = c * 32;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off)), vmean);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 8)), vmean);
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 16)), vmean);
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(off + 24)), vmean);
            sq0 = _mm256_fmadd_ps(d0, d0, sq0);
            sq1 = _mm256_fmadd_ps(d1, d1, sq1);
            sq2 = _mm256_fmadd_ps(d2, d2, sq2);
            sq3 = _mm256_fmadd_ps(d3, d3, sq3);
        }
        let combined_sq = _mm256_add_ps(_mm256_add_ps(sq0, sq1), _mm256_add_ps(sq2, sq3));
        let mut total_sq = hsum_m256(combined_sq);
        for i in (chunks * 32)..hidden {
            let d = *ptr.add(i) - mean;
            total_sq += d * d;
        }
        let variance = total_sq * inv_n;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // --- Pass 3: normalize. Byte-for-byte the same as `layer_norm_avx2`'s pass 3.
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

#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn layer_norm_accepts_valid_lengths() {
        let hidden = 4;
        let gamma = vec![1.0; hidden];
        let beta = vec![0.0; hidden];
        let mut x = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]; // 2 rows
        layer_norm(&mut x, &gamma, &beta, hidden, 1e-5);
        assert_eq!(x.len(), 8);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "gamma length must equal hidden")]
    fn layer_norm_rejects_short_gamma() {
        // hidden=4 needs gamma.len()==4; a short gamma would OOB the unsafe SIMD kernel.
        let mut x = vec![0.0; 8];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 4];
        layer_norm(&mut x, &gamma, &beta, 4, 1e-5);
    }

    #[test]
    fn residual_add_layer_norm_accepts_valid_lengths() {
        let hidden = 4;
        let gamma = vec![1.0; hidden];
        let beta = vec![0.0; hidden];
        let mut out = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]; // 2 rows
        let dense = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        residual_add_layer_norm(&mut out, &dense, &gamma, &beta, hidden, 1e-5);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "gamma length must equal hidden")]
    fn residual_add_layer_norm_rejects_short_gamma() {
        let mut out = vec![0.0; 8];
        let dense = vec![0.0; 8];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 4];
        residual_add_layer_norm(&mut out, &dense, &gamma, &beta, 4, 1e-5);
    }

    #[test]
    #[should_panic(expected = "dense length must equal out length")]
    fn residual_add_layer_norm_rejects_mismatched_dense() {
        let mut out = vec![0.0; 8];
        let dense = vec![0.0; 4]; // short: 1 row instead of 2
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        residual_add_layer_norm(&mut out, &dense, &gamma, &beta, 4, 1e-5);
    }
}

#[cfg(test)]
mod residual_add_layer_norm_tests {
    use super::*;

    /// Small deterministic LCG, local to this test module so it does not
    /// depend on the sibling `tests` module's private helpers.
    fn lcg_f32_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 32) as f32 / u32::MAX as f32;
            out.push(unit * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn residual_add_layer_norm_simd_matches_scalar() {
        for &hidden in &[30usize, 100, 384, 768] {
            let rows = 4usize;
            let out_proto = lcg_f32_vec(rows * hidden, 0xABCD_0001 ^ hidden as u64);
            let dense = lcg_f32_vec(rows * hidden, 0xABCD_0002 ^ hidden as u64);
            let gamma = {
                let raw = lcg_f32_vec(hidden, 0xABCD_0003 ^ hidden as u64);
                raw.into_iter().map(|v| 1.0 + v * 0.2).collect::<Vec<_>>()
            };
            let beta = lcg_f32_vec(hidden, 0xABCD_0004 ^ hidden as u64);
            let eps = 1e-6_f32;

            let mut out_dispatched = out_proto.clone();
            residual_add_layer_norm(&mut out_dispatched, &dense, &gamma, &beta, hidden, eps);

            let mut out_scalar = out_proto.clone();
            residual_add_layer_norm_scalar(&mut out_scalar, &dense, &gamma, &beta, hidden, eps);

            for i in 0..(rows * hidden) {
                let a = out_dispatched[i];
                let b = out_scalar[i];
                let diff = (a - b).abs();
                let threshold = 1e-4 * a.abs().max(b.abs()) + 1e-5;
                assert!(
                    diff <= threshold,
                    "residual_add_layer_norm_parity hidden={hidden}[{i}]: {a} vs {b}, diff={diff:.3e} > {threshold:.3e}"
                );
            }
        }
    }

    #[test]
    fn residual_add_layer_norm_is_bit_exact_vs_add_then_layer_norm() {
        // This is the mutation-sensitivity anchor for the fusion: the fused
        // primitive must feed the residual-summed value into the identical
        // reduction order `layer_norm` uses, so it must match the two-step
        // add-then-layer_norm reference exactly, bit for bit.
        for &hidden in &[30usize, 100, 384, 768] {
            let rows = 4usize;
            let out0 = lcg_f32_vec(rows * hidden, 0xF00D_1001 ^ hidden as u64);
            let dense = lcg_f32_vec(rows * hidden, 0xF00D_1002 ^ hidden as u64);
            let gamma = {
                let raw = lcg_f32_vec(hidden, 0xF00D_1003 ^ hidden as u64);
                raw.into_iter().map(|v| 1.0 + v * 0.2).collect::<Vec<_>>()
            };
            let beta = lcg_f32_vec(hidden, 0xF00D_1004 ^ hidden as u64);
            let eps = 1e-6_f32;

            let mut fused = out0.clone();
            residual_add_layer_norm(&mut fused, &dense, &gamma, &beta, hidden, eps);

            let mut reference = out0.clone();
            for i in 0..(rows * hidden) {
                reference[i] += dense[i];
            }
            layer_norm(&mut reference, &gamma, &beta, hidden, eps);

            assert_eq!(
                fused, reference,
                "hidden={hidden}: fused result must be bit-identical to add-then-layer_norm"
            );
        }
    }
}
