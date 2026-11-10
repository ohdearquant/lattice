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
        let mean_f64 = row.iter().map(|&v| f64::from(v)).sum::<f64>() / hidden as f64;
        let mean = mean_f64 as f32;
        let mean_residual = mean_f64 - f64::from(mean);
        let variance = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / hidden as f32;
        // Widen the correction so an overflowing f32 variance stays infinite.
        let variance =
            (f64::from(variance) - mean_residual * mean_residual).clamp(0.0, f64::INFINITY) as f32;
        let mean_residual = mean_residual as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        for i in 0..hidden {
            row[i] = ((row[i] - mean) - mean_residual) * inv_std * gamma[i] + beta[i];
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

        // --- Pass 1: widen four input streams before accumulating the mean ---
        let mut sum0 = vdupq_n_f64(0.0);
        let mut sum1 = vdupq_n_f64(0.0);
        let mut sum2 = vdupq_n_f64(0.0);
        let mut sum3 = vdupq_n_f64(0.0);
        let mut sum4 = vdupq_n_f64(0.0);
        let mut sum5 = vdupq_n_f64(0.0);
        let mut sum6 = vdupq_n_f64(0.0);
        let mut sum7 = vdupq_n_f64(0.0);
        for c in 0..chunks {
            let off = c * 16;
            let x0 = vld1q_f32(ptr.add(off));
            let x1 = vld1q_f32(ptr.add(off + 4));
            let x2 = vld1q_f32(ptr.add(off + 8));
            let x3 = vld1q_f32(ptr.add(off + 12));
            sum0 = vaddq_f64(sum0, vcvt_f64_f32(vget_low_f32(x0)));
            sum1 = vaddq_f64(sum1, vcvt_f64_f32(vget_high_f32(x0)));
            sum2 = vaddq_f64(sum2, vcvt_f64_f32(vget_low_f32(x1)));
            sum3 = vaddq_f64(sum3, vcvt_f64_f32(vget_high_f32(x1)));
            sum4 = vaddq_f64(sum4, vcvt_f64_f32(vget_low_f32(x2)));
            sum5 = vaddq_f64(sum5, vcvt_f64_f32(vget_high_f32(x2)));
            sum6 = vaddq_f64(sum6, vcvt_f64_f32(vget_low_f32(x3)));
            sum7 = vaddq_f64(sum7, vcvt_f64_f32(vget_high_f32(x3)));
        }
        #[cfg(test)]
        if chunks > 0 {
            simd_mean_tests::record_route(simd_mean_tests::PLAIN_NEON);
        }
        let combined_sum = vaddq_f64(
            vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3)),
            vaddq_f64(vaddq_f64(sum4, sum5), vaddq_f64(sum6, sum7)),
        );
        let mut total_sum = vaddvq_f64(combined_sum);
        for i in (chunks * 16)..hidden {
            total_sum += f64::from(*ptr.add(i));
        }
        let mean_f64 = total_sum / hidden as f64;
        let mean = mean_f64 as f32;
        let mean_residual = mean_f64 - f64::from(mean);

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
        let variance =
            (f64::from(variance) - mean_residual * mean_residual).clamp(0.0, f64::INFINITY) as f32;
        let mean_residual = mean_residual as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // --- Pass 3: normalize using SIMD: ((x - mean) - mean_residual) * inv_std * gamma + beta ---
        let vinv_std = vdupq_n_f32(inv_std);
        let vmean_residual = vdupq_n_f32(mean_residual);
        let out_ptr = row.as_mut_ptr();
        let g_ptr = gamma.as_ptr();
        let b_ptr = beta.as_ptr();

        for c in 0..chunks {
            let off = c * 16;
            let d0 = vsubq_f32(vld1q_f32(out_ptr.add(off) as *const f32), vmean);
            let n0 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off)),
                vmulq_f32(vsubq_f32(d0, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off)),
            );
            vst1q_f32(out_ptr.add(off), n0);

            let d1 = vsubq_f32(vld1q_f32(out_ptr.add(off + 4) as *const f32), vmean);
            let n1 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 4)),
                vmulq_f32(vsubq_f32(d1, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off + 4)),
            );
            vst1q_f32(out_ptr.add(off + 4), n1);

            let d2 = vsubq_f32(vld1q_f32(out_ptr.add(off + 8) as *const f32), vmean);
            let n2 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 8)),
                vmulq_f32(vsubq_f32(d2, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off + 8)),
            );
            vst1q_f32(out_ptr.add(off + 8), n2);

            let d3 = vsubq_f32(vld1q_f32(out_ptr.add(off + 12) as *const f32), vmean);
            let n3 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 12)),
                vmulq_f32(vsubq_f32(d3, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off + 12)),
            );
            vst1q_f32(out_ptr.add(off + 12), n3);
        }
        for i in (chunks * 16)..hidden {
            *out_ptr.add(i) = ((*out_ptr.add(i) - mean) - mean_residual) * inv_std * *g_ptr.add(i)
                + *b_ptr.add(i);
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

        // --- Pass 1: widen four input streams before accumulating the mean ---
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();
        let mut sum4 = _mm256_setzero_pd();
        let mut sum5 = _mm256_setzero_pd();
        let mut sum6 = _mm256_setzero_pd();
        let mut sum7 = _mm256_setzero_pd();
        for c in 0..chunks {
            let off = c * 32;
            let x0 = _mm256_loadu_ps(ptr.add(off));
            let x1 = _mm256_loadu_ps(ptr.add(off + 8));
            let x2 = _mm256_loadu_ps(ptr.add(off + 16));
            let x3 = _mm256_loadu_ps(ptr.add(off + 24));
            sum0 = _mm256_add_pd(sum0, _mm256_cvtps_pd(_mm256_castps256_ps128(x0)));
            sum1 = _mm256_add_pd(sum1, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x0)));
            sum2 = _mm256_add_pd(sum2, _mm256_cvtps_pd(_mm256_castps256_ps128(x1)));
            sum3 = _mm256_add_pd(sum3, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x1)));
            sum4 = _mm256_add_pd(sum4, _mm256_cvtps_pd(_mm256_castps256_ps128(x2)));
            sum5 = _mm256_add_pd(sum5, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x2)));
            sum6 = _mm256_add_pd(sum6, _mm256_cvtps_pd(_mm256_castps256_ps128(x3)));
            sum7 = _mm256_add_pd(sum7, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x3)));
        }
        #[cfg(test)]
        if chunks > 0 {
            simd_mean_tests::record_route(simd_mean_tests::PLAIN_AVX2);
        }
        let combined_sum = _mm256_add_pd(
            _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3)),
            _mm256_add_pd(_mm256_add_pd(sum4, sum5), _mm256_add_pd(sum6, sum7)),
        );
        let sum128 = _mm_add_pd(
            _mm256_castpd256_pd128(combined_sum),
            _mm256_extractf128_pd::<1>(combined_sum),
        );
        let mut total_sum = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum128, sum128));
        for i in (chunks * 32)..hidden {
            total_sum += f64::from(*ptr.add(i));
        }
        let mean_f64 = total_sum / hidden as f64;
        let mean = mean_f64 as f32;
        let mean_residual = mean_f64 - f64::from(mean);

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
        let variance =
            (f64::from(variance) - mean_residual * mean_residual).clamp(0.0, f64::INFINITY) as f32;
        let mean_residual = mean_residual as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // --- Pass 3: normalize ---
        let vinv_std = _mm256_set1_ps(inv_std);
        let vmean_residual = _mm256_set1_ps(mean_residual);
        let out_ptr = row.as_mut_ptr();
        let g_ptr = gamma.as_ptr();
        let b_ptr = beta.as_ptr();

        for c in 0..chunks {
            let off = c * 32;
            for lane_off in (0..32).step_by(8) {
                let p = off + lane_off;
                let d = _mm256_sub_ps(_mm256_loadu_ps(out_ptr.add(p) as *const f32), vmean);
                let scaled = _mm256_mul_ps(_mm256_sub_ps(d, vmean_residual), vinv_std);
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
            *out_ptr.add(i) = ((*out_ptr.add(i) - mean) - mean_residual) * inv_std * *g_ptr.add(i)
                + *b_ptr.add(i);
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

        let mean_f64 = row.iter().map(|&v| f64::from(v)).sum::<f64>() / hidden as f64;
        let mean = mean_f64 as f32;
        let mean_residual = mean_f64 - f64::from(mean);
        let variance = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / hidden as f32;
        let variance =
            (f64::from(variance) - mean_residual * mean_residual).clamp(0.0, f64::INFINITY) as f32;
        let mean_residual = mean_residual as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        for i in 0..hidden {
            row[i] = ((row[i] - mean) - mean_residual) * inv_std * gamma[i] + beta[i];
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

        // --- Pass 1: fused f32 residual add + f64 mean, four input streams.
        // Each lane loads `out` and `dense`, adds them, stores the sum back into
        // `out` (so passes 2/3 read the summed value), and folds the same sum
        // into the mean accumulator -- identical accumulator order to
        // `layer_norm_neon`'s pass 1, just with a fused source value.
        let mut sum0 = vdupq_n_f64(0.0);
        let mut sum1 = vdupq_n_f64(0.0);
        let mut sum2 = vdupq_n_f64(0.0);
        let mut sum3 = vdupq_n_f64(0.0);
        let mut sum4 = vdupq_n_f64(0.0);
        let mut sum5 = vdupq_n_f64(0.0);
        let mut sum6 = vdupq_n_f64(0.0);
        let mut sum7 = vdupq_n_f64(0.0);
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
            sum0 = vaddq_f64(sum0, vcvt_f64_f32(vget_low_f32(x0)));
            sum1 = vaddq_f64(sum1, vcvt_f64_f32(vget_high_f32(x0)));
            sum2 = vaddq_f64(sum2, vcvt_f64_f32(vget_low_f32(x1)));
            sum3 = vaddq_f64(sum3, vcvt_f64_f32(vget_high_f32(x1)));
            sum4 = vaddq_f64(sum4, vcvt_f64_f32(vget_low_f32(x2)));
            sum5 = vaddq_f64(sum5, vcvt_f64_f32(vget_high_f32(x2)));
            sum6 = vaddq_f64(sum6, vcvt_f64_f32(vget_low_f32(x3)));
            sum7 = vaddq_f64(sum7, vcvt_f64_f32(vget_high_f32(x3)));
        }
        #[cfg(test)]
        if chunks > 0 {
            simd_mean_tests::record_route(simd_mean_tests::FUSED_NEON);
        }
        let combined_sum = vaddq_f64(
            vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3)),
            vaddq_f64(vaddq_f64(sum4, sum5), vaddq_f64(sum6, sum7)),
        );
        let mut total_sum = vaddvq_f64(combined_sum);
        for i in (chunks * 16)..hidden {
            let x = *out_ptr.add(i) + *d_ptr.add(i);
            *out_ptr.add(i) = x;
            total_sum += f64::from(x);
        }
        let mean_f64 = total_sum / hidden as f64;
        let mean = mean_f64 as f32;
        let mean_residual = mean_f64 - f64::from(mean);

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
        let variance =
            (f64::from(variance) - mean_residual * mean_residual).clamp(0.0, f64::INFINITY) as f32;
        let mean_residual = mean_residual as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // --- Pass 3: normalize using SIMD: ((x - mean) - mean_residual) * inv_std * gamma + beta.
        // Byte-for-byte the same as `layer_norm_neon`'s pass 3.
        let vinv_std = vdupq_n_f32(inv_std);
        let vmean_residual = vdupq_n_f32(mean_residual);
        let out_ptr = row.as_mut_ptr();
        let g_ptr = gamma.as_ptr();
        let b_ptr = beta.as_ptr();

        for c in 0..chunks {
            let off = c * 16;
            let d0 = vsubq_f32(vld1q_f32(out_ptr.add(off) as *const f32), vmean);
            let n0 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off)),
                vmulq_f32(vsubq_f32(d0, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off)),
            );
            vst1q_f32(out_ptr.add(off), n0);

            let d1 = vsubq_f32(vld1q_f32(out_ptr.add(off + 4) as *const f32), vmean);
            let n1 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 4)),
                vmulq_f32(vsubq_f32(d1, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off + 4)),
            );
            vst1q_f32(out_ptr.add(off + 4), n1);

            let d2 = vsubq_f32(vld1q_f32(out_ptr.add(off + 8) as *const f32), vmean);
            let n2 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 8)),
                vmulq_f32(vsubq_f32(d2, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off + 8)),
            );
            vst1q_f32(out_ptr.add(off + 8), n2);

            let d3 = vsubq_f32(vld1q_f32(out_ptr.add(off + 12) as *const f32), vmean);
            let n3 = vfmaq_f32(
                vld1q_f32(b_ptr.add(off + 12)),
                vmulq_f32(vsubq_f32(d3, vmean_residual), vinv_std),
                vld1q_f32(g_ptr.add(off + 12)),
            );
            vst1q_f32(out_ptr.add(off + 12), n3);
        }
        for i in (chunks * 16)..hidden {
            *out_ptr.add(i) = ((*out_ptr.add(i) - mean) - mean_residual) * inv_std * *g_ptr.add(i)
                + *b_ptr.add(i);
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

        // --- Pass 1: fused f32 residual add + f64 mean, four input streams.
        // Same accumulator order as `layer_norm_avx2`'s pass 1, fused source value.
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();
        let mut sum4 = _mm256_setzero_pd();
        let mut sum5 = _mm256_setzero_pd();
        let mut sum6 = _mm256_setzero_pd();
        let mut sum7 = _mm256_setzero_pd();
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
            sum0 = _mm256_add_pd(sum0, _mm256_cvtps_pd(_mm256_castps256_ps128(x0)));
            sum1 = _mm256_add_pd(sum1, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x0)));
            sum2 = _mm256_add_pd(sum2, _mm256_cvtps_pd(_mm256_castps256_ps128(x1)));
            sum3 = _mm256_add_pd(sum3, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x1)));
            sum4 = _mm256_add_pd(sum4, _mm256_cvtps_pd(_mm256_castps256_ps128(x2)));
            sum5 = _mm256_add_pd(sum5, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x2)));
            sum6 = _mm256_add_pd(sum6, _mm256_cvtps_pd(_mm256_castps256_ps128(x3)));
            sum7 = _mm256_add_pd(sum7, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(x3)));
        }
        #[cfg(test)]
        if chunks > 0 {
            simd_mean_tests::record_route(simd_mean_tests::FUSED_AVX2);
        }
        let combined_sum = _mm256_add_pd(
            _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3)),
            _mm256_add_pd(_mm256_add_pd(sum4, sum5), _mm256_add_pd(sum6, sum7)),
        );
        let sum128 = _mm_add_pd(
            _mm256_castpd256_pd128(combined_sum),
            _mm256_extractf128_pd::<1>(combined_sum),
        );
        let mut total_sum = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum128, sum128));
        for i in (chunks * 32)..hidden {
            let x = *out_ptr.add(i) + *d_ptr.add(i);
            *out_ptr.add(i) = x;
            total_sum += f64::from(x);
        }
        let mean_f64 = total_sum / hidden as f64;
        let mean = mean_f64 as f32;
        let mean_residual = mean_f64 - f64::from(mean);

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
        let variance =
            (f64::from(variance) - mean_residual * mean_residual).clamp(0.0, f64::INFINITY) as f32;
        let mean_residual = mean_residual as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // --- Pass 3: normalize. Byte-for-byte the same as `layer_norm_avx2`'s pass 3.
        let vinv_std = _mm256_set1_ps(inv_std);
        let vmean_residual = _mm256_set1_ps(mean_residual);
        let out_ptr = row.as_mut_ptr();
        let g_ptr = gamma.as_ptr();
        let b_ptr = beta.as_ptr();

        for c in 0..chunks {
            let off = c * 32;
            for lane_off in (0..32).step_by(8) {
                let p = off + lane_off;
                let d = _mm256_sub_ps(_mm256_loadu_ps(out_ptr.add(p) as *const f32), vmean);
                let scaled = _mm256_mul_ps(_mm256_sub_ps(d, vmean_residual), vinv_std);
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
            *out_ptr.add(i) = ((*out_ptr.add(i) - mean) - mean_residual) * inv_std * *g_ptr.add(i)
                + *b_ptr.add(i);
        }
    }
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn scalar_layer_norm_constant_rows_preserve_beta() {
        let hidden = 16;
        let constant = f32::from_bits(0x3f80_0003);
        let gamma: Vec<_> = (0..hidden).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let beta: Vec<_> = (0..hidden).map(|i| (i as f32 - 7.0) * 0.25).collect();
        let mut x: Vec<_> = [constant, -constant, 0.0]
            .into_iter()
            .flat_map(|value| vec![value; hidden])
            .collect();

        layer_norm_scalar(&mut x, &gamma, &beta, hidden, 1e-12);

        for row in x.chunks_exact(hidden) {
            assert_eq!(row, beta.as_slice());
        }
    }

    #[test]
    fn scalar_residual_layer_norm_constant_rows_preserve_beta() {
        let hidden = 16;
        let constant = f32::from_bits(0x3f80_0003);
        let gamma: Vec<_> = (0..hidden).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let beta: Vec<_> = (0..hidden).map(|i| (i as f32 - 7.0) * 0.25).collect();
        let mut out = Vec::new();
        let mut dense = Vec::new();
        for (value, residual) in [(constant, 1.0), (-constant, -1.0), (0.0, 1.0)] {
            out.extend(vec![value - residual; hidden]);
            dense.extend(vec![residual; hidden]);
        }

        residual_add_layer_norm_scalar(&mut out, &dense, &gamma, &beta, hidden, 1e-12);

        for row in out.chunks_exact(hidden) {
            assert_eq!(row, beta.as_slice());
        }
    }

    #[test]
    fn scalar_layer_norm_affine_nonconstant_control() {
        let hidden = 16;
        let gamma = vec![2.0; hidden];
        let beta = vec![0.5; hidden];
        let mut plain: Vec<_> = (0..hidden)
            .map(|i| if i % 2 == 0 { 1.0 } else { 3.0 })
            .collect();
        let mut fused = vec![1.0; hidden];
        let dense: Vec<_> = (0..hidden)
            .map(|i| if i % 2 == 0 { 0.0 } else { 2.0 })
            .collect();
        let expected: Vec<_> = (0..hidden)
            .map(|i| if i % 2 == 0 { -1.5 } else { 2.5 })
            .collect();

        layer_norm_scalar(&mut plain, &gamma, &beta, hidden, 1e-12);
        residual_add_layer_norm_scalar(&mut fused, &dense, &gamma, &beta, hidden, 1e-12);

        assert_eq!(plain, expected);
        assert_eq!(fused, expected);
    }

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

            let mut fused_scalar = out0.clone();
            residual_add_layer_norm_scalar(&mut fused_scalar, &dense, &gamma, &beta, hidden, eps);
            let mut reference_scalar = out0;
            for i in 0..(rows * hidden) {
                reference_scalar[i] += dense[i];
            }
            layer_norm_scalar(&mut reference_scalar, &gamma, &beta, hidden, eps);
            for (i, (fused, reference)) in fused_scalar.iter().zip(&reference_scalar).enumerate() {
                assert_eq!(
                    fused.to_bits(),
                    reference.to_bits(),
                    "hidden={hidden}[{i}]: scalar fusion must be bit-identical"
                );
            }
        }
    }
}

#[cfg(test)]
mod simd_mean_tests {
    use super::*;
    use std::cell::Cell;

    #[cfg(target_arch = "aarch64")]
    pub(super) const PLAIN_NEON: u8 = 1;
    #[cfg(target_arch = "aarch64")]
    pub(super) const FUSED_NEON: u8 = 2;
    #[cfg(target_arch = "x86_64")]
    pub(super) const PLAIN_AVX2: u8 = 4;
    #[cfg(target_arch = "x86_64")]
    pub(super) const FUSED_AVX2: u8 = 8;

    std::thread_local! {
        static ROUTES: Cell<u8> = const { Cell::new(0) };
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    pub(super) fn record_route(route: u8) {
        ROUTES.with(|seen| seen.set(seen.get() | route));
    }

    fn take_routes() -> u8 {
        ROUTES.with(|seen| seen.replace(0))
    }

    fn expected_routes(_hidden: usize) -> (u8, u8) {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        let config = simd_config();
        #[cfg(target_arch = "aarch64")]
        if config.neon_enabled && _hidden >= 16 {
            return (PLAIN_NEON, FUSED_NEON);
        }
        #[cfg(target_arch = "x86_64")]
        if config.avx2_enabled && config.fma_enabled && _hidden >= 32 {
            return (PLAIN_AVX2, FUSED_AVX2);
        }
        (0, 0)
    }

    fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
        let mut sum: f64 = 0.0;
        let mut correction = 0.0;
        for value in values {
            let next = sum + value;
            correction += if sum.abs() >= value.abs() {
                (sum - next) + value
            } else {
                (value - next) + sum
            };
            sum = next;
        }
        sum + correction
    }

    fn reference_moments(row: &[f32]) -> (f64, f64) {
        let mean = compensated_sum(row.iter().map(|&v| f64::from(v))) / row.len() as f64;
        let variance = compensated_sum(row.iter().map(|&v| {
            let delta = f64::from(v) - mean;
            delta * delta
        })) / row.len() as f64;
        (mean, variance)
    }

    fn assert_bits_equal(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "index={index}");
        }
    }

    fn check_public_pair(
        out: &[f32],
        dense: &[f32],
        gamma: &[f32],
        beta: &[f32],
        hidden: usize,
        eps: f32,
    ) -> Vec<f32> {
        let summed: Vec<f32> = out.iter().zip(dense).map(|(&a, &b)| a + b).collect();
        let expected = expected_routes(hidden);
        take_routes();
        let mut plain = summed.clone();
        layer_norm(&mut plain, gamma, beta, hidden, eps);
        assert_eq!(take_routes(), expected.0, "plain route hidden={hidden}");
        let mut fused = out.to_vec();
        residual_add_layer_norm(&mut fused, dense, gamma, beta, hidden, eps);
        assert_eq!(take_routes(), expected.1, "fused route hidden={hidden}");
        assert_bits_equal(&fused, &plain);

        for (row_index, (input, output)) in summed
            .chunks_exact(hidden)
            .zip(plain.chunks_exact(hidden))
            .enumerate()
        {
            let (mean, variance) = reference_moments(input);
            let denominator = (variance + f64::from(eps)).sqrt();
            for (index, (&value, &got)) in input.iter().zip(output).enumerate() {
                let want = (f64::from(value) - mean) / denominator * f64::from(gamma[index])
                    + f64::from(beta[index]);
                let error = (f64::from(got) - want).abs();
                let limit = 1e-4 * f64::from(got).abs().max(want.abs()) + 1e-5;
                assert!(
                    error <= limit,
                    "hidden={hidden} row={row_index} index={index}: got={got}, reference={want}, error={error}, limit={limit}"
                );
            }
        }
        plain
    }

    fn affine(hidden: usize) -> (Vec<f32>, Vec<f32>) {
        let gamma = (0..hidden).map(|i| (i % 9) as f32 * 0.25 - 0.75).collect();
        let beta = (0..hidden).map(|i| (i % 7) as f32 * 0.125 - 0.25).collect();
        (gamma, beta)
    }

    fn rounded_mean_arm(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        simd: bool,
        fused: bool,
    ) -> Vec<f32> {
        let hidden = input.len();
        let mut output = input.to_vec();
        let mut dense = Vec::new();
        if fused {
            for value in &mut output {
                *value *= 0.5;
            }
            dense = output.clone();
            for (&half, &value) in output.iter().zip(input) {
                assert_eq!((half + half).to_bits(), value.to_bits());
            }
        }
        if !simd {
            if fused {
                residual_add_layer_norm_scalar(&mut output, &dense, gamma, beta, hidden, 1e-5);
            } else {
                layer_norm_scalar(&mut output, gamma, beta, hidden, 1e-5);
            }
            return output;
        }

        take_routes();
        #[cfg(target_arch = "aarch64")]
        {
            assert!(simd_config().neon_enabled, "rounded mean requires NEON");
            // SAFETY: NEON is enabled and all fixtures contain one complete row.
            unsafe {
                if fused {
                    residual_add_layer_norm_neon(&mut output, &dense, gamma, beta, hidden, 1e-5);
                } else {
                    layer_norm_neon(&mut output, gamma, beta, hidden, 1e-5);
                }
            }
            assert_eq!(take_routes(), if fused { FUSED_NEON } else { PLAIN_NEON });
        }
        #[cfg(target_arch = "x86_64")]
        {
            let config = simd_config();
            assert!(
                config.avx2_enabled && config.fma_enabled,
                "rounded mean requires AVX2+FMA"
            );
            // SAFETY: AVX2+FMA are enabled and all fixtures contain one complete row.
            unsafe {
                if fused {
                    residual_add_layer_norm_avx2(&mut output, &dense, gamma, beta, hidden, 1e-5);
                } else {
                    layer_norm_avx2(&mut output, gamma, beta, hidden, 1e-5);
                }
            }
            assert_eq!(take_routes(), if fused { FUSED_AVX2 } else { PLAIN_AVX2 });
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert!(!simd, "rounded mean requires a supported SIMD target");
        eprintln!(
            "rounded mean route: arch={} fused={fused} hidden={hidden}",
            std::env::consts::ARCH
        );
        output
    }

    fn check_rounded_mean_analytic(simd: bool, fused: bool) {
        for hidden in [896, 898] {
            let gamma = vec![1.0; hidden];
            let beta = vec![0.0; hidden];
            for low in [33_554_448.0f32, 33_554_452.0] {
                let input: Vec<_> = (0..hidden).map(|i| low + (i % 2) as f32 * 4.0).collect();
                assert_eq!(reference_moments(&input), (f64::from(low) + 2.0, 4.0));
                let output = rounded_mean_arm(&input, &gamma, &beta, simd, fused);
                for (index, &got) in output.iter().enumerate() {
                    let want = if index % 2 == 0 { -1.0 } else { 1.0 };
                    assert!(
                        (got - want).abs() <= 1e-5,
                        "rounded mean analytic simd={simd} fused={fused} hidden={hidden} low={low} index={index}: got={got}, want={want}"
                    );
                }
            }
        }
    }

    #[test]
    fn rounded_mean_scalar_plain_matches_analytic() {
        check_rounded_mean_analytic(false, false);
    }

    #[test]
    fn rounded_mean_scalar_fused_matches_analytic() {
        check_rounded_mean_analytic(false, true);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn rounded_mean_simd_plain_matches_analytic() {
        check_rounded_mean_analytic(true, false);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn rounded_mean_simd_fused_matches_analytic() {
        check_rounded_mean_analytic(true, true);
    }

    fn check_rounded_mean_equivalence(ordinary: bool) {
        for hidden in [896, 898] {
            let (input, gamma, beta) = if ordinary {
                let input = (0..hidden)
                    .map(|i| (((i * 13) % 31) as f32 - 15.0) * 0.125)
                    .collect::<Vec<_>>();
                let (gamma, beta) = affine(hidden);
                (input, gamma, beta)
            } else {
                let input = (0..hidden)
                    .map(|i| 33_554_448.0 + (i % 2) as f32 * 4.0)
                    .collect();
                (input, vec![1.0; hidden], vec![0.0; hidden])
            };
            let (mean, variance) = reference_moments(&input);
            let scalar = rounded_mean_arm(&input, &gamma, &beta, false, false);
            let scalar_fused = rounded_mean_arm(&input, &gamma, &beta, false, true);
            assert_bits_equal(&scalar, &scalar_fused);
            let halves: Vec<_> = input.iter().map(|&value| value * 0.5).collect();
            let public = check_public_pair(&halves, &halves, &gamma, &beta, hidden, 1e-5);
            if ordinary {
                eprintln!(
                    "rounded mean ordinary arch={} hidden={hidden} scalar_bits={:?} public_bits={:?}",
                    std::env::consts::ARCH,
                    scalar
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    public
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            for (index, ((&got, &scalar), &value)) in
                public.iter().zip(&scalar).zip(&input).enumerate()
            {
                assert!(
                    scalar.is_finite() && got.is_finite(),
                    "rounded mean finite ordinary={ordinary} hidden={hidden} index={index}"
                );
                let want = (f64::from(value) - mean) / (variance + f64::from(1e-5f32)).sqrt()
                    * f64::from(gamma[index])
                    + f64::from(beta[index]);
                let limit = 1e-4 * f64::from(scalar).abs().max(want.abs()) + 1e-5;
                assert!(
                    (f64::from(scalar) - want).abs() <= limit,
                    "rounded mean scalar reference ordinary={ordinary} hidden={hidden} index={index}"
                );
                let limit = 1e-4 * got.abs().max(scalar.abs()) + 1e-5;
                assert!(
                    (got - scalar).abs() <= limit,
                    "rounded mean equivalence ordinary={ordinary} hidden={hidden} index={index}: got={got}, scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn rounded_mean_ordinary_rows_preserve_equivalence() {
        check_rounded_mean_equivalence(true);
    }

    #[test]
    fn rounded_mean_issue_rows_preserve_equivalence() {
        check_rounded_mean_equivalence(false);
    }

    #[test]
    fn rounded_mean_overflow_keeps_infinite_variance() {
        let low = f32::from_bits(0x7180_0000);
        let input: Vec<_> = (0..64)
            .map(|i| if i % 2 == 0 { low } else { low.next_up() })
            .collect();
        let gamma = vec![1.0; input.len()];
        let beta = vec![0.0; input.len()];
        for simd in [
            false,
            cfg!(any(target_arch = "aarch64", target_arch = "x86_64")),
        ] {
            for fused in [false, true] {
                let output = rounded_mean_arm(&input, &gamma, &beta, simd, fused);
                assert!(
                    output.iter().all(|&value| value == 0.0),
                    "rounded mean correction must preserve infinite variance simd={simd} fused={fused}"
                );
            }
        }
    }

    #[test]
    fn independent_moments_match_analytic_controls() {
        assert_eq!(reference_moments(&[-3.0, -1.0, 1.0, 3.0]), (0.0, 5.0));
        for center in [1.5f32, -1.5, 0.1, 0.3, 1.7, -2.3, 123.456] {
            let mut row = vec![center; 48];
            row[0] = center.next_down();
            row[47] = center.next_up();
            let step = f64::from(center.next_up()) - f64::from(center);
            assert_eq!(
                reference_moments(&row),
                (f64::from(center), 2.0 * step * step / 48.0)
            );
        }
    }

    #[test]
    fn public_constant_rows_preserve_positive_beta() {
        for hidden in [1, 15, 16, 17, 30, 31, 32, 33, 64, 65, 384, 768, 1024] {
            let gamma = vec![1.0; hidden];
            let beta = vec![0.5; hidden];
            for constant in [0.1f32, 0.3, 1.7, -2.3, 123.456, f32::from_bits(0x3f80_0003)] {
                let out: Vec<f32> = [constant, -constant]
                    .into_iter()
                    .flat_map(|value| vec![value * 0.5; hidden])
                    .collect();
                let output = check_public_pair(&out, &out, &gamma, &beta, hidden, 1e-12);
                for value in output {
                    assert_eq!(
                        value.to_bits(),
                        0.5f32.to_bits(),
                        "hidden={hidden} constant={constant}"
                    );
                }
            }
        }
        eprintln!("simd layer norm: completed independent constant controls");
    }

    #[test]
    fn public_ordinary_and_balanced_rows_match_mathematical_reference() {
        for hidden in (1..=65).chain([100, 127, 128, 129, 384, 768, 896, 1024, 2048, 4096]) {
            let (gamma, beta) = affine(hidden);
            let input: Vec<f32> = (0..3 * hidden)
                .map(|i| (((i * 13 + i / hidden * 7) % 31) as f32 - 15.0) * 0.125)
                .collect();
            let dense: Vec<f32> = (0..input.len())
                .map(|i| ((i % 7) as f32 - 3.0) * 0.125)
                .collect();
            let out: Vec<f32> = input.iter().zip(&dense).map(|(&x, &d)| x - d).collect();
            for eps in [1e-12, 1e-6] {
                let _ = check_public_pair(&out, &dense, &gamma, &beta, hidden, eps);
            }
            if hidden > 1 {
                let _ = check_public_pair(&out, &dense, &gamma, &beta, hidden, 0.0);
            }

            let mut out = Vec::new();
            let mut dense = Vec::new();
            for (row_index, center) in [1.5f32, -1.5, 0.1, 0.3, 1.7, -2.3, 123.456]
                .into_iter()
                .enumerate()
            {
                let mut row = vec![center; hidden];
                if hidden > 1 {
                    let left = row_index * 7 % hidden;
                    let right = (left + hidden - 1) % hidden;
                    row[left] = center.next_down();
                    row[right] = center.next_up();
                }
                let residual = center * 0.5;
                out.extend(row.iter().map(|&value| value - residual));
                dense.extend(vec![residual; hidden]);
            }
            for eps in [1e-12, 1e-6] {
                let _ = check_public_pair(&out, &dense, &gamma, &beta, hidden, eps);
            }
        }
        eprintln!("simd layer norm: completed independent ordinary/balanced and tail controls");
    }

    #[test]
    fn fused_residual_rounding_matches_explicit_f32_addition() {
        let values = [16_777_216.0f32, -16_777_216.0, 4.0, -8.0, 0.25, -0.5];
        let residuals = [1.0f32, -1.0, 0.125, -0.25, 1_048_576.0, -1_048_576.0];
        assert_eq!((values[0] + residuals[0]).to_bits(), values[0].to_bits());
        assert_eq!((values[1] + residuals[1]).to_bits(), values[1].to_bits());
        for hidden in [31, 32, 33, 65, 128] {
            let (gamma, beta) = affine(hidden);
            let out: Vec<f32> = (0..2 * hidden).map(|i| values[i % values.len()]).collect();
            let dense: Vec<f32> = (0..2 * hidden)
                .map(|i| residuals[i % residuals.len()])
                .collect();
            let _ = check_public_pair(&out, &dense, &gamma, &beta, hidden, 1e-6);
        }
        eprintln!("simd layer norm: completed rounded-residual and fusion controls");
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn public_vector_routes_and_direct_arms_agree() {
        let hidden = 64;
        let expected = expected_routes(hidden);
        if expected == (0, 0) {
            return;
        }
        let (gamma, beta) = affine(hidden);
        let out: Vec<f32> = (0..2 * hidden)
            .map(|i| (i % 13) as f32 * 0.25 - 1.0)
            .collect();
        let dense: Vec<f32> = (0..2 * hidden).map(|i| (i % 3) as f32 * 0.125).collect();
        let public = check_public_pair(&out, &dense, &gamma, &beta, hidden, 1e-6);
        let mut direct: Vec<f32> = out.iter().zip(&dense).map(|(&a, &b)| a + b).collect();
        let mut fused = out.clone();

        #[cfg(target_arch = "aarch64")]
        {
            assert!(simd_config().neon_enabled);
            // SAFETY: NEON is enabled, and the fixtures have complete rows/affine arrays.
            unsafe {
                layer_norm_neon(&mut direct, &gamma, &beta, hidden, 1e-6);
                residual_add_layer_norm_neon(&mut fused, &dense, &gamma, &beta, hidden, 1e-6);
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let config = simd_config();
            assert!(config.avx2_enabled && config.fma_enabled);
            // SAFETY: both features are enabled, and all fixture lengths match.
            unsafe {
                layer_norm_avx2(&mut direct, &gamma, &beta, hidden, 1e-6);
                residual_add_layer_norm_avx2(&mut fused, &dense, &gamma, &beta, hidden, 1e-6);
            }
        }
        assert_eq!(take_routes(), expected.0 | expected.1);
        assert_bits_equal(&direct, &public);
        assert_bits_equal(&fused, &public);
        eprintln!(
            "simd layer norm: completed public vector routes and direct controls on {}",
            std::env::consts::ARCH
        );
    }
}

#[cfg(test)]
mod fused_correction_tests;

#[cfg(test)]
mod pivot_mean_tests;
