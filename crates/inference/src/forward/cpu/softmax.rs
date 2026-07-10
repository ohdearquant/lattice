//! Softmax attention helpers and fast exponential approximation.
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
    use crate::attention::softmax_row::{
        finalize_row, is_masked_neg_inf, row_fails_closed_pre_exp, row_max_and_any_nan,
    };

    for h in 0..num_heads {
        for s in 0..seq_len {
            let start = (h * seq_len + s) * seq_len;
            let row = &mut x[start..start + seq_len];
            let (max_val, any_nan) = row_max_and_any_nan(row);
            if row_fails_closed_pre_exp(max_val, any_nan) {
                // Either every score in this row is masked (-inf, no valid key to
                // attend to) or a NaN score is present (ADR-080 C1 / #740: a NaN
                // score used to survive as a single bad lane instead of failing
                // the whole row closed). Zero the whole row before computing any
                // exp — this keeps every backend fail-closed and identical.
                row.fill(0.0);
                continue;
            }
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                // `fast_exp` clamps its input to a finite floor, so a masked
                // `-inf` lane would otherwise round-trip into a small nonzero
                // weight (#740) instead of exact zero. Guard it explicitly.
                *val = if is_masked_neg_inf(*val) {
                    0.0
                } else {
                    fast_exp(*val - max_val)
                };
                sum += *val;
            }
            finalize_row(row, sum);
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
/// Intentionally kept instead of the more accurate polynomial exp from elementwise — the
/// Schraudolph systematic bias cancels in softmax normalization, and it's ~3x faster.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn neon_exp_f32(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
    let x = vminq_f32(x, vdupq_n_f32(88.0));
    let t = vfmaq_f32(vdupq_n_f32(1_065_353_216.0), x, vdupq_n_f32(12_102_203.0));
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

    const UNROLL: usize = 4;
    const CHUNK: usize = 4 * UNROLL;

    for h in 0..num_heads {
        for s in 0..seq_len {
            let start = (h * seq_len + s) * seq_len;
            let row = &mut x[start..start + seq_len];
            let ptr = row.as_mut_ptr();
            let n = row.len();
            let chunks = n / CHUNK;

            // --- Step 1: Find row max with 4x unrolling ---
            // ADR-080 C1 (#785 round-1 perf): use the plain FMAX intrinsics
            // (vmaxq/vmaxvq), which PROPAGATE a NaN operand, instead of the
            // maxNum FMAXNM intrinsics (vmaxnmq/vmaxnmvq) paired with an
            // explicit `vceqq`/`vandq` NaN-tracking mask. The C1 contract
            // requires a NaN score to fail the WHOLE row closed (not survive
            // as a single bad lane, #740) — the same outcome FMAX gives for
            // free via `!max_val.is_finite()` below, mirroring the scalar
            // reference (`row_max_and_any_nan`/`row_fails_closed_pre_exp`) and
            // `attention/decode.rs`'s `softmax_decode_neon`. The prior
            // FMAXNM-drops-NaN + explicit-track approach was redundant work
            // once the scalar path itself started failing closed on NaN;
            // there is no longer a "normalize around the NaN" case to
            // preserve. `-inf` masked lanes still flow through `max`
            // correctly (they are not NaN and always lose to a real value).
            let mut m0 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut m1 = m0;
            let mut m2 = m0;
            let mut m3 = m0;
            for c in 0..chunks {
                let base = c * CHUNK;
                m0 = vmaxq_f32(m0, vld1q_f32(ptr.add(base) as *const f32));
                m1 = vmaxq_f32(m1, vld1q_f32(ptr.add(base + 4) as *const f32));
                m2 = vmaxq_f32(m2, vld1q_f32(ptr.add(base + 8) as *const f32));
                m3 = vmaxq_f32(m3, vld1q_f32(ptr.add(base + 12) as *const f32));
            }
            let vmax = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3));
            let mut max_val = vmaxvq_f32(vmax);
            for i in (chunks * CHUNK)..n {
                let v = *ptr.add(i);
                // `f32::max` returns the finite operand when the other is
                // NaN, which would drop a tail NaN AND erase an already-NaN
                // chunk max behind a later finite tail lane. Force NaN to
                // propagate across the chunk/tail boundary explicitly,
                // exactly like `softmax_decode_neon`.
                if max_val.is_nan() || v.is_nan() {
                    max_val = f32::NAN;
                    break;
                }
                max_val = max_val.max(v);
            }

            // All-masked row (max stayed -inf), a `+inf` max, or any NaN score
            // (NaN propagated through FMAX above, so `is_finite()` alone now
            // catches it): zero it and skip, matching the scalar reference
            // (ADR-080 C1 contract).
            if !max_val.is_finite() {
                row.fill(0.0);
                continue;
            }

            // --- Step 2: Subtract max + fast exp + accumulate sum ---
            // A structurally-masked `-inf` lane must produce exact `0.0`, not the
            // small nonzero value `neon_exp`'s finite-floor clamp would otherwise
            // produce (#740); `vceqq_f32(v, neg_inf)` + `vbslq_f32` blends that lane
            // to exact zero without a scalar fallback.
            let vmax_val = vdupq_n_f32(max_val);
            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
            let zero = vdupq_n_f32(0.0);
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = s0;
            let mut s2 = s0;
            let mut s3 = s0;
            for c in 0..chunks {
                let base = c * CHUNK;
                let raw0 = vld1q_f32(ptr.add(base) as *const f32);
                let raw1 = vld1q_f32(ptr.add(base + 4) as *const f32);
                let raw2 = vld1q_f32(ptr.add(base + 8) as *const f32);
                let raw3 = vld1q_f32(ptr.add(base + 12) as *const f32);
                let e0 = vbslq_f32(
                    vceqq_f32(raw0, neg_inf),
                    zero,
                    neon_exp_f32(vsubq_f32(raw0, vmax_val)),
                );
                let e1 = vbslq_f32(
                    vceqq_f32(raw1, neg_inf),
                    zero,
                    neon_exp_f32(vsubq_f32(raw1, vmax_val)),
                );
                let e2 = vbslq_f32(
                    vceqq_f32(raw2, neg_inf),
                    zero,
                    neon_exp_f32(vsubq_f32(raw2, vmax_val)),
                );
                let e3 = vbslq_f32(
                    vceqq_f32(raw3, neg_inf),
                    zero,
                    neon_exp_f32(vsubq_f32(raw3, vmax_val)),
                );
                vst1q_f32(ptr.add(base), e0);
                vst1q_f32(ptr.add(base + 4), e1);
                vst1q_f32(ptr.add(base + 8), e2);
                vst1q_f32(ptr.add(base + 12), e3);
                s0 = vaddq_f32(s0, e0);
                s1 = vaddq_f32(s1, e1);
                s2 = vaddq_f32(s2, e2);
                s3 = vaddq_f32(s3, e3);
            }
            let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            for i in (chunks * CHUNK)..n {
                let raw = *ptr.add(i);
                let e = if raw == f32::NEG_INFINITY {
                    0.0
                } else {
                    fast_exp(raw - max_val)
                };
                *ptr.add(i) = e;
                sum += e;
            }

            // --- Step 3: Divide by sum with 4x unrolling ---
            if sum.is_finite() && sum > 0.0 {
                let vinv = vdupq_n_f32(1.0 / sum);
                for c in 0..chunks {
                    let base = c * CHUNK;
                    vst1q_f32(
                        ptr.add(base),
                        vmulq_f32(vld1q_f32(ptr.add(base) as *const f32), vinv),
                    );
                    vst1q_f32(
                        ptr.add(base + 4),
                        vmulq_f32(vld1q_f32(ptr.add(base + 4) as *const f32), vinv),
                    );
                    vst1q_f32(
                        ptr.add(base + 8),
                        vmulq_f32(vld1q_f32(ptr.add(base + 8) as *const f32), vinv),
                    );
                    vst1q_f32(
                        ptr.add(base + 12),
                        vmulq_f32(vld1q_f32(ptr.add(base + 12) as *const f32), vinv),
                    );
                }
                for i in (chunks * CHUNK)..n {
                    *ptr.add(i) *= 1.0 / sum;
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

            // --- Step 1: Find row max using SIMD, plus an explicit NaN scan ---
            // `_mm256_max_ps` (MAXPS) drops a lone NaN operand like `f32::max`, so a
            // NaN score would otherwise survive undetected as a single bad lane
            // (ADR-080 C1 / #740) instead of failing the whole row closed.
            // `_mm256_cmp_ps(v, v, _CMP_EQ_OQ)` is all-ones for every non-NaN lane
            // (ordered-equal comparisons are false for NaN), so ANDing that mask
            // across chunks gives an explicit any-NaN flag.
            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut nan_free = _mm256_set1_ps(f32::from_bits(u32::MAX));
            for c in 0..chunks {
                // SAFETY: c * 8 + 7 < chunks * 8 <= n, within row bounds.
                let v = _mm256_loadu_ps(ptr.add(c * 8) as *const f32);
                vmax = _mm256_max_ps(vmax, v);
                nan_free = _mm256_and_ps(nan_free, _mm256_cmp_ps(v, v, _CMP_EQ_OQ));
            }
            // Horizontal max reduction
            let hi128 = _mm256_extractf128_ps(vmax, 1);
            let lo128 = _mm256_castps256_ps128(vmax);
            let max128 = _mm_max_ps(lo128, hi128);
            let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
            let max32 = _mm_max_ps(max64, _mm_movehdup_ps(max64));
            let mut max_val = _mm_cvtss_f32(max32);
            // Horizontal AND reduction of the NaN-free mask (as integers).
            let nan_free_i = _mm256_castps_si256(nan_free);
            let hi128_i = _mm256_extractf128_si256(nan_free_i, 1);
            let lo128_i = _mm256_castsi256_si128(nan_free_i);
            let and128 = _mm_and_si128(lo128_i, hi128_i);
            let and64 = _mm_and_si128(and128, _mm_srli_si128(and128, 8));
            let and32 = _mm_and_si128(and64, _mm_srli_si128(and64, 4));
            let mut any_nan = (_mm_cvtsi128_si32(and32) as u32) != u32::MAX;
            for i in (chunks * 8)..n {
                let v = *ptr.add(i);
                if v.is_nan() {
                    any_nan = true;
                } else {
                    max_val = max_val.max(v);
                }
            }

            // All-masked row (max stayed -inf), a `+inf` max, or any NaN score: zero
            // it and skip, matching the scalar reference (ADR-080 C1 contract).
            if any_nan || !max_val.is_finite() {
                row.fill(0.0);
                continue;
            }

            // --- Step 2: Subtract max + fast exp + accumulate sum (all SIMD) ---
            // A structurally-masked `-inf` lane must produce exact `0.0`, not the
            // small nonzero value `fast_exp_avx2`'s finite-floor clamp would
            // otherwise produce (#740); blend that lane to exact zero via the
            // NaN-free-style ordered-equal compare against `-inf`.
            let vmax_val = _mm256_set1_ps(max_val);
            let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut vsum = _mm256_setzero_ps();
            for c in 0..chunks {
                let off = c * 8;
                let raw = _mm256_loadu_ps(ptr.add(off) as *const f32);
                let is_masked = _mm256_cmp_ps(raw, neg_inf, _CMP_EQ_OQ);
                let e_full = fast_exp_avx2(_mm256_sub_ps(raw, vmax_val));
                let e = _mm256_andnot_ps(is_masked, e_full);
                _mm256_storeu_ps(ptr.add(off), e);
                vsum = _mm256_add_ps(vsum, e);
            }
            let mut sum = hsum_m256(vsum);
            for i in (chunks * 8)..n {
                let raw = *ptr.add(i);
                let e = if raw == f32::NEG_INFINITY {
                    0.0
                } else {
                    fast_exp(raw - max_val)
                };
                *ptr.add(i) = e;
                sum += e;
            }

            // --- Step 3: Divide by sum using SIMD ---
            if sum.is_finite() && sum > 0.0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[seq_len, seq_len]` single-head score matrix where query row 0
    /// holds `row` (padded/truncated to `seq_len`) and every other query row
    /// holds benign, well-formed finite scores. Returns `(matrix, seq_len)`.
    fn single_row_matrix(row: &[f32]) -> (Vec<f32>, usize) {
        let seq_len = row.len();
        let mut x = vec![0.0f32; seq_len * seq_len];
        x[..seq_len].copy_from_slice(row);
        for qi in 1..seq_len {
            for ki in 0..seq_len {
                x[qi * seq_len + ki] = 0.1 * (qi as f32 + 1.0) + 0.01 * (ki as f32);
            }
        }
        (x, seq_len)
    }

    /// A single NaN score anywhere in the row must zero the WHOLE row, not just
    /// that lane (#740). RED before the fix: the row max fold silently dropped
    /// the NaN operand, so only the NaN's own output lane stayed corrupted while
    /// the rest of the row normalized around the (correct) finite max.
    #[test]
    fn scalar_nan_lane_zeros_whole_row() {
        let (mut x, seq_len) = single_row_matrix(&[1.0, f32::NAN, 2.0, 0.5]);
        softmax_attention_scalar(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// A `+inf` score must zero the whole row (fail-closed on non-finite max).
    #[test]
    fn scalar_pos_inf_zeros_whole_row() {
        let (mut x, seq_len) = single_row_matrix(&[1.0, f32::INFINITY, 2.0]);
        softmax_attention_scalar(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// An all-masked (`-inf`) row must zero out rather than compute `fast_exp(NaN)`.
    #[test]
    fn scalar_all_neg_inf_zeros_row() {
        let (mut x, seq_len) = single_row_matrix(&[f32::NEG_INFINITY; 4]);
        softmax_attention_scalar(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// A masked (`-inf`) lane mixed with valid scores must produce EXACT `0.0`,
    /// not the small nonzero value `fast_exp`'s finite-floor clamp would
    /// otherwise leak (#740). RED before the fix: `fast_exp(-inf - max)` clamped
    /// to `fast_exp(-87.0)`, a tiny but nonzero weight.
    #[test]
    fn scalar_masked_lane_is_exact_zero() {
        let (mut x, seq_len) = single_row_matrix(&[1.0, f32::NEG_INFINITY, 0.5]);
        softmax_attention_scalar(&mut x, seq_len, 1);
        assert_eq!(x[1], 0.0, "masked lane must be exact zero, got {}", x[1]);
        assert!(x[..seq_len].iter().all(|v| v.is_finite()));
        let sum: f32 = x[..seq_len].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row must still sum to ~1: {:?}",
            &x[..seq_len]
        );
    }

    /// Multiple heads/rows in one call: only the poisoned row zeros, sibling
    /// rows in the same batched buffer stay well-formed.
    #[test]
    fn scalar_only_poisoned_row_zeros() {
        let seq_len = 2;
        let num_heads = 1;
        // head 0, row 0 (query 0): NaN present. head 0, row 1 (query 1): clean.
        let mut x = vec![f32::NAN, 1.0, 0.3, 0.7];
        softmax_attention_scalar(&mut x, seq_len, num_heads);
        assert!(x[0..2].iter().all(|&v| v == 0.0), "row 0 must zero: {x:?}");
        assert!(x[2..4].iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use super::*;

        fn run_neon(x: &mut [f32], seq_len: usize, num_heads: usize) {
            // SAFETY: NEON is always available on aarch64.
            unsafe {
                softmax_attention_neon(x, seq_len, num_heads);
            }
        }

        /// NEON must match the scalar reference: a single NaN lane zeros the
        /// whole row. Uses a 17-wide row (one 16-wide vectorized chunk + a
        /// scalar tail) so both the chunked path and the tail path run.
        #[test]
        fn neon_nan_in_chunk_zeros_whole_row() {
            let mut row = vec![0.5f32; 17];
            row[5] = f32::NAN; // inside the 16-wide vectorized chunk
            let (mut x, seq_len) = single_row_matrix(&row);
            run_neon(&mut x, seq_len, 1);
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        /// NaN in the scalar tail (not caught by the vectorized chunk NaN scan)
        /// must also zero the whole row.
        #[test]
        fn neon_nan_in_tail_zeros_whole_row() {
            let mut row = vec![0.5f32; 17];
            row[16] = f32::NAN; // scalar tail lane
            let (mut x, seq_len) = single_row_matrix(&row);
            run_neon(&mut x, seq_len, 1);
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        /// A `+inf` score zeros the whole row.
        #[test]
        fn neon_pos_inf_zeros_whole_row() {
            let mut row = vec![0.5f32; 17];
            row[3] = f32::INFINITY;
            let (mut x, seq_len) = single_row_matrix(&row);
            run_neon(&mut x, seq_len, 1);
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        /// An all-masked (`-inf`) row zeros out.
        #[test]
        fn neon_all_neg_inf_zeros_row() {
            let row = vec![f32::NEG_INFINITY; 17];
            let (mut x, seq_len) = single_row_matrix(&row);
            run_neon(&mut x, seq_len, 1);
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        /// A masked (`-inf`) lane inside the vectorized chunk must be exact
        /// zero, not `neon_exp`'s finite-floor-clamp leakage (#740).
        #[test]
        fn neon_masked_lane_in_chunk_is_exact_zero() {
            let mut row = vec![0.5f32; 17];
            row[2] = f32::NEG_INFINITY;
            let (mut x, seq_len) = single_row_matrix(&row);
            run_neon(&mut x, seq_len, 1);
            assert_eq!(x[2], 0.0, "masked lane must be exact zero, got {}", x[2]);
            assert!(x[..seq_len].iter().all(|v| v.is_finite()));
            let sum: f32 = x[..seq_len].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "row must still sum to ~1: {:?}",
                &x[..seq_len]
            );
        }

        /// NEON must agree with the scalar reference on well-formed (all
        /// finite) rows, matching the scalar/NEON parity the file's own
        /// comments already document.
        #[test]
        fn neon_matches_scalar_on_finite_row() {
            let row: Vec<f32> = (0..20).map(|i| (i as f32 * 0.37).sin()).collect();
            let (mut x_scalar, seq_len) = single_row_matrix(&row);
            let mut x_neon = x_scalar.clone();
            softmax_attention_scalar(&mut x_scalar, seq_len, 1);
            run_neon(&mut x_neon, seq_len, 1);
            for (a, b) in x_scalar[..seq_len].iter().zip(x_neon[..seq_len].iter()) {
                assert!((a - b).abs() < 1e-4, "scalar={a} neon={b}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod avx2 {
        use super::*;

        fn run_avx2(x: &mut [f32], seq_len: usize, num_heads: usize) -> bool {
            if !simd_config().avx2_enabled {
                return false;
            }
            // SAFETY: gated on the runtime AVX2 feature check above.
            unsafe {
                softmax_attention_avx2(x, seq_len, num_heads);
            }
            true
        }

        #[test]
        fn avx2_nan_in_chunk_zeros_whole_row() {
            let mut row = vec![0.5f32; 9]; // one 8-wide chunk + a scalar tail
            row[3] = f32::NAN;
            let (mut x, seq_len) = single_row_matrix(&row);
            if !run_avx2(&mut x, seq_len, 1) {
                return;
            }
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        #[test]
        fn avx2_nan_in_tail_zeros_whole_row() {
            let mut row = vec![0.5f32; 9];
            row[8] = f32::NAN;
            let (mut x, seq_len) = single_row_matrix(&row);
            if !run_avx2(&mut x, seq_len, 1) {
                return;
            }
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        #[test]
        fn avx2_pos_inf_zeros_whole_row() {
            let mut row = vec![0.5f32; 9];
            row[0] = f32::INFINITY;
            let (mut x, seq_len) = single_row_matrix(&row);
            if !run_avx2(&mut x, seq_len, 1) {
                return;
            }
            assert!(
                x[..seq_len].iter().all(|&v| v == 0.0),
                "expected all-zero row: {:?}",
                &x[..seq_len]
            );
        }

        #[test]
        fn avx2_masked_lane_is_exact_zero() {
            let mut row = vec![0.5f32; 9];
            row[4] = f32::NEG_INFINITY;
            let (mut x, seq_len) = single_row_matrix(&row);
            if !run_avx2(&mut x, seq_len, 1) {
                return;
            }
            assert_eq!(x[4], 0.0, "masked lane must be exact zero, got {}", x[4]);
        }

        #[test]
        fn avx2_matches_scalar_on_finite_row() {
            let row: Vec<f32> = (0..20).map(|i| (i as f32 * 0.37).sin()).collect();
            let (mut x_scalar, seq_len) = single_row_matrix(&row);
            let mut x_avx2 = x_scalar.clone();
            softmax_attention_scalar(&mut x_scalar, seq_len, 1);
            if !run_avx2(&mut x_avx2, seq_len, 1) {
                return;
            }
            for (a, b) in x_scalar[..seq_len].iter().zip(x_avx2[..seq_len].iter()) {
                assert!((a - b).abs() < 1e-4, "scalar={a} avx2={b}");
            }
        }
    }
}
