//! Decode-time single-token attention kernel.
//!
//! Single-token (seq_len=1) attention against a full cached K/V sequence.
//! NEON SIMD path accelerates QK dot products, softmax, and V accumulation.

use crate::attention::gqa::GqaConfig;
#[cfg(target_arch = "aarch64")]
use crate::forward::cpu::simd_config;

/// **Unstable**: compute decode-time attention scores for one query token.
///
/// Layouts:
/// - `q_buf`: `[num_heads * head_dim]`
/// - `k_buf`: `[kv_seq_len, num_kv_heads * head_dim]`
/// - `scores`: at least `num_heads * score_stride`
///
/// Writes scaled QK scores to `scores[h * score_stride..][..kv_seq_len]`.
/// Panics if shapes are inconsistent or `num_heads` is not divisible by
/// `num_kv_heads`.
pub fn decode_attention_scores(
    q_buf: &[f32],
    k_buf: &[f32],
    scores: &mut [f32],
    kv_seq_len: usize,
    cfg: GqaConfig,
    score_stride: usize,
) {
    assert!(cfg.num_kv_heads > 0, "num_kv_heads must be > 0");
    assert_eq!(
        cfg.num_heads % cfg.num_kv_heads,
        0,
        "num_heads must be divisible by num_kv_heads"
    );
    assert_eq!(q_buf.len(), cfg.q_dim(), "q_buf length mismatch");
    assert_eq!(
        k_buf.len(),
        kv_seq_len * cfg.kv_dim(),
        "k_buf length mismatch"
    );
    assert!(
        scores.len() >= cfg.num_heads * score_stride,
        "scores buffer too small"
    );
    assert!(score_stride >= kv_seq_len, "score_stride < kv_seq_len");

    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            unsafe {
                decode_scores_neon(q_buf, k_buf, scores, kv_seq_len, cfg, score_stride);
            }
            return;
        }
    }

    decode_scores_scalar(q_buf, k_buf, scores, kv_seq_len, cfg, score_stride);
}

/// **Unstable**: compute full decode-time attention for one query token.
///
/// Layouts:
/// - `q_buf`: `[num_heads * head_dim]`
/// - `k_buf`: `[kv_seq_len, num_kv_heads * head_dim]`
/// - `v_buf`: `[kv_seq_len, num_kv_heads * head_dim]`
/// - `attn_out`: `[num_heads * head_dim]`
/// - `scores`: at least `num_heads * score_stride`
///
/// Computes scores, applies row softmax in-place to `scores`, then writes
/// `attn_out[h, d] = sum_i softmax(scores[h, i]) * V[i, kv_h, d]`.
/// Panics if shapes are inconsistent or `num_heads` is not divisible by
/// `num_kv_heads`.
pub fn decode_attention(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    scores: &mut [f32],
    kv_seq_len: usize,
    cfg: GqaConfig,
    score_stride: usize,
) {
    assert!(cfg.num_kv_heads > 0, "num_kv_heads must be > 0");
    assert_eq!(
        cfg.num_heads % cfg.num_kv_heads,
        0,
        "num_heads must be divisible by num_kv_heads"
    );
    assert_eq!(q_buf.len(), cfg.q_dim(), "q_buf length mismatch");
    assert_eq!(attn_out.len(), cfg.q_dim(), "attn_out length mismatch");
    assert!(
        scores.len() >= cfg.num_heads * score_stride,
        "scores buffer too small"
    );

    if kv_seq_len == 0 {
        attn_out.fill(0.0);
        return;
    }

    assert_eq!(
        k_buf.len(),
        kv_seq_len * cfg.kv_dim(),
        "k_buf length mismatch"
    );
    assert_eq!(
        v_buf.len(),
        kv_seq_len * cfg.kv_dim(),
        "v_buf length mismatch"
    );
    assert!(score_stride >= kv_seq_len, "score_stride < kv_seq_len");

    #[cfg(target_arch = "aarch64")]
    {
        if simd_config().neon_enabled {
            unsafe {
                decode_scores_neon(q_buf, k_buf, scores, kv_seq_len, cfg, score_stride);
                softmax_decode_neon(scores, cfg.num_heads, kv_seq_len, score_stride);
                attn_out.fill(0.0);
                accumulate_decode_v_neon(attn_out, scores, v_buf, kv_seq_len, cfg, score_stride);
            }
            return;
        }
    }

    decode_scores_scalar(q_buf, k_buf, scores, kv_seq_len, cfg, score_stride);
    softmax_decode_scores(scores, cfg.num_heads, kv_seq_len, score_stride);
    attn_out.fill(0.0);
    accumulate_decode_v_scalar(attn_out, scores, v_buf, kv_seq_len, cfg, score_stride);
}

fn decode_scores_scalar(
    q_buf: &[f32],
    k_buf: &[f32],
    scores: &mut [f32],
    kv_seq_len: usize,
    cfg: GqaConfig,
    score_stride: usize,
) {
    let groups = cfg.groups();
    let head_dim = cfg.head_dim;
    let kv_dim = cfg.kv_dim();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for kv_h in 0..cfg.num_kv_heads {
        let group_start = kv_h * groups;
        for ki in 0..kv_seq_len {
            let k_off = ki * kv_dim + kv_h * head_dim;
            for gi in 0..groups {
                let h = group_start + gi;
                let q_off = h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_buf[q_off + d] * k_buf[k_off + d];
                }
                scores[h * score_stride + ki] = dot * scale;
            }
        }
    }
}

fn softmax_decode_scores(
    scores: &mut [f32],
    num_heads: usize,
    kv_seq_len: usize,
    score_stride: usize,
) {
    for h in 0..num_heads {
        let row = &mut scores[h * score_stride..h * score_stride + kv_seq_len];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        for &s in row.iter() {
            let m_new = m.max(s);
            #[allow(clippy::float_cmp)]
            let alpha = if m == f32::NEG_INFINITY {
                0.0
            } else {
                (m - m_new).exp()
            };
            l = l * alpha + (s - m_new).exp();
            m = m_new;
        }
        if l > 0.0 {
            let inv_l = 1.0 / l;
            for s in row.iter_mut() {
                *s = (*s - m).exp() * inv_l;
            }
        } else {
            row.fill(0.0);
        }
    }
}

fn accumulate_decode_v_scalar(
    attn_out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    kv_seq_len: usize,
    cfg: GqaConfig,
    score_stride: usize,
) {
    let groups = cfg.groups();
    let head_dim = cfg.head_dim;
    let kv_dim = cfg.kv_dim();

    for h in 0..cfg.num_heads {
        let kv_h = h / groups;
        let out_off = h * head_dim;
        let score_off = h * score_stride;
        for ki in 0..kv_seq_len {
            let w = scores[score_off + ki];
            let v_off = ki * kv_dim + kv_h * head_dim;
            for d in 0..head_dim {
                attn_out[out_off + d] += w * v_buf[v_off + d];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fast_exp_decode(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let val = (12_102_203.0f32 * x + 1_065_353_216.0f32) as i32;
    f32::from_bits(val as u32)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fast_exp_neon_decode(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
    let x = vminq_f32(x, vdupq_n_f32(88.0));
    let t = vfmaq_f32(vdupq_n_f32(1_065_353_216.0), x, vdupq_n_f32(12_102_203.0));
    vreinterpretq_f32_s32(vcvtq_s32_f32(t))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_scores_neon(
    q_buf: &[f32],
    k_buf: &[f32],
    scores: &mut [f32],
    kv_seq_len: usize,
    cfg: GqaConfig,
    score_stride: usize,
) {
    use std::arch::aarch64::*;

    let groups = cfg.groups();
    let head_dim = cfg.head_dim;
    let kv_dim = cfg.kv_dim();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let hd_chunks = head_dim / 16;

    for kv_h in 0..cfg.num_kv_heads {
        let group_start = kv_h * groups;
        for gi in 0..groups {
            let h = group_start + gi;
            let q_ptr = q_buf.as_ptr().add(h * head_dim);

            for ki in 0..kv_seq_len {
                let k_ptr = k_buf.as_ptr().add(ki * kv_dim + kv_h * head_dim);

                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                for c in 0..hd_chunks {
                    let off = c * 16;
                    acc0 = vfmaq_f32(acc0, vld1q_f32(q_ptr.add(off)), vld1q_f32(k_ptr.add(off)));
                    acc1 = vfmaq_f32(
                        acc1,
                        vld1q_f32(q_ptr.add(off + 4)),
                        vld1q_f32(k_ptr.add(off + 4)),
                    );
                    acc2 = vfmaq_f32(
                        acc2,
                        vld1q_f32(q_ptr.add(off + 8)),
                        vld1q_f32(k_ptr.add(off + 8)),
                    );
                    acc3 = vfmaq_f32(
                        acc3,
                        vld1q_f32(q_ptr.add(off + 12)),
                        vld1q_f32(k_ptr.add(off + 12)),
                    );
                }
                let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                let mut dot = vaddvq_f32(sum);

                for d in (hd_chunks * 16)..head_dim {
                    dot += *q_ptr.add(d) * *k_ptr.add(d);
                }

                scores[h * score_stride + ki] = dot * scale;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_decode_neon(
    scores: &mut [f32],
    num_heads: usize,
    kv_seq_len: usize,
    score_stride: usize,
) {
    use std::arch::aarch64::*;

    let chunks = kv_seq_len / 16;

    for h in 0..num_heads {
        let ptr = scores.as_mut_ptr().add(h * score_stride);

        // Pass 1: find max with 4x unroll
        let mut m0 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut m1 = m0;
        let mut m2 = m0;
        let mut m3 = m0;
        for c in 0..chunks {
            let base = c * 16;
            m0 = vmaxq_f32(m0, vld1q_f32(ptr.add(base) as *const f32));
            m1 = vmaxq_f32(m1, vld1q_f32(ptr.add(base + 4) as *const f32));
            m2 = vmaxq_f32(m2, vld1q_f32(ptr.add(base + 8) as *const f32));
            m3 = vmaxq_f32(m3, vld1q_f32(ptr.add(base + 12) as *const f32));
        }
        let vmax = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3));
        let mut max_val = vmaxvq_f32(vmax);
        for i in (chunks * 16)..kv_seq_len {
            max_val = max_val.max(*ptr.add(i));
        }

        // Pass 2: exp(x - max) + accumulate sum
        let vmax_bcast = vdupq_n_f32(max_val);
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = s0;
        let mut s2 = s0;
        let mut s3 = s0;
        for c in 0..chunks {
            let base = c * 16;
            let e0 = fast_exp_neon_decode(vsubq_f32(
                vld1q_f32(ptr.add(base) as *const f32),
                vmax_bcast,
            ));
            let e1 = fast_exp_neon_decode(vsubq_f32(
                vld1q_f32(ptr.add(base + 4) as *const f32),
                vmax_bcast,
            ));
            let e2 = fast_exp_neon_decode(vsubq_f32(
                vld1q_f32(ptr.add(base + 8) as *const f32),
                vmax_bcast,
            ));
            let e3 = fast_exp_neon_decode(vsubq_f32(
                vld1q_f32(ptr.add(base + 12) as *const f32),
                vmax_bcast,
            ));
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
        for i in (chunks * 16)..kv_seq_len {
            let e = fast_exp_decode(*ptr.add(i) - max_val);
            *ptr.add(i) = e;
            sum += e;
        }

        // Pass 3: normalize
        if sum > 0.0 {
            let vinv = vdupq_n_f32(1.0 / sum);
            for c in 0..chunks {
                let base = c * 16;
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
            let inv = 1.0 / sum;
            for i in (chunks * 16)..kv_seq_len {
                *ptr.add(i) *= inv;
            }
        } else {
            for i in 0..kv_seq_len {
                *ptr.add(i) = 0.0;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_decode_v_neon(
    attn_out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    kv_seq_len: usize,
    cfg: GqaConfig,
    score_stride: usize,
) {
    use std::arch::aarch64::*;

    let groups = cfg.groups();
    let head_dim = cfg.head_dim;
    let kv_dim = cfg.kv_dim();
    let hd_chunks = head_dim / 16;

    for h in 0..cfg.num_heads {
        let kv_h = h / groups;
        let out_ptr = attn_out.as_mut_ptr().add(h * head_dim);
        let score_off = h * score_stride;

        for ki in 0..kv_seq_len {
            let w = scores[score_off + ki];
            if w == 0.0 {
                continue;
            }
            let vw = vdupq_n_f32(w);
            let v_ptr = v_buf.as_ptr().add(ki * kv_dim + kv_h * head_dim);

            for c in 0..hd_chunks {
                let off = c * 16;
                vst1q_f32(
                    out_ptr.add(off),
                    vfmaq_f32(
                        vld1q_f32(out_ptr.add(off) as *const f32),
                        vw,
                        vld1q_f32(v_ptr.add(off)),
                    ),
                );
                vst1q_f32(
                    out_ptr.add(off + 4),
                    vfmaq_f32(
                        vld1q_f32(out_ptr.add(off + 4) as *const f32),
                        vw,
                        vld1q_f32(v_ptr.add(off + 4)),
                    ),
                );
                vst1q_f32(
                    out_ptr.add(off + 8),
                    vfmaq_f32(
                        vld1q_f32(out_ptr.add(off + 8) as *const f32),
                        vw,
                        vld1q_f32(v_ptr.add(off + 8)),
                    ),
                );
                vst1q_f32(
                    out_ptr.add(off + 12),
                    vfmaq_f32(
                        vld1q_f32(out_ptr.add(off + 12) as *const f32),
                        vw,
                        vld1q_f32(v_ptr.add(off + 12)),
                    ),
                );
            }
            for d in (hd_chunks * 16)..head_dim {
                *out_ptr.add(d) += w * *v_ptr.add(d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_data(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u32)
                    .wrapping_mul(seed)
                    .wrapping_add(1_013_904_223 ^ seed.wrapping_mul(0x9E37_79B9));
                x ^= x >> 16;
                x = x.wrapping_mul(0x7FEB_352D);
                x ^= x >> 16;
                ((x & 0x00FF_FFFF) as f32 / 16_777_216.0 - 0.5) * 0.125
            })
            .collect()
    }

    #[test]
    fn test_decode_attention_zero_kvlen() {
        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
        };
        let q = small_data(cfg.q_dim(), 1);
        let mut out = vec![99.0f32; cfg.q_dim()];
        let mut scores = vec![0.0f32; cfg.num_heads];
        decode_attention(&q, &[], &[], &mut out, &mut scores, 0, cfg, 1);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_decode_attention_mha() {
        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
        };
        let kv_seq_len = 5;
        let q = small_data(cfg.q_dim(), 1);
        let k = small_data(kv_seq_len * cfg.kv_dim(), 2);
        let v = small_data(kv_seq_len * cfg.kv_dim(), 3);
        let mut out = vec![0.0f32; cfg.q_dim()];
        let mut scores = vec![0.0f32; cfg.num_heads * kv_seq_len];
        decode_attention(
            &q,
            &k,
            &v,
            &mut out,
            &mut scores,
            kv_seq_len,
            cfg,
            kv_seq_len,
        );
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_decode_attention_gqa() {
        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
        };
        let kv_seq_len = 5;
        let q = small_data(cfg.q_dim(), 1);
        let k = small_data(kv_seq_len * cfg.kv_dim(), 2);
        let v = small_data(kv_seq_len * cfg.kv_dim(), 3);
        let mut out = vec![0.0f32; cfg.q_dim()];
        let mut scores = vec![0.0f32; cfg.num_heads * kv_seq_len];
        decode_attention(
            &q,
            &k,
            &v,
            &mut out,
            &mut scores,
            kv_seq_len,
            cfg,
            kv_seq_len,
        );
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_decode_scores_only() {
        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
        };
        let kv_seq_len = 3;
        let q = small_data(cfg.q_dim(), 1);
        let k = small_data(kv_seq_len * cfg.kv_dim(), 2);
        let mut scores = vec![0.0f32; cfg.num_heads * kv_seq_len];
        decode_attention_scores(&q, &k, &mut scores, kv_seq_len, cfg, kv_seq_len);
        assert!(scores.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[should_panic(expected = "num_heads must be divisible by num_kv_heads")]
    fn test_non_divisible_heads_panics() {
        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 3,
            head_dim: 8,
        };
        let q = vec![0.0f32; cfg.q_dim()];
        let mut out = vec![0.0f32; cfg.q_dim()];
        let mut scores = vec![0.0f32; 4 * 4];
        decode_attention(&q, &[], &[], &mut out, &mut scores, 0, cfg, 1);
    }

    /// FP-NEON-SOFTMAX: NEON decode path (Schraudolph fast_exp) vs scalar path parity.
    ///
    /// The NEON softmax uses the Schraudolph bit-trick exp (~5% per-element error).
    /// Bias cancels approximately in normalization, so the final attention output
    /// should agree with the scalar path within 5% relative tolerance.
    ///
    /// Tested with adversarial score patterns: uniform, monotonic ramp, one-hot spike.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_decode_softmax_neon_vs_scalar_parity() {
        // LCG data generator matching the pattern used elsewhere in this file.
        fn lcg_data(n: usize, seed: u32) -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let mut x = (i as u32)
                        .wrapping_mul(seed)
                        .wrapping_add(1_013_904_223 ^ seed.wrapping_mul(0x9E37_79B9));
                    x ^= x >> 16;
                    x = x.wrapping_mul(0x7FEB_352D);
                    x ^= x >> 16;
                    ((x & 0x00FF_FFFF) as f32 / 16_777_216.0 - 0.5) * 2.0
                })
                .collect()
        }

        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
        };

        // Adversarial score patterns: uniform, monotonic ramp, one-hot spike.
        let kv_seq_lens: &[(usize, &str)] = &[(16, "uniform"), (32, "ramp"), (8, "one-hot")];

        for &(kv_seq_len, pattern_name) in kv_seq_lens {
            let q = lcg_data(cfg.q_dim(), 42);
            let v = lcg_data(kv_seq_len * cfg.kv_dim(), 44);

            // Construct adversarial Q/K pairs for each score pattern.
            // We run both paths end-to-end from Q/K/V to exercise the full pipeline.
            let (q_patched, k_patched) = match pattern_name {
                "uniform" => {
                    // All-equal scores: uniform Q so all QK dots are equal.
                    let q_eq = vec![1.0f32 / (cfg.head_dim as f32).sqrt(); cfg.q_dim()];
                    let k_eq = vec![1.0f32; kv_seq_len * cfg.kv_dim()];
                    (q_eq, k_eq)
                }
                "ramp" => {
                    // Monotonic ramp: K[i] increases with i so scores are monotone.
                    let q_r = q.clone();
                    let k_r: Vec<f32> = (0..kv_seq_len * cfg.kv_dim())
                        .map(|i| (i / cfg.kv_dim()) as f32 * 0.1)
                        .collect();
                    (q_r, k_r)
                }
                _ => {
                    // One-hot spike: only position 0 has a large K, rest are zero.
                    let q_s = q.clone();
                    let mut k_s = vec![0.0f32; kv_seq_len * cfg.kv_dim()];
                    for d in 0..cfg.kv_dim() {
                        k_s[d] = 10.0;
                    }
                    (q_s, k_s)
                }
            };

            // Run NEON path (dispatches to NEON on aarch64 when neon_enabled).
            let mut out_neon = vec![0.0f32; cfg.q_dim()];
            let mut scores_neon = vec![0.0f32; cfg.num_heads * kv_seq_len];
            decode_attention(
                &q_patched,
                &k_patched,
                &v,
                &mut out_neon,
                &mut scores_neon,
                kv_seq_len,
                cfg,
                kv_seq_len,
            );

            // Run scalar path directly.
            let mut out_scalar = vec![0.0f32; cfg.q_dim()];
            let mut scores_scalar = vec![0.0f32; cfg.num_heads * kv_seq_len];
            decode_scores_scalar(
                &q_patched,
                &k_patched,
                &mut scores_scalar,
                kv_seq_len,
                cfg,
                kv_seq_len,
            );
            softmax_decode_scores(&mut scores_scalar, cfg.num_heads, kv_seq_len, kv_seq_len);
            out_scalar.fill(0.0);
            accumulate_decode_v_scalar(
                &mut out_scalar,
                &scores_scalar,
                &v,
                kv_seq_len,
                cfg,
                kv_seq_len,
            );

            // Assert outputs match within a mixed tolerance budget:
            //   atol=5e-3 (absolute floor; Schraudolph fast_exp ~5% error on [-1, 1] weights
            //              contributes an absolute error floor ≈ max_weight * 5%)
            //   rtol=0.05 (5% relative, accounting for Schraudolph fast_exp error)
            //
            // We use |a-b| <= atol + rtol * max(|a|, |b|) which avoids division-by-zero
            // instability when both values are near zero (e.g. tail positions after softmax).
            let atol = 5e-3f32;
            let rtol = 0.05f32;
            for (i, (&n, &s)) in out_neon.iter().zip(out_scalar.iter()).enumerate() {
                assert!(
                    n.is_finite() && s.is_finite(),
                    "decode_attention pattern={pattern_name} idx={i}: non-finite output neon={n} scalar={s}"
                );
                let abs_err = (n - s).abs();
                let tol = atol + rtol * n.abs().max(s.abs());
                assert!(
                    abs_err <= tol,
                    "decode_attention NEON vs scalar parity failure: pattern={pattern_name} \
                     idx={i} neon={n:.6} scalar={s:.6} abs_err={abs_err:.6} tol={tol:.6}"
                );
            }
        }
    }
}
