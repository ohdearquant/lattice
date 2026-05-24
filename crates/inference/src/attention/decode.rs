//! Decode-time single-token attention kernel.
//!
//! Scalar baseline for seq_len=1 against a full cached K/V sequence.
//! SIMD paths are added in a subsequent optimization pass.

use crate::attention::gqa::GqaConfig;

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
}
