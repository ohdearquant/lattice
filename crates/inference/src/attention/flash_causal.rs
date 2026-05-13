//! Flash Attention v2 for causal decoder inference with grouped-query attention (GQA).
//!
//! This module is designed as a drop-in replacement for the standard per-head
//! causal attention loop used in decoder-only models. It never materializes the
//! full `[seq_len, seq_len]` score matrix. Instead it processes attention in
//! `[Br, Bc]` tiles and maintains numerically stable online-softmax statistics.
//!
//! The implementation keeps an *unnormalized* output accumulator `O~` together
//! with the running row-wise max `m` and denominator `l`, which is the forward
//! pass simplification highlighted by FlashAttention-2.
//!
//! Small GEMMs dispatch to [`crate::forward::cpu::matmul_bt`] and
//! [`crate::forward::cpu::matmul_into`], which use Accelerate BLAS on macOS and a
//! scalar fallback elsewhere.

use std::cmp;

use crate::forward::cpu::{matmul_bt, matmul_into};

/// **Unstable**: default query-row block size for Flash Attention tiling; tuning in progress.
pub const DEFAULT_BR: usize = 64;
/// **Unstable**: default key/value-column block size for Flash Attention tiling; tuning in progress.
pub const DEFAULT_BC: usize = 64;

/// **Unstable**: tile-size configuration for Flash Attention; block sizes subject to tuning.
#[derive(Clone, Copy, Debug)]
pub struct FlashAttentionConfig {
    /// Query-row block size.
    pub br: usize,
    /// Key/value-column block size.
    pub bc: usize,
    /// Fall back to the simple reference path for very short sequences.
    pub fallback_below: usize,
}

impl FlashAttentionConfig {
    /// **Unstable**: construct config from head dimension; heuristic may change.
    #[inline]
    pub fn for_head_dim(head_dim: usize) -> Self {
        let block = head_dim.clamp(1, DEFAULT_BR);
        Self {
            br: block,
            bc: head_dim.clamp(1, DEFAULT_BC),
            fallback_below: block,
        }
    }
}

#[derive(Debug)]
struct FlashAttentionScratch {
    scores_block: Vec<f32>, // [br, bc]
    p_block: Vec<f32>,      // [br, bc]
    q_block: Vec<f32>,      // [br, d]
    k_block: Vec<f32>,      // [bc, d]
    v_block: Vec<f32>,      // [bc, d]
    o_block: Vec<f32>,      // [br, d] -- unnormalized accumulator
    pv_block: Vec<f32>,     // [br, d]
    row_m_new: Vec<f32>,    // [br]
    row_l_new: Vec<f32>,    // [br]
    row_alpha: Vec<f32>,    // [br]
}

impl FlashAttentionScratch {
    fn new(br: usize, bc: usize, head_dim: usize) -> Self {
        Self {
            scores_block: vec![0.0; br * bc],
            p_block: vec![0.0; br * bc],
            q_block: vec![0.0; br * head_dim],
            k_block: vec![0.0; bc * head_dim],
            v_block: vec![0.0; bc * head_dim],
            o_block: vec![0.0; br * head_dim],
            pv_block: vec![0.0; br * head_dim],
            row_m_new: vec![0.0; br],
            row_l_new: vec![0.0; br],
            row_alpha: vec![0.0; br],
        }
    }
}

/// **Unstable**: Flash Attention v2 causal kernel; tile sizes and dispatch path under tuning.
///
/// Flash Attention v2 for causal decoder inference with grouped-query attention.
///
/// Layouts:
/// * `q_buf`: `[seq_len, num_heads * head_dim]` (interleaved heads per token)
/// * `k_buf`: `[seq_len, num_kv_heads * head_dim]`
/// * `v_buf`: `[seq_len, num_kv_heads * head_dim]`
/// * `attn_out`: `[seq_len, num_heads * head_dim]`
///
/// The implementation is exact up to floating-point roundoff and should match a
/// standard causal attention implementation to within small numerical tolerance.
pub fn flash_attention_causal(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) {
    flash_attention_causal_with_config(
        q_buf,
        k_buf,
        v_buf,
        attn_out,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        FlashAttentionConfig::for_head_dim(head_dim),
    );
}

/// **Unstable**: Flash Attention v2 causal kernel with explicit tile configuration.
///
/// Same as [`flash_attention_causal`] but with explicit tile sizes.
pub fn flash_attention_causal_with_config(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    config: FlashAttentionConfig,
) {
    if seq_len == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return;
    }

    assert!(
        num_heads % num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads"
    );

    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q_buf.len(), seq_len * q_dim, "bad q_buf shape");
    assert_eq!(k_buf.len(), seq_len * kv_dim, "bad k_buf shape");
    assert_eq!(v_buf.len(), seq_len * kv_dim, "bad v_buf shape");
    assert_eq!(attn_out.len(), seq_len * q_dim, "bad attn_out shape");

    let groups = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let br = cmp::max(1, config.br);
    let bc = cmp::max(1, config.bc);

    if seq_len < config.fallback_below {
        standard_attention_causal(
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );
        return;
    }

    let mut scratch = FlashAttentionScratch::new(br, bc, head_dim);
    let mut group_m = vec![f32::NEG_INFINITY; groups * seq_len];
    let mut group_l = vec![0.0f32; groups * seq_len];

    let num_q_blocks = ceil_div(seq_len, br);
    let num_k_blocks = ceil_div(seq_len, bc);

    for kv_h in 0..num_kv_heads {
        group_m.fill(f32::NEG_INFINITY);
        group_l.fill(0.0);

        let q_head_base = kv_h * groups;
        for g in 0..groups {
            zero_head_output(attn_out, seq_len, q_dim, head_dim, q_head_base + g);
        }

        for k_block_idx in 0..num_k_blocks {
            let k_start = k_block_idx * bc;
            let bc_cur = cmp::min(bc, seq_len - k_start);

            extract_head_block(
                k_buf,
                kv_dim,
                head_dim,
                kv_h,
                k_start,
                bc_cur,
                &mut scratch.k_block[..bc_cur * head_dim],
            );
            extract_head_block(
                v_buf,
                kv_dim,
                head_dim,
                kv_h,
                k_start,
                bc_cur,
                &mut scratch.v_block[..bc_cur * head_dim],
            );

            for g in 0..groups {
                let q_h = q_head_base + g;
                let m_head = &mut group_m[g * seq_len..(g + 1) * seq_len];
                let l_head = &mut group_l[g * seq_len..(g + 1) * seq_len];

                for q_block_idx in 0..num_q_blocks {
                    let q_start = q_block_idx * br;
                    let br_cur = cmp::min(br, seq_len - q_start);
                    let q_end = q_start + br_cur - 1;

                    // Entire K block lies strictly in the future of the entire Q block.
                    if k_start > q_end {
                        continue;
                    }

                    extract_head_block(
                        q_buf,
                        q_dim,
                        head_dim,
                        q_h,
                        q_start,
                        br_cur,
                        &mut scratch.q_block[..br_cur * head_dim],
                    );
                    load_head_output_block(
                        attn_out,
                        q_dim,
                        head_dim,
                        q_h,
                        q_start,
                        br_cur,
                        &mut scratch.o_block[..br_cur * head_dim],
                    );

                    flash_update_block(
                        &mut scratch,
                        &mut m_head[q_start..q_start + br_cur],
                        &mut l_head[q_start..q_start + br_cur],
                        q_start,
                        k_start,
                        br_cur,
                        bc_cur,
                        head_dim,
                        scale,
                    );

                    store_head_output_block(
                        attn_out,
                        q_dim,
                        head_dim,
                        q_h,
                        q_start,
                        br_cur,
                        &scratch.o_block[..br_cur * head_dim],
                    );
                }
            }
        }

        for g in 0..groups {
            normalize_head_output(
                attn_out,
                q_dim,
                head_dim,
                q_head_base + g,
                seq_len,
                &group_l[g * seq_len..(g + 1) * seq_len],
            );
        }
    }
}

#[inline]
fn ceil_div(n: usize, d: usize) -> usize {
    debug_assert!(d > 0);
    n / d + usize::from(n % d != 0)
}

fn extract_head_block(
    src: &[f32],
    row_stride: usize,
    head_dim: usize,
    head_idx: usize,
    start_row: usize,
    rows: usize,
    dst: &mut [f32],
) {
    debug_assert!(dst.len() >= rows * head_dim);
    let head_off = head_idx * head_dim;
    for r in 0..rows {
        let src_off = (start_row + r) * row_stride + head_off;
        let dst_off = r * head_dim;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
}

fn load_head_output_block(
    src: &[f32],
    row_stride: usize,
    head_dim: usize,
    head_idx: usize,
    start_row: usize,
    rows: usize,
    dst: &mut [f32],
) {
    debug_assert!(dst.len() >= rows * head_dim);
    let head_off = head_idx * head_dim;
    for r in 0..rows {
        let src_off = (start_row + r) * row_stride + head_off;
        let dst_off = r * head_dim;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
}

fn store_head_output_block(
    dst: &mut [f32],
    row_stride: usize,
    head_dim: usize,
    head_idx: usize,
    start_row: usize,
    rows: usize,
    src: &[f32],
) {
    debug_assert!(src.len() >= rows * head_dim);
    let head_off = head_idx * head_dim;
    for r in 0..rows {
        let dst_off = (start_row + r) * row_stride + head_off;
        let src_off = r * head_dim;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
}

fn zero_head_output(
    attn_out: &mut [f32],
    seq_len: usize,
    q_dim: usize,
    head_dim: usize,
    head_idx: usize,
) {
    let head_off = head_idx * head_dim;
    for row in 0..seq_len {
        let off = row * q_dim + head_off;
        attn_out[off..off + head_dim].fill(0.0);
    }
}

fn normalize_head_output(
    attn_out: &mut [f32],
    q_dim: usize,
    head_dim: usize,
    head_idx: usize,
    seq_len: usize,
    l: &[f32],
) {
    let head_off = head_idx * head_dim;
    for row in 0..seq_len {
        let inv_l = if l[row] > 0.0 { 1.0 / l[row] } else { 0.0 };
        let off = row * q_dim + head_off;
        for d in 0..head_dim {
            attn_out[off + d] *= inv_l;
        }
    }
}

fn flash_update_block(
    scratch: &mut FlashAttentionScratch,
    m_rows: &mut [f32],
    l_rows: &mut [f32],
    q_start: usize,
    k_start: usize,
    br_cur: usize,
    bc_cur: usize,
    head_dim: usize,
    scale: f32,
) {
    let q_slice = &scratch.q_block[..br_cur * head_dim];
    let k_slice = &scratch.k_block[..bc_cur * head_dim];
    let v_slice = &scratch.v_block[..bc_cur * head_dim];
    let scores = &mut scratch.scores_block[..br_cur * bc_cur];
    let p = &mut scratch.p_block[..br_cur * bc_cur];
    let o = &mut scratch.o_block[..br_cur * head_dim];
    let pv = &mut scratch.pv_block[..br_cur * head_dim];

    matmul_bt(q_slice, k_slice, scores, br_cur, head_dim, bc_cur);

    let block_is_strictly_lower = k_start + bc_cur <= q_start;

    for row in 0..br_cur {
        let abs_q = q_start + row;
        let row_scores = &mut scores[row * bc_cur..(row + 1) * bc_cur];
        let row_p = &mut p[row * bc_cur..(row + 1) * bc_cur];

        let mut block_row_max = f32::NEG_INFINITY;

        if block_is_strictly_lower {
            for score in row_scores.iter_mut() {
                *score *= scale;
                if *score > block_row_max {
                    block_row_max = *score;
                }
            }
        } else {
            for col in 0..bc_cur {
                let abs_k = k_start + col;
                let score = &mut row_scores[col];
                if abs_k > abs_q {
                    *score = f32::NEG_INFINITY;
                } else {
                    *score *= scale;
                    if *score > block_row_max {
                        block_row_max = *score;
                    }
                }
            }
        }

        let m_old = m_rows[row];
        let m_new = if m_old > block_row_max {
            m_old
        } else {
            block_row_max
        };
        scratch.row_m_new[row] = m_new;

        let alpha = if m_old.is_finite() && m_new.is_finite() {
            (m_old - m_new).exp()
        } else {
            0.0
        };
        scratch.row_alpha[row] = alpha;

        let mut row_sum = 0.0f32;
        if m_new.is_finite() {
            for col in 0..bc_cur {
                let score = row_scores[col];
                let value = if score.is_finite() {
                    (score - m_new).exp()
                } else {
                    0.0
                };
                row_p[col] = value;
                row_sum += value;
            }
        } else {
            row_p.fill(0.0);
        }

        scratch.row_l_new[row] = alpha * l_rows[row] + row_sum;
    }

    matmul_into(p, v_slice, pv, br_cur, bc_cur, head_dim);

    for row in 0..br_cur {
        let alpha = scratch.row_alpha[row];
        let out_row = &mut o[row * head_dim..(row + 1) * head_dim];
        let pv_row = &pv[row * head_dim..(row + 1) * head_dim];
        for d in 0..head_dim {
            out_row[d] = alpha * out_row[d] + pv_row[d];
        }
        m_rows[row] = scratch.row_m_new[row];
        l_rows[row] = scratch.row_l_new[row];
    }
}

/// Simple reference path for very short sequences.
///
/// This keeps the code path small when `seq_len < block_size`, where a Flash
/// tiled implementation typically has little or no benefit.
fn standard_attention_causal(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) {
    let groups = num_heads / num_kv_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0.0f32; seq_len];
    let mut probs = vec![0.0f32; seq_len];

    for q_h in 0..num_heads {
        let kv_h = q_h / groups;
        let q_head_off = q_h * head_dim;
        let kv_head_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let q_off = qi * q_dim + q_head_off;
            let q_row = &q_buf[q_off..q_off + head_dim];

            let mut max_score = f32::NEG_INFINITY;
            for ki in 0..seq_len {
                let score = if ki > qi {
                    f32::NEG_INFINITY
                } else {
                    let k_off = ki * kv_dim + kv_head_off;
                    let k_row = &k_buf[k_off..k_off + head_dim];
                    scale * dot(q_row, k_row)
                };
                scores[ki] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut denom = 0.0f32;
            for ki in 0..seq_len {
                let p = if scores[ki].is_finite() {
                    (scores[ki] - max_score).exp()
                } else {
                    0.0
                };
                probs[ki] = p;
                denom += p;
            }

            let out_off = qi * q_dim + q_head_off;
            let out_row = &mut attn_out[out_off..out_off + head_dim];
            out_row.fill(0.0);

            if denom > 0.0 {
                let inv = 1.0 / denom;
                for ki in 0..=qi {
                    let weight = probs[ki] * inv;
                    let v_off = ki * kv_dim + kv_head_off;
                    let v_row = &v_buf[v_off..v_off + head_dim];
                    for d in 0..head_dim {
                        out_row[d] += weight * v_row[d];
                    }
                }
            }
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_attention_causal(
        q_buf: &[f32],
        k_buf: &[f32],
        v_buf: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_dim = num_heads * head_dim;
        let mut out = vec![0.0; seq_len * q_dim];
        standard_attention_causal(
            q_buf,
            k_buf,
            v_buf,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );
        out
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff <= tol,
                "mismatch at {idx}: got {x}, expected {y}, |diff|={diff} > {tol}"
            );
        }
    }

    fn lcg_fill(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
            out.push(v * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn small_known_input_matches_expected_values() {
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;

        let q_buf = vec![
            0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.20, 0.10, 0.40, 0.30, 0.60, 0.50,
            0.80, 0.70, 0.30, 0.40, 0.10, 0.20, 0.70, 0.80, 0.50, 0.60, 0.40, 0.30, 0.20, 0.10,
            0.80, 0.70, 0.60, 0.50,
        ];
        let k_buf = vec![
            0.11, 0.21, 0.31, 0.41, 0.22, 0.12, 0.42, 0.32, 0.33, 0.43, 0.13, 0.23, 0.44, 0.34,
            0.24, 0.14,
        ];
        let v_buf = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        let expected = vec![
            1.0000000, 2.0000000, 3.0000000, 4.0000000, 1.0000000, 2.0000000, 3.0000000, 4.0000000,
            3.0149997, 4.0149999, 5.0149999, 6.0149999, 3.0229990, 4.0229990, 5.0229990, 6.0229990,
            5.0673227, 6.0673227, 7.0673227, 8.0673227, 5.0888591, 6.0888591, 7.0888591, 8.0888591,
            7.1149790, 8.1149790, 9.1149790, 10.1149790, 7.1549511, 8.1549511, 9.1549511,
            10.1549511,
        ];

        let mut out = vec![0.0; seq_len * q_dim];
        flash_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );

        approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    fn causal_mask_prevents_future_attention() {
        let seq_len = 3;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let q_buf = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let k_buf = vec![1.0, 0.0, 10.0, 0.0, 20.0, 0.0];
        let v_buf = vec![3.0, 4.0, 30.0, 40.0, 300.0, 400.0];

        let mut out = vec![0.0; seq_len * head_dim];
        flash_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );

        // Token 0 may attend only to itself, so the output must equal V[0].
        approx_eq(&out[0..2], &[3.0, 4.0], 1e-6);

        let expected = reference_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );
        approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    fn gqa_matches_reference() {
        let seq_len = 17;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let q_buf = lcg_fill(seq_len * num_heads * head_dim, 1);
        let k_buf = lcg_fill(seq_len * num_kv_heads * head_dim, 2);
        let v_buf = lcg_fill(seq_len * num_kv_heads * head_dim, 3);

        let expected = reference_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );
        let mut out = vec![0.0; seq_len * num_heads * head_dim];

        flash_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );

        approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    fn unequal_tile_sizes_match_reference() {
        let seq_len = 96;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let q_buf = lcg_fill(seq_len * num_heads * head_dim, 101);
        let k_buf = lcg_fill(seq_len * num_kv_heads * head_dim, 202);
        let v_buf = lcg_fill(seq_len * num_kv_heads * head_dim, 303);
        let expected = reference_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );
        let mut out = vec![0.0; seq_len * num_heads * head_dim];

        flash_attention_causal_with_config(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            FlashAttentionConfig {
                br: 64,
                bc: 32,
                fallback_below: 1,
            },
        );

        approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    fn multi_block_random_matches_reference() {
        let seq_len = 73;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let q_buf = lcg_fill(seq_len * num_heads * head_dim, 11);
        let k_buf = lcg_fill(seq_len * num_kv_heads * head_dim, 22);
        let v_buf = lcg_fill(seq_len * num_kv_heads * head_dim, 33);
        let mut out = vec![0.0; seq_len * num_heads * head_dim];

        flash_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );

        let expected = reference_attention_causal(
            &q_buf,
            &k_buf,
            &v_buf,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        );

        approx_eq(&out, &expected, 1e-4);
    }
}
