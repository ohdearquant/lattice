use super::*;

fn deterministic_data(len: usize) -> Vec<f32> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 7;
        state ^= state >> 9;
        state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mantissa = ((state >> 41) as u32) & 0x007f_ffff;
        let x = f32::from_bits(0x3f80_0000 | mantissa) - 1.5;
        out.push(x);
    }
    out
}

/// Compare with combined abs+rel tolerance. Batched BLAS uses different
/// matrix dimensions (M=groups*seq vs M=seq), causing different internal
/// tiling in Accelerate. Softmax then amplifies these differences,
/// especially for near-zero values where relative error is misleading.
fn nearly_eq(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len(), "length mismatch");
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let abs_diff = (a - b).abs();
        // Absolute tolerance for small values, relative for large.
        let tol = 1e-5_f32 + 1e-4 * a.abs().max(b.abs());
        assert!(
            abs_diff <= tol,
            "mismatch at index {idx}: {a:e} vs {b:e} (abs_diff={abs_diff:e}, tol={tol:e})"
        );
    }
}

fn bit_eq(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len(), "length mismatch");
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit mismatch at index {idx}: {a:?} vs {b:?}"
        );
    }
}

/// Reference per-head implementation (matches existing qwen_model.rs attention).
fn reference_attention_per_head(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    seq_len: usize,
    cfg: GqaConfig,
) {
    let groups = cfg.groups();
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut q_head = vec![0.0f32; seq_len * head_dim];
    let mut k_head = vec![0.0f32; seq_len * head_dim];
    let mut scores = vec![0.0f32; seq_len * seq_len];
    let mut v_head_t = vec![0.0f32; head_dim * seq_len];
    let mut context = vec![0.0f32; seq_len * head_dim];

    for h in 0..cfg.num_heads {
        let kv_h = h / groups;

        for pos in 0..seq_len {
            let src_off = pos * q_dim + h * head_dim;
            let dst_off = pos * head_dim;
            q_head[dst_off..dst_off + head_dim]
                .copy_from_slice(&q_buf[src_off..src_off + head_dim]);
        }

        extract_k_head(k_buf, seq_len, kv_dim, head_dim, kv_h, &mut k_head);
        matmul_bt(&q_head, &k_head, &mut scores, seq_len, head_dim, seq_len);

        // Scale + causal mask + softmax (reference).
        for qi in 0..seq_len {
            let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
            for ki in 0..seq_len {
                if ki > qi {
                    row[ki] = f32::NEG_INFINITY;
                } else {
                    row[ki] *= scale;
                }
            }
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }

        transpose_v_head(v_buf, seq_len, kv_dim, head_dim, kv_h, &mut v_head_t);
        matmul_bt(&scores, &v_head_t, &mut context, seq_len, seq_len, head_dim);

        for pos in 0..seq_len {
            let src_off = pos * head_dim;
            let dst_off = pos * q_dim + h * head_dim;
            attn_out[dst_off..dst_off + head_dim]
                .copy_from_slice(&context[src_off..src_off + head_dim]);
        }
    }
}

fn run_bitexact(seq_len: usize) {
    let cfg = GqaConfig {
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 128,
    };
    let q = deterministic_data(seq_len * cfg.q_dim());
    let k = deterministic_data(seq_len * cfg.kv_dim());
    let v = deterministic_data(seq_len * cfg.kv_dim());

    let mut out_ref = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut out_opt = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut scratch = GqaScratch::default();

    reference_attention_per_head(&q, &k, &v, &mut out_ref, seq_len, cfg);
    apply_gqa_attention(&q, &k, &v, &mut out_opt, seq_len, cfg, &mut scratch);

    nearly_eq(&out_ref, &out_opt);
}

#[test]
fn bitexact_seq_len_1() {
    run_bitexact(1);
}

#[test]
fn bitexact_seq_len_5() {
    run_bitexact(5);
}

#[test]
fn bitexact_seq_len_60() {
    run_bitexact(60);
}

#[cfg_attr(not(target_os = "macos"), ignore = "slow without Accelerate")]
#[test]
fn bitexact_seq_len_1100() {
    run_bitexact(1100);
}

#[test]
fn fused_softmax_matches_reference() {
    let seq_len = 11usize;
    let scale = 1.0 / (128.0f32).sqrt();
    let groups = 2usize;
    let batch_rows = groups * seq_len;

    let mut fused = deterministic_data(batch_rows * seq_len);
    let mut ref_scores = fused.clone();

    apply_scaled_causal_softmax_fused(&mut fused, batch_rows, seq_len, scale);

    // Reference: process each group's seq_len rows independently.
    for row_idx in 0..batch_rows {
        let qi = row_idx % seq_len;
        let row = &mut ref_scores[row_idx * seq_len..(row_idx + 1) * seq_len];
        for ki in 0..seq_len {
            if ki > qi {
                row[ki] = f32::NEG_INFINITY;
            } else {
                row[ki] *= scale;
            }
        }
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    bit_eq(&fused, &ref_scores);
}

#[test]
#[should_panic(expected = "num_heads must be divisible by num_kv_heads")]
fn non_divisible_head_config_panics() {
    // A non-divisible head topology (num_heads % num_kv_heads != 0) must fail
    // fast at the kernel entry rather than silently process only the first
    // `num_kv_heads * floor(groups)` query heads and leave the remainder stale.
    // The hard assert fires in release too (unlike the prior `debug_assert` in
    // `groups()`), matching `decode_attention_scores`. RED before the entry
    // guard: `groups()`'s debug_assert panics with the default message, which
    // does not contain this expected substring.
    let cfg = GqaConfig {
        num_heads: 3,
        num_kv_heads: 2,
        head_dim: 4,
    };
    let seq_len = 4usize;
    let q = vec![0.0f32; seq_len * cfg.q_dim()];
    let k = vec![0.0f32; seq_len * cfg.kv_dim()];
    let v = vec![0.0f32; seq_len * cfg.kv_dim()];
    let mut out = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut scratch = GqaScratch::default();
    apply_gqa_attention(&q, &k, &v, &mut out, seq_len, cfg, &mut scratch);
}

#[test]
#[should_panic(expected = "k_buf length mismatch")]
fn short_buffer_panics_before_unsafe_blas() {
    // A too-short K buffer must fail fast at the release-active length assert at
    // the kernel entry, BEFORE the macOS unsafe strided-BLAS path derives a
    // pointer into k_buf. The guard is OS-independent (it precedes the
    // `cfg(macos)` block), so a panic here proves the macOS unsafe FFI cannot be
    // reached with an out-of-bounds K pointer in release. RED before the fix:
    // the length check was `debug_assert_eq!`, a no-op in release.
    let cfg = GqaConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 1,
    };
    let seq_len = 1usize;
    let q = vec![0.0f32; seq_len * cfg.q_dim()];
    let k: Vec<f32> = Vec::new(); // too short: expected seq_len * kv_dim == 1
    let v = vec![0.0f32; seq_len * cfg.kv_dim()];
    let mut out = vec![0.0f32; seq_len * cfg.q_dim()];
    let mut scratch = GqaScratch::default();
    apply_gqa_attention(&q, &k, &v, &mut out, seq_len, cfg, &mut scratch);
}

#[test]
fn fused_softmax_all_neg_inf_row_is_zero_not_nan() {
    // A query row whose every valid score is -inf (fully masked / non-finite)
    // must not propagate NaN: `max_val` stays -inf, `exp(-inf - -inf)` is NaN,
    // and the row must fall back to a zero-probability distribution, matching
    // decode.rs and flash-causal. RED before the fix: the row stayed NaN.
    let seq_len = 2usize;
    let batch_rows = 2usize; // groups = 1
    let mut scores = vec![f32::NEG_INFINITY; batch_rows * seq_len];
    apply_scaled_causal_softmax_fused(&mut scores, batch_rows, seq_len, 0.125);
    assert!(
        scores.iter().all(|v| v.is_finite()),
        "row contains non-finite"
    );
    assert!(
        scores.iter().all(|&v| v == 0.0),
        "degenerate row must be zero"
    );
}

#[test]
fn fused_softmax_pos_inf_row_is_zero_not_nan() {
    // A +inf score also yields NaN through `exp(+inf - +inf)`; the degenerate
    // fallback (sum not positive-finite) must zero the row rather than emit NaN.
    let seq_len = 3usize;
    let batch_rows = 3usize;
    let mut scores = vec![f32::INFINITY; batch_rows * seq_len];
    apply_scaled_causal_softmax_fused(&mut scores, batch_rows, seq_len, 0.125);
    assert!(
        scores.iter().all(|v| v.is_finite()),
        "row contains non-finite"
    );
    assert!(
        scores.iter().all(|&v| v == 0.0),
        "degenerate row must be zero"
    );
}

/// ADR-080 C1: a lone NaN score anywhere in the visible range must zero
/// the whole row via the shared `row_fails_closed_pre_exp` pre-scan, not
/// just leave the NaN lane corrupted. Row 1 (qi=1, valid=2) is poisoned;
/// row 0 (qi=0, valid=1, its own well-formed score) is unaffected.
#[test]
fn fused_softmax_nan_lane_zeros_whole_row() {
    let seq_len = 2usize;
    let batch_rows = 2usize; // one query group of seq_len rows
    let mut scores = vec![1.0f32, 0.0, f32::NAN, 2.0];
    apply_scaled_causal_softmax_fused(&mut scores, batch_rows, seq_len, 1.0);
    assert!(
        scores[seq_len..].iter().all(|&v| v == 0.0),
        "NaN-poisoned row must zero out: {:?}",
        &scores[seq_len..]
    );
    assert!(
        scores[..seq_len].iter().all(|v| v.is_finite()),
        "sibling row must be unaffected: {:?}",
        &scores[..seq_len]
    );
}
