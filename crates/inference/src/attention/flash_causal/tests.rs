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
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.20, 0.10, 0.40, 0.30, 0.60, 0.50, 0.80,
        0.70, 0.30, 0.40, 0.10, 0.20, 0.70, 0.80, 0.50, 0.60, 0.40, 0.30, 0.20, 0.10, 0.80, 0.70,
        0.60, 0.50,
    ];
    let k_buf = vec![
        0.11, 0.21, 0.31, 0.41, 0.22, 0.12, 0.42, 0.32, 0.33, 0.43, 0.13, 0.23, 0.44, 0.34, 0.24,
        0.14,
    ];
    let v_buf = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    let expected = vec![
        1.0000000, 2.0000000, 3.0000000, 4.0000000, 1.0000000, 2.0000000, 3.0000000, 4.0000000,
        3.0149997, 4.015, 5.015, 6.015, 3.022_999, 4.022_999, 5.022_999, 6.022_999, 5.0673227,
        6.0673227, 7.0673227, 8.067_323, 5.088_859, 6.088_859, 7.088_859, 8.088_86, 7.114_979,
        8.114_979, 9.114_979, 10.114_979, 7.154_951, 8.154_951, 9.154_951, 10.154_951,
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

#[test]
#[should_panic(expected = "flash shape overflow")]
fn entry_shape_overflow_is_rejected_not_silently_accepted() {
    // num_heads * head_dim wraps in release: (2^63 + 1) * 2 == 2 (mod 2^64),
    // which without the guard makes the length asserts accept 4-element buffers
    // and then traps later from a derived offset. The guard rejects it up front.
    let seq_len = 2;
    let num_heads = (1usize << 63) + 1;
    let num_kv_heads = 1;
    let head_dim = 2;
    let q_buf = vec![0.0; 4];
    let k_buf = vec![0.0; 4];
    let v_buf = vec![0.0; 4];
    let mut attn_out = vec![0.0; 4];

    flash_attention_causal(
        &q_buf,
        &k_buf,
        &v_buf,
        &mut attn_out,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    );
}

#[test]
#[should_panic(expected = "flash scratch overflow")]
fn tile_config_scratch_overflow_is_rejected() {
    // External shapes are valid, but a caller-supplied huge `bc` wraps the
    // scratch products (br * bc) to 0, allocating undersized scratch buffers
    // that would trap on an internal slice. The scratch guard rejects it.
    let seq_len = 2;
    let num_heads = 1;
    let num_kv_heads = 1;
    let head_dim = 2;
    let q_buf = vec![0.0; 4];
    let k_buf = vec![0.0; 4];
    let v_buf = vec![0.0; 4];
    let mut attn_out = vec![0.0; 4];

    flash_attention_causal_with_config(
        &q_buf,
        &k_buf,
        &v_buf,
        &mut attn_out,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        FlashAttentionConfig {
            br: 2,
            bc: 1usize << 63,
            fallback_below: 0,
        },
    );
}

/// ADR-080 C1 (#780): `standard_attention_causal` (the short-sequence
/// fallback path, used here via `head_dim=4 > seq_len=3`). A NaN key
/// score at position 1 must fail-close every row that causally attends
/// to it (query 2), while query 0 (attends only to itself, never sees
/// position 1) stays unaffected.
#[test]
fn standard_path_nan_key_fails_closed_for_dependent_rows_only() {
    let seq_len = 3;
    let num_heads = 1;
    let num_kv_heads = 1;
    let head_dim = 4; // seq_len(3) < fallback_below(head_dim=4) -> standard path
    let q_buf = [1.0f32, 0.0, 0.0, 0.0].repeat(seq_len);
    let mut k_buf = vec![0.0f32; seq_len * head_dim];
    k_buf[0] = 1.0; // K0
    k_buf[head_dim] = f32::NAN; // K1: poison
    k_buf[2 * head_dim] = 0.5; // K2
    let mut v_buf = vec![0.0f32; seq_len * head_dim];
    v_buf[0] = 10.0; // V0
    v_buf[head_dim] = 20.0; // V1
    v_buf[2 * head_dim] = 30.0; // V2
    let mut out = vec![1.0f32; seq_len * head_dim]; // pre-fill nonzero to prove overwrite

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

    let pos0 = &out[0..head_dim];
    assert!(
        (pos0[0] - 10.0).abs() < 1e-4 && pos0[1..].iter().all(|&v| v == 0.0),
        "query 0 (self-attends only) must be unaffected: {pos0:?}"
    );
    let pos2 = &out[2 * head_dim..3 * head_dim];
    assert!(
        pos2.iter().all(|&v| v == 0.0),
        "query 2 (causally attends to the NaN-poisoned position 1) must \
             fail closed to exact zero: {pos2:?}"
    );
}

/// ADR-080 C1 (#780): the tiled Flash Attention path threads a per-row
/// `bad` flag across K-blocks (`flash_update_block`'s `bad_rows` /
/// `normalize_head_output`'s `bad`) so a NaN key discovered in one block
/// fails closed every row that causally depends on it, even in a LATER
/// block. `br=2, bc=2` forces `seq_len=4` into 2 K-blocks and 2 Q-blocks
/// with `fallback_below=0` (never use the short-sequence fallback), so
/// this specifically exercises `flash_update_block`/`normalize_head_output`,
/// not `standard_attention_causal`. Poison sits in K-block 0 (position 1);
/// query 0 (block 0, self-attends only) must stay clean, while queries
/// 1, 2, and 3 (which all causally see position 1, queries 2/3 via a
/// LATER Q-block re-processing the FIRST K-block) must all fail closed.
#[test]
fn tiled_path_nan_key_fails_closed_across_blocks() {
    let seq_len = 4;
    let num_heads = 1;
    let num_kv_heads = 1;
    let head_dim = 4;
    let q_buf = [1.0f32, 0.0, 0.0, 0.0].repeat(seq_len);
    let mut k_buf = vec![0.0f32; seq_len * head_dim];
    k_buf[0] = 1.0; // K0
    k_buf[head_dim] = f32::NAN; // K1: poison
    k_buf[2 * head_dim] = 0.5; // K2
    k_buf[3 * head_dim] = 0.3; // K3
    let mut v_buf = vec![0.0f32; seq_len * head_dim];
    v_buf[0] = 10.0; // V0
    v_buf[head_dim] = 20.0; // V1
    v_buf[2 * head_dim] = 30.0; // V2
    v_buf[3 * head_dim] = 40.0; // V3
    let mut out = vec![1.0f32; seq_len * head_dim];

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
            br: 2,
            bc: 2,
            fallback_below: 0,
        },
    );

    let pos0 = &out[0..head_dim];
    assert!(
        (pos0[0] - 10.0).abs() < 1e-4 && pos0[1..].iter().all(|&v| v == 0.0),
        "query 0 (self-attends only) must be unaffected: {pos0:?}"
    );
    for qi in 1..seq_len {
        let row = &out[qi * head_dim..(qi + 1) * head_dim];
        assert!(
            row.iter().all(|&v| v == 0.0),
            "query {qi} (causally attends to the NaN-poisoned position 1) \
                 must fail closed to exact zero: {row:?}"
        );
    }
}
