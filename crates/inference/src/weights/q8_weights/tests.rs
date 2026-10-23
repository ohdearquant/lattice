use super::*;
use crate::forward::cpu::matmul_bt;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn uniform_signed(state: &mut u32) -> f32 {
    let x = xorshift32(state);
    let u = x as f32 / u32::MAX as f32;
    u * 2.0 - 1.0
}

#[test]
fn test_quantize_identity() {
    let rows = 4;
    let cols = 4;
    let w = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 1.0, // row 3
    ];

    let q = quantize_matrix(&w, rows, cols).unwrap();
    let dq = dequantize_matrix(&q);

    assert_eq!(q.rows, rows);
    assert_eq!(q.cols, cols);
    assert_eq!(q.scales.len(), rows);
    assert_eq!(q.data.len(), rows * cols);

    for &scale in &q.scales {
        assert!(approx_eq(scale, 1.0 / 127.0, 1e-8));
    }
    for (orig, recon) in w.iter().zip(dq.iter()) {
        assert!(approx_eq(*orig, *recon, 1e-6));
    }
}

/// Build a GatedDeltaNetWeights whose decay params are internally consistent at
/// value-head granularity: a_log/dt_bias length == in_proj_b_rows == in_proj_a_rows.
/// `value_head_dim` is fixed at 2 (`z_rows = value_heads * 2`) so `matching_gdn_cfg`
/// can derive `linear_output_dim() == z_rows` for any `value_heads`.
const GDN_VALUE_HEAD_DIM: usize = 2;

fn valid_gdn_weights(value_heads: usize, hidden: usize) -> GatedDeltaNetWeights {
    let qkv_rows = value_heads * 4;
    let z_rows = value_heads * GDN_VALUE_HEAD_DIM;
    GatedDeltaNetWeights {
        in_proj_qkv: vec![0.0; qkv_rows * hidden],
        in_proj_qkv_rows: qkv_rows,
        in_proj_qkv_cols: hidden,
        in_proj_z: vec![0.0; z_rows * hidden],
        in_proj_z_rows: z_rows,
        in_proj_z_cols: hidden,
        in_proj_b: vec![0.0; value_heads * hidden],
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: vec![0.0; value_heads * hidden],
        in_proj_a_rows: value_heads,
        in_proj_a_cols: hidden,
        a_log: vec![0.0; value_heads],
        dt_bias: vec![0.0; value_heads],
        conv1d_weight: vec![0.0; qkv_rows * 4],
        conv_dim: qkv_rows,
        kernel_size: 4,
        norm_weight: vec![1.0; GDN_VALUE_HEAD_DIM],
        out_proj: vec![0.0; hidden * z_rows],
        out_proj_rows: hidden,
        out_proj_cols: z_rows,
    }
}

/// `Qwen35Config` whose GatedDeltaNet dimensions match `valid_gdn_weights(3, 8)`'s
/// derived geometry: `linear_num_value_heads() == 3` (matching `in_proj_b_rows` /
/// `in_proj_a_rows` / `a_log.len()`), `linear_qkv_dim() == 12` (matching `qkv_rows`),
/// `linear_output_dim() == 6` (matching `z_rows`), and `linear_value_head_dim == 2`
/// (matching `norm_weight`).
fn matching_gdn_cfg() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 8,
        linear_num_key_heads: 1,
        linear_key_head_dim: 3,
        linear_num_value_heads: Some(3),
        linear_value_head_dim: GDN_VALUE_HEAD_DIM,
        linear_conv_kernel_dim: 4,
        ..Qwen35Config::qwen35_2b()
    }
}

#[test]
fn quantize_gdn_weights_accepts_consistent_value_head_shapes() {
    let w = valid_gdn_weights(3, 8);
    let q = quantize_gdn_weights(&w, &matching_gdn_cfg())
        .expect("consistent value-head shapes must quantize");
    assert_eq!(q.a_log.len(), 3);
    assert_eq!(q.dt_bias.len(), 3);
}

#[test]
fn quantize_gdn_weights_rejects_decay_shape_mismatch() {
    // a_log length (4) disagrees with the value-head row count (3). The #262 loader
    // bug would have masked an asymmetric mismatch; the quantizer must fail closed
    // (return Err — not panic via quantize_matrix's assert, not silently proceed).
    let mut w = valid_gdn_weights(3, 8);
    w.a_log = vec![0.0; 4];
    match quantize_gdn_weights(&w, &matching_gdn_cfg()) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(msg.contains("a_log"), "error must name a_log, got: {msg}");
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for decay shape mismatch, got Ok"),
    }
}

/// A malformed checkpoint whose GDN `conv1d_weight` does not match the
/// `[qkv_dim, kernel_size]` geometry the forward pass (`cpu_q8.rs`
/// `conv1d_silu_fused`) indexes with must be rejected at Q8 ingress, not
/// reach generation and panic on out-of-bounds indexing on the first token.
///
/// Rejects a `conv1d_weight` whose length disagrees with the
/// `[qkv_dim, kernel_size]` geometry.
#[test]
fn quantize_gdn_weights_rejects_malformed_conv1d_weight() {
    let mut w = valid_gdn_weights(3, 8);
    w.conv1d_weight.pop(); // one element short of qkv_dim * kernel_size
    match quantize_gdn_weights(&w, &matching_gdn_cfg()) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("conv1d_weight"),
                "error must name conv1d_weight, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for malformed conv1d_weight, got Ok"),
    }
}

/// A checkpoint-supplied `conv_dim` that disagrees with `cfg.linear_qkv_dim()` must be
/// rejected even though `conv1d_weight.len()` is internally consistent with the bogus
/// `conv_dim` (so the `conv1d_weight` length guard above does not catch it). The NEON
/// forward path indexes `qkv_proj`/`conv_buffer` through `0..weights.conv_dim`; an
/// oversized `conv_dim` panics on the first decoded token instead of failing at ingress.
///
/// Rejects a checkpoint whose self-reported `conv_dim` disagrees with `qkv_dim`,
/// even when `conv1d_weight.len()` is internally consistent with the bogus value.
#[test]
fn quantize_gdn_weights_rejects_conv_dim_disagreeing_with_cfg() {
    let mut w = valid_gdn_weights(3, 8);
    // `conv1d_weight` itself is left cfg-correct (qkv_dim(12) * kernel_size(4) = 48
    // elements), so the existing `conv1d_weight` length guard alone would pass this —
    // only the separate self-reported `conv_dim` field (consumed directly by the NEON
    // forward path's `0..conv_dim` indexing) lies about being larger than `qkv_dim`.
    w.conv_dim = 16;
    match quantize_gdn_weights(&w, &matching_gdn_cfg()) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("conv_dim"),
                "error must name conv_dim, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for conv_dim disagreeing with cfg, got Ok"),
    }
}

#[test]
fn quantize_gdn_weights_accepts_conv_dim_matching_cfg() {
    // valid_gdn_weights already sets conv_dim == qkv_dim (12); this is an explicit
    // positive-control counterpart to the mismatch-rejection test above.
    let w = valid_gdn_weights(3, 8);
    assert_eq!(w.conv_dim, 12);
    assert!(quantize_gdn_weights(&w, &matching_gdn_cfg()).is_ok());
}

/// A malformed checkpoint whose GDN `norm_weight` (gated RMSNorm gamma) is
/// shorter than `value_dim` must be rejected at Q8 ingress, not reach
/// generation and panic on the `norm_weight[..value_dim]` slice in
/// `cpu_q8.rs` on the first token.
///
/// Rejects a `norm_weight` (gated RMSNorm gamma) shorter than `value_dim`.
#[test]
fn quantize_gdn_weights_rejects_malformed_norm_weight() {
    let mut w = valid_gdn_weights(3, 8);
    w.norm_weight = vec![1.0; 1]; // shorter than value_dim (GDN_VALUE_HEAD_DIM = 2)
    match quantize_gdn_weights(&w, &matching_gdn_cfg()) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("norm_weight"),
                "error must name norm_weight, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for malformed norm_weight, got Ok"),
    }
}

/// A checkpoint-supplied `GatedDeltaNetWeights` whose `in_proj_b_rows` /
/// `in_proj_a_rows` / `a_log` / `dt_bias` are self-consistent with each other but
/// disagree with `cfg.linear_num_value_heads()` must be rejected by the public
/// `quantize_gdn_weights`, not just by internal self-consistency checks. Before this
/// guard, only `in_proj_a_rows == in_proj_b_rows` was checked — a value that agrees
/// with itself but not with the runtime config still produced Q8 matrices
/// `matmul_bt_q8` asserts on during inference.
///
/// Rejects `in_proj_b`/`in_proj_a` row counts that are internally consistent with
/// each other but disagree with `cfg.linear_num_value_heads()`.
#[test]
fn quantize_gdn_weights_rejects_value_heads_disagreeing_with_cfg() {
    // Internally consistent (in_proj_b_rows == in_proj_a_rows == a_log.len() ==
    // dt_bias.len() == 5), but cfg says linear_num_value_heads() == 3.
    let mut w = valid_gdn_weights(3, 8);
    w.in_proj_b = vec![0.0; 5 * 8];
    w.in_proj_b_rows = 5;
    w.in_proj_b_cols = 8;
    w.in_proj_a = vec![0.0; 5 * 8];
    w.in_proj_a_rows = 5;
    w.in_proj_a_cols = 8;
    w.a_log = vec![0.0; 5];
    w.dt_bias = vec![0.0; 5];
    match quantize_gdn_weights(&w, &matching_gdn_cfg()) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("in_proj_b"),
                "error must name in_proj_b, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for value-head count disagreeing with cfg, got Ok \
                 (Q8 matrices incompatible with cfg would have been produced)"
        ),
    }
}

/// `a_log`/`dt_bias`/`conv1d_weight`/`norm_weight` are retained as f32 (never
/// quantized), so they never pass through `quantize_named_matrix`'s finite-value scan.
/// A NaN or `+inf` in any of the four must be rejected, not silently reach the GDN
/// decay/conv/gated-norm recurrence and poison recurrent state.
///
/// Rejects a NaN or `+inf` in any of `a_log`/`dt_bias`/`conv1d_weight`/`norm_weight`,
/// the four GDN vectors retained as f32 and never quantized.
#[test]
fn quantize_gdn_weights_rejects_non_finite_retained_vectors() {
    let cfg = matching_gdn_cfg();

    let mut w = valid_gdn_weights(3, 8);
    w.a_log[0] = f32::NAN;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.a_log[0] = f32::INFINITY;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.dt_bias[0] = f32::NAN;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.dt_bias[0] = f32::INFINITY;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.conv1d_weight[0] = f32::NAN;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.conv1d_weight[0] = f32::INFINITY;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.norm_weight[0] = f32::NAN;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));

    let mut w = valid_gdn_weights(3, 8);
    w.norm_weight[0] = f32::INFINITY;
    assert!(matches!(
        quantize_gdn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn quantize_gdn_weights_accepts_all_finite_retained_vectors() {
    let w = valid_gdn_weights(3, 8);
    assert!(quantize_gdn_weights(&w, &matching_gdn_cfg()).is_ok());
}

#[test]
fn checked_full_q_dim_rejects_overflow() {
    let cfg = Qwen35Config {
        num_attention_heads: 1 << 63,
        head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    assert!(matches!(
        cfg.checked_full_q_dim(),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn checked_full_q_dim_accepts_valid_config() {
    let cfg = Qwen35Config {
        num_attention_heads: 16,
        head_dim: 64,
        ..Qwen35Config::qwen35_2b()
    };
    assert_eq!(cfg.checked_full_q_dim().unwrap(), 16 * 64);
}

#[test]
fn checked_full_kv_dim_rejects_overflow() {
    let cfg = Qwen35Config {
        num_key_value_heads: 1 << 63,
        head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    assert!(matches!(
        cfg.checked_full_kv_dim(),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn checked_full_kv_dim_accepts_valid_config() {
    let cfg = Qwen35Config {
        num_key_value_heads: 2,
        head_dim: 64,
        ..Qwen35Config::qwen35_2b()
    };
    assert_eq!(cfg.checked_full_kv_dim().unwrap(), 2 * 64);
}

#[test]
fn checked_linear_qkv_dim_rejects_overflow() {
    let cfg = Qwen35Config {
        linear_num_key_heads: 1 << 63,
        linear_key_head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    assert!(matches!(
        cfg.checked_linear_qkv_dim(),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn checked_linear_qkv_dim_accepts_valid_config() {
    let cfg = matching_gdn_cfg();
    assert_eq!(cfg.checked_linear_qkv_dim().unwrap(), cfg.linear_qkv_dim());
    assert_eq!(cfg.checked_linear_qkv_dim().unwrap(), 12);
}

#[test]
fn checked_linear_output_dim_rejects_overflow() {
    let cfg = Qwen35Config {
        linear_num_value_heads: Some(1 << 63),
        linear_value_head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    assert!(matches!(
        cfg.checked_linear_output_dim(),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn checked_linear_output_dim_accepts_valid_config() {
    let cfg = matching_gdn_cfg();
    assert_eq!(
        cfg.checked_linear_output_dim().unwrap(),
        cfg.linear_output_dim()
    );
    assert_eq!(cfg.checked_linear_output_dim().unwrap(), 6);
}

#[test]
fn checked_linear_conv_len_rejects_overflow() {
    let cfg = Qwen35Config {
        linear_num_key_heads: 1 << 63,
        linear_key_head_dim: 2,
        linear_conv_kernel_dim: 4,
        ..Qwen35Config::qwen35_2b()
    };
    assert!(matches!(
        cfg.checked_linear_conv_len(),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn checked_linear_conv_len_accepts_valid_config() {
    let cfg = matching_gdn_cfg();
    assert_eq!(
        cfg.checked_linear_conv_len().unwrap(),
        cfg.linear_qkv_dim() * cfg.linear_conv_kernel_dim
    );
}

/// A config with `num_attention_heads = 2^63, head_dim = 2` (which passes
/// `Qwen35Config::validate`'s existing checks) must be rejected by
/// `infer_full_attention_shapes` via checked arithmetic instead of silently wrapping
/// `full_q_dim()` to a small/zero value that then passes geometry checks and reaches
/// indexing.
#[test]
fn quantize_full_attn_weights_rejects_overflowing_config() {
    let cfg = Qwen35Config {
        num_attention_heads: 1 << 63,
        num_key_value_heads: 1,
        head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    let w = FullAttentionLayerWeights {
        q_proj: vec![],
        k_proj: vec![],
        v_proj: vec![],
        o_proj: vec![],
        q_norm: vec![0.0; cfg.head_dim],
        k_norm: vec![0.0; cfg.head_dim],
    };
    assert!(matches!(
        quantize_full_attn_weights(&w, &cfg),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
fn test_quantize_roundtrip_accuracy() {
    let rows = 32;
    let cols = 64;
    let mut seed = 0x1234_5678u32;
    let mut w = Vec::with_capacity(rows * cols);

    for _ in 0..rows * cols {
        w.push(uniform_signed(&mut seed) * 0.03);
    }

    let q = quantize_matrix(&w, rows, cols).unwrap();
    let dq = dequantize_matrix(&q);

    let mut max_abs_err = 0.0f32;
    let mut mean_abs_err = 0.0f32;
    for (orig, recon) in w.iter().zip(dq.iter()) {
        let err = (orig - recon).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        mean_abs_err += err;
    }
    mean_abs_err /= (rows * cols) as f32;

    assert!(max_abs_err < 2.0e-4, "max_abs_err={max_abs_err}");
    assert!(mean_abs_err < 7.0e-5, "mean_abs_err={mean_abs_err}");
}

#[test]
fn test_matmul_bt_q8_matches_f32() {
    let m = 2;
    let k = 256;
    let n = 16;

    let mut seed = 0xCAFE_BABEu32;
    let mut a = Vec::with_capacity(m * k);
    let mut b = Vec::with_capacity(n * k);

    for _ in 0..m * k {
        a.push(uniform_signed(&mut seed) * 0.1);
    }

    for row in 0..n {
        let scale = (row as f32 + 1.0) / 8_192.0;
        b.push(127.0 * scale);
        for col in 1..k {
            let qv = ((row * 17 + col * 13) % 255) as i32 - 127;
            b.push(qv as f32 * scale);
        }
    }

    let b_q = quantize_matrix(&b, n, k).unwrap();

    let mut c_f32 = vec![0.0f32; m * n];
    let mut c_q8 = vec![0.0f32; m * n];
    matmul_bt(&a, &b, &mut c_f32, m, k, n);
    matmul_bt_q8(&a, &b_q, &mut c_q8, m, k, n);

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut large_outputs = 0usize;

    for (&ref_v, &q_v) in c_f32.iter().zip(c_q8.iter()) {
        let abs_err = (ref_v - q_v).abs();
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if ref_v.abs() > 0.01 {
            large_outputs += 1;
            let rel_err = abs_err / ref_v.abs();
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }
    }

    assert!(large_outputs > 0, "expected some non-trivial outputs");
    assert!(max_abs_err < 1.0e-4, "max_abs_err={max_abs_err}");
    assert!(max_rel_err < 0.01, "max_rel_err={max_rel_err}");
}

#[test]
fn test_matmul_bt_q8_known_values() {
    let a = vec![1.0, 2.0, -1.0]; // [1, 3]
    let b = vec![
        1.27, -0.64, 0.0, // row 0, scale = 0.01, q = [127, -64, 0]
        -2.54, 0.0, 1.28, // row 1, scale = 0.02, q = [-127, 0, 64]
    ];

    let q = quantize_matrix(&b, 2, 3).unwrap();
    assert!(approx_eq(q.scales[0], 0.01, 1e-8));
    assert!(approx_eq(q.scales[1], 0.02, 1e-8));
    assert_eq!(&q.data[0..3], &[127i8, -64i8, 0i8]);
    assert_eq!(&q.data[3..6], &[-127i8, 0i8, 64i8]);

    let mut c = vec![0.0f32; 2];
    matmul_bt_q8(&a, &q, &mut c, 1, 3, 2);

    let expected0 = 0.01 * (1.0 * 127.0 + 2.0 * -64.0 + -0.0);
    let expected1 = 0.02 * (1.0 * -127.0 + 2.0 * 0.0 + -64.0);

    assert!(approx_eq(c[0], expected0, 1e-6));
    assert!(approx_eq(c[1], expected1, 1e-6));
    assert!(approx_eq(c[0], -0.01, 1e-6));
    assert!(approx_eq(c[1], -3.82, 1e-6));
}

/// Tiny valid Q8Matrix for the overflow-guard tests: the guards fire before
/// `b_q` is inspected, so shape need only be self-consistent.
fn dummy_q8_1x1() -> Q8Matrix {
    Q8Matrix {
        data: vec![0i8],
        scales: vec![0.0f32],
        rows: 1,
        cols: 1,
    }
}

/// Overflow guard: `m * k` must be rejected with a dedicated overflow message
/// before the length `assert_eq!` can wrap it and panic with a misleading
/// "length does not match" message instead.
#[test]
#[should_panic(expected = "matmul shape overflow: m*k")]
fn test_matmul_bt_q8_rejects_mk_overflow() {
    let q = dummy_q8_1x1();
    let mut c = vec![0.0f32; 1];
    // m*k overflows; n*k and m*n do not.
    matmul_bt_q8(&[], &q, &mut c, usize::MAX, 2, 1);
}

/// Overflow guard: `n * k` (m*k passes first).
#[test]
#[should_panic(expected = "matmul shape overflow: n*k")]
fn test_matmul_bt_q8_rejects_nk_overflow() {
    let q = dummy_q8_1x1();
    let mut c = vec![0.0f32; 1];
    // m*k = 1*2 ok; n*k = MAX*2 overflows.
    matmul_bt_q8(&[0.0, 0.0], &q, &mut c, 1, 2, usize::MAX);
}

/// Overflow guard: `m * n` (m*k and n*k pass first).
#[test]
#[should_panic(expected = "matmul shape overflow: m*n")]
fn test_matmul_bt_q8_rejects_mn_overflow() {
    let q = dummy_q8_1x1();
    let mut c = vec![0.0f32; 1];
    // m*k = MAX*1 ok; n*k = 2*1 ok; m*n = MAX*2 overflows.
    matmul_bt_q8(&[], &q, &mut c, usize::MAX, 1, 2);
}

/// Overflow guard for the macOS tile-scratch size `TILE_N * k`. With
/// `m=0, k=MAX, n=0` every earlier product (`m*k`, `n*k`, `m*n`) and every
/// length assert is zero/empty, so control reaches the tile block; `TILE_N*k`
/// then overflows and `Q8MatmulScratch::prepare` must reject it with the
/// dedicated message rather than silently wrapping the scratch length.
#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "matmul shape overflow: TILE_N*k")]
fn test_matmul_bt_q8_rejects_tile_nk_overflow() {
    let q = Q8Matrix {
        data: vec![],
        scales: vec![],
        rows: 0,
        cols: usize::MAX,
    };
    matmul_bt_q8(&[], &q, &mut [], 0, usize::MAX, 0);
}

/// Overflow guard for the macOS tile-scratch size `m * TILE_N`. With
/// `m=MAX, k=0, n=0` the earlier products and length asserts are all
/// zero/empty, so control reaches the tile block; `m*TILE_N` then overflows
/// and must be rejected with a dedicated message, same as `TILE_N*k` above.
#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "matmul shape overflow: m*TILE_N")]
fn test_matmul_bt_q8_rejects_m_tile_n_overflow() {
    let q = Q8Matrix {
        data: vec![],
        scales: vec![],
        rows: 0,
        cols: 0,
    };
    matmul_bt_q8(&[], &q, &mut [], usize::MAX, 0, 0);
}

#[test]
fn test_zero_row_handling() {
    let b = vec![
        0.0, 0.0, 0.0, 0.0, // zero row
        0.5, -0.5, 0.25, -0.25,
    ];
    let q = quantize_matrix(&b, 2, 4).unwrap();

    assert!(approx_eq(q.scales[0], 1.0, 1e-8));
    assert_eq!(&q.data[0..4], &[0i8, 0i8, 0i8, 0i8]);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0f32; 2];
    matmul_bt_q8(&a, &q, &mut c, 1, 4, 2);

    assert!(approx_eq(c[0], 0.0, 1e-8));
    assert!(c[1].abs() > 0.0);
}

#[test]
fn test_scale_computation() {
    let w = vec![
        1.27, -0.63, 0.0, // max abs = 1.27 -> scale = 0.01
        -2.54, 2.0, 0.0, // max abs = 2.54 -> scale = 0.02
    ];
    let q = quantize_matrix(&w, 2, 3).unwrap();

    assert!(approx_eq(q.scales[0], 0.01, 1e-8));
    assert!(approx_eq(q.scales[1], 0.02, 1e-8));
    assert_eq!(&q.data[0..3], &[127i8, -63i8, 0i8]);
    assert_eq!(&q.data[3..6], &[-127i8, 100i8, 0i8]);
}

#[test]
fn test_memory_report() {
    let cfg = Qwen35Config::qwen35_2b();
    let (f32_bytes, q8_bytes, ratio) = memory_report(&cfg);

    assert_eq!(f32_bytes, 5_490_868_224);
    assert_eq!(q8_bytes, 1_375_004_928);
    assert!(q8_bytes < f32_bytes);
    assert!(ratio > 3.9 && ratio < 4.05, "ratio={ratio}");
}

#[test]
fn test_quantize_large_matrix() {
    let rows = 6_144;
    let cols = 2_048;
    let mut w = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        let row_scale = 0.005 + (r % 17) as f32 * 0.0005;
        for c in 0..cols {
            let bucket = ((r * 131 + c * 17) % 257) as i32 - 128;
            w.push(bucket as f32 * row_scale / 128.0);
        }
    }

    let q = quantize_matrix(&w, rows, cols).unwrap();

    assert_eq!(q.rows, rows);
    assert_eq!(q.cols, cols);
    assert_eq!(q.data.len(), rows * cols);
    assert_eq!(q.scales.len(), rows);
    assert!(q.scales.iter().all(|&s| s.is_finite() && s > 0.0));
    assert!(q.data.iter().any(|&v| v != 0));

    let first_row_scale = q.scales[0];
    let last_row_scale = q.scales[rows - 1];
    assert!(first_row_scale > 0.0);
    assert!(last_row_scale > 0.0);
}

#[test]
fn test_quantize_common_weights_moe_returns_err_not_panic() {
    use crate::error::InferenceError;
    use crate::model::qwen35::{
        CommonLayerWeights, FeedForwardWeights, MoeLayerWeights, MoeRouter, RoutedExperts,
        SharedExpert,
    };

    let hidden = 4usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let inter = 2usize;

    let router = MoeRouter::new(
        vec![0.0f32; num_experts * hidden],
        num_experts,
        num_experts_per_tok,
        hidden,
    )
    .expect("valid router shape");

    let experts = RoutedExperts::new(
        vec![0.0f32; num_experts * 2 * inter * hidden],
        vec![0.0f32; num_experts * hidden * inter],
        num_experts,
        hidden,
        inter,
    )
    .expect("valid experts shape");

    let shared_expert = SharedExpert::new(
        vec![0.0f32; inter * hidden],
        vec![0.0f32; inter * hidden],
        vec![0.0f32; hidden * inter],
        vec![0.0f32; hidden],
        hidden,
        inter,
    )
    .expect("valid shared expert shape");

    let moe_common = CommonLayerWeights {
        input_layernorm: vec![0.0f32; hidden],
        post_attention_layernorm: vec![0.0f32; hidden],
        ffn: FeedForwardWeights::Moe(MoeLayerWeights {
            router,
            experts,
            shared_expert,
        }),
    };

    let cfg = Qwen35Config {
        hidden_size: hidden,
        intermediate_size: inter,
        ..Qwen35Config::qwen35_2b()
    };
    let result = quantize_common_weights(&moe_common, &cfg);
    assert!(
        matches!(result, Err(InferenceError::UnsupportedModel(_))),
        "expected Err(UnsupportedModel), got: {result:?}"
    );
}

/// `quantize_matrix` must reject a weight matrix whose source row contains
/// `+inf` or `NaN`. Without ingress validation such a row can produce a
/// non-finite scale or silently quantize NaN to zero.
#[test]
fn test_quantize_matrix_rejects_nonfinite_source_row() {
    // Row 0 is finite; row 1 contains +inf.
    let w = vec![
        1.0,
        2.0,
        3.0, // row 0: finite, scale ≈ 3/127
        1.0,
        f32::INFINITY,
        0.0, // row 1: non-finite source
    ];
    match quantize_matrix(&w, 2, 3) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error must describe the non-finite source; got: {msg}"
            );
            assert!(msg.contains("element index 4"), "wrong attribution: {msg}");
        }
        Err(e) => panic!("expected InvalidInput for non-finite source, got: {e}"),
        Ok(_) => panic!("expected Err for non-finite source row, got Ok"),
    }
}

/// `quantize_matrix` must reject a weight matrix that contains `NaN`.
///
/// Unlike `±inf`, NaN does NOT propagate through the `> max_abs` comparison
/// in IEEE 754 — `NaN > max_abs` evaluates to `false` — so a NaN value
/// never updates `max_abs`.  The scale then stays finite and the NaN element
/// is silently quantized to 0 via Rust's saturating `f32 as i8` cast.
#[test]
fn test_quantize_matrix_rejects_nan_input() {
    // Row 0 is finite; row 1 contains NaN in lane 1.
    let w = vec![
        1.0,
        2.0,
        3.0, // row 0: finite
        1.0,
        f32::NAN,
        0.0, // row 1: NaN lane
    ];
    match quantize_matrix(&w, 2, 3) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error must describe the non-finite value; got: {msg}"
            );
            assert!(msg.contains("element index 4"), "wrong attribution: {msg}");
        }
        Err(e) => panic!("expected InvalidInput for NaN input, got: {e}"),
        Ok(_) => panic!("expected Err for NaN input, got Ok"),
    }
}

#[test]
fn test_quantize_matrix_rejects_shape_overflow() {
    match quantize_matrix(&[], usize::MAX, 2) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("overflows usize"),
                "error must describe the geometry overflow; got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput for shape overflow, got: {e}"),
        Ok(_) => panic!("expected Err for shape overflow, got Ok"),
    }
}

#[test]
fn test_quantize_matrix_rejects_scale_underflow() {
    let w = [f32::from_bits(1)];
    match quantize_matrix(&w, 1, 1) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-positive scale 0"),
                "error must describe the underflowed scale; got: {msg}"
            );
            assert!(msg.contains("row 0"), "wrong attribution: {msg}");
        }
        Err(e) => panic!("expected InvalidInput for scale underflow, got: {e}"),
        Ok(_) => panic!("expected Err for scale underflow, got Ok"),
    }
}

/// `quantize_gdn_weights` propagates the shared ingress error from its
/// named `in_proj_qkv` conversion. A single `+inf` value is enough.
#[test]
fn test_quantize_gdn_weights_rejects_nonfinite_source() {
    let mut w = valid_gdn_weights(3, 8);
    // The first in_proj_qkv element is non-finite.
    w.in_proj_qkv[0] = f32::INFINITY;
    match quantize_gdn_weights(&w, &matching_gdn_cfg()) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error must describe the non-finite source; got: {msg}"
            );
            assert!(
                msg.contains("in_proj_qkv"),
                "error must name the offending tensor; got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput for non-finite source, got: {e}"),
        Ok(_) => panic!("expected Err for non-finite source weight, got Ok"),
    }
}

#[test]
fn test_quantize_common_weights_dense_returns_ok() {
    use crate::model::qwen35::{CommonLayerWeights, DenseFfnWeights, FeedForwardWeights};

    let hidden = 4usize;
    let inter = 2usize;

    let dense_common = CommonLayerWeights {
        input_layernorm: vec![0.0f32; hidden],
        post_attention_layernorm: vec![0.0f32; hidden],
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![0.0f32; inter * hidden],
            up_proj: vec![0.0f32; inter * hidden],
            down_proj: vec![0.0f32; hidden * inter],
        }),
    };

    let cfg = Qwen35Config {
        hidden_size: hidden,
        intermediate_size: inter,
        ..Qwen35Config::qwen35_2b()
    };
    let result = quantize_common_weights(&dense_common, &cfg);
    assert!(
        result.is_ok(),
        "expected Ok for dense layer, got: {result:?}"
    );
}

fn valid_full_attention_weights(
    cfg: &Qwen35Config,
    head_dim: usize,
    hidden: usize,
) -> FullAttentionLayerWeights {
    let q_dim = cfg.num_attention_heads * head_dim;
    let kv_dim = cfg.num_key_value_heads * head_dim;
    let q_rows = 2 * q_dim;
    FullAttentionLayerWeights {
        q_proj: vec![0.0; q_rows * hidden],
        k_proj: vec![0.0; kv_dim * hidden],
        v_proj: vec![0.0; kv_dim * hidden],
        o_proj: vec![0.0; hidden * q_dim],
        q_norm: vec![1.0; head_dim],
        k_norm: vec![1.0; head_dim],
    }
}

/// An empty `q_norm` must be rejected at Q8 ingress, not reach
/// `infer_full_attention_shapes`'s `head_dim > 0` invariant as a panic
/// (checkpoint-file DoS on Q8 conversion).
#[test]
fn quantize_full_attn_weights_rejects_empty_q_norm() {
    let cfg = Qwen35Config {
        head_dim: 4,
        hidden_size: 6,
        ..Qwen35Config::qwen35_2b()
    };
    let mut w = valid_full_attention_weights(&cfg, 4, 6);
    w.q_norm = vec![];
    match quantize_full_attn_weights(&w, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(msg.contains("q_norm"), "error must name q_norm, got: {msg}");
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for empty q_norm, got Ok"),
    }
}

/// A `k_norm` whose length disagrees with `q_norm`/`head_dim` must be
/// rejected at Q8 ingress rather than panicking via the `k_norm.len() ==
/// head_dim` invariant in `infer_full_attention_shapes`.
#[test]
fn quantize_full_attn_weights_rejects_mismatched_k_norm() {
    let cfg = Qwen35Config {
        head_dim: 4,
        hidden_size: 6,
        ..Qwen35Config::qwen35_2b()
    };
    let mut w = valid_full_attention_weights(&cfg, 4, 6);
    w.k_norm = vec![1.0; 2]; // head_dim is 4; k_norm must match
    match quantize_full_attn_weights(&w, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(msg.contains("k_norm"), "error must name k_norm, got: {msg}");
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for mismatched k_norm, got Ok"),
    }
}

/// A `q_norm`/`k_norm` pair that agree with each other but disagree with
/// `cfg.head_dim` must be rejected. Before this guard, `infer_full_attention_shapes`
/// derived `head_dim` FROM `q_norm.len()` instead of `cfg.head_dim`, so a
/// consistently-malformed checkpoint (`q_norm.len() == k_norm.len() != cfg.head_dim`)
/// passed Q8 conversion and panicked in `matmul_bt_q8` on the first full-attention
/// token — a checkpoint-triggerable process DoS.
#[test]
fn quantize_full_attn_weights_rejects_q_norm_disagreeing_with_cfg_head_dim() {
    let cfg = Qwen35Config {
        head_dim: 4,
        hidden_size: 6,
        ..Qwen35Config::qwen35_2b()
    };
    // q_norm and k_norm agree with each other (both length 5) but neither matches
    // cfg.head_dim (4).
    let mut w = valid_full_attention_weights(&cfg, 4, 6);
    w.q_norm = vec![1.0; 5];
    w.k_norm = vec![1.0; 5];
    match quantize_full_attn_weights(&w, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(msg.contains("q_norm"), "error must name q_norm, got: {msg}");
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for q_norm/k_norm disagreeing with cfg.head_dim, got Ok \
                 (would panic in matmul_bt_q8 on the first full-attention token)"
        ),
    }
}

/// An empty `input_layernorm` disagreeing with `cfg.hidden_size` must be
/// rejected at Q8 ingress instead of reaching `infer_dense_shapes` with an
/// hidden dimension the checkpoint never declared.
#[test]
fn quantize_common_weights_rejects_empty_input_layernorm() {
    use crate::model::qwen35::{CommonLayerWeights, DenseFfnWeights, FeedForwardWeights};

    let cfg = Qwen35Config {
        hidden_size: 4,
        intermediate_size: 2,
        ..Qwen35Config::qwen35_2b()
    };
    let common = CommonLayerWeights {
        input_layernorm: vec![],
        post_attention_layernorm: vec![0.0f32; 4],
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![0.0f32; 8],
            up_proj: vec![0.0f32; 8],
            down_proj: vec![0.0f32; 8],
        }),
    };

    match quantize_common_weights(&common, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("input_layernorm") || msg.contains("hidden"),
                "error must describe the empty hidden dimension, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for empty input_layernorm, got Ok"),
    }
}

/// `cfg.hidden_size == 0` with all-empty dense tensors satisfies every
/// `validate_cfg_len`/`validate_source_geometry` check vacuously (`0 == 0`), so it
/// must be rejected by an explicit `hidden == 0` guard, not left to reach
/// `qwen35_rms_norm`'s `x.len() / hidden` division on the first forward pass.
#[test]
fn quantize_common_weights_rejects_zero_hidden_size() {
    use crate::model::qwen35::{CommonLayerWeights, DenseFfnWeights, FeedForwardWeights};

    let cfg = Qwen35Config {
        hidden_size: 0,
        intermediate_size: 0,
        ..Qwen35Config::qwen35_2b()
    };
    let common = CommonLayerWeights {
        input_layernorm: vec![],
        post_attention_layernorm: vec![],
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![],
            up_proj: vec![],
            down_proj: vec![],
        }),
    };

    match quantize_common_weights(&common, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("hidden_size") || msg.contains("hidden"),
                "error must describe the zero hidden dimension, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for hidden_size == 0, got Ok (would panic on x.len() / hidden \
                 in qwen35_rms_norm on the first forward pass)"
        ),
    }
}

/// A short `post_attention_layernorm` (shorter than `cfg.hidden_size`, with a
/// correctly-sized `input_layernorm`) must be rejected, not silently accepted. In
/// release builds `qwen35_rms_norm` zips gamma against the activation, so a short
/// gamma would leave trailing hidden values unnormalized instead of failing loudly —
/// a silent-wrong-output bug, not a panic.
#[test]
fn quantize_common_weights_rejects_short_post_attention_layernorm() {
    use crate::model::qwen35::{CommonLayerWeights, DenseFfnWeights, FeedForwardWeights};

    let cfg = Qwen35Config {
        hidden_size: 4,
        intermediate_size: 2,
        ..Qwen35Config::qwen35_2b()
    };
    let common = CommonLayerWeights {
        input_layernorm: vec![1.0f32; 4],
        post_attention_layernorm: vec![1.0f32; 2], // shorter than cfg.hidden_size (4)
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![0.0f32; 8],
            up_proj: vec![0.0f32; 8],
            down_proj: vec![0.0f32; 8],
        }),
    };

    match quantize_common_weights(&common, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("post_attention_layernorm"),
                "error must name post_attention_layernorm, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for short post_attention_layernorm, got Ok (would silently \
                 leave trailing hidden values unnormalized)"
        ),
    }
}

/// Reference two-pass implementation predating the ingress fusion: a full
/// finite-value scan over the source matrix, followed by the max-abs +
/// quantize loop as two independent passes. Kept only to prove the fused
/// single-pass `quantize_named_matrix` produces byte-identical output.
fn quantize_matrix_two_pass_reference(w: &[f32], rows: usize, cols: usize) -> Q8Matrix {
    assert!(
        w.iter().all(|v| v.is_finite()),
        "reference requires finite input"
    );

    let mut data = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);

    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let row = &w[start..end];

        let mut max_abs = 0.0f32;
        for &v in row {
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales.push(scale);

        for &v in row {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q);
        }
    }

    Q8Matrix {
        data,
        scales,
        rows,
        cols,
    }
}

/// Proves the fused single-pass `quantize_matrix` is byte-identical to the
/// pre-fusion two-pass reference across representative shapes, including a
/// forced zero row (the `max_abs == 0.0` branch). max(|x_i|) is
/// order-independent and the per-element quantization involves no FP
/// accumulation, so fusing the finite-value scan into the max-abs loop
/// must not change any output byte or scale.
#[test]
fn test_fused_pass_matches_two_pass_reference_bit_exact() {
    let cases: [(usize, usize, u32); 4] = [
        (4, 4, 0x1111_1111),
        (32, 64, 0x1234_5678),
        (17, 33, 0xDEAD_BEEF),
        (6, 4, 0xCAFE_BABE),
    ];

    for (rows, cols, seed0) in cases {
        let mut seed = seed0;
        let mut w = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            w.push(uniform_signed(&mut seed) * 0.37);
        }
        // Force a zero row to exercise the max_abs == 0.0 branch identically
        // in both the fused and reference paths.
        for v in &mut w[cols..2 * cols] {
            *v = 0.0;
        }

        let fused = quantize_matrix(&w, rows, cols).expect("valid finite matrix must quantize");
        let reference = quantize_matrix_two_pass_reference(&w, rows, cols);

        assert_eq!(
            fused.data, reference.data,
            "quantized bytes diverged for {rows}x{cols}"
        );
        assert_eq!(
            fused.scales, reference.scales,
            "scales diverged for {rows}x{cols}"
        );
    }
}
