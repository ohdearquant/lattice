use super::*;
use approx::assert_relative_eq;

#[test]
fn test_matmul_small_known_matrices() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    let c = matmul(&a, &b, 2, 3, 2);
    assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_bt_small_known_matrices() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3, used as B^T logically
    let mut c = vec![0.0; 4];
    matmul_bt(&a, &b, &mut c, 2, 3, 2);
    assert_eq!(c, vec![50.0, 68.0, 122.0, 167.0]);
}

#[test]
fn test_matmul_bt_simd_matches_scalar_large() {
    // Test with sizes that exercise the 4-accumulator SIMD paths
    // (k=384 exercises NEON 16-wide and AVX2 32-wide main loops).
    let m = 16;
    let k = 384;
    let n = 384;
    let a = make_deterministic_vec(m * k, 0xABCD);
    let b = make_deterministic_vec(n * k, 0x1234);
    let mut c_simd = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    matmul_bt(&a, &b, &mut c_simd, m, k, n);
    matmul_bt_scalar(&a, &b, &mut c_scalar, m, k, n);

    for i in 0..(m * n) {
        assert_relative_eq!(c_simd[i], c_scalar[i], epsilon = 1e-3);
    }
}

#[test]
fn test_matmul_bt_tiled_ffn_up() {
    // FFN up-projection: 16×384 @ (1536×384)^T = 16×1536
    // This exercises the tiled path (16*384*1536 = 9,437,184 >> 1M)
    // with dimensions that are not multiples of TILE_I=4 but are multiples of TILE_J=8.
    let m = 16;
    let k = 384;
    let n = 1536;
    let a = make_deterministic_vec(m * k, 0xFFD1);
    let b = make_deterministic_vec(n * k, 0xFFD2);
    let mut c_tiled = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    matmul_bt(&a, &b, &mut c_tiled, m, k, n);
    matmul_bt_scalar(&a, &b, &mut c_scalar, m, k, n);

    for i in 0..(m * n) {
        assert_relative_eq!(c_tiled[i], c_scalar[i], epsilon = 1e-3);
    }
}

#[test]
fn test_matmul_bt_tiled_ffn_down() {
    // FFN down-projection: 16×1536 @ (384×1536)^T = 16×384
    // Exercises the tiled path with large K dimension.
    let m = 16;
    let k = 1536;
    let n = 384;
    let a = make_deterministic_vec(m * k, 0xFFD3);
    let b = make_deterministic_vec(n * k, 0xFFD4);
    let mut c_tiled = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    matmul_bt(&a, &b, &mut c_tiled, m, k, n);
    matmul_bt_scalar(&a, &b, &mut c_scalar, m, k, n);

    for i in 0..(m * n) {
        assert_relative_eq!(c_tiled[i], c_scalar[i], epsilon = 1e-3);
    }
}

#[test]
fn test_matmul_bt_tiled_edge_dimensions() {
    // Test with dimensions that are NOT multiples of tile sizes (TILE_I=4, TILE_J=8, TILE_K=128).
    // m=17 (not div by 4), k=500 (not div by 128), n=130 (not div by 8).
    // total_work = 17*500*130 = 1,105,000 > 1M => exercises the tiled path
    // with edge-tile scalar fallback at all three tile boundaries.
    let m = 17;
    let k = 500;
    let n = 130;
    let a = make_deterministic_vec(m * k, 0xED01);
    let b = make_deterministic_vec(n * k, 0xED02);
    let mut c_tiled = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    matmul_bt(&a, &b, &mut c_tiled, m, k, n);
    matmul_bt_scalar(&a, &b, &mut c_scalar, m, k, n);

    for i in 0..(m * n) {
        assert_relative_eq!(c_tiled[i], c_scalar[i], epsilon = 1e-3);
    }
}

#[test]
fn test_matmul_bt_tiled_long_seq() {
    // Long sequence attention projection: 128×384 @ (384×384)^T = 128×384
    // 128*384*384 = 18,874,368 >> 1M
    let m = 128;
    let k = 384;
    let n = 384;
    let a = make_deterministic_vec(m * k, 0xBEEF);
    let b = make_deterministic_vec(n * k, 0xCAFE);
    let mut c_tiled = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    matmul_bt(&a, &b, &mut c_tiled, m, k, n);
    matmul_bt_scalar(&a, &b, &mut c_scalar, m, k, n);

    for i in 0..(m * n) {
        assert_relative_eq!(c_tiled[i], c_scalar[i], epsilon = 1e-3);
    }
}

#[test]
fn test_layer_norm_known_pair() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    layer_norm(&mut x, &gamma, &beta, 2, 0.0);
    assert_relative_eq!(x[0], -1.0, epsilon = 1e-6);
    assert_relative_eq!(x[1], 1.0, epsilon = 1e-6);
    assert_relative_eq!(x[2], -1.0, epsilon = 1e-6);
    assert_relative_eq!(x[3], 1.0, epsilon = 1e-6);
}

#[test]
fn test_layer_norm_simd_matches_scalar() {
    // Test with a hidden size that exercises SIMD paths (384 = 24 chunks of 16 NEON).
    let hidden = 384;
    let rows = 8;
    let mut x_simd = make_deterministic_vec(rows * hidden, 0xF00D);
    let mut x_scalar = x_simd.clone();
    let gamma = make_deterministic_vec_range(hidden, 0xAA01, 0.8, 1.2);
    let beta = make_deterministic_vec_range(hidden, 0xBE01, -0.1, 0.1);
    let eps = 1e-12;

    layer_norm(&mut x_simd, &gamma, &beta, hidden, eps);
    layer_norm_scalar(&mut x_scalar, &gamma, &beta, hidden, eps);

    for i in 0..(rows * hidden) {
        assert_relative_eq!(x_simd[i], x_scalar[i], epsilon = 1e-4);
    }
}

#[test]
fn test_fast_tanh_precision() {
    // Verify our Padé (7,6) rational approximation against std tanh.
    // Core region |x| <= 3: error < 4e-5 (Padé convergence is excellent).
    // Tail region |x| > 3: error < 2e-4 (rational slightly overshoots
    //   before the |x| >= 10 clamp kicks in). The clamp catches it.
    let core_values: [f32; 13] = [
        -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0,
    ];
    let tail_values: [f32; 4] = [-10.0, -5.0, 5.0, 10.0];

    for &x in &core_values {
        let expected = x.tanh();
        let actual = fast_tanh(x);
        let abs_err = (actual - expected).abs();
        assert!(
            abs_err < 4e-5,
            "fast_tanh({}) = {}, expected {}, abs_err = {} (exceeds 4e-5 in core)",
            x,
            actual,
            expected,
            abs_err,
        );
    }

    for &x in &tail_values {
        let expected = x.tanh();
        let actual = fast_tanh(x);
        let abs_err = (actual - expected).abs();
        assert!(
            abs_err < 2e-4,
            "fast_tanh({}) = {}, expected {}, abs_err = {} (exceeds 2e-4 in tail)",
            x,
            actual,
            expected,
            abs_err,
        );
    }
}

#[test]
fn test_gelu_known_values() {
    let mut x = vec![-1.0, 0.0, 1.0];
    gelu(&mut x);
    assert_relative_eq!(x[0], -0.1588, epsilon = 1e-3);
    assert_relative_eq!(x[1], 0.0, epsilon = 1e-6);
    assert_relative_eq!(x[2], 0.8412, epsilon = 1e-3);
}

#[test]
fn test_gelu_simd_matches_scalar() {
    // Test with size that exercises SIMD paths.
    // Tolerance tightened to 1e-5: NEON now uses vdivq_f32 (true division)
    // matching AVX2's _mm256_div_ps, so the only difference vs scalar is
    // FMA instruction ordering (fused multiply-add rounding).
    let n = 1536;
    let mut x_simd = make_deterministic_vec(n, 0xEE10);
    let mut x_scalar = x_simd.clone();

    gelu(&mut x_simd);
    gelu_scalar(&mut x_scalar);

    for i in 0..n {
        assert_relative_eq!(x_simd[i], x_scalar[i], epsilon = 1e-5);
    }
}

#[test]
fn test_softmax_attention_rows_sum_to_one_and_are_stable() {
    let mut x = vec![1000.0, 1000.0, 0.0, 1.0];
    softmax_attention(&mut x, 2, 1);
    assert_relative_eq!(x[0] + x[1], 1.0, epsilon = 1e-6);
    assert_relative_eq!(x[2] + x[3], 1.0, epsilon = 1e-6);
    assert_relative_eq!(x[0], 0.5, epsilon = 1e-6);
    assert!(x[3] > x[2]);
}

#[test]
fn test_softmax_simd_matches_scalar() {
    // 12 heads, seq_len=16 -> exercises SIMD softmax on row length 16
    let num_heads = 12;
    let seq_len = 16;
    let n = num_heads * seq_len * seq_len;
    let mut x_simd = make_deterministic_vec(n, 0x50F7);
    let mut x_scalar = x_simd.clone();

    softmax_attention(&mut x_simd, seq_len, num_heads);
    softmax_attention_scalar(&mut x_scalar, seq_len, num_heads);

    for i in 0..n {
        assert_relative_eq!(x_simd[i], x_scalar[i], epsilon = 1e-3);
    }

    // Verify rows still sum to 1
    for h in 0..num_heads {
        for s in 0..seq_len {
            let start = (h * seq_len + s) * seq_len;
            let row_sum: f32 = x_simd[start..start + seq_len].iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-3);
        }
    }
}

#[test]
fn test_fast_exp_accuracy() {
    // Verify fast_exp matches std exp within acceptable tolerance for softmax.
    // Schraudolph's method has ~5-6% relative error which is fine because
    // softmax normalizes by the sum, so systematic bias cancels out.
    for &x in &[-10.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
        let expected = x.exp();
        let actual = fast_exp(x);
        let rel_err = ((actual - expected) / expected).abs();
        assert!(
            rel_err < 0.08,
            "fast_exp({}) = {}, expected {}, rel_err = {}",
            x,
            actual,
            expected,
            rel_err
        );
    }
}

#[test]
fn test_add_bias_gelu_matches_separate_ops() {
    // Verify fused add_bias_gelu produces the same result as add_bias + gelu.
    let dim = 1536; // intermediate_size for BGE-small
    let rows = 16;
    let n = rows * dim;
    let mut x_fused = make_deterministic_vec(n, 0xFB01);
    let mut x_separate = x_fused.clone();
    let bias = make_deterministic_vec_range(dim, 0xFB02, -0.01, 0.01);

    add_bias_gelu(&mut x_fused, &bias, dim);

    add_bias(&mut x_separate, &bias, dim);
    gelu(&mut x_separate);

    for i in 0..n {
        assert_relative_eq!(x_fused[i], x_separate[i], epsilon = 1e-5);
    }
}

#[test]
fn test_add_bias_gelu_scalar_matches_fused() {
    // Ensure the SIMD path matches the scalar path for the fused operation.
    // Tightened to 1e-5 after switching NEON from vrecpeq_f32 to vdivq_f32.
    let dim = 384;
    let rows = 8;
    let n = rows * dim;
    let mut x_dispatched = make_deterministic_vec(n, 0xFB03);
    let mut x_scalar = x_dispatched.clone();
    let bias = make_deterministic_vec_range(dim, 0xFB04, -0.01, 0.01);

    add_bias_gelu(&mut x_dispatched, &bias, dim);
    add_bias_gelu_scalar(&mut x_scalar, &bias, dim);

    for i in 0..n {
        assert_relative_eq!(x_dispatched[i], x_scalar[i], epsilon = 1e-5);
    }
}

// --- Test helpers ---

/// Deterministic pseudo-random vector for reproducible tests.
fn make_deterministic_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
    if state == 0 {
        state = 0xA341_316C;
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let unit = state as f32 / u32::MAX as f32;
        out.push(unit * 0.04 - 0.02);
    }
    out
}

/// Deterministic pseudo-random vector in a custom range.
fn make_deterministic_vec_range(len: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
    if state == 0 {
        state = 0xA341_316C;
    }
    let range = hi - lo;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let unit = state as f32 / u32::MAX as f32;
        out.push(lo + unit * range);
    }
    out
}
