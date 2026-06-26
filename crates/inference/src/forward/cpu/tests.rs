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
            "fast_tanh({x}) = {actual}, expected {expected}, abs_err = {abs_err} (exceeds 4e-5 in core)",
        );
    }

    for &x in &tail_values {
        let expected = x.tanh();
        let actual = fast_tanh(x);
        let abs_err = (actual - expected).abs();
        assert!(
            abs_err < 2e-4,
            "fast_tanh({x}) = {actual}, expected {expected}, abs_err = {abs_err} (exceeds 2e-4 in tail)",
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

// A NaN in an attention-score row must not make the NEON fast path diverge from the
// scalar reference. seq_len must be >= CHUNK (16) so the row actually enters the NEON
// vector max loop; a shorter row falls to the scalar tail and matches trivially. With
// FMAX (the pre-fix reduction) the NaN propagates -> max = NaN -> every exp underflows
// to 0 -> the row is all zeros. With FMAXNM (maxNum) the NaN is dropped, matching the
// scalar path which normalizes the finite logits and gives the NaN lane zero weight.
// aarch64-only: the AVX2 path uses a different (position-dependent) NaN reduction and
// its maxNum/NaN-delta parity is a deliberate, separately-tracked follow-up.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_softmax_neon_nan_row_matches_scalar() {
    let num_heads = 1;
    let seq_len = 16;
    let n = num_heads * seq_len * seq_len;
    let mut x_simd = vec![0.0f32; n];
    for (i, v) in x_simd.iter_mut().enumerate() {
        *v = (i % seq_len) as f32;
    }
    // Poison the first lane of row 0 with NaN (the lane FMAX would propagate from).
    x_simd[0] = f32::NAN;
    let mut x_scalar = x_simd.clone();

    softmax_attention(&mut x_simd, seq_len, num_heads);
    softmax_attention_scalar(&mut x_scalar, seq_len, num_heads);

    // The NaN lane gets zero weight in both paths (fast_exp(NaN) -> 0.0).
    assert_eq!(x_simd[0], 0.0, "NEON NaN lane must be 0.0, not propagated");
    assert_eq!(x_scalar[0], 0.0);

    // Row 0 must be a real distribution (sums to 1), NOT the pre-fix all-zeros.
    let row0_sum: f32 = x_simd[0..seq_len].iter().sum();
    assert_relative_eq!(row0_sum, 1.0, epsilon = 1e-3);

    // NEON and scalar agree element-wise across every lane of the NaN row.
    for i in 0..seq_len {
        assert_relative_eq!(x_simd[i], x_scalar[i], epsilon = 1e-3);
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
            "fast_exp({x}) = {actual}, expected {expected}, rel_err = {rel_err}"
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

// ===================================================================
// Elementwise SIMD vs. scalar comparison tests
// ===================================================================

#[test]
fn test_rms_norm_known_values() {
    // Single token, hidden=4, gamma=all-ones, eps=0.
    // rms = sqrt(mean([1,2,3,4]^2)) = sqrt(7.5) ≈ 2.7386.
    // Expected: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    rms_norm(&mut x, &gamma, 4, 1e-8);
    let expected_rms = (7.5f32 + 1e-8).sqrt();
    for (i, &v) in x.iter().enumerate() {
        let expected = (i + 1) as f32 / expected_rms;
        assert_relative_eq!(v, expected, epsilon = 1e-5);
    }
}

#[test]
fn test_rms_norm_simd_matches_scalar() {
    // Two sizes: one that exercises SIMD main loops, one with a remainder tail.
    for &hidden in &[896usize, 2048, 4096, 13] {
        let rows = 4;
        let mut x_simd = make_deterministic_vec(rows * hidden, 0xC0DE);
        let mut x_scalar = x_simd.clone();
        let gamma = make_deterministic_vec_range(hidden, 0xC0DF, 0.9, 1.1);
        let eps = 1e-6;

        rms_norm(&mut x_simd, &gamma, hidden, eps);
        rms_norm_scalar(&mut x_scalar, &gamma, hidden, eps);

        for i in 0..(rows * hidden) {
            let diff = (x_simd[i] - x_scalar[i]).abs();
            assert!(
                diff <= 1e-4,
                "rms_norm mismatch at hidden={hidden} index={i}: simd={} scalar={} diff={diff}",
                x_simd[i],
                x_scalar[i],
            );
        }
    }
}

#[test]
fn test_silu_known_values() {
    // silu(0) = 0 * 0.5 = 0.
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311.
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689.
    let mut x = vec![0.0f32, 1.0, -1.0];
    silu_inplace(&mut x);
    assert_relative_eq!(x[0], 0.0, epsilon = 1e-5);
    // Use a loose tolerance because we use fast_exp (Schraudolph) in SIMD paths.
    assert_relative_eq!(x[1], 0.731_059_f32, epsilon = 5e-2);
    assert_relative_eq!(x[2], -0.268_941_f32, epsilon = 5e-2);
}

#[test]
fn test_silu_simd_matches_scalar() {
    for &n in &[896usize, 2048, 4096, 17] {
        let mut x_simd = make_deterministic_vec(n, 0x51C0);
        let mut x_scalar = x_simd.clone();

        silu_inplace(&mut x_simd);
        silu_inplace_scalar(&mut x_scalar);

        for i in 0..n {
            let diff = (x_simd[i] - x_scalar[i]).abs();
            assert!(
                diff <= 5e-2,
                "silu mismatch at n={n} index={i}: simd={} scalar={} diff={diff}",
                x_simd[i],
                x_scalar[i],
            );
        }
    }
}

#[test]
fn test_elementwise_mul_known_values() {
    let mut a = vec![2.0f32, 3.0, -1.0, 0.0];
    let b = vec![4.0f32, -2.0, 5.0, 7.0];
    elementwise_mul(&mut a, &b);
    assert_relative_eq!(a[0], 8.0, epsilon = 1e-6);
    assert_relative_eq!(a[1], -6.0, epsilon = 1e-6);
    assert_relative_eq!(a[2], -5.0, epsilon = 1e-6);
    assert_relative_eq!(a[3], 0.0, epsilon = 1e-6);
}

#[test]
fn test_elementwise_mul_simd_matches_scalar() {
    for &n in &[896usize, 2048, 4096, 11] {
        let mut a_simd = make_deterministic_vec(n, 0xE1A1);
        let b = make_deterministic_vec(n, 0xE1A2);
        let mut a_scalar = a_simd.clone();

        elementwise_mul(&mut a_simd, &b);
        elementwise_mul_scalar(&mut a_scalar, &b);

        for i in 0..n {
            // elementwise_mul is exact (single multiply), tolerance is floating-point rounding.
            let diff = (a_simd[i] - a_scalar[i]).abs();
            assert!(
                diff <= 1e-6,
                "elementwise_mul mismatch at n={n} index={i}: simd={} scalar={} diff={diff}",
                a_simd[i],
                a_scalar[i],
            );
        }
    }
}

// ===================================================================
// SIMD/scalar parity regression tests — LCG generator + relative error
// ===================================================================
//
// These tests use a second independent generator (LCG) and a proper
// relative-error comparison to catch silent numerical regressions when
// NEON SIMD kernels are modified.  They are intentionally separate from
// the xorshift-based tests above so that both data-generation paths are
// exercised.

/// LCG-based deterministic f32 vector in the range [-0.02, 0.02].
/// Uses a different bit-mixing strategy than `make_deterministic_vec`
/// so the two generators produce uncorrelated data at the same index.
fn lcg_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x6c62_272e_07bb_0142;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 32) as f32 / u32::MAX as f32;
        out.push(unit * 0.04 - 0.02);
    }
    out
}

/// Mixed absolute + relative error comparison.
///
/// Passes when `|a - b| <= rtol * max(|a|, |b|) + atol`.
///
/// Using a non-zero `atol` prevents near-zero elements from triggering
/// false failures when the *absolute* error is within f32 precision but
/// the *relative* error is large simply because both values approach zero.
fn assert_close_mixed(a: &[f32], b: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let abs_diff = (x - y).abs();
        let threshold = rtol * x.abs().max(y.abs()) + atol;
        assert!(
            abs_diff <= threshold,
            "{label}[{i}]: {x} vs {y}, abs_err={abs_diff:.3e} > rtol*mag+atol={threshold:.3e}"
        );
    }
}

/// Pure relative-error comparison (atol=0).  For each element the
/// denominator is `max(|a|, |b|, 1e-8)` to avoid division by zero.
fn assert_close(a: &[f32], b: &[f32], rtol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let denom = x.abs().max(y.abs()).max(1e-8);
        let rel = (x - y).abs() / denom;
        assert!(
            rel <= rtol,
            "{label}[{i}]: {x} vs {y}, rel_err={rel} > {rtol}"
        );
    }
}

#[test]
fn test_silu_simd_scalar_parity() {
    // Verify silu_inplace() (SIMD dispatch) matches silu_inplace_scalar()
    // at hidden sizes representative of real model widths.
    // Tolerance 1e-4 relative — silu uses fast_exp (Schraudolph) in NEON.
    for &n in &[896usize, 2048, 4096] {
        let mut x_simd = lcg_f32_vec(n, 0x5100_0001_u64 ^ (n as u64));
        let mut x_scalar = x_simd.clone();

        silu_inplace(&mut x_simd);
        silu_inplace_scalar(&mut x_scalar);

        assert_close(&x_simd, &x_scalar, 1e-4, &format!("silu_parity n={n}"));
    }
}

#[test]
fn test_gelu_simd_scalar_parity() {
    // Verify gelu() (SIMD dispatch) matches gelu_scalar().
    // Tolerance 1e-4 relative — NEON uses vdivq_f32 so error is only FMA order.
    for &n in &[896usize, 2048, 4096] {
        let mut x_simd = lcg_f32_vec(n, 0x9E10_0001_u64 ^ (n as u64));
        let mut x_scalar = x_simd.clone();

        gelu(&mut x_simd);
        gelu_scalar(&mut x_scalar);

        assert_close(&x_simd, &x_scalar, 1e-4, &format!("gelu_parity n={n}"));
    }
}

#[test]
fn test_rms_norm_simd_scalar_parity() {
    // 8 tokens × three hidden sizes.  Tolerance 1e-5 relative —
    // rms_norm uses ieee-accurate rsqrt so error is only sum-reduction order.
    let rows = 8usize;
    for &hidden in &[896usize, 2048, 4096] {
        let mut x_simd = lcg_f32_vec(rows * hidden, 0xC0DE_0001_u64 ^ (hidden as u64));
        let mut x_scalar = x_simd.clone();
        let gamma = {
            let raw = lcg_f32_vec(hidden, 0xC0DE_0002_u64 ^ (hidden as u64));
            // Shift into [0.9, 1.1] for realistic scale factors.
            raw.into_iter().map(|v| 1.0 + v * 5.0).collect::<Vec<_>>()
        };
        let eps = 1e-6_f32;

        rms_norm(&mut x_simd, &gamma, hidden, eps);
        rms_norm_scalar(&mut x_scalar, &gamma, hidden, eps);

        assert_close(
            &x_simd,
            &x_scalar,
            1e-5,
            &format!("rms_norm_parity hidden={hidden}"),
        );
    }
}

#[test]
fn test_layer_norm_simd_scalar_parity() {
    // 8 tokens × three hidden sizes.  Tolerance 1e-5 relative.
    let rows = 8usize;
    for &hidden in &[896usize, 2048, 4096] {
        let mut x_simd = lcg_f32_vec(rows * hidden, 0xF00D_0001_u64 ^ (hidden as u64));
        let mut x_scalar = x_simd.clone();
        let gamma = {
            let raw = lcg_f32_vec(hidden, 0xF00D_0002_u64 ^ (hidden as u64));
            raw.into_iter().map(|v| 1.0 + v * 5.0).collect::<Vec<_>>()
        };
        let beta = lcg_f32_vec(hidden, 0xF00D_0003_u64 ^ (hidden as u64));
        let eps = 1e-12_f32;

        layer_norm(&mut x_simd, &gamma, &beta, hidden, eps);
        layer_norm_scalar(&mut x_scalar, &gamma, &beta, hidden, eps);

        // layer_norm involves two horizontal reductions (mean + variance)
        // whose NEON pairwise-summation order diverges from scalar sequential
        // sum at large hidden sizes.  Near-zero output elements (gamma×z+beta≈0)
        // can have large *relative* error from tiny absolute differences.
        // We use a mixed criterion: |a-b| <= 1e-4*max(|a|,|b|) + 1e-5
        // so near-zero outputs fall back to absolute tolerance (1e-5 >> f32 ULP).
        assert_close_mixed(
            &x_simd,
            &x_scalar,
            1e-4,
            1e-5,
            &format!("layer_norm_parity hidden={hidden}"),
        );
    }
}

#[test]
fn test_softmax_simd_scalar_parity() {
    // seq_len in {32, 64, 128} with 8 heads.
    // Tolerance 1e-3 relative — fast_exp approximation introduces ~1-2% error
    // per row; normalisation cancels systematic bias but relative error remains.
    let num_heads = 8usize;
    for &seq_len in &[32usize, 64, 128] {
        let n = num_heads * seq_len * seq_len;
        let mut x_simd = lcg_f32_vec(n, 0x50F7_0001_u64 ^ (seq_len as u64));
        // Scale to realistic attention logit magnitudes [-2, 2].
        for v in &mut x_simd {
            *v *= 100.0;
        }
        let mut x_scalar = x_simd.clone();

        softmax_attention(&mut x_simd, seq_len, num_heads);
        softmax_attention_scalar(&mut x_scalar, seq_len, num_heads);

        assert_close(
            &x_simd,
            &x_scalar,
            1e-3,
            &format!("softmax_parity seq_len={seq_len}"),
        );

        // Sanity: every row must still sum to 1.0.
        for h in 0..num_heads {
            for s in 0..seq_len {
                let start = (h * seq_len + s) * seq_len;
                let row_sum: f32 = x_simd[start..start + seq_len].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-3,
                    "softmax_parity seq_len={seq_len} head={h} row={s}: sum={row_sum}"
                );
            }
        }
    }
}

#[test]
fn test_elementwise_mul_simd_scalar_parity() {
    // elementwise_mul is a single multiply per element — expect exact match
    // (both paths issue the same FP operation, just via SIMD registers vs scalar).
    // Using rtol=0 would be fragile across compilers; use 1e-7 for rounding
    // differences between vmulq_f32 and scalar `*`.
    for &n in &[896usize, 2048, 4096] {
        let mut a_simd = lcg_f32_vec(n, 0xE1A1_0001_u64 ^ (n as u64));
        let b = lcg_f32_vec(n, 0xE1A2_0001_u64 ^ (n as u64));
        let mut a_scalar = a_simd.clone();

        elementwise_mul(&mut a_simd, &b);
        elementwise_mul_scalar(&mut a_scalar, &b);

        assert_close(
            &a_simd,
            &a_scalar,
            1e-7,
            &format!("elementwise_mul_parity n={n}"),
        );
    }
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
