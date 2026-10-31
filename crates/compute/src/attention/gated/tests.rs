use super::*;

/// Deterministic pseudo-random data (xorshift64 + float extraction).
fn det_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
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

// NaN-honest max|a-b|: a `.fold(0.0, f32::max)` silently drops a NaN/Inf
// operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
// wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if !d.is_finite() {
            return d;
        }
        if d > max {
            max = d;
        }
    }
    max
}

// ---------------------------------------------------------------
// deinterleave_q_gate
// ---------------------------------------------------------------

#[test]
fn test_deinterleave_roundtrip() {
    // Build a packed [Q|gate] buffer: head h occupies [h*2*D..(h+1)*2*D]
    // with Q in the first half and gate in the second half.
    let num_heads = 4usize;
    let head_dim = 8usize;
    let q_dim = num_heads * head_dim;
    let packed = det_data(2 * q_dim, 42);

    let mut q_buf = vec![0.0f32; q_dim];
    let mut gate_buf = vec![0.0f32; q_dim];
    deinterleave_q_gate(&packed, &mut q_buf, &mut gate_buf, num_heads, head_dim);

    // Re-interleave manually and compare with original.
    let mut reinterleaved = vec![0.0f32; 2 * q_dim];
    for h in 0..num_heads {
        let dst = h * head_dim * 2;
        let src = h * head_dim;
        reinterleaved[dst..dst + head_dim].copy_from_slice(&q_buf[src..src + head_dim]);
        reinterleaved[dst + head_dim..dst + head_dim * 2]
            .copy_from_slice(&gate_buf[src..src + head_dim]);
    }
    for (i, (&orig, &rebuilt)) in packed.iter().zip(reinterleaved.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            rebuilt.to_bits(),
            "bit mismatch at index {i}: {orig:?} vs {rebuilt:?}"
        );
    }
}

#[test]
fn test_deinterleave_values() {
    // Verify that Q and gate slices come from the correct positions.
    // packed = [Q0_0, Q0_1, G0_0, G0_1, Q1_0, Q1_1, G1_0, G1_1] for 2 heads, head_dim=2
    let num_heads = 2usize;
    let head_dim = 2usize;
    let packed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut q_buf = vec![0.0f32; 4];
    let mut gate_buf = vec![0.0f32; 4];
    deinterleave_q_gate(&packed, &mut q_buf, &mut gate_buf, num_heads, head_dim);
    assert_eq!(q_buf, [1.0, 2.0, 5.0, 6.0]);
    assert_eq!(gate_buf, [3.0, 4.0, 7.0, 8.0]);
}

#[test]
#[should_panic(expected = "q_and_gate.len()")]
fn test_deinterleave_rejects_overlong_packed_input() {
    // A too-LONG packed buffer does not index-panic in the copy loop (the
    // source has enough elements), so without the release-active length
    // assert it would silently process only the first 2*q_dim and succeed.
    // The guard must reject it. Mutation-sensitive: removing the assert_eq!
    // makes this test fail (no panic on the silent partial result).
    let num_heads = 2usize;
    let head_dim = 2usize;
    let q_dim = num_heads * head_dim;
    let overlong = vec![0.0f32; 2 * q_dim + 2];
    let mut q_buf = vec![0.0f32; q_dim];
    let mut gate_buf = vec![0.0f32; q_dim];
    deinterleave_q_gate(&overlong, &mut q_buf, &mut gate_buf, num_heads, head_dim);
}

// ---------------------------------------------------------------
// apply_sigmoid_gate_scalar — correctness
// ---------------------------------------------------------------

#[test]
fn test_sigmoid_gate_correctness() {
    // sigmoid(0) = 0.5, sigmoid(large) ≈ 1.0, sigmoid(-large) ≈ 0.0
    let gate: Vec<f32> = vec![0.0, 100.0, -100.0, 2.0, -2.0];
    let mut ctx = vec![1.0f32; 5];
    apply_sigmoid_gate_scalar(&mut ctx, &gate);

    // context * sigmoid(gate)
    // sigmoid(0) = 0.5
    let diff0 = (ctx[0] - 0.5).abs();
    assert!(diff0 < 1e-6, "sigmoid(0): expected 0.5, got {}", ctx[0]);
    // sigmoid(100) ≈ 1.0
    assert!(
        ctx[1] > 0.999,
        "sigmoid(100) should be ≈1.0, got {}",
        ctx[1]
    );
    // sigmoid(-100) ≈ 0.0
    assert!(
        ctx[2] < 1e-6,
        "sigmoid(-100) should be ≈0.0, got {}",
        ctx[2]
    );
    // sigmoid(2) ≈ 0.880797
    let expected_sig2 = 1.0 / (1.0 + (-2.0f32).exp());
    assert!(
        (ctx[3] - expected_sig2).abs() < 1e-6,
        "sigmoid(2): expected {expected_sig2}, got {}",
        ctx[3]
    );
    // sigmoid(-2) ≈ 0.119203
    let expected_sig_neg2 = 1.0 / (1.0 + 2.0f32.exp());
    assert!(
        (ctx[4] - expected_sig_neg2).abs() < 1e-6,
        "sigmoid(-2): expected {expected_sig_neg2}, got {}",
        ctx[4]
    );
}

#[test]
fn test_gate_identity_when_zeros() {
    // gate of all zeros → context * sigmoid(0) = context * 0.5
    let n = 32usize;
    let mut ctx = det_data(n, 7);
    let expected: Vec<f32> = ctx.iter().map(|&c| c * 0.5).collect();
    let gate = vec![0.0f32; n];
    apply_sigmoid_gate_scalar(&mut ctx, &gate);
    for (i, (&got, &exp)) in ctx.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "index {i}: expected {exp}, got {got}"
        );
    }
}

// ---------------------------------------------------------------
// SIMD vs scalar agreement
// ---------------------------------------------------------------

#[test]
fn test_sigmoid_gate_simd_matches_scalar() {
    let n = 64usize;
    let gate = det_data(n, 1234);
    let ctx_base = det_data(n, 5678);

    let mut ctx_scalar = ctx_base.clone();
    apply_sigmoid_gate_scalar(&mut ctx_scalar, &gate);

    let mut ctx_simd = ctx_base.clone();
    apply_sigmoid_gate_simd(&mut ctx_simd, &gate);

    // The SIMD path uses the Schraudolph fast-exp bit trick (~5-6% relative
    // error on exp), which compounds into ~1% absolute error on sigmoid.
    // Tolerance is set to 2e-2 to cover this approximation budget.
    let diff = max_abs_diff(&ctx_scalar, &ctx_simd);
    assert!(
        diff <= 2e-2,
        "SIMD vs scalar max_abs_diff={diff} exceeds 2e-2"
    );
}

/// Run SIMD vs scalar on a length that is not a multiple of the SIMD lane
/// width (4 for NEON, 8 for AVX2) to exercise the scalar tail path.
#[test]
fn test_sigmoid_gate_simd_tail_matches_scalar() {
    for n in [1usize, 3, 7, 9, 13, 17, 31, 33] {
        let gate = det_data(n, n as u64);
        let ctx_base = det_data(n, n as u64 + 1000);

        let mut ctx_scalar = ctx_base.clone();
        apply_sigmoid_gate_scalar(&mut ctx_scalar, &gate);

        let mut ctx_simd = ctx_base.clone();
        apply_sigmoid_gate_simd(&mut ctx_simd, &gate);

        let diff = max_abs_diff(&ctx_scalar, &ctx_simd);
        assert!(
            diff <= 2e-2,
            "n={n}: SIMD vs scalar max_abs_diff={diff} exceeds 2e-2"
        );
    }
}

/// Explicit NEON path test (aarch64 only).
#[cfg(target_arch = "aarch64")]
#[test]
fn test_sigmoid_gate_neon_matches_scalar() {
    let n = 64usize;
    let gate = det_data(n, 9999);
    let ctx_base = det_data(n, 8888);

    let mut ctx_scalar = ctx_base.clone();
    apply_sigmoid_gate_scalar(&mut ctx_scalar, &gate);

    let mut ctx_neon = ctx_base.clone();
    // SAFETY: we're on aarch64 where NEON is always present.
    unsafe { apply_sigmoid_gate_neon(&mut ctx_neon, &gate) };

    let diff = max_abs_diff(&ctx_scalar, &ctx_neon);
    assert!(
        diff <= 2e-2,
        "NEON vs scalar max_abs_diff={diff} exceeds 2e-2"
    );
}

/// Explicit AVX2 path test (x86_64 only, requires AVX2).
#[cfg(target_arch = "x86_64")]
#[test]
fn test_sigmoid_gate_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return; // skip on machines without AVX2
    }
    let n = 64usize;
    let gate = det_data(n, 1111);
    let ctx_base = det_data(n, 2222);

    let mut ctx_scalar = ctx_base.clone();
    apply_sigmoid_gate_scalar(&mut ctx_scalar, &gate);

    let mut ctx_avx2 = ctx_base.clone();
    // SAFETY: we just checked AVX2 is available.
    unsafe { apply_sigmoid_gate_avx2(&mut ctx_avx2, &gate) };

    let diff = max_abs_diff(&ctx_scalar, &ctx_avx2);
    assert!(
        diff <= 2e-2,
        "AVX2 vs scalar max_abs_diff={diff} exceeds 2e-2"
    );
}
