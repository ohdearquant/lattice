//! Gated attention primitives for Qwen3.5-style Q+gate projection.
//!
//! Qwen3.5 fuses the Q projection and a per-element sigmoid gate into a single
//! weight matrix of shape `[2*q_dim, hidden]`. After the matmul the packed
//! `[Q|gate]` output must be deinterleaved per head before Q can be used for
//! attention. After the attention context is computed, the gate is applied
//! element-wise via `context[i] *= sigmoid(gate[i])`.
//!
//! This module extracts both operations from the inline model code into
//! reusable, tested functions with SIMD fast paths.

use crate::forward::cpu::simd_config;

// ===================================================================
// Deinterleave
// ===================================================================

/// **Unstable**: deinterleave packed `[Q|gate]` per-head layout into separate buffers.
///
/// The Q projection weight has shape `[2*q_dim, hidden]`. After the matmul the
/// result is laid out as `[q_h0|g_h0|q_h1|g_h1|...]` — each head's Q slice
/// immediately followed by that head's gate slice. This function splits them
/// into two flat buffers of length `q_dim = num_heads * head_dim`.
///
/// # Panics
///
/// Panics (debug-only) if `q_and_gate.len() != 2 * num_heads * head_dim`,
/// `q_buf.len() != num_heads * head_dim`, or `gate_buf.len() != num_heads * head_dim`.
#[inline]
pub fn deinterleave_q_gate(
    q_and_gate: &[f32],
    q_buf: &mut [f32],
    gate_buf: &mut [f32],
    num_heads: usize,
    head_dim: usize,
) {
    let q_dim = num_heads * head_dim;
    debug_assert_eq!(q_and_gate.len(), 2 * q_dim);
    debug_assert_eq!(q_buf.len(), q_dim);
    debug_assert_eq!(gate_buf.len(), q_dim);

    for h in 0..num_heads {
        let src = h * head_dim * 2;
        let dst = h * head_dim;
        q_buf[dst..dst + head_dim].copy_from_slice(&q_and_gate[src..src + head_dim]);
        gate_buf[dst..dst + head_dim]
            .copy_from_slice(&q_and_gate[src + head_dim..src + head_dim * 2]);
    }
}

// ===================================================================
// Sigmoid gate — scalar
// ===================================================================

/// **Unstable**: apply elementwise sigmoid gate to `context`.
///
/// Computes `context[i] *= 1 / (1 + exp(-gate[i]))` for every element.
/// This is the G1 SDPA-output gating from the Gated Attention paper.
///
/// Uses the SIMD path when the target architecture supports it; falls back
/// to the scalar path otherwise.
///
/// # Panics
///
/// Panics if `context.len() != gate.len()`.
#[inline]
pub fn apply_sigmoid_gate(context: &mut [f32], gate: &[f32]) {
    assert_eq!(
        context.len(),
        gate.len(),
        "apply_sigmoid_gate: context and gate must have equal length"
    );
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is always present on aarch64; runtime gate confirms it.
            unsafe {
                apply_sigmoid_gate_neon(context, gate);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled {
            // SAFETY: The runtime feature check guarantees AVX2 support.
            unsafe {
                apply_sigmoid_gate_avx2(context, gate);
                return;
            }
        }
    }

    // Suppress unused-variable warning on architectures where no SIMD is used.
    let _ = config;

    apply_sigmoid_gate_scalar(context, gate);
}

/// **Unstable**: scalar elementwise sigmoid gate; used as fallback and in tests.
#[inline]
pub fn apply_sigmoid_gate_scalar(context: &mut [f32], gate: &[f32]) {
    assert_eq!(
        context.len(),
        gate.len(),
        "apply_sigmoid_gate_scalar: context.len()={} != gate.len()={}",
        context.len(),
        gate.len(),
    );
    for (c, &g) in context.iter_mut().zip(gate.iter()) {
        let sig = 1.0 / (1.0 + (-g).exp());
        *c *= sig;
    }
}

// ===================================================================
// Sigmoid gate — NEON (aarch64)
// ===================================================================

/// **Unstable**: NEON sigmoid gate; dispatch via [`apply_sigmoid_gate`].
///
/// Uses the Schraudolph fast-exp bit trick (same as `softmax.rs`) for the
/// sigmoid denominator. Accuracy: ~5-6% relative error on the sigmoid, which
/// is sufficient for gating.
///
/// # Safety
///
/// - Caller must ensure NEON is available (guaranteed by `simd_config()`).
/// - `context.len()` must equal `gate.len()`; the SIMD loop accesses both
///   slices at the same offsets without bounds checks. Upheld by the
///   `assert_eq!` in [`apply_sigmoid_gate`] before dispatch.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_sigmoid_gate_neon(context: &mut [f32], gate: &[f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(context.len(), gate.len());
    let n = context.len();
    let chunks = n / 4;

    let ctx_ptr = context.as_mut_ptr();
    let gate_ptr = gate.as_ptr();

    let ones = vdupq_n_f32(1.0);

    for c in 0..chunks {
        let off = c * 4;
        // SAFETY: off + 3 < chunks * 4 <= n, within slice bounds for both pointers.
        let g = vld1q_f32(gate_ptr.add(off));
        let ctx = vld1q_f32(ctx_ptr.add(off) as *const f32);

        // sigmoid(g) = 1 / (1 + exp(-g))
        let neg_g = vnegq_f32(g);
        let exp_neg_g = fast_exp_neon(neg_g);
        let denom = vaddq_f32(ones, exp_neg_g);
        // reciprocal estimate (good enough — same approach as softmax inv_sum)
        let sig = vdivq_f32(ones, denom);
        let out = vmulq_f32(ctx, sig);
        vst1q_f32(ctx_ptr.add(off), out);
    }

    // Scalar tail
    for i in (chunks * 4)..n {
        // SAFETY: i < n, within slice bounds.
        let g = *gate_ptr.add(i);
        let sig = 1.0 / (1.0 + (-g).exp());
        *ctx_ptr.add(i) *= sig;
    }
}

/// NEON fast-exp approximation (Schraudolph bit trick), 4 lanes.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fast_exp_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
    let x = vminq_f32(x, vdupq_n_f32(88.0));
    let t = vfmaq_f32(vdupq_n_f32(1_065_353_216.0), x, vdupq_n_f32(12_102_203.0));
    vreinterpretq_f32_s32(vcvtq_s32_f32(t))
}

// ===================================================================
// Sigmoid gate — AVX2 (x86_64)
// ===================================================================

/// **Unstable**: AVX2 sigmoid gate; dispatch via [`apply_sigmoid_gate`].
///
/// Uses the same Schraudolph fast-exp bit trick as the NEON path.
///
/// # Safety
///
/// - Caller must ensure AVX2 is available (guaranteed by `simd_config()`).
/// - `context.len()` must equal `gate.len()`; the SIMD loop accesses both
///   slices at the same offsets without bounds checks. Upheld by the
///   `assert_eq!` in [`apply_sigmoid_gate`] before dispatch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn apply_sigmoid_gate_avx2(context: &mut [f32], gate: &[f32]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(context.len(), gate.len());
    let n = context.len();
    let chunks = n / 8;

    let ctx_ptr = context.as_mut_ptr();
    let gate_ptr = gate.as_ptr();

    let ones = _mm256_set1_ps(1.0);

    for c in 0..chunks {
        let off = c * 8;
        // SAFETY: off + 7 < chunks * 8 <= n, within slice bounds for both pointers.
        let g = _mm256_loadu_ps(gate_ptr.add(off));
        let ctx = _mm256_loadu_ps(ctx_ptr.add(off) as *const f32);

        // sigmoid(g) = 1 / (1 + exp(-g))
        let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        let exp_neg_g = fast_exp_avx2(neg_g);
        let denom = _mm256_add_ps(ones, exp_neg_g);
        // Use reciprocal approximation + Newton-Raphson for accuracy
        let rcp = _mm256_rcp_ps(denom);
        // Newton-Raphson: rcp2 = rcp * (2 - denom * rcp)
        let nr = _mm256_mul_ps(denom, rcp);
        let two_minus_nr = _mm256_sub_ps(_mm256_set1_ps(2.0), nr);
        let sig = _mm256_mul_ps(rcp, two_minus_nr);
        let out = _mm256_mul_ps(ctx, sig);
        _mm256_storeu_ps(ctx_ptr.add(off), out);
    }

    // Scalar tail
    for i in (chunks * 8)..n {
        // SAFETY: i < n, within slice bounds.
        let g = *gate_ptr.add(i);
        let sig = 1.0 / (1.0 + (-g).exp());
        *ctx_ptr.add(i) *= sig;
    }
}

/// AVX2 fast-exp approximation (Schraudolph bit trick), 8 lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));
    let scale = _mm256_set1_ps(12_102_203.0);
    let bias = _mm256_set1_ps(1_065_353_216.0);
    let t = _mm256_add_ps(_mm256_mul_ps(x, scale), bias);
    _mm256_castsi256_ps(_mm256_cvtps_epi32(t))
}

// ===================================================================
// SIMD dispatch alias (public, for benchmarks)
// ===================================================================

/// **Unstable**: apply elementwise sigmoid gate with SIMD acceleration.
///
/// This is an alias for [`apply_sigmoid_gate`]; the dispatch to NEON/AVX2/scalar
/// is handled inside. Exposed separately so benchmarks can compare paths explicitly.
#[inline]
pub fn apply_sigmoid_gate_simd(context: &mut [f32], gate: &[f32]) {
    apply_sigmoid_gate(context, gate);
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
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

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
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
}
