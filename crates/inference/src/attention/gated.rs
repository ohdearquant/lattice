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
/// Panics if `num_heads * head_dim` or `2 * q_dim` overflows `usize`, or if
/// `q_and_gate.len() != 2 * num_heads * head_dim`, `q_buf.len() != num_heads * head_dim`,
/// or `gate_buf.len() != num_heads * head_dim`.
#[inline]
pub fn deinterleave_q_gate(
    q_and_gate: &[f32],
    q_buf: &mut [f32],
    gate_buf: &mut [f32],
    num_heads: usize,
    head_dim: usize,
) {
    assert!(
        num_heads.checked_mul(head_dim).is_some(),
        "deinterleave_q_gate shape overflow: num_heads * head_dim"
    );
    let q_dim = num_heads * head_dim;
    assert!(
        q_dim.checked_mul(2).is_some(),
        "deinterleave_q_gate shape overflow: 2 * q_dim"
    );
    assert_eq!(
        q_and_gate.len(),
        2 * q_dim,
        "deinterleave_q_gate: q_and_gate.len()={} != 2 * num_heads * head_dim={}",
        q_and_gate.len(),
        2 * q_dim
    );
    assert_eq!(
        q_buf.len(),
        q_dim,
        "deinterleave_q_gate: q_buf.len()={} != num_heads * head_dim={}",
        q_buf.len(),
        q_dim
    );
    assert_eq!(
        gate_buf.len(),
        q_dim,
        "deinterleave_q_gate: gate_buf.len()={} != num_heads * head_dim={}",
        gate_buf.len(),
        q_dim
    );

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

/// **Unstable**: apply elementwise sigmoid gate to `context` using exact `exp`.
///
/// Computes `context[i] *= 1 / (1 + exp(-gate[i]))` for every element.
/// This is the G1 SDPA-output gating from the Gated Attention paper.
///
/// Uses exact `f32::exp` to match all other Qwen3.5 execution paths (prefill,
/// f16, q8, Metal, speculative). See [`apply_sigmoid_gate_fast`] for the
/// approximate SIMD variant.
///
/// # Panics
///
/// Panics if `context.len() != gate.len()`.
#[inline]
pub fn apply_sigmoid_gate(context: &mut [f32], gate: &[f32]) {
    apply_sigmoid_gate_scalar(context, gate);
}

/// **Unstable**: apply elementwise sigmoid gate with approximate SIMD fast-exp.
///
/// Uses the Schraudolph bit-trick for `exp` (~5–6% relative error), dispatching
/// to NEON or AVX2 when available. Achieves 6–8× speedup over the scalar path
/// but produces different numerical output.
///
/// Use this only when all comparable execution paths (prefill, decode, etc.)
/// agree on the same approximation. For refactoring where exact parity is
/// required, use [`apply_sigmoid_gate`] instead.
///
/// # Panics
///
/// Panics if `context.len() != gate.len()`.
#[inline]
pub fn apply_sigmoid_gate_fast(context: &mut [f32], gate: &[f32]) {
    assert_eq!(
        context.len(),
        gate.len(),
        "apply_sigmoid_gate_fast: context and gate must have equal length"
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

/// **Unstable**: NEON sigmoid gate; dispatch via [`apply_sigmoid_gate_fast`].
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
///   `assert_eq!` in [`apply_sigmoid_gate_fast`] before dispatch.
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

/// **Unstable**: AVX2 sigmoid gate; dispatch via [`apply_sigmoid_gate_fast`].
///
/// Uses the same Schraudolph fast-exp bit trick as the NEON path.
///
/// # Safety
///
/// - Caller must ensure AVX2 is available (guaranteed by `simd_config()`).
/// - `context.len()` must equal `gate.len()`; the SIMD loop accesses both
///   slices at the same offsets without bounds checks. Upheld by the
///   `assert_eq!` in [`apply_sigmoid_gate_fast`] before dispatch.
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

/// **Unstable**: alias for [`apply_sigmoid_gate_fast`].
///
/// Exposed so benchmarks can compare scalar vs SIMD paths by name.
#[inline]
pub fn apply_sigmoid_gate_simd(context: &mut [f32], gate: &[f32]) {
    apply_sigmoid_gate_fast(context, gate);
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests;
