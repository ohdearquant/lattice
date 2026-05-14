//! Self-contained integration test for gated attention.
//!
//! Generates reference data deterministically using xorshift64 PRNG and
//! exact `f32::exp`. Tests both the exact (`apply_sigmoid_gate`) and
//! approximate (`apply_sigmoid_gate_fast`) paths.
//!
//! Run:
//!   cargo test --test gated_attention_test

use lattice_inference::attention::gated::{
    apply_sigmoid_gate, apply_sigmoid_gate_fast, apply_sigmoid_gate_scalar, deinterleave_q_gate,
};

// ===================================================================
// Constants (match Python reference script parameters)
// ===================================================================

/// Number of Q heads used in the reference fixture.
const NUM_HEADS: usize = 4;
/// Per-head dimension used in the reference fixture.
const HEAD_DIM: usize = 8;
/// Total Q dimension = NUM_HEADS * HEAD_DIM.
const Q_DIM: usize = NUM_HEADS * HEAD_DIM;

// Seeds used by the Python reference script (xorshift64).
const SEED_Q_AND_GATE: u64 = 42;
const SEED_CONTEXT_IN: u64 = 99;

// ===================================================================
// Deterministic pseudo-random data (xorshift64 + float extraction)
//
// Matches the generator used in the unit tests in gated.rs and in the
// Python reference script that originally produced the .bin fixtures.
// ===================================================================

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

// ===================================================================
// Reference computation (exact exp — no fast-exp approximation)
// ===================================================================

/// Compute the expected deinterleave result directly.
///
/// The reference layout is `[Q_h0 | G_h0 | Q_h1 | G_h1 | ...]`.
/// Returns `(q_buf, gate_buf)`.
fn ref_deinterleave(q_and_gate: &[f32], num_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let q_dim = num_heads * head_dim;
    let mut q_buf = vec![0.0f32; q_dim];
    let mut gate_buf = vec![0.0f32; q_dim];
    for h in 0..num_heads {
        let src = h * head_dim * 2;
        let dst = h * head_dim;
        q_buf[dst..dst + head_dim].copy_from_slice(&q_and_gate[src..src + head_dim]);
        gate_buf[dst..dst + head_dim]
            .copy_from_slice(&q_and_gate[src + head_dim..src + head_dim * 2]);
    }
    (q_buf, gate_buf)
}

/// Compute `context[i] *= sigmoid(gate[i])` using exact `f32::exp`.
fn ref_apply_sigmoid_gate(context: &[f32], gate: &[f32]) -> Vec<f32> {
    context
        .iter()
        .zip(gate.iter())
        .map(|(&c, &g)| {
            let sig = 1.0 / (1.0 + (-g).exp());
            c * sig
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ===================================================================
// Tests
// ===================================================================

/// Verify `deinterleave_q_gate` matches the reference within 1e-6.
///
/// Both implementations use exact f32 copy — no approximation — so the result
/// should be bit-exact. Tolerance is 1e-6 to allow for any future compiler
/// differences.
#[test]
fn test_deinterleave_matches_reference() {
    let q_and_gate = det_data(2 * Q_DIM, SEED_Q_AND_GATE);
    let (ref_q_buf, ref_gate_buf) = ref_deinterleave(&q_and_gate, NUM_HEADS, HEAD_DIM);

    let mut rust_q_buf = vec![0.0f32; Q_DIM];
    let mut rust_gate_buf = vec![0.0f32; Q_DIM];
    deinterleave_q_gate(
        &q_and_gate,
        &mut rust_q_buf,
        &mut rust_gate_buf,
        NUM_HEADS,
        HEAD_DIM,
    );

    let diff_q = max_abs_diff(&rust_q_buf, &ref_q_buf);
    let diff_gate = max_abs_diff(&rust_gate_buf, &ref_gate_buf);

    assert!(
        diff_q <= 1e-6,
        "q_buf: max_abs_diff={diff_q} vs reference exceeds 1e-6"
    );
    assert!(
        diff_gate <= 1e-6,
        "gate_buf: max_abs_diff={diff_gate} vs reference exceeds 1e-6"
    );
}

/// Verify `apply_sigmoid_gate_scalar` (exact exp) matches the reference
/// within 1e-6.
///
/// The scalar implementation and the reference both use `f32::exp` — the only
/// difference can be floating-point evaluation order, which is within 1e-6.
#[test]
fn test_sigmoid_gate_scalar_matches_reference() {
    let q_and_gate = det_data(2 * Q_DIM, SEED_Q_AND_GATE);
    let (_ref_q_buf, ref_gate_buf) = ref_deinterleave(&q_and_gate, NUM_HEADS, HEAD_DIM);
    let context_in = det_data(Q_DIM, SEED_CONTEXT_IN);
    let ref_context_out = ref_apply_sigmoid_gate(&context_in, &ref_gate_buf);

    let mut rust_context = context_in.clone();
    apply_sigmoid_gate_scalar(&mut rust_context, &ref_gate_buf);

    let diff = max_abs_diff(&rust_context, &ref_context_out);
    assert!(
        diff <= 1e-6,
        "apply_sigmoid_gate_scalar: max_abs_diff={diff} exceeds 1e-6"
    );
}

/// Verify `apply_sigmoid_gate` (exact, production default) matches the
/// reference within 1e-6 — same tolerance as the scalar test.
#[test]
fn test_sigmoid_gate_exact_matches_reference() {
    let q_and_gate = det_data(2 * Q_DIM, SEED_Q_AND_GATE);
    let (_ref_q_buf, ref_gate_buf) = ref_deinterleave(&q_and_gate, NUM_HEADS, HEAD_DIM);
    let context_in = det_data(Q_DIM, SEED_CONTEXT_IN);
    let ref_context_out = ref_apply_sigmoid_gate(&context_in, &ref_gate_buf);

    let mut rust_context = context_in.clone();
    apply_sigmoid_gate(&mut rust_context, &ref_gate_buf);

    let diff = max_abs_diff(&rust_context, &ref_context_out);
    assert!(
        diff <= 1e-6,
        "apply_sigmoid_gate (exact): max_abs_diff={diff} exceeds 1e-6"
    );
}

/// Verify `apply_sigmoid_gate_fast` (approximate SIMD) matches the exact
/// reference within 2e-2.
///
/// The fast path uses the Schraudolph fast-exp bit trick (~5-6% relative
/// error on exp), which translates to ~1% absolute error on sigmoid.
#[test]
fn test_sigmoid_gate_fast_matches_reference() {
    let q_and_gate = det_data(2 * Q_DIM, SEED_Q_AND_GATE);
    let (_ref_q_buf, ref_gate_buf) = ref_deinterleave(&q_and_gate, NUM_HEADS, HEAD_DIM);
    let context_in = det_data(Q_DIM, SEED_CONTEXT_IN);
    let ref_context_out = ref_apply_sigmoid_gate(&context_in, &ref_gate_buf);

    let mut rust_context = context_in.clone();
    apply_sigmoid_gate_fast(&mut rust_context, &ref_gate_buf);

    let diff = max_abs_diff(&rust_context, &ref_context_out);
    assert!(
        diff <= 2e-2,
        "apply_sigmoid_gate_fast: max_abs_diff={diff} exceeds 2e-2"
    );
}

/// Verify the fast sigmoid path stays within 2% of exact over the realistic
/// gate range [-5, 5] where the Schraudolph approximation has the most impact.
#[test]
fn test_sigmoid_gate_fast_realistic_range() {
    let n = 32usize;
    let gate: Vec<f32> = (0..n)
        .map(|i| -5.0 + (i as f32) * (10.0 / (n - 1) as f32))
        .collect();
    let context_in = vec![1.0f32; n];
    let ref_out = ref_apply_sigmoid_gate(&context_in, &gate);

    let mut fast = context_in.clone();
    apply_sigmoid_gate_fast(&mut fast, &gate);

    let diff = max_abs_diff(&fast, &ref_out);
    assert!(
        diff <= 2e-2,
        "realistic gate range [-5, 5]: max_abs_diff={diff} exceeds 2e-2"
    );
}

/// End-to-end test: deinterleave then apply exact gate.
///
/// Chains both operations as the production decode path does.
#[test]
fn test_full_gated_pipeline_matches_reference() {
    let q_and_gate = det_data(2 * Q_DIM, SEED_Q_AND_GATE);
    let context_in = det_data(Q_DIM, SEED_CONTEXT_IN);

    let (_ref_q_buf, ref_gate_buf) = ref_deinterleave(&q_and_gate, NUM_HEADS, HEAD_DIM);
    let ref_context_out = ref_apply_sigmoid_gate(&context_in, &ref_gate_buf);

    let mut q_buf = vec![0.0f32; Q_DIM];
    let mut gate_buf = vec![0.0f32; Q_DIM];
    deinterleave_q_gate(&q_and_gate, &mut q_buf, &mut gate_buf, NUM_HEADS, HEAD_DIM);

    let mut context = context_in.clone();
    apply_sigmoid_gate(&mut context, &gate_buf);

    let diff = max_abs_diff(&context, &ref_context_out);
    assert!(
        diff <= 1e-6,
        "full pipeline: max_abs_diff={diff} exceeds 1e-6"
    );
}
