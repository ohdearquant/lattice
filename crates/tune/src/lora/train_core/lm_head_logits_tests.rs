use super::*;

// Pre-#737 form: one scalar dot product per vocab row. `matmul_bt`
// reassociates the reduction (Accelerate/AMX on macOS, tiled/SIMD
// elsewhere), so parity is checked to a tight relative tolerance
// (matches PARITY_TOL in `lattice_inference::backward::ops`), not
// bit-exact — see that module's tests for the same rationale.
fn lm_head_logits_scalar(
    lm_head: &[f32],
    final_normed: &[f32],
    hidden: usize,
    vocab: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab];
    for (v, out) in logits.iter_mut().enumerate() {
        let row = &lm_head[v * hidden..(v + 1) * hidden];
        let mut acc = 0.0f32;
        for j in 0..hidden {
            acc += row[j] * final_normed[j];
        }
        *out = acc;
    }
    logits
}

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let diff_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum();
    let norm_sq: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    (diff_sq / norm_sq.max(1e-30)).sqrt()
}

#[test]
fn lm_head_logits_parity() {
    // Odd, unequal dims on purpose: a wrong transpose or swapped
    // dim-order (e.g. treating lm_head as [hidden, vocab] instead of
    // the actual [vocab, hidden]) would either panic on an out-of-bounds
    // slice or produce values that disagree with the scalar oracle —
    // it cannot silently pass by square-shape symmetry.
    const HIDDEN: usize = 7;
    const VOCAB: usize = 5;
    const PARITY_TOL: f64 = 1e-4;

    let lm_head: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| (i as f32 + 1.0) * 0.037 - 0.6)
        .collect();
    let final_normed: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 + 1.0) * 0.11 - 0.3).collect();

    let vectorized = lm_head_logits(&lm_head, &final_normed, HIDDEN, VOCAB);
    let scalar = lm_head_logits_scalar(&lm_head, &final_normed, HIDDEN, VOCAB);

    assert_eq!(vectorized.len(), VOCAB);
    let err = rel_err(&vectorized, &scalar);
    eprintln!("lm_head_logits parity rel_err={err:.2e}");
    assert!(
        err < PARITY_TOL,
        "lm_head_logits vectorized vs scalar rel_err {err:.2e} >= {PARITY_TOL:.2e}"
    );
}
