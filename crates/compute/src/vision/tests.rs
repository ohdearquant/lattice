use super::*;

// ADR-080 C1 / #741: `softmax_inplace` previously had NO fail-closed guard
// at all — a NaN or `+inf` logit divided every element by a NaN/Inf `sum`,
// propagating NaN through the whole row instead of zeroing it. RED before
// the fix: these assertions failed with a NaN-populated row.
#[test]
fn softmax_inplace_nan_fails_closed() {
    let mut x = vec![1.0f32, f32::NAN, 2.0, 0.5];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_pos_inf_fails_closed() {
    let mut x = vec![1.0f32, f32::INFINITY, 2.0];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_all_neg_inf_fails_closed() {
    let mut x = vec![f32::NEG_INFINITY; 4];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_normal_row_still_normalizes() {
    let mut x = vec![1.0f32, 2.0, 3.0];
    softmax_inplace(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "row must sum to ~1: {x:?}");
    assert!(x.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
#[should_panic(expected = "a too short for m*k")]
fn vit_batch_matvec_rejects_short_x() {
    let a = [0.0f32; 4]; // rows*cols = 4
    let x = [0.0f32; 3]; // needs n*cols = 4
    batch_matvec(&a, &x, 2, 2, 2);
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn vit_batch_matvec_rejects_short_a() {
    let a = [0.0f32; 3]; // needs rows*cols = 4
    let x = [0.0f32; 4];
    batch_matvec(&a, &x, 2, 2, 2);
}

#[test]
#[should_panic(expected = "vit_batch_matvec: output shape overflow: n*rows")]
fn vit_batch_matvec_rejects_output_overflow() {
    let a = [0.0f32; 4];
    let x = [0.0f32; 4];
    batch_matvec(&a, &x, usize::MAX, 2, 2);
}

#[test]
fn erf_matches_known_values() {
    // Reference values from a standard erf table.
    assert!((erf(0.0) - 0.0).abs() < 1e-6);
    assert!((erf(1.0) - 0.842_700_79).abs() < 1e-6);
    assert!((erf(-1.0) + 0.842_700_79).abs() < 1e-6);
    assert!((erf(2.0) - 0.995_322_3).abs() < 1e-6);
}

#[test]
fn gelu_exact_matches_known_values() {
    // gelu_exact(0) = 0; gelu_exact is odd-ish but not exactly odd.
    assert!((gelu_exact(0.0)).abs() < 1e-6);
    // gelu_exact(1.0) ~= 0.8413447
    assert!((gelu_exact(1.0) - 0.841_344_7).abs() < 1e-4);
    // gelu_exact(-1.0) ~= -0.1586553
    assert!((gelu_exact(-1.0) + 0.158_655_3).abs() < 1e-4);
}

fn batch_matvec_reference(a: &[f32], x: &[f32], n: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * rows];
    for i in 0..n {
        for r in 0..rows {
            let mut acc = 0.0f64;
            for c in 0..cols {
                acc += f64::from(a[r * cols + c]) * f64::from(x[i * cols + c]);
            }
            out[i * rows + r] = acc as f32;
        }
    }
    out
}

fn filled(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

#[test]
fn vit_batch_matvec_matches_scalar_reference() {
    // An odd shape exercises the kernel tails; the larger one is past the row-split
    // threshold of the x86 dispatch, so the threaded path is compared too.
    for &(n, rows, cols) in &[(37, 29, 67), (200, 960, 640)] {
        let a = filled(rows * cols, 7);
        let x = filled(n * cols, 11);
        let got = batch_matvec(&a, &x, n, rows, cols);
        let want = batch_matvec_reference(&a, &x, n, rows, cols);
        assert_eq!(got.len(), n * rows);
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-4 * (1.0 + w.abs());
            assert!(
                (g - w).abs() <= tol,
                "n={n} rows={rows} cols={cols} index {idx}: got {g}, want {w}"
            );
        }
    }
}
