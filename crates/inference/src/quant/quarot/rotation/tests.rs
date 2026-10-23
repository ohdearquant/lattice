use super::*;

/// Tiny matvec: y[i] = sum_j W[i,j] * x[j].
fn matvec(weight: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), cols);
    let mut y = vec![0.0_f32; rows];
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        y[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    y
}

// NaN-honest max|a-b|: a `.fold(0.0, f32::max)` silently drops a NaN/Inf
// operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
// wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut max = 0.0_f32;
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

fn synthetic_weight(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as u32;
            (bits as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
fn input_side_absorption_preserves_forward_pass() {
    let rows = 16;
    let cols = 64;
    let r = RandomizedHadamard::new(0xCAFE_BABE, cols).unwrap();
    let weight = synthetic_weight(rows, cols, 1);
    let x: Vec<f32> = synthetic_weight(1, cols, 2).into_iter().collect();

    let y_original = matvec(&weight, rows, cols, &x);

    let mut weight_abs = weight.clone();
    absorb_input_rotation(&mut weight_abs, rows, cols, &r).unwrap();
    let mut x_rotated = x.clone();
    r.apply(&mut x_rotated).unwrap();
    let y_after = matvec(&weight_abs, rows, cols, &x_rotated);

    let delta = max_abs_diff(&y_original, &y_after);
    assert!(
        delta < 1e-4,
        "input-side absorption diverged: max delta {delta}"
    );
}

#[test]
fn output_side_absorption_rotates_forward_pass() {
    let rows = 64;
    let cols = 16;
    let r = RandomizedHadamard::new(0xFEED_FACE, rows).unwrap();
    let weight = synthetic_weight(rows, cols, 3);
    let x: Vec<f32> = synthetic_weight(1, cols, 4).into_iter().collect();

    let y_original = matvec(&weight, rows, cols, &x);
    let mut y_rotated_expected = y_original.clone();
    r.apply(&mut y_rotated_expected).unwrap();

    let mut weight_abs = weight.clone();
    absorb_output_rotation(&mut weight_abs, rows, cols, &r).unwrap();
    let y_after = matvec(&weight_abs, rows, cols, &x);

    let delta = max_abs_diff(&y_rotated_expected, &y_after);
    assert!(
        delta < 1e-4,
        "output-side absorption diverged: max delta {delta}"
    );
}

#[test]
fn two_layer_round_trip_preserves_output() {
    // Layer A: y_A = W_A · x   (cols=64, rows=32)
    // Layer B: y_B = W_B · y_A (cols=32, rows=8)
    // After absorbing R on the boundary between A and B:
    //   W_A' = R · W_A (output-side on A; produces R·y_A as the intermediate)
    //   W_B' = W_B · R^T (input-side on B; consumes R·y_A back to original output)
    // The end-to-end y_B must equal the original y_B.
    let cols_a = 64;
    let rows_a = 32; // = cols_b
    let rows_b = 8;

    let r = RandomizedHadamard::new(0xDEAD_F00D, rows_a).unwrap();
    let w_a = synthetic_weight(rows_a, cols_a, 5);
    let w_b = synthetic_weight(rows_b, rows_a, 6);
    let x: Vec<f32> = synthetic_weight(1, cols_a, 7).into_iter().collect();

    let y_a_original = matvec(&w_a, rows_a, cols_a, &x);
    let y_b_original = matvec(&w_b, rows_b, rows_a, &y_a_original);

    let mut w_a_abs = w_a.clone();
    absorb_output_rotation(&mut w_a_abs, rows_a, cols_a, &r).unwrap();
    let mut w_b_abs = w_b.clone();
    absorb_input_rotation(&mut w_b_abs, rows_b, rows_a, &r).unwrap();

    let y_a_rotated = matvec(&w_a_abs, rows_a, cols_a, &x);
    let y_b_after = matvec(&w_b_abs, rows_b, rows_a, &y_a_rotated);

    let delta = max_abs_diff(&y_b_original, &y_b_after);
    assert!(
        delta < 5e-4,
        "two-layer round trip diverged: max delta {delta}"
    );
}

#[test]
fn input_side_dimension_mismatch_rejected() {
    let r = RandomizedHadamard::new(1, 32).unwrap();
    let mut w = vec![0.0_f32; 16 * 64];
    assert!(absorb_input_rotation(&mut w, 16, 64, &r).is_err());
}

#[test]
fn output_side_dimension_mismatch_rejected() {
    let r = RandomizedHadamard::new(1, 32).unwrap();
    let mut w = vec![0.0_f32; 16 * 64];
    assert!(absorb_output_rotation(&mut w, 16, 64, &r).is_err());
}

#[test]
fn input_side_weight_length_mismatch_rejected() {
    let r = RandomizedHadamard::new(1, 64).unwrap();
    let mut w = vec![0.0_f32; 100];
    assert!(absorb_input_rotation(&mut w, 16, 64, &r).is_err());
}

#[test]
fn rows_cols_overflow_rejected() {
    let r = RandomizedHadamard::new(1, 64).unwrap();
    let mut w = vec![0.0_f32; 100];
    let err = absorb_input_rotation(&mut w, usize::MAX, 64, &r).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("overflow"),
        "expected overflow error, got: {msg}"
    );
}

fn synthetic_weight_f64(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    synthetic_weight(rows, cols, seed)
        .into_iter()
        .map(f64::from)
        .collect()
}

fn matvec_f64(weight: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), cols);
    let mut y = vec![0.0_f64; rows];
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        y[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    y
}

// NaN-honest max|a-b|: a `.fold(0.0, f64::max)` silently drops a NaN/Inf
// operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
// wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut max = 0.0_f64;
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

#[test]
fn input_side_f64_absorption_preserves_forward_pass() {
    let rows = 16;
    let cols = 64;
    let r = RandomizedHadamard::new(0xCAFE_BABE, cols).unwrap();
    let weight = synthetic_weight_f64(rows, cols, 1);
    let x: Vec<f64> = synthetic_weight_f64(1, cols, 2);

    let y_original = matvec_f64(&weight, rows, cols, &x);

    let mut weight_abs = weight.clone();
    absorb_input_rotation_f64(&mut weight_abs, rows, cols, &r).unwrap();
    let mut x_rotated = x.clone();
    r.apply_f64(&mut x_rotated).unwrap();
    let y_after = matvec_f64(&weight_abs, rows, cols, &x_rotated);

    let delta = max_abs_diff_f64(&y_original, &y_after);
    assert!(
        delta < 1e-12,
        "f64 input-side absorption diverged: max delta {delta}"
    );
}

#[test]
fn output_side_f64_absorption_rotates_forward_pass() {
    let rows = 64;
    let cols = 16;
    let r = RandomizedHadamard::new(0xFEED_FACE, rows).unwrap();
    let weight = synthetic_weight_f64(rows, cols, 3);
    let x: Vec<f64> = synthetic_weight_f64(1, cols, 4);

    let y_original = matvec_f64(&weight, rows, cols, &x);
    let mut y_rotated_expected = y_original.clone();
    r.apply_f64(&mut y_rotated_expected).unwrap();

    let mut weight_abs = weight.clone();
    absorb_output_rotation_f64(&mut weight_abs, rows, cols, &r).unwrap();
    let y_after = matvec_f64(&weight_abs, rows, cols, &x);

    let delta = max_abs_diff_f64(&y_rotated_expected, &y_after);
    assert!(
        delta < 1e-12,
        "f64 output-side absorption diverged: max delta {delta}"
    );
}

#[test]
fn f64_absorption_more_precise_than_f32() {
    let rows = 32;
    let cols = 128;
    let r = RandomizedHadamard::new(42, cols).unwrap();
    let weight_f32 = synthetic_weight(rows, cols, 9);
    let weight_f64 = weight_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let x_f32: Vec<f32> = synthetic_weight(1, cols, 10);
    let x_f64: Vec<f64> = x_f32.iter().map(|&v| f64::from(v)).collect();

    let y_original_f64 = matvec_f64(&weight_f64, rows, cols, &x_f64);

    let mut w_f32_abs = weight_f32.clone();
    absorb_input_rotation(&mut w_f32_abs, rows, cols, &r).unwrap();
    let mut x_f32_rot = x_f32.clone();
    r.apply(&mut x_f32_rot).unwrap();
    let y_f32 = matvec(&w_f32_abs, rows, cols, &x_f32_rot);

    let mut w_f64_abs = weight_f64.clone();
    absorb_input_rotation_f64(&mut w_f64_abs, rows, cols, &r).unwrap();
    let mut x_f64_rot = x_f64.clone();
    r.apply_f64(&mut x_f64_rot).unwrap();
    let y_f64 = matvec_f64(&w_f64_abs, rows, cols, &x_f64_rot);

    let err_f32 = y_f32
        .iter()
        .zip(y_original_f64.iter())
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0_f64, f64::max);
    let err_f64 = y_f64
        .iter()
        .zip(y_original_f64.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        err_f64 < err_f32,
        "f64 absorption should be no less precise than f32: f32 err {err_f32}, f64 err {err_f64}"
    );
}
