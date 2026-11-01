use super::*;

#[test]
#[should_panic(expected = "a too short for m*k")]
fn test_matmul_bt_f16_rejects_short_a() {
    let m = 1usize;
    let k = 4usize;
    let n = 2usize;
    let a = vec![0.0f32; k - 1]; // too short
    let b = vec![0u16; n * k];
    let mut c = vec![0.0f32; m * n];
    matmul_bt_f16(&a, &b, &mut c, m, k, n);
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn test_matmul_bt_f16_rejects_short_b() {
    let m = 1usize;
    let k = 4usize;
    let n = 2usize;
    let a = vec![0.0f32; k];
    let b = vec![0u16; n * k - 1]; // too short
    let mut c = vec![0.0f32; m * n];
    matmul_bt_f16(&a, &b, &mut c, m, k, n);
}

#[test]
fn test_matmul_bt_f16_accepts_oversized_a() {
    // ADR-080 C4: matmul_bt_f16 previously used `assert_eq!` (exact length) instead of
    // `>=` (oversized-scratch-prefix allowed), unlike matmul_bt/matmul_bt_q8. A caller
    // passing an oversized `a` must not panic.
    let m = 1usize;
    let k = 4usize;
    let n = 2usize;
    let a = vec![1.0f32; k + 3]; // oversized
    let b_f32 = vec![1.0f32; n * k];
    let b: Vec<u16> = b_f32.iter().map(|&v| F16::from_f32(v).0).collect();
    let mut c = vec![0.0f32; m * n];
    matmul_bt_f16(&a, &b, &mut c, m, k, n);
    for &v in &c {
        assert!((v - k as f32).abs() < 1e-3);
    }
}

#[test]
#[should_panic(expected = "shape overflow: n*k")]
fn test_matmul_bt_f16_rejects_overflow() {
    let a = [0.0f32; 2];
    let b = [0u16; 2];
    let mut c = [0.0f32; 2];
    matmul_bt_f16(&a, &b, &mut c, 2, 2, usize::MAX);
}

#[test]
fn test_matmul_bt_f16_vs_f32() {
    let m = 1usize;
    let k = 2048usize;
    let n = 6144usize;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 13) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let b_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i * 11 + 17) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let b_f16: Vec<u16> = b_f32.iter().map(|&v| F16::from_f32(v).0).collect();

    let mut c_f32 = vec![0.0f32; m * n];
    let mut c_f16 = vec![0.0f32; m * n];

    matmul_bt_f32_reference(&a, &b_f32, &mut c_f32, m, k, n);
    matmul_bt_f16(&a, &b_f16, &mut c_f16, m, k, n);

    let max_diff = c_f32
        .iter()
        .zip(c_f16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 0.1,
        "max absolute difference too large: {max_diff}"
    );
}

#[test]
fn test_matmul_bt_f16_matches_scalar_reference_exactly_for_same_half_weights() {
    let m = 2usize;
    let k = 65usize;
    let n = 17usize;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 5 + 3) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let b_half: Vec<u16> = (0..n * k)
        .map(|i| F16::from_f32(((i * 9 + 1) % 113) as f32 / 113.0 - 0.5).0)
        .collect();

    let mut c_simd = vec![0.0f32; m * n];
    let mut c_scalar = vec![0.0f32; m * n];

    matmul_bt_f16(&a, &b_half, &mut c_simd, m, k, n);

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let c_row = &mut c_scalar[row * n..(row + 1) * n];
        for out in 0..n {
            let b_row = &b_half[out * k..(out + 1) * k];
            c_row[out] = dot_f16_f32_scalar(a_row, b_row);
        }
    }

    // Allow small FP rounding differences: BLAS (AMX/FMA) accumulation order
    // differs from the scalar reference, causing ~1e-7 relative drift.
    let max_diff = c_simd
        .iter()
        .zip(c_scalar.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "max diff {max_diff} exceeds tolerance 1e-5"
    );
}

#[test]
fn test_matmul_bt_f16_near_zero_weights() {
    let m = 1usize;
    let k = 128usize;
    let n = 64usize;

    let a = vec![1.0f32; m * k];
    let b_f32: Vec<f32> = (0..n * k).map(|_| 1e-6f32).collect();
    let b_f16: Vec<u16> = b_f32.iter().map(|&v| F16::from_f32(v).0).collect();

    let mut c = vec![0.0f32; m * n];
    matmul_bt_f16(&a, &b_f16, &mut c, m, k, n);

    for &v in &c {
        assert!(
            (v - 0.000128).abs() < 0.001,
            "near-zero weight output wrong: {v}"
        );
    }
}
