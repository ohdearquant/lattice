use super::*;
use lattice_compute::forward::cpu::matmul_bt;
use lattice_weights::q8_weights::quantize_matrix;
use lattice_weights::q8_weights::test_support::{approx_eq, uniform_signed};

#[test]
fn test_matmul_bt_q8_matches_f32() {
    let m = 2;
    let k = 256;
    let n = 16;

    let mut seed = 0xCAFE_BABEu32;
    let mut a = Vec::with_capacity(m * k);
    let mut b = Vec::with_capacity(n * k);

    for _ in 0..m * k {
        a.push(uniform_signed(&mut seed) * 0.1);
    }

    for row in 0..n {
        let scale = (row as f32 + 1.0) / 8_192.0;
        b.push(127.0 * scale);
        for col in 1..k {
            let qv = ((row * 17 + col * 13) % 255) as i32 - 127;
            b.push(qv as f32 * scale);
        }
    }

    let b_q = quantize_matrix(&b, n, k).unwrap();

    let mut c_f32 = vec![0.0f32; m * n];
    let mut c_q8 = vec![0.0f32; m * n];
    matmul_bt(&a, &b, &mut c_f32, m, k, n);
    matmul_bt_q8(&a, &b_q, &mut c_q8, m, k, n);

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut large_outputs = 0usize;

    for (&ref_v, &q_v) in c_f32.iter().zip(c_q8.iter()) {
        let abs_err = (ref_v - q_v).abs();
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if ref_v.abs() > 0.01 {
            large_outputs += 1;
            let rel_err = abs_err / ref_v.abs();
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }
    }

    assert!(large_outputs > 0, "expected some non-trivial outputs");
    assert!(max_abs_err < 1.0e-4, "max_abs_err={max_abs_err}");
    assert!(max_rel_err < 0.01, "max_rel_err={max_rel_err}");
}

#[test]
fn test_matmul_bt_q8_known_values() {
    let a = vec![1.0, 2.0, -1.0]; // [1, 3]
    let b = vec![
        1.27, -0.64, 0.0, // row 0, scale = 0.01, q = [127, -64, 0]
        -2.54, 0.0, 1.28, // row 1, scale = 0.02, q = [-127, 0, 64]
    ];

    let q = quantize_matrix(&b, 2, 3).unwrap();
    assert!(approx_eq(q.scales[0], 0.01, 1e-8));
    assert!(approx_eq(q.scales[1], 0.02, 1e-8));
    assert_eq!(&q.data[0..3], &[127i8, -64i8, 0i8]);
    assert_eq!(&q.data[3..6], &[-127i8, 0i8, 64i8]);

    let mut c = vec![0.0f32; 2];
    matmul_bt_q8(&a, &q, &mut c, 1, 3, 2);

    let expected0 = 0.01 * (1.0 * 127.0 + 2.0 * -64.0 + -0.0);
    let expected1 = 0.02 * (1.0 * -127.0 + 2.0 * 0.0 + -64.0);

    assert!(approx_eq(c[0], expected0, 1e-6));
    assert!(approx_eq(c[1], expected1, 1e-6));
    assert!(approx_eq(c[0], -0.01, 1e-6));
    assert!(approx_eq(c[1], -3.82, 1e-6));
}

/// Tiny valid Q8Matrix for the overflow-guard tests: the guards fire before
/// `b_q` is inspected, so shape need only be self-consistent.
fn dummy_q8_1x1() -> Q8Matrix {
    Q8Matrix {
        data: vec![0i8],
        scales: vec![0.0f32],
        rows: 1,
        cols: 1,
    }
}

/// Overflow guard: `m * k` must be rejected with a dedicated overflow message
/// before the length `assert_eq!` can wrap it and panic with a misleading
/// "length does not match" message instead.
#[test]
#[should_panic(expected = "matmul shape overflow: m*k")]
fn test_matmul_bt_q8_rejects_mk_overflow() {
    let q = dummy_q8_1x1();
    let mut c = vec![0.0f32; 1];
    // m*k overflows; n*k and m*n do not.
    matmul_bt_q8(&[], &q, &mut c, usize::MAX, 2, 1);
}

/// Overflow guard: `n * k` (m*k passes first).
#[test]
#[should_panic(expected = "matmul shape overflow: n*k")]
fn test_matmul_bt_q8_rejects_nk_overflow() {
    let q = dummy_q8_1x1();
    let mut c = vec![0.0f32; 1];
    // m*k = 1*2 ok; n*k = MAX*2 overflows.
    matmul_bt_q8(&[0.0, 0.0], &q, &mut c, 1, 2, usize::MAX);
}

/// Overflow guard: `m * n` (m*k and n*k pass first).
#[test]
#[should_panic(expected = "matmul shape overflow: m*n")]
fn test_matmul_bt_q8_rejects_mn_overflow() {
    let q = dummy_q8_1x1();
    let mut c = vec![0.0f32; 1];
    // m*k = MAX*1 ok; n*k = 2*1 ok; m*n = MAX*2 overflows.
    matmul_bt_q8(&[], &q, &mut c, usize::MAX, 1, 2);
}

/// Overflow guard for the macOS tile-scratch size `TILE_N * k`. With
/// `m=0, k=MAX, n=0` every earlier product (`m*k`, `n*k`, `m*n`) and every
/// length assert is zero/empty, so control reaches the tile block; `TILE_N*k`
/// then overflows and `Q8MatmulScratch::prepare` must reject it with the
/// dedicated message rather than silently wrapping the scratch length.
#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "matmul shape overflow: TILE_N*k")]
fn test_matmul_bt_q8_rejects_tile_nk_overflow() {
    let q = Q8Matrix {
        data: vec![],
        scales: vec![],
        rows: 0,
        cols: usize::MAX,
    };
    matmul_bt_q8(&[], &q, &mut [], 0, usize::MAX, 0);
}

/// Overflow guard for the macOS tile-scratch size `m * TILE_N`. With
/// `m=MAX, k=0, n=0` the earlier products and length asserts are all
/// zero/empty, so control reaches the tile block; `m*TILE_N` then overflows
/// and must be rejected with a dedicated message, same as `TILE_N*k` above.
#[cfg(target_os = "macos")]
#[test]
#[should_panic(expected = "matmul shape overflow: m*TILE_N")]
fn test_matmul_bt_q8_rejects_m_tile_n_overflow() {
    let q = Q8Matrix {
        data: vec![],
        scales: vec![],
        rows: 0,
        cols: 0,
    };
    matmul_bt_q8(&[], &q, &mut [], usize::MAX, 0, 0);
}

#[test]
fn test_zero_row_handling() {
    let b = vec![
        0.0, 0.0, 0.0, 0.0, // zero row
        0.5, -0.5, 0.25, -0.25,
    ];
    let q = quantize_matrix(&b, 2, 4).unwrap();

    assert!(approx_eq(q.scales[0], 1.0, 1e-8));
    assert_eq!(&q.data[0..4], &[0i8, 0i8, 0i8, 0i8]);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0f32; 2];
    matmul_bt_q8(&a, &q, &mut c, 1, 4, 2);

    assert!(approx_eq(c[0], 0.0, 1e-8));
    assert!(c[1].abs() > 0.0);
}
