use super::test_support::{approx_eq, uniform_signed};
use super::*;

#[test]
fn test_quantize_identity() {
    let rows = 4;
    let cols = 4;
    let w = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 1.0, // row 3
    ];

    let q = quantize_matrix(&w, rows, cols).unwrap();
    let dq = dequantize_matrix(&q);

    assert_eq!(q.rows, rows);
    assert_eq!(q.cols, cols);
    assert_eq!(q.scales.len(), rows);
    assert_eq!(q.data.len(), rows * cols);

    for &scale in &q.scales {
        assert!(approx_eq(scale, 1.0 / 127.0, 1e-8));
    }
    for (orig, recon) in w.iter().zip(dq.iter()) {
        assert!(approx_eq(*orig, *recon, 1e-6));
    }
}

#[test]
fn test_quantize_roundtrip_accuracy() {
    let rows = 32;
    let cols = 64;
    let mut seed = 0x1234_5678u32;
    let mut w = Vec::with_capacity(rows * cols);

    for _ in 0..rows * cols {
        w.push(uniform_signed(&mut seed) * 0.03);
    }

    let q = quantize_matrix(&w, rows, cols).unwrap();
    let dq = dequantize_matrix(&q);

    let mut max_abs_err = 0.0f32;
    let mut mean_abs_err = 0.0f32;
    for (orig, recon) in w.iter().zip(dq.iter()) {
        let err = (orig - recon).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        mean_abs_err += err;
    }
    mean_abs_err /= (rows * cols) as f32;

    assert!(max_abs_err < 2.0e-4, "max_abs_err={max_abs_err}");
    assert!(mean_abs_err < 7.0e-5, "mean_abs_err={mean_abs_err}");
}

#[test]
fn test_scale_computation() {
    let w = vec![
        1.27, -0.63, 0.0, // max abs = 1.27 -> scale = 0.01
        -2.54, 2.0, 0.0, // max abs = 2.54 -> scale = 0.02
    ];
    let q = quantize_matrix(&w, 2, 3).unwrap();

    assert!(approx_eq(q.scales[0], 0.01, 1e-8));
    assert!(approx_eq(q.scales[1], 0.02, 1e-8));
    assert_eq!(&q.data[0..3], &[127i8, -63i8, 0i8]);
    assert_eq!(&q.data[3..6], &[-127i8, 100i8, 0i8]);
}

#[test]
fn test_quantize_large_matrix() {
    let rows = 6_144;
    let cols = 2_048;
    let mut w = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        let row_scale = 0.005 + (r % 17) as f32 * 0.0005;
        for c in 0..cols {
            let bucket = ((r * 131 + c * 17) % 257) as i32 - 128;
            w.push(bucket as f32 * row_scale / 128.0);
        }
    }

    let q = quantize_matrix(&w, rows, cols).unwrap();

    assert_eq!(q.rows, rows);
    assert_eq!(q.cols, cols);
    assert_eq!(q.data.len(), rows * cols);
    assert_eq!(q.scales.len(), rows);
    assert!(q.scales.iter().all(|&s| s.is_finite() && s > 0.0));
    assert!(q.data.iter().any(|&v| v != 0));

    let first_row_scale = q.scales[0];
    let last_row_scale = q.scales[rows - 1];
    assert!(first_row_scale > 0.0);
    assert!(last_row_scale > 0.0);
}

/// `quantize_matrix` must reject a weight matrix whose source row contains
/// `+inf` or `NaN`. Without ingress validation such a row can produce a
/// non-finite scale or silently quantize NaN to zero.
#[test]
fn test_quantize_matrix_rejects_nonfinite_source_row() {
    // Row 0 is finite; row 1 contains +inf.
    let w = vec![
        1.0,
        2.0,
        3.0, // row 0: finite, scale ≈ 3/127
        1.0,
        f32::INFINITY,
        0.0, // row 1: non-finite source
    ];
    match quantize_matrix(&w, 2, 3) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error must describe the non-finite source; got: {msg}"
            );
            assert!(msg.contains("element index 4"), "wrong attribution: {msg}");
        }
        Err(e) => panic!("expected InvalidInput for non-finite source, got: {e}"),
        Ok(_) => panic!("expected Err for non-finite source row, got Ok"),
    }
}

/// `quantize_matrix` must reject a weight matrix that contains `NaN`.
///
/// Unlike `±inf`, NaN does NOT propagate through the `> max_abs` comparison
/// in IEEE 754 — `NaN > max_abs` evaluates to `false` — so a NaN value
/// never updates `max_abs`.  The scale then stays finite and the NaN element
/// is silently quantized to 0 via Rust's saturating `f32 as i8` cast.
#[test]
fn test_quantize_matrix_rejects_nan_input() {
    // Row 0 is finite; row 1 contains NaN in lane 1.
    let w = vec![
        1.0,
        2.0,
        3.0, // row 0: finite
        1.0,
        f32::NAN,
        0.0, // row 1: NaN lane
    ];
    match quantize_matrix(&w, 2, 3) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error must describe the non-finite value; got: {msg}"
            );
            assert!(msg.contains("element index 4"), "wrong attribution: {msg}");
        }
        Err(e) => panic!("expected InvalidInput for NaN input, got: {e}"),
        Ok(_) => panic!("expected Err for NaN input, got Ok"),
    }
}

#[test]
fn test_quantize_matrix_rejects_shape_overflow() {
    match quantize_matrix(&[], usize::MAX, 2) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("overflows usize"),
                "error must describe the geometry overflow; got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput for shape overflow, got: {e}"),
        Ok(_) => panic!("expected Err for shape overflow, got Ok"),
    }
}

#[test]
fn test_quantize_matrix_rejects_scale_underflow() {
    let w = [f32::from_bits(1)];
    match quantize_matrix(&w, 1, 1) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-positive scale 0"),
                "error must describe the underflowed scale; got: {msg}"
            );
            assert!(msg.contains("row 0"), "wrong attribution: {msg}");
        }
        Err(e) => panic!("expected InvalidInput for scale underflow, got: {e}"),
        Ok(_) => panic!("expected Err for scale underflow, got Ok"),
    }
}

/// Reference two-pass implementation predating the ingress fusion: a full
/// finite-value scan over the source matrix, followed by the max-abs +
/// quantize loop as two independent passes. Kept only to prove the fused
/// single-pass `quantize_named_matrix` produces byte-identical output.
fn quantize_matrix_two_pass_reference(w: &[f32], rows: usize, cols: usize) -> Q8Matrix {
    assert!(
        w.iter().all(|v| v.is_finite()),
        "reference requires finite input"
    );

    let mut data = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);

    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let row = &w[start..end];

        let mut max_abs = 0.0f32;
        for &v in row {
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales.push(scale);

        for &v in row {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q);
        }
    }

    Q8Matrix {
        data,
        scales,
        rows,
        cols,
    }
}

/// Proves the fused single-pass `quantize_matrix` is byte-identical to the
/// pre-fusion two-pass reference across representative shapes, including a
/// forced zero row (the `max_abs == 0.0` branch). max(|x_i|) is
/// order-independent and the per-element quantization involves no FP
/// accumulation, so fusing the finite-value scan into the max-abs loop
/// must not change any output byte or scale.
#[test]
fn test_fused_pass_matches_two_pass_reference_bit_exact() {
    let cases: [(usize, usize, u32); 4] = [
        (4, 4, 0x1111_1111),
        (32, 64, 0x1234_5678),
        (17, 33, 0xDEAD_BEEF),
        (6, 4, 0xCAFE_BABE),
    ];

    for (rows, cols, seed0) in cases {
        let mut seed = seed0;
        let mut w = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            w.push(uniform_signed(&mut seed) * 0.37);
        }
        // Force a zero row to exercise the max_abs == 0.0 branch identically
        // in both the fused and reference paths.
        for v in &mut w[cols..2 * cols] {
            *v = 0.0;
        }

        let fused = quantize_matrix(&w, rows, cols).expect("valid finite matrix must quantize");
        let reference = quantize_matrix_two_pass_reference(&w, rows, cols);

        assert_eq!(
            fused.data, reference.data,
            "quantized bytes diverged for {rows}x{cols}"
        );
        assert_eq!(
            fused.scales, reference.scales,
            "scales diverged for {rows}x{cols}"
        );
    }
}
