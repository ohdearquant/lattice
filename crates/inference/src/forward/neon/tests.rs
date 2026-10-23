use super::*;

#[test]
fn test_quantize_vec_zeros() {
    let (q, s) = quantize_vec_q8(&[0.0; 64]);
    assert_eq!(s, 0.0);
    assert!(q.iter().all(|&v| v == 0));
}

#[test]
fn test_quantize_vec_roundtrip() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
    let (q, s) = quantize_vec_q8(&x);
    assert!(s > 0.0);
    for (i, (&orig, &quant)) in x.iter().zip(q.iter()).enumerate() {
        let recovered = quant as f32 * s;
        let diff = (orig - recovered).abs();
        assert!(diff < s + 1e-6, "mismatch at {i}: {orig} vs {recovered}");
    }
}

#[test]
fn test_pack_weights_roundtrip() {
    let n = 4;
    let k = 64;
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let packed = pack_weights_q8(&w, n, k).unwrap();
    assert_eq!(packed.len(), n * (k / QK8_0) * Q8_0_BLOCK_BYTES);
}

#[test]
fn test_scalar_matches_reference() {
    let n = 8;
    let k = 64;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 - 32.0) * 0.02).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 7 + 3) % 19) as f32 * 0.05 - 0.45)
        .collect();

    // Reference: f32 matmul
    let mut ref_out = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0;
        for col in 0..k {
            sum += x[col] * w[row * k + col];
        }
        ref_out[row] = sum;
    }

    // Q8 scalar
    let (x_q, x_scale) = quantize_vec_q8(&x);
    let packed = pack_weights_q8(&w, n, k).unwrap();
    let mut q8_out = vec![0.0f32; n];
    matvec_q8_scalar(&x_q, x_scale, &packed, n, k, &mut q8_out);

    for i in 0..n {
        let diff = (ref_out[i] - q8_out[i]).abs();
        let tol = 0.05 * ref_out[i].abs().max(1.0);
        assert!(
            diff < tol,
            "row {i}: ref={} q8={} diff={}",
            ref_out[i],
            q8_out[i],
            diff
        );
    }
}

#[test]
fn test_neon_matches_scalar() {
    let n = 17; // not divisible by 4 — tests tail handling
    let k = 128;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 13 + 5) % 23) as f32 * 0.04 - 0.46)
        .collect();

    let (x_q, x_scale) = quantize_vec_q8(&x);
    let packed = pack_weights_q8(&w, n, k).unwrap();

    let mut scalar_out = vec![0.0f32; n];
    matvec_q8_scalar(&x_q, x_scale, &packed, n, k, &mut scalar_out);

    let neon_out = matmul_q8_neon(&x, &packed, n, k);

    for i in 0..n {
        let diff = (scalar_out[i] - neon_out[i]).abs();
        assert!(
            diff < 1e-4,
            "row {i}: scalar={} neon={} diff={}",
            scalar_out[i],
            neon_out[i],
            diff
        );
    }
}

#[test]
fn test_safe_wrapper_basic() {
    let n = 4;
    let k = 32;
    let x = vec![1.0f32; k];
    let w: Vec<f32> = (0..n * k)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let packed = pack_weights_q8(&w, n, k).unwrap();
    let out = matmul_q8_neon(&x, &packed, n, k);
    assert_eq!(out.len(), n);
    // Each row alternates 0.5/-0.5, dot with all-1s = 0.0
    for &v in &out {
        assert!(v.abs() < 0.5, "expected near-zero, got {v}");
    }
}

#[test]
fn test_large_matrix() {
    let n = 2048;
    let k = 2048;
    let x: Vec<f32> = (0..k).map(|i| ((i % 100) as f32 - 50.0) * 0.001).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.01)
        .collect();
    let packed = pack_weights_q8(&w, n, k).unwrap();
    let out = matmul_q8_neon(&x, &packed, n, k);
    assert_eq!(out.len(), n);
    // Smoke test: no NaN/Inf
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_matmul_q8_neon_into_matches_allocating_wrapper() {
    let n = 96;
    let k = 64;
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 7 + 3) % 19) as f32 * 0.05 - 0.45)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 13 + 7) % 23) as f32 * 0.04 - 0.46)
        .collect();
    let packed = pack_weights_q8(&w, n, k).unwrap();

    let expected = matmul_q8_neon(&x, &packed, n, k);

    let mut out = vec![0.0f32; n];
    let mut x_q = Vec::new();
    matmul_q8_neon_into(&x, &packed, n, k, &mut out, &mut x_q);

    assert_eq!(
        expected, out,
        "matmul_q8_neon_into must produce bitwise-identical output to matmul_q8_neon"
    );
}

/// Rejects a NaN block element instead of quantizing it silently. NaN never
/// updates `max_abs` via `>`, so without this guard the scale would stay
/// finite and the corrupt element would clamp to an arbitrary i8 value.
#[test]
fn pack_weights_q8_rejects_nan() {
    let n = 1;
    let k = 32;
    let mut w = vec![0.5f32; n * k];
    w[15] = f32::NAN;
    assert!(
        pack_weights_q8(&w, n, k).is_err(),
        "pack_weights_q8 must return Err when a block element is NaN"
    );
}

/// Rejects a +Inf block element instead of propagating it into the quantized
/// scale. `f32::INFINITY > max_abs` is true, so without this guard `max_abs`
/// and the resulting scale would both become +Inf, corrupting every row.
#[test]
fn pack_weights_q8_rejects_inf() {
    let n = 1;
    let k = 32;
    let mut w = vec![0.5f32; n * k];
    w[0] = f32::INFINITY;
    assert!(
        pack_weights_q8(&w, n, k).is_err(),
        "pack_weights_q8 must return Err when a block element is +Inf"
    );
}

/// A finite tensor whose element count disagrees with the declared `[n, k]` shape
/// must return a typed `Err`, not panic. Before this guard, `pack_weights_q8` used
/// `assert_eq!(weights.len(), n * k)`, so a malformed-but-finite checkpoint tensor
/// reaching the public NEON `quantize_model` conversion path (e.g. through
/// `pack_gdn_weights`) would abort the process instead of propagating an error.
#[test]
fn pack_weights_q8_rejects_element_count_mismatch_instead_of_panicking() {
    let n = 2;
    let k = 32;
    let w = vec![0.5f32; n * k - 1]; // one element short, all finite
    match pack_weights_q8(&w, n, k) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("does not match"),
                "error must describe the shape mismatch, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for element count mismatch, got Ok"),
    }
}

/// A `k` that is not a multiple of the Q8_0 block size (32) must return a typed
/// `Err`, not panic via `assert_eq!(k % QK8_0, 0)`.
#[test]
fn pack_weights_q8_rejects_non_block_aligned_k_instead_of_panicking() {
    let n = 1;
    let k = 33; // not a multiple of 32
    let w = vec![0.5f32; n * k];
    match pack_weights_q8(&w, n, k) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("multiple of"),
                "error must describe the alignment mismatch, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!("expected Err for non-block-aligned k, got Ok"),
    }
}
