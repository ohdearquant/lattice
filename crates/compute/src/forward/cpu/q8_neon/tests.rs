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

#[test]
fn test_quantize_vec_pins_scale_and_values() {
    // max |x| = 254, so the scale is exactly 2 and every value is exact or a half step.
    let x = [254.0, -254.0, 3.0, -3.0, 1.0, 0.0, 127.0];
    let (q, s) = quantize_vec_q8(&x);
    assert_eq!(s, 2.0);
    assert_eq!(q, vec![127, -127, 2, -2, 1, 0, 64]);
}

#[test]
fn test_quantize_vec_into_zero_input_clears_reused_scratch() {
    let mut scratch = Vec::new();
    assert_eq!(
        quantize_vec_q8_into(&[254.0, -254.0, 3.0], &mut scratch),
        2.0
    );
    assert_eq!(scratch, vec![127, -127, 2]);
    assert_eq!(quantize_vec_q8_into(&[0.0, 0.0], &mut scratch), 0.0);
    assert_eq!(scratch, vec![0, 0]);
}

fn raw_blocks(
    n: usize,
    k: usize,
    scale: impl Fn(usize, usize) -> f32,
    weight: impl Fn(usize, usize, usize) -> i8,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in 0..n {
        for block in 0..k / 32 {
            bytes.extend_from_slice(&scale(row, block).to_ne_bytes());
            bytes.extend((0..32).map(|lane| weight(row, block, lane) as u8));
        }
    }
    bytes
}

fn assert_same_floats(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (row, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "row {row}: {actual} must be NaN");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "row {row}");
        }
    }
}

fn check_wrappers(x: &[f32], weights: &[u8], n: usize) {
    let k = x.len();
    let (x_q, x_scale) = quantize_vec_q8(x);
    let mut expected = vec![0.0; n];
    matvec_q8_scalar(&x_q, x_scale, weights, n, k, &mut expected);
    assert_same_floats(&matmul_q8_neon(x, weights, n, k), &expected);
    let mut output = vec![12345.25; n + 4];
    let mut scratch = Vec::with_capacity(k + 8);
    let pointer = scratch.as_ptr();
    let capacity = scratch.capacity();
    matmul_q8_neon_into(x, weights, n, k, &mut output[2..], &mut scratch);
    assert_same_floats(&output[2..n + 2], &expected);
    assert_eq!(&output[..2], &[12345.25; 2]);
    assert_eq!(&output[n + 2..], &[12345.25; 2]);
    assert_eq!(scratch, x_q);
    assert_eq!(scratch.as_ptr(), pointer);
    assert_eq!(scratch.capacity(), capacity);
}

#[cfg(target_arch = "x86_64")]
std::thread_local! {
    static AVX2_BLOCKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FORCE_SCALAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(target_arch = "x86_64")]
pub(super) fn record_avx2_block() {
    AVX2_BLOCKS.with(|blocks| blocks.set(blocks.get() + 1));
}

#[cfg(target_arch = "x86_64")]
pub(super) fn force_scalar() -> bool {
    FORCE_SCALAR.with(std::cell::Cell::get)
}

#[cfg(target_arch = "x86_64")]
fn with_scalar_dispatch(f: impl FnOnce()) {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_SCALAR.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(FORCE_SCALAR.with(|flag| flag.replace(true)));
    f();
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_scalar_fallback_routes_both_wrappers_without_vector_blocks() {
    with_scalar_dispatch(|| {
        AVX2_BLOCKS.with(|blocks| blocks.set(0));
        let weights = raw_blocks(5, 64, |_, b| (b + 1) as f32 * 0.25, |_, _, _| -128);
        check_wrappers(&[1.0; 64], &weights, 5);
        assert_eq!(AVX2_BLOCKS.with(std::cell::Cell::get), 0);
    });
}

#[cfg(target_arch = "x86_64")]
fn call_avx2(x: &[i8], x_scale: f32, weights: &[u8], n: usize, k: usize, output: &mut [f32]) {
    assert!(
        std::is_x86_feature_detected!("avx2"),
        "the AVX2 host gate requires AVX2; scalar fallback is not acceptance"
    );
    // SAFETY: The assertion establishes AVX2. The kernel checks all buffer extents.
    unsafe { matvec_q8_avx2_inner(x, x_scale, weights, n, k, output) }
}

#[cfg(target_arch = "x86_64")]
fn check_avx2(x: &[i8], x_scale: f32, weights: &[u8], n: usize, expected: &[f32]) {
    let mut output = vec![12345.25; n + 4];
    AVX2_BLOCKS.with(|blocks| blocks.set(0));
    call_avx2(x, x_scale, weights, n, x.len(), &mut output[2..]);
    assert_eq!(AVX2_BLOCKS.with(std::cell::Cell::get), n * (x.len() / 32));
    assert_same_floats(&output[2..n + 2], expected);
    assert_eq!(&output[..2], &[12345.25; 2]);
    assert_eq!(&output[n + 2..], &[12345.25; 2]);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "requires an AVX2 host; run explicitly with --ignored --exact"]
fn avx2_signed_blocks_cover_all_lanes_and_unaligned_row_tails() {
    for (x, w, dot) in [
        (-128, -128, 524_288.0),
        (-128, 127, -520_192.0),
        (127, -128, -520_192.0),
        (127, 127, 516_128.0),
    ] {
        let weights = raw_blocks(5, 32, |_, _| 1.0, |_, _, _| w);
        check_avx2(&[x; 32], 1.0, &weights, 5, &[dot; 5]);
    }
    for lane in 0..32 {
        let mut x = [0; 32];
        x[lane] = -128;
        let weights = raw_blocks(5, 32, |_, _| 1.0, |row, _, col| col as i8 - 16 + row as i8);
        let expected: Vec<_> = (0..5)
            .map(|row| (-128 * (lane as i32 - 16 + row)) as f32)
            .collect();
        check_avx2(&x, 1.0, &weights, 5, &expected);
    }
    check_unaligned_avx2_shapes();
    eprintln!("Q8_AVX2_EXERCISED signed blocks, every lane, unaligned inputs and row tails");
}

#[cfg(target_arch = "x86_64")]
fn check_unaligned_avx2_shapes() {
    for n in [0, 1, 2, 3, 4, 5, 17] {
        for k in [0, 32, 64, 96, 128] {
            let mut x_storage = vec![0i8; k + 16];
            let x_start = (16 - x_storage.as_ptr() as usize % 16) % 16 + 1;
            let x = &mut x_storage[x_start..x_start + k];
            for (i, value) in x.iter_mut().enumerate() {
                *value = ((i * 47 + 83) % 256) as u8 as i8;
            }
            let packed = raw_blocks(
                n,
                k,
                |row, block| (row + block + 1) as f32 * 0.125,
                |row, block, lane| ((row * 73 + block * 41 + lane * 19) % 256) as u8 as i8,
            );
            let mut storage = vec![0u8; packed.len() + 16];
            let start = (16 - storage.as_ptr() as usize % 16) % 16 + 1;
            storage[start..start + packed.len()].copy_from_slice(&packed);
            let weights = &storage[start..start + packed.len()];
            assert_eq!(x.as_ptr() as usize % 16, 1);
            assert_eq!(weights.as_ptr() as usize % 16, 1);
            let mut expected = vec![0.0; n];
            matvec_q8_scalar(x, 0.375, weights, n, k, &mut expected);
            check_avx2(x, 0.375, weights, n, &expected);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "requires an AVX2 host; run explicitly with --ignored --exact"]
fn avx2_preserves_block_order_scale_association_and_special_values() {
    let weights = raw_blocks(
        5,
        96,
        |_, b| [1e20, -1e20, 1.0][b],
        |_, _, lane| if lane == 0 { 1 } else { 0 },
    );
    check_avx2(&[1; 96], 1.0, &weights, 5, &[1.0; 5]);
    let weights = raw_blocks(
        5,
        32,
        |_, _| 65504.0,
        |_, _, lane| if lane % 2 == 0 { 1 } else { -1 },
    );
    check_avx2(&[1; 32], f32::MAX, &weights, 5, &[0.0; 5]);
    for x_scale in [0.0, -0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        for scale in [
            0.0,
            -0.0,
            1.0,
            -2.0,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let weights = raw_blocks(
                5,
                64,
                |_, _| scale,
                |row, b, _| if row + b == 0 { 0 } else { -128 },
            );
            let mut expected = vec![0.0; 5];
            matvec_q8_scalar(&[-128; 64], x_scale, &weights, 5, 64, &mut expected);
            check_avx2(&[-128; 64], x_scale, &weights, 5, &expected);
        }
    }
    eprintln!("Q8_AVX2_EXERCISED block order, scale association and special values");
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "requires an AVX2 host; run explicitly with --ignored --exact"]
fn avx2_rejects_invalid_extents_before_loading() {
    assert!(std::is_x86_feature_detected!("avx2"), "AVX2 host required");
    for (x_len, weights_len, n, k, output_len) in [
        (32, 36, 1, 31, 1),
        (31, 36, 1, 32, 1),
        (32, 35, 1, 32, 1),
        (32, 36, 1, 32, 0),
        (0, 0, 1, usize::MAX - 31, 1),
        (32, 36, usize::MAX, 32, 1),
    ] {
        AVX2_BLOCKS.with(|blocks| blocks.set(0));
        let result = std::panic::catch_unwind(|| {
            call_avx2(
                &vec![0; x_len],
                1.0,
                &vec![0; weights_len],
                n,
                k,
                &mut vec![0.0; output_len],
            );
        });
        assert!(
            result.is_err(),
            "invalid extent accepted: {:?}",
            (x_len, weights_len, n, k, output_len)
        );
        assert_eq!(AVX2_BLOCKS.with(std::cell::Cell::get), 0);
    }
    eprintln!("Q8_AVX2_VALIDATION_EXERCISED invalid extents before vector loads");
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "requires an AVX2 host; run explicitly with --ignored --exact"]
fn avx2_both_wrappers_execute_vector_blocks() {
    assert!(std::is_x86_feature_detected!("avx2"), "AVX2 host required");
    for exceptional in [
        None,
        Some(f32::NAN),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
    ] {
        let mut x: Vec<_> = (0..64).map(|i| (i % 23) as f32 - 11.0).collect();
        if let Some(value) = exceptional {
            x[3] = value;
        }
        let weights = raw_blocks(
            5,
            64,
            |_, b| (b + 1) as f32 * 0.125,
            |row, _, lane| (row + lane) as i8 - 16,
        );
        let (x_q, x_scale) = quantize_vec_q8(&x);
        let mut expected = vec![0.0; 5];
        matvec_q8_scalar(&x_q, x_scale, &weights, 5, 64, &mut expected);
        AVX2_BLOCKS.with(|blocks| blocks.set(0));
        assert_same_floats(&matmul_q8_neon(&x, &weights, 5, 64), &expected);
        assert_eq!(
            AVX2_BLOCKS.with(std::cell::Cell::get),
            10,
            "allocating wrapper fell back"
        );
        AVX2_BLOCKS.with(|blocks| blocks.set(0));
        let mut output = [12345.25; 7];
        let mut scratch = Vec::with_capacity(64);
        let pointer = scratch.as_ptr();
        matmul_q8_neon_into(&x, &weights, 5, 64, &mut output, &mut scratch);
        assert_same_floats(&output[..5], &expected);
        assert_eq!(&output[5..], &[12345.25; 2]);
        assert_eq!(scratch.as_ptr(), pointer);
        assert_eq!(
            AVX2_BLOCKS.with(std::cell::Cell::get),
            10,
            "into wrapper fell back"
        );
    }
    eprintln!("Q8_AVX2_EXERCISED both public wrappers");
}

#[test]
fn wrappers_preserve_empty_shapes_row_tails_and_output_suffix() {
    for n in [0, 1, 2, 3, 4, 5, 17] {
        for k in [0, 32, 64, 96, 128] {
            let x: Vec<f32> = (0..k).map(|i| (i % 23) as f32 - 11.0).collect();
            let weights = raw_blocks(
                n,
                k,
                |row, block| (row + block + 1) as f32 * 0.125,
                |row, block, lane| ((row * 73 + block * 41 + lane * 19) % 256) as u8 as i8,
            );
            check_wrappers(&x, &weights, n);
        }
    }
}
