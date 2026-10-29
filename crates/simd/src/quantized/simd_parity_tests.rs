use super::*;

fn gen_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ ((dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..dim)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                .wrapping_add(i as u64);
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn test_i8_quantize_explicit_simd_matches_scalar_and_is_dispatched() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
        let mut input = gen_vec(dim, 900 + dim as u64);
        if dim > 0 {
            input[0] = f32::NAN;
        }
        if dim > 1 {
            input[1] = f32::INFINITY;
        }
        if dim > 2 {
            input[2] = f32::NEG_INFINITY;
        }
        if dim > 3 {
            input[3] = 0.25;
        }
        if dim > 4 {
            input[4] = -0.25;
        }
        if dim > 5 {
            input[5] = f32::from_bits(0.25f32.to_bits() - 1);
        }
        if dim > 6 {
            input[6] = f32::from_bits(0.25f32.to_bits() + 1);
        }

        let scalar = quantize_i8_scalar(&input, 2.0);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
        let simd = unsafe { quantize_i8_neon(&input, 2.0) };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 was detected above; the kernel bounds every access.
        let simd = unsafe { quantize_i8_avx2(&input, 2.0) };
        assert_eq!(simd, scalar, "explicit SIMD mismatch at dim={dim}");
    }

    let input = gen_vec(385, 1_063);
    let before = I8_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
    let quantized = QuantizedVector::from_f32(&input);
    let after = I8_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
    assert_eq!(
        after,
        before + 1,
        "QuantizedVector::from_f32 did not execute its explicit SIMD quantizer"
    );
    assert_eq!(
        quantized.data,
        quantize_i8_scalar(&input, quantized.params.scale)
    );
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn test_finite_minmax_explicit_simd_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
        let mut input = gen_vec(dim, 1_100 + dim as u64);
        if dim > 0 {
            input[0] = f32::NAN;
        }
        if dim > 1 {
            input[1] = f32::INFINITY;
        }
        if dim > 2 {
            input[2] = f32::NEG_INFINITY;
        }
        if dim > 3 {
            input[3] = -0.0;
        }
        if dim > 4 {
            input[4] = 0.0;
        }

        let scalar = minmax_finite_scalar(&input);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
        let simd = unsafe { minmax_finite_neon(&input) };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 was detected above; the kernel bounds every access.
        let simd = unsafe { minmax_finite_avx2(&input) };
        assert_eq!(
            (simd.0.to_bits(), simd.1.to_bits()),
            (scalar.0.to_bits(), scalar.1.to_bits()),
            "finite min/max mismatch at dim={dim}"
        );
    }

    let mut min_zero_input = vec![1.0f32; 16];
    min_zero_input[0] = -0.0;
    min_zero_input[8] = 0.0;
    let mut max_zero_input = vec![-1.0f32; 16];
    max_zero_input[0] = 0.0;
    max_zero_input[8] = -0.0;
    for input in [&min_zero_input, &max_zero_input] {
        let scalar = minmax_finite_scalar(input);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
        let simd = unsafe { minmax_finite_neon(input) };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 was detected above; the kernel bounds every access.
        let simd = unsafe { minmax_finite_avx2(input) };
        assert_eq!(
            (simd.0.to_bits(), simd.1.to_bits()),
            (scalar.0.to_bits(), scalar.1.to_bits()),
            "finite min/max signed-zero mismatch"
        );
    }

    let input = gen_vec(385, 1_063);
    let before = I8_MINMAX_SIMD_HITS.with(std::cell::Cell::get);
    let params = QuantizationParams::from_vector(&input);
    let after = I8_MINMAX_SIMD_HITS.with(std::cell::Cell::get);
    assert_eq!(
        after,
        before + 1,
        "QuantizationParams::from_vector did not execute its explicit SIMD reducer"
    );
    let scalar = minmax_finite_scalar(&input);
    assert_eq!(
        (params.min_val.to_bits(), params.max_val.to_bits()),
        (scalar.0.to_bits(), scalar.1.to_bits())
    );
}

/// The signed-zero tie-break is a contract, not whatever the reduction happened to
/// pick. Parity tests can only compare kernels against each other, so they pass on
/// any architecture whose kernels agree by accident; this pins the value itself and
/// runs everywhere, including targets with no explicit SIMD path at all.
///
/// The first block drives `pin_zero_signs` directly with bounds carrying the wrong
/// sign. That matters: on aarch64 the scalar fold already returns the contracted
/// signs, so assertions routed through `minmax_finite_scalar` alone still pass with
/// the sign pass deleted. Only x86-64 breaks the tie the other way, and a guard that
/// can be removed without any local test noticing is not a guard.
#[test]
fn test_minmax_finite_pins_zero_signs() {
    let both_signs = [-0.0f32, 0.0, -1.0];
    assert_eq!(
        pin_zero_signs(&both_signs, -1.0, -0.0).1.to_bits(),
        0.0f32.to_bits(),
        "a max of -0.0 must be rewritten to +0.0 when +0.0 is present"
    );
    assert_eq!(
        pin_zero_signs(&both_signs, 0.0, -1.0).0.to_bits(),
        (-0.0f32).to_bits(),
        "a min of +0.0 must be rewritten to -0.0 when -0.0 is present"
    );

    let max_ties = [-0.0f32, 0.0, -1.0];
    let (_, max_val) = minmax_finite_scalar(&max_ties);
    assert_eq!(
        max_val.to_bits(),
        0.0f32.to_bits(),
        "max must take +0.0 when both zero signs are present"
    );

    let min_ties = [0.0f32, -0.0, 1.0];
    let (min_val, _) = minmax_finite_scalar(&min_ties);
    assert_eq!(
        min_val.to_bits(),
        (-0.0f32).to_bits(),
        "min must take -0.0 when both zero signs are present"
    );

    // A bound that is zero with only one sign available keeps that sign.
    let (only_neg_min, only_neg_max) = minmax_finite_scalar(&[-0.0f32, -1.0]);
    assert_eq!(only_neg_max.to_bits(), (-0.0f32).to_bits());
    assert_eq!(only_neg_min.to_bits(), (-1.0f32).to_bits());
    let (only_pos_min, only_pos_max) = minmax_finite_scalar(&[0.0f32, 1.0]);
    assert_eq!(only_pos_min.to_bits(), 0.0f32.to_bits());
    assert_eq!(only_pos_max.to_bits(), 1.0f32.to_bits());

    // Non-zero bounds are returned untouched, and an all-nonfinite input keeps the
    // identity pair rather than being rewritten by the sign pass.
    assert_eq!(
        minmax_finite_scalar(&[]),
        (f32::INFINITY, f32::NEG_INFINITY)
    );
    assert_eq!(
        minmax_finite_scalar(&[f32::NAN, f32::INFINITY]),
        (f32::INFINITY, f32::NEG_INFINITY)
    );
}

// FP-034: NEON SDOT vs scalar parity for INT8 dot product.
// Gated on dotprod: SDOT is FEAT_DotProd, not baseline NEON.
#[test]
fn test_i8_neon_scalar_parity() {
    #[cfg(target_arch = "aarch64")]
    {
        if !super::super::SimdConfig::detect().dotprod_enabled {
            eprintln!("skipping SDOT parity test: dotprod not available");
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    for dim in [7usize, 16, 64, 128, 384, 768] {
        let a_q = QuantizedVector::from_f32(&gen_vec(dim, 200 + dim as u64));
        let b_q = QuantizedVector::from_f32(&gen_vec(dim, 300 + dim as u64));

        // SAFETY: dotprod confirmed above; slices have equal length from from_f32.
        let neon = unsafe { dot_product_i8_neon_unrolled(&a_q.data, &b_q.data) };
        let scalar: f32 = a_q
            .data
            .iter()
            .zip(b_q.data.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum::<i32>() as f32;

        let diff = (neon - scalar).abs();
        assert!(
            diff == 0.0,
            "NEON vs scalar i8 dot product dim={dim}: neon={neon} scalar={scalar} diff={diff}"
        );
    }
}

// FP-034: AVX2 vs scalar parity for INT8 dot product.
#[test]
fn test_i8_avx2_scalar_parity() {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        for dim in [7usize, 16, 64, 128, 384, 768] {
            let a_q = QuantizedVector::from_f32(&gen_vec(dim, 400 + dim as u64));
            let b_q = QuantizedVector::from_f32(&gen_vec(dim, 500 + dim as u64));

            // SAFETY: AVX2 verified by is_x86_feature_detected! above; slices have equal length.
            let avx2 = unsafe { dot_product_i8_avx2_unrolled(&a_q.data, &b_q.data) };
            let scalar: f32 = a_q
                .data
                .iter()
                .zip(b_q.data.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>() as f32;

            let diff = (avx2 - scalar).abs();
            assert!(
                diff == 0.0,
                "AVX2 vs scalar i8 dot product dim={dim}: avx2={avx2} scalar={scalar} diff={diff}"
            );
        }
    } else {
        eprintln!("skipping AVX2 parity test: avx2 not available");
    }
}

// FP-034: AVX-512 VNNI vs scalar parity for INT8 dot product.
// Compiled unconditionally on x86_64 (the kernel is no longer feature-gated);
// runs only when the host advertises AVX-512F+BW+VNNI, matching the kernel's
// safety contract and SimdConfig::avx512vnni_enabled.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_i8_avx512vnni_scalar_parity() {
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni")
    {
        for dim in [7usize, 16, 64, 128, 384, 768] {
            let a_q = QuantizedVector::from_f32(&gen_vec(dim, 600 + dim as u64));
            let b_q = QuantizedVector::from_f32(&gen_vec(dim, 700 + dim as u64));

            // SAFETY: AVX-512F+BW+VNNI verified above; slices have equal length from from_f32.
            let vnni = unsafe { dot_product_i8_avx512vnni(&a_q.data, &b_q.data) };
            let scalar: f32 = a_q
                .data
                .iter()
                .zip(b_q.data.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>() as f32;

            let diff = (vnni - scalar).abs();
            assert!(
                diff == 0.0,
                "VNNI vs scalar i8 dot product dim={dim}: vnni={vnni} scalar={scalar} diff={diff}"
            );
        }
    } else {
        eprintln!("skipping VNNI parity test: avx512f/bw/vnni not available");
    }
}
