use super::*;

/// Absolute error bound of `fast_tanh` against `f64::tanh`, over every sweep input.
const TANH_BOUND: f64 = 1e-6;

/// Inputs: a dense grid over [-12, 12], the `f32` neighbours of the clamp points,
/// zeros, tiny, subnormal and huge magnitudes, and a stride over every finite `f32`
/// bit pattern (all exponents, both signs).
fn sweep_inputs() -> Vec<f32> {
    let mut xs: Vec<f32> = (-120_000..=120_000)
        .map(|i| (f64::from(i) * 1e-4) as f32)
        .collect();
    for c in [TANH_CLAMP, -TANH_CLAMP] {
        let bits = c.to_bits();
        xs.extend([c, f32::from_bits(bits - 1), f32::from_bits(bits + 1)]);
    }
    xs.extend([
        0.0,
        -0.0,
        1e-30,
        -1e-30,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        1e10,
        -1e10,
        f32::MAX,
        f32::MIN,
    ]);
    xs.extend(
        (0..=u32::MAX)
            .step_by(4099)
            .map(f32::from_bits)
            .filter(|v| v.is_finite()),
    );
    xs
}

fn gelu_f64(v: f32) -> f64 {
    let v = f64::from(v);
    let inner = (2.0 / std::f64::consts::PI).sqrt() * (v + 0.044_715 * v * v * v);
    0.5 * v * (1.0 + inner.tanh())
}

/// Runs the SIMD `fast_tanh` over `xs` in full vectors, zero-padding the last vector so
/// remainder lanes go through the same kernel. `None` when the CPU lacks the feature.
#[cfg(target_arch = "x86_64")]
fn simd_fast_tanh(xs: &[f32]) -> Option<Vec<f32>> {
    use std::arch::x86_64::*;
    let config = simd_config();
    if !(config.avx2_enabled && config.fma_enabled) {
        return None;
    }
    let mut out = vec![0.0f32; xs.len()];
    for (src, dst) in xs.chunks(8).zip(out.chunks_mut(8)) {
        let mut lane = [0.0f32; 8];
        lane[..src.len()].copy_from_slice(src);
        // SAFETY: AVX2 and FMA were detected above; `lane` holds 8 f32.
        unsafe {
            let r = fast_tanh_avx2(_mm256_loadu_ps(lane.as_ptr()));
            _mm256_storeu_ps(lane.as_mut_ptr(), r);
        }
        dst.copy_from_slice(&lane[..src.len()]);
    }
    Some(out)
}

#[cfg(target_arch = "aarch64")]
fn simd_fast_tanh(xs: &[f32]) -> Option<Vec<f32>> {
    use std::arch::aarch64::*;
    if !simd_config().neon_enabled {
        return None;
    }
    let mut out = vec![0.0f32; xs.len()];
    for (src, dst) in xs.chunks(4).zip(out.chunks_mut(4)) {
        let mut lane = [0.0f32; 4];
        lane[..src.len()].copy_from_slice(src);
        // SAFETY: NEON is available on aarch64; `lane` holds 4 f32.
        unsafe {
            let r = fast_tanh_neon(vld1q_f32(lane.as_ptr()));
            vst1q_f32(lane.as_mut_ptr(), r);
        }
        dst.copy_from_slice(&lane[..src.len()]);
    }
    Some(out)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn simd_fast_tanh(_xs: &[f32]) -> Option<Vec<f32>> {
    None
}

#[test]
fn fast_tanh_scalar_matches_f64_tanh_over_full_range() {
    let xs = sweep_inputs();
    let mut worst = (0.0f64, 0.0f32);
    for &x in &xs {
        let y = fast_tanh(x);
        let err = (f64::from(y) - f64::from(x).tanh()).abs();
        assert!(
            err <= TANH_BOUND,
            "fast_tanh({x:e}) = {y:e}, f64 tanh = {:e}, abs_err = {err:e} (bound {TANH_BOUND:e})",
            f64::from(x).tanh(),
        );
        assert!(y.abs() <= 1.0, "fast_tanh({x:e}) = {y:e} leaves [-1, 1]");
        if err > worst.0 {
            worst = (err, x);
        }
    }
    eprintln!(
        "fast_tanh scalar: {} inputs, max abs err {:e} at x = {:e}",
        xs.len(),
        worst.0,
        worst.1
    );
}

#[test]
fn fast_tanh_special_values() {
    assert_eq!(fast_tanh(f32::INFINITY), 1.0);
    assert_eq!(fast_tanh(f32::NEG_INFINITY), -1.0);
    assert!(fast_tanh(f32::NAN).is_nan(), "NaN must propagate");
    let pz = fast_tanh(0.0);
    assert!(pz == 0.0 && pz.is_sign_positive());
    let nz = fast_tanh(-0.0);
    assert!(nz == 0.0 && nz.is_sign_negative());
    for k in 1..=40 {
        let x = k as f32 * 0.25;
        assert_eq!(fast_tanh(-x), -fast_tanh(x), "odd symmetry at {x}");
    }
}

#[test]
fn fast_tanh_simd_matches_scalar_and_f64() {
    let xs = sweep_inputs();
    let Some(simd) = simd_fast_tanh(&xs) else {
        eprintln!("fast_tanh SIMD path unavailable on this CPU; skipped");
        return;
    };
    let mut worst_vs_scalar = (0.0f32, 0.0f32);
    let mut worst_vs_f64 = (0.0f64, 0.0f32);
    for (&x, &y) in xs.iter().zip(&simd) {
        let err64 = (f64::from(y) - f64::from(x).tanh()).abs();
        if err64 > worst_vs_f64.0 {
            worst_vs_f64 = (err64, x);
        }
        assert!(
            err64 <= TANH_BOUND,
            "simd fast_tanh({x:e}) = {y:e}, abs_err vs f64 = {err64:e}"
        );
        let d = (y - fast_tanh(x)).abs();
        assert!(
            d <= 1e-6,
            "simd vs scalar at {x:e}: {y:e} vs {:e}",
            fast_tanh(x)
        );
        if d > worst_vs_scalar.0 {
            worst_vs_scalar = (d, x);
        }
    }
    eprintln!(
        "fast_tanh simd vs scalar: max diff {:e} at x = {:e}; vs f64: max abs err {:e} at x = {:e}",
        worst_vs_scalar.0, worst_vs_scalar.1, worst_vs_f64.0, worst_vs_f64.1
    );
}

#[test]
fn fast_tanh_simd_remainder_lanes_and_special_values() {
    let pool = [
        -1e10,
        -9.5,
        -9.0,
        -3.25,
        -0.5,
        -1e-30,
        0.0,
        1e-30,
        0.5,
        3.25,
        9.0,
        9.5,
        1e10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.125,
        -0.125,
    ];
    for n in 0..=pool.len() {
        let Some(simd) = simd_fast_tanh(&pool[..n]) else {
            return;
        };
        for (&x, &y) in pool[..n].iter().zip(&simd) {
            assert!((y - fast_tanh(x)).abs() <= 1e-6, "len {n}, x = {x:e}");
        }
    }
    let nan = simd_fast_tanh(&[f32::NAN, 1.0, f32::NAN]).unwrap();
    assert!(
        nan[0].is_nan() && nan[2].is_nan(),
        "NaN lanes must propagate"
    );
    assert!((nan[1] - 1.0f32.tanh()).abs() <= 1e-6);
    let signed_zero = simd_fast_tanh(&[-0.0]).unwrap();
    assert!(signed_zero[0] == 0.0 && signed_zero[0].is_sign_negative());
}

#[test]
fn gelu_matches_f64_tanh_form_over_minus8_to_8() {
    // 16001 values: not a multiple of 8 or 4, so the SIMD remainder loops run.
    let vs: Vec<f32> = (-8000..=8000).map(|i| i as f32 * 1e-3).collect();
    assert_ne!(vs.len() % 8, 0);
    assert_ne!(vs.len() % 4, 0);

    let mut dispatched = vs.clone();
    gelu(&mut dispatched);
    let mut scalar = vs.clone();
    gelu_scalar(&mut scalar);
    let mut fused = vs.clone();
    let zero_bias = vec![0.0f32; vs.len()];
    add_bias_gelu(&mut fused, &zero_bias, vs.len());

    let mut worst = 0.0f64;
    for (i, &v) in vs.iter().enumerate() {
        let expected = gelu_f64(v);
        for (name, got) in [
            ("gelu", dispatched[i]),
            ("gelu_scalar", scalar[i]),
            ("add_bias_gelu", fused[i]),
        ] {
            let err = (f64::from(got) - expected).abs();
            assert!(
                err <= 2e-6,
                "{name}({v}) = {got:e}, f64 reference {expected:e}, abs_err = {err:e}"
            );
            worst = worst.max(err);
        }
    }
    eprintln!("gelu over [-8, 8]: max abs err {worst:e}");
}
