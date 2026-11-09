use super::{layer_norm, residual_add_layer_norm};

fn assert_finite_beta(output: &[f32]) {
    assert!(
        output.iter().all(|value| value.is_finite()),
        "large constant normalization must remain finite"
    );
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() == 0.5f32.to_bits()),
        "large constant normalization must preserve beta bits"
    );
}

#[test]
fn public_large_constant_plain_preserves_finite_beta() {
    let constant = f32::from_bits(0x4c00_0005);
    for hidden in [896, 1792] {
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let mut output = vec![constant; hidden];

        layer_norm(&mut output, &gamma, &beta, hidden, 1e-12);

        assert_finite_beta(&output);
    }
    eprintln!("simd layer norm E: completed large constant plain controls");
}

#[test]
fn public_large_constant_fused_preserves_finite_beta() {
    let constant = f32::from_bits(0x4c00_0005);
    let half = constant * 0.5;
    assert_eq!(
        (half + half).to_bits(),
        constant.to_bits(),
        "large constant fixture must preserve the explicit f32 sum"
    );
    for hidden in [896, 1792] {
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let mut output = vec![half; hidden];
        let dense = vec![half; hidden];

        residual_add_layer_norm(&mut output, &dense, &gamma, &beta, hidden, 1e-12);

        assert_finite_beta(&output);
    }
    eprintln!("simd layer norm E: completed large constant fused controls");
}

const NEAR_CONSTANT_VALUES: [f32; 2] = [33_554_448.0, 33_554_456.0];
const ANALYTIC_CENTER: f64 = 33_554_452.0;
const ANALYTIC_VARIANCE: f64 = 16.0;

fn near_constant_row(hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|index| NEAR_CONSTANT_VALUES[index % 2])
        .collect()
}

fn assert_analytic_centered_variance(input: &[f32], output: &[f32], eps: f32) {
    assert_eq!(input.len(), output.len());
    assert!(
        output.iter().all(|value| value.is_finite()),
        "near-constant normalization must remain finite"
    );
    let denominator = (ANALYTIC_VARIANCE + f64::from(eps)).sqrt();
    for (&value, &got) in input.iter().zip(output) {
        let want = (f64::from(value) - ANALYTIC_CENTER) / denominator + 0.5;
        let error = (f64::from(got) - want).abs();
        let limit = 1e-4 * f64::from(got).abs().max(want.abs()) + 1e-5;
        assert!(
            error <= limit,
            "corrected variance must match analytic centered variance"
        );
    }
}

#[test]
fn near_constant_fixture_has_exact_analytic_moments() {
    let [low, high] = NEAR_CONSTANT_VALUES.map(f64::from);
    assert_eq!(
        (low + high) * 0.5,
        ANALYTIC_CENTER,
        "near-constant fixture must have the exact analytic center"
    );
    let low_delta = low - ANALYTIC_CENTER;
    let high_delta = high - ANALYTIC_CENTER;
    assert_eq!(
        (low_delta * low_delta + high_delta * high_delta) * 0.5,
        ANALYTIC_VARIANCE,
        "near-constant fixture must have the exact analytic variance"
    );
}

#[test]
fn public_near_constant_plain_matches_analytic_variance() {
    let hidden = 896;
    let eps = 1e-12;
    let gamma = vec![1.0; hidden];
    let beta = vec![0.5; hidden];
    let input = near_constant_row(hidden);
    let mut output = input.clone();

    layer_norm(&mut output, &gamma, &beta, hidden, eps);

    assert_analytic_centered_variance(&input, &output, eps);
    eprintln!("simd layer norm E: completed analytic near-constant plain controls");
}

#[test]
fn public_near_constant_fused_matches_analytic_variance() {
    let hidden = 896;
    let eps = 1e-12;
    let gamma = vec![1.0; hidden];
    let beta = vec![0.5; hidden];
    let input = near_constant_row(hidden);
    let mut output: Vec<f32> = input.iter().map(|&value| value * 0.5).collect();
    let dense = output.clone();
    for ((&left, &right), &value) in output.iter().zip(&dense).zip(&input) {
        assert_eq!(
            (left + right).to_bits(),
            value.to_bits(),
            "near-constant fixture must preserve the explicit f32 sum"
        );
    }

    residual_add_layer_norm(&mut output, &dense, &gamma, &beta, hidden, eps);

    assert_analytic_centered_variance(&input, &output, eps);
    eprintln!("simd layer norm E: completed analytic near-constant fused controls");
}

fn sparse_near_constant_row(hidden: usize, stride: usize) -> Vec<f32> {
    let center = f32::from_bits(0x4c00_0005);
    let mut input = vec![center; hidden];
    input[0] = center + 4.0;
    input[stride] = center - 4.0;
    input
}

fn assert_sparse_near_constant_tolerance(input: &[f32], output: &[f32], eps: f32) {
    assert_eq!(input.len(), output.len());
    assert!(
        output.iter().all(|value| value.is_finite()),
        "sparse near-constant normalization must remain finite"
    );
    let center = f64::from(f32::from_bits(0x4c00_0005));
    let variance = 32.0 / input.len() as f64;
    let denominator = (variance + f64::from(eps)).sqrt();
    for (&value, &got) in input.iter().zip(output) {
        let want = (f64::from(value) - center) / denominator + 0.5;
        let error = (f64::from(got) - want).abs();
        let limit = 1e-4 * f64::from(got).abs().max(want.abs()) + 1e-5;
        assert!(
            error <= limit,
            "sparse near-constant row must satisfy the existing numerical tolerance"
        );
    }
}

#[test]
fn public_sparse_near_constant_plain_matches_analytic_variance() {
    let eps = 1e-12;
    for (hidden, stride) in [(896, 16), (1792, 32)] {
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let input = sparse_near_constant_row(hidden, stride);
        let mut output = input.clone();

        layer_norm(&mut output, &gamma, &beta, hidden, eps);

        assert_sparse_near_constant_tolerance(&input, &output, eps);
    }
    eprintln!("simd layer norm E: completed sparse near-constant plain controls");
}

#[test]
fn public_sparse_near_constant_fused_matches_analytic_variance() {
    let eps = 1e-12;
    for (hidden, stride) in [(896, 16), (1792, 32)] {
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let input = sparse_near_constant_row(hidden, stride);
        let mut output: Vec<f32> = input.iter().map(|&value| value * 0.5).collect();
        let dense = output.clone();
        for ((&left, &right), &value) in output.iter().zip(&dense).zip(&input) {
            assert_eq!(
                (left + right).to_bits(),
                value.to_bits(),
                "sparse near-constant fixture must preserve the explicit f32 sum"
            );
        }

        residual_add_layer_norm(&mut output, &dense, &gamma, &beta, hidden, eps);

        assert_sparse_near_constant_tolerance(&input, &output, eps);
    }
    eprintln!("simd layer norm E: completed sparse near-constant fused controls");
}

#[test]
fn scalar_sparse_near_constant_control_matches_analytic_variance() {
    let eps = 1e-12;
    let center = f64::from(f32::from_bits(0x4c00_0005));
    for (hidden, stride) in [(896, 16), (1792, 32)] {
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let input = sparse_near_constant_row(hidden, stride);
        let mean = input.iter().map(|&value| f64::from(value)).sum::<f64>() / hidden as f64;
        assert_eq!(mean, center, "sparse fixture must have the exact center");
        let centered_sum = input
            .iter()
            .map(|&value| {
                let delta = f64::from(value) - center;
                delta * delta
            })
            .sum::<f64>();
        assert_eq!(
            centered_sum, 32.0,
            "sparse fixture must have centered sum 32"
        );
        let mut plain = input.clone();
        let mut fused: Vec<f32> = input.iter().map(|&value| value * 0.5).collect();
        let dense = fused.clone();

        super::layer_norm_scalar(&mut plain, &gamma, &beta, hidden, eps);
        super::residual_add_layer_norm_scalar(&mut fused, &dense, &gamma, &beta, hidden, eps);

        assert_sparse_near_constant_tolerance(&input, &plain, eps);
        assert_sparse_near_constant_tolerance(&input, &fused, eps);
    }
    eprintln!("simd layer norm E: completed scalar sparse near-constant controls");
}
