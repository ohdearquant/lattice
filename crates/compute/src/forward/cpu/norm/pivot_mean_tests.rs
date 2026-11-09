use super::{layer_norm, residual_add_layer_norm};

const EPS: f32 = 1e-12;
const OUTLIER_WIDTHS: [usize; 9] = [16, 17, 31, 32, 33, 65, 896, 1792, 4096];
const SPARSE_WIDTH: usize = 4096;

fn outlier_row(hidden: usize) -> Vec<f32> {
    let mut input = vec![-1.0; hidden];
    input[0] = (hidden - 1) as f32;
    input
}

fn sparse_zero_sum_row() -> Vec<f32> {
    let pivot = f32::from_bits(0x3f80_0005);
    let mut input = vec![0.0; SPARSE_WIDTH];
    input[0] = pivot;
    input[32] = -pivot;
    input
}

fn sparse_variance() -> f64 {
    let pivot = f64::from(f32::from_bits(0x3f80_0005));
    2.0 * pivot * pivot / SPARSE_WIDTH as f64
}

fn assert_zero_mean_normalization(input: &[f32], output: &[f32], variance: f64, beta: f32) {
    assert_eq!(input.len(), output.len());
    assert!(
        output.iter().all(|value| value.is_finite()),
        "pivot-centered normalization must remain finite"
    );
    let denominator = (variance + f64::from(EPS)).sqrt();
    for (&value, &got) in input.iter().zip(output) {
        let want = f64::from(value) / denominator + f64::from(beta);
        let error = (f64::from(got) - want).abs();
        let limit = 1e-4 * f64::from(got).abs().max(want.abs()) + 1e-5;
        assert!(
            error <= limit,
            "pivot-centered normalization must satisfy the existing numerical tolerance"
        );
    }
}

fn exact_halves(input: &[f32]) -> Vec<f32> {
    let halves: Vec<f32> = input.iter().map(|&value| value * 0.5).collect();
    for (&half, &value) in halves.iter().zip(input) {
        assert_eq!(
            (half + half).to_bits(),
            value.to_bits(),
            "pivot fixture must preserve the explicit f32 sum"
        );
    }
    halves
}

fn assert_fixture_moments(input: &[f32], expected_variance: f64) {
    let mean = input.iter().map(|&value| f64::from(value)).sum::<f64>() / input.len() as f64;
    assert_eq!(mean, 0.0, "pivot fixture must have exact zero mean");
    let variance = input
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / input.len() as f64;
    assert_eq!(
        variance, expected_variance,
        "pivot fixture must have the exact analytic variance"
    );
}

#[test]
fn pivot_fixtures_have_exact_analytic_moments() {
    for hidden in OUTLIER_WIDTHS {
        assert_fixture_moments(&outlier_row(hidden), (hidden - 1) as f64);
    }
    assert_fixture_moments(&sparse_zero_sum_row(), sparse_variance());
}

#[test]
fn public_pivot_outlier_plain_matches_analytic_variance() {
    for hidden in OUTLIER_WIDTHS {
        let input = outlier_row(hidden);
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let mut output = input.clone();

        layer_norm(&mut output, &gamma, &beta, hidden, EPS);

        assert_zero_mean_normalization(&input, &output, (hidden - 1) as f64, 0.5);
    }
    eprintln!("simd layer norm: completed pivot outlier plain controls");
}

#[test]
fn public_pivot_outlier_fused_matches_analytic_variance() {
    for hidden in OUTLIER_WIDTHS {
        let input = outlier_row(hidden);
        let gamma = vec![1.0; hidden];
        let beta = vec![0.5; hidden];
        let mut output = exact_halves(&input);
        let dense = output.clone();

        residual_add_layer_norm(&mut output, &dense, &gamma, &beta, hidden, EPS);

        assert_zero_mean_normalization(&input, &output, (hidden - 1) as f64, 0.5);
    }
    eprintln!("simd layer norm: completed pivot outlier fused controls");
}

#[test]
fn public_sparse_zero_sum_plain_matches_analytic_variance() {
    let input = sparse_zero_sum_row();
    let gamma = vec![1.0; SPARSE_WIDTH];
    let beta = vec![0.0; SPARSE_WIDTH];
    let mut output = input.clone();

    layer_norm(&mut output, &gamma, &beta, SPARSE_WIDTH, EPS);

    assert_zero_mean_normalization(&input, &output, sparse_variance(), 0.0);
    eprintln!("simd layer norm: completed sparse zero-sum plain controls");
}

#[test]
fn public_sparse_zero_sum_fused_matches_analytic_variance() {
    let input = sparse_zero_sum_row();
    let gamma = vec![1.0; SPARSE_WIDTH];
    let beta = vec![0.0; SPARSE_WIDTH];
    let mut output = exact_halves(&input);
    let dense = output.clone();

    residual_add_layer_norm(&mut output, &dense, &gamma, &beta, SPARSE_WIDTH, EPS);

    assert_zero_mean_normalization(&input, &output, sparse_variance(), 0.0);
    eprintln!("simd layer norm: completed sparse zero-sum fused controls");
}
