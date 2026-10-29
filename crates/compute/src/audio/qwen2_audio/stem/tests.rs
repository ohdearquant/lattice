use super::*;

fn identity_weights() -> StemWeights<'static> {
    StemWeights {
        conv1: &[0.0, 1.0, 0.0],
        bias1: &[0.0],
        conv2: &[0.0, 1.0, 0.0],
        bias2: &[0.0],
    }
}

#[test]
fn gelu_uses_error_function_not_tanh_approximation() {
    assert!((exact_gelu(1.0) - 0.841_344_8).abs() < 2e-7);
    assert!((exact_gelu(-1.0) - (-0.158_655_26)).abs() < 2e-7);
    assert_eq!(exact_gelu(0.0), 0.0);
}

#[test]
fn convolution_respects_edge_padding_and_stride() {
    let weights = StemWeights {
        conv1: &[1.0, 10.0, 100.0],
        bias1: &[0.0],
        conv2: &[0.0, 1.0, 0.0],
        bias2: &[0.0],
    };
    let out = run(&[1.0, 2.0, 3.0], 1, 3, 1, &weights).unwrap();
    assert_eq!(out.first, vec![210.0, 321.0, 32.0]);
    assert_eq!(out.second, vec![210.0, 32.0]);
    assert_eq!(out.output_frames, 2);
}

#[test]
fn channels_weights_bias_and_even_odd_lengths() {
    let w = StemWeights {
        conv1: &[0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
        bias1: &[1.0],
        conv2: &[0.0, 1.0, 0.0],
        bias2: &[0.0],
    };
    let out = run(&[2.0, 3.0, 5.0, 7.0], 2, 2, 1, &w).unwrap();
    assert_eq!(out.first, vec![13.0, 18.0]);
    assert_eq!(out.second, vec![13.0]);
    assert_eq!(out.output_frames, 1);
    assert_eq!(
        run(&[1.0], 1, 1, 1, &identity_weights())
            .unwrap()
            .output_frames,
        1
    );
}

#[test]
fn rejects_shapes_nonfinite_overflow_and_work_limit() {
    let w = identity_weights();
    assert!(run(&[], 1, 0, 1, &w).is_err());
    assert!(run(&[1.0, 2.0], 1, 1, 1, &w).is_err());
    assert!(run(&[f32::NAN], 1, 1, 1, &w).is_err());
    assert!(run(&[1.0], usize::MAX, 2, 1, &w).is_err());
    let invalid = StemWeights {
        conv1: &[0.0, f32::INFINITY, 0.0],
        ..w
    };
    assert!(run(&[1.0], 1, 1, 1, &invalid).is_err());
}

#[test]
fn rejects_over_budget_before_running_convolution() {
    let input = vec![0.0; 16_000];
    let conv1 = vec![0.0; 3_000];
    let conv2 = vec![0.0; 3_000_000];
    let bias = vec![0.0; 1_000];
    let weights = StemWeights {
        conv1: &conv1,
        bias1: &bias,
        conv2: &conv2,
        bias2: &bias,
    };
    let err = match run(&input, 1, 16_000, 1_000, &weights) {
        Ok(_) => panic!("expected work budget refusal"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("work budget"));
}
