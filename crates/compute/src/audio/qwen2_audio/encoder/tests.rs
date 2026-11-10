use super::*;

fn config() -> EncoderConfig {
    EncoderConfig {
        input_channels: 1,
        frames: 5,
        valid_frames: 3,
        hidden: 3,
        heads: 1,
        ffn: 3,
        output: 2,
    }
}

fn weights<'a>(
    conv1: &'a [f32],
    conv2: &'a [f32],
    bias: &'a [f32],
    positions: &'a [f32],
    layers: &'a [layer::LayerWeights<'a>],
    projector: &'a [f32],
    out_bias: &'a [f32],
) -> EncoderWeights<'a> {
    EncoderWeights {
        stem: stem::StemWeights {
            conv1,
            bias1: bias,
            conv2,
            bias2: bias,
        },
        positions,
        layers,
        norm_weight: &[1.0, 1.0, 1.0],
        norm_bias: bias,
        projector_weight: projector,
        projector_bias: out_bias,
    }
}

#[test]
fn positions_pool_before_norm_projector_and_odd_frame_drop() {
    let c = config();
    let conv1 = [0.0; 9];
    let conv2 = [0.0; 27];
    let bias = [0.0; 3];
    let positions = [0.0, 0.0, 0.0, 3.0, 6.0, 12.0, 100.0, -20.0, 40.0];
    let projector = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let out_bias = [0.25, -0.5];
    let w = weights(
        &conv1,
        &conv2,
        &bias,
        &positions,
        &[],
        &projector,
        &out_bias,
    );
    let out = run(&[0.0; 5], &c, &w).unwrap();
    assert_eq!(out.time, 3);
    assert_eq!(out.valid_tokens, 1);
    assert_eq!(out.positioned, positions);
    assert_eq!(out.layers, positions);
    assert_eq!(out.pooled, [1.5, 3.0, 6.0]);
    assert!((out.normalized[0] + 1.069_044).abs() < 2e-5);
    assert!((out.normalized[1] + 0.267_261).abs() < 2e-5);
    assert!((out.normalized[2] - 1.336_306).abs() < 2e-5);
    assert!((out.projected[0] - (out.normalized[0] + 0.25)).abs() < 1e-6);
    assert!((out.projected[1] - (out.normalized[1] - 0.5)).abs() < 1e-6);
}

#[test]
fn stem_channels_are_transposed_before_positions_are_added() {
    let c = config();
    let conv1 = [0.0, 1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.25, 0.0];
    let mut conv2 = [0.0; 27];
    for h in 0..3 {
        conv2[(h * 3 + h) * 3 + 1] = 1.0;
    }
    let bias = [0.0; 3];
    let pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let projector = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let out_bias = [0.0; 2];
    let w = weights(&conv1, &conv2, &bias, &pos, &[], &projector, &out_bias);
    let out = run(&[1.0, 2.0, -1.0, 3.0, 0.0], &c, &w).unwrap();
    assert_ne!(out.stem[1], out.stem[3]);
    for t in 0..out.time {
        for h in 0..c.hidden {
            assert_eq!(
                out.positioned[t * c.hidden + h],
                out.stem[h * out.time + t] + pos[t * c.hidden + h]
            );
        }
    }
}

#[test]
fn rejects_bad_shapes_nonfinite_valid_counts_and_overflow() {
    let mut c = config();
    let conv1 = [0.0; 9];
    let conv2 = [0.0; 27];
    let bias = [0.0; 3];
    let pos = [0.0; 9];
    let projector = [0.0; 6];
    let out_bias = [0.0; 2];
    let w = weights(&conv1, &conv2, &bias, &pos, &[], &projector, &out_bias);
    assert!(run(&[0.0; 4], &c, &w).is_err());
    assert!(run(&[0.0, f32::NAN, 0.0, 0.0, 0.0], &c, &w).is_err());
    c.valid_frames = 0;
    assert!(run(&[0.0; 5], &c, &w).is_err());
    c.valid_frames = 6;
    assert!(run(&[0.0; 5], &c, &w).is_err());
    c.valid_frames = 3;
    c.frames = usize::MAX;
    assert!(run(&[0.0; 5], &c, &w).is_err());
    c.frames = 1;
    c.valid_frames = 1;
    assert!(run(&[0.0], &c, &w).is_err());
}

#[test]
fn layer_stack_receives_positioned_states_and_valid_key_mask() {
    let mut c = config();
    let conv1 = [0.0; 9];
    let conv2 = [0.0; 27];
    let bias = [0.0; 3];
    let one = [1.0; 3];
    let zero = [0.0; 9];
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let pos = [-1.0, 0.0, 1.0, 1.0, 1.0, -2.0, 2.0, -1.0, -1.0];
    let projector = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let out_bias = [0.0; 2];
    let layers = [layer::LayerWeights {
        norm1_weight: &one,
        norm1_bias: &bias,
        q_weight: &zero,
        q_bias: &bias,
        k_weight: &zero,
        v_weight: &identity,
        v_bias: &bias,
        out_weight: &identity,
        out_bias: &bias,
        norm2_weight: &one,
        norm2_bias: &bias,
        fc1_weight: &zero,
        fc1_bias: &bias,
        fc2_weight: &zero,
        fc2_bias: &bias,
    }];
    let w = weights(&conv1, &conv2, &bias, &pos, &layers, &projector, &out_bias);
    let masked = run(&[0.0; 5], &c, &w).unwrap();
    let expected = layer::run(&pos, 3, 3, 1, 3, 2, &layers[0]).unwrap();
    assert_eq!(masked.layers, expected.output);
    assert_ne!(masked.layers, masked.positioned);
    c.valid_frames = 5;
    let unmasked = run(&[0.0; 5], &c, &w).unwrap();
    assert!((masked.layers[0] - unmasked.layers[0]).abs() > 0.1);
}

#[test]
fn aggregate_work_budget_counts_every_layer_before_computation() {
    let c = EncoderConfig {
        input_channels: 128,
        frames: 3000,
        valid_frames: 3000,
        hidden: 512,
        heads: 8,
        ffn: 2048,
        output: 32,
    };
    let conv1 = 512 * 128 * 3;
    let conv2 = 512 * 512 * 3;
    let projector = 32 * 512;
    assert!(check_work(&c, 1500, 1500, conv1, conv2, projector, 1).is_ok());
    assert!(check_work(&c, 1500, 1500, conv1, conv2, projector, 2).is_err());
    assert!(check_work(&c, usize::MAX, 1500, conv1, conv2, projector, 2).is_err());
}

#[test]
fn aggregate_work_budget_matches_non_square_layer_cost() {
    let c = EncoderConfig {
        frames: 8,
        valid_frames: 4,
        ffn: 5,
        ..config()
    };
    assert_eq!(layer::work_macs(4, 3, 5, 2), Some(312));
    for layer_count in [0, 1, 2] {
        let conv1_len = usize::try_from((MAX_MACS - 312 * layer_count as u64) / 8).unwrap();
        assert!(check_work(&c, 4, 2, conv1_len, 0, 0, layer_count).is_ok());
        let error = check_work(&c, 4, 2, conv1_len, 1, 0, layer_count).unwrap_err();
        assert!(matches!(
            error,
            InferenceError::InvalidInput(message)
                if message == "audio encoder exceeds aggregate work budget"
        ));
    }
}

fn post_pool_output(row: &[f32], gamma: &[f32], beta: &[f32]) -> EncoderOutput {
    let hidden = row.len();
    let c = EncoderConfig {
        input_channels: 1,
        frames: 4,
        valid_frames: 4,
        hidden,
        heads: 1,
        ffn: 1,
        output: 2,
    };
    let conv1 = vec![0.0; hidden * 3];
    let conv2 = vec![0.0; hidden * hidden * 3];
    let zero = vec![0.0; hidden];
    let positions = row.repeat(2);
    let mut projector = vec![0.0; hidden * 2];
    projector[0] = 1.0;
    projector[hidden * 2 - 1] = 1.0;
    let w = EncoderWeights {
        stem: stem::StemWeights {
            conv1: &conv1,
            bias1: &zero,
            conv2: &conv2,
            bias2: &zero,
        },
        positions: &positions,
        layers: &[],
        norm_weight: gamma,
        norm_bias: beta,
        projector_weight: &projector,
        projector_bias: &[0.0; 2],
    };
    let out = run(&[0.0; 4], &c, &w).unwrap();
    assert_eq!(out.time, 2, "post-pool fixture time");
    assert_eq!(out.valid_tokens, 1, "post-pool fixture valid tokens");
    assert!(out.stem.iter().all(|&v| v == 0.0), "post-pool zero stem");
    assert_eq!(out.positioned, positions, "post-pool positioned rows");
    assert_eq!(out.layers, positions, "post-pool empty layer stack");
    assert_eq!(out.pooled.len(), hidden, "post-pool row length");
    assert_eq!(out.normalized.len(), hidden, "post-pool normalized length");
    for (i, (&actual, &expected)) in out.pooled.iter().zip(row).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "post-pool row bits {i}"
        );
    }
    assert_eq!(out.projected.len(), 2, "post-pool projected length");
    for (output, lane) in [0, hidden - 1].into_iter().enumerate() {
        assert_eq!(
            out.projected[output].to_bits(),
            out.normalized[lane].to_bits(),
            "post-pool one-hot projector output {output}"
        );
    }
    out
}

fn assert_post_pool_f64_reference(
    out: &EncoderOutput,
    row: &[f32],
    gamma: &[f32],
    beta: &[f32],
    case: &str,
) {
    let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / row.len() as f64;
    let variance = row
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / row.len() as f64;
    let denominator = (variance + f64::from(1e-5_f32)).sqrt();
    for (i, &value) in row.iter().enumerate() {
        let expected =
            (f64::from(value) - mean) / denominator * f64::from(gamma[i]) + f64::from(beta[i]);
        let actual = f64::from(out.normalized[i]);
        assert!(
            actual.is_finite() && expected.is_finite(),
            "post-pool finite {case} lane={i}: actual={actual} expected={expected}"
        );
        assert!(
            (actual - expected).abs() <= 1e-5,
            "post-pool f64 reference {case} lane={i}: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn post_pool_constant_row_preserves_bias_bits() {
    let row = [f32::from_bits(0x4280_0003); 16];
    let gamma: Vec<f32> = (0..16).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
    let beta: Vec<f32> = (0..16).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
    let out = post_pool_output(&row, &gamma, &beta);
    for (i, (&actual, &expected)) in out.normalized.iter().zip(&beta).enumerate() {
        assert!(actual.is_finite(), "post-pool finite constant lane={i}");
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "post-pool constant bias bits lane={i}"
        );
    }
}

#[test]
fn post_pool_large_offset_row_matches_f64_reference() {
    let row: Vec<f32> = (0..896)
        .map(|i| {
            if i % 2 == 0 {
                33_554_448.0
            } else {
                33_554_452.0
            }
        })
        .collect();
    let gamma = vec![1.0; 896];
    let beta = vec![0.0; 896];
    let out = post_pool_output(&row, &gamma, &beta);
    assert_post_pool_f64_reference(&out, &row, &gamma, &beta, "large-offset896");
}

#[test]
fn post_pool_affine_row_matches_f64_reference() {
    let row: Vec<f32> = (0..896)
        .map(|i| {
            if i % 2 == 0 {
                33_554_448.0
            } else {
                33_554_452.0
            }
        })
        .collect();
    let gamma: Vec<f32> = (0..896).map(|i| [0.5, -1.0, 2.0, -0.5][i % 4]).collect();
    let beta: Vec<f32> = (0..896)
        .map(|i| [-0.125, 0.0, 0.125, 0.0625][i % 4])
        .collect();
    let out = post_pool_output(&row, &gamma, &beta);
    assert_post_pool_f64_reference(&out, &row, &gamma, &beta, "large-offset-affine896");
}

#[test]
fn post_pool_ordinary_affine_row_matches_f64_reference() {
    let row: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.125).collect();
    let gamma: Vec<f32> = (0..16).map(|i| [0.5, -1.0, 2.0, -0.5][i % 4]).collect();
    let beta: Vec<f32> = (0..16)
        .map(|i| [-0.125, 0.0, 0.125, 0.0625][i % 4])
        .collect();
    let out = post_pool_output(&row, &gamma, &beta);
    assert_post_pool_f64_reference(&out, &row, &gamma, &beta, "ordinary-affine16");
}
