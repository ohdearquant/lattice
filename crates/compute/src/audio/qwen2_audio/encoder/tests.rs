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
