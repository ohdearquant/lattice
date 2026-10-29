use super::*;

fn weights<'a>(
    zero4: &'a [f32],
    zero2: &'a [f32],
    ones: &'a [f32],
    identity: &'a [f32],
) -> LayerWeights<'a> {
    LayerWeights {
        norm1_weight: ones,
        norm1_bias: zero2,
        q_weight: zero4,
        q_bias: zero2,
        k_weight: zero4,
        v_weight: identity,
        v_bias: zero2,
        out_weight: identity,
        out_bias: zero2,
        norm2_weight: ones,
        norm2_bias: zero2,
        fc1_weight: zero4,
        fc1_bias: zero2,
        fc2_weight: zero4,
        fc2_bias: zero2,
    }
}

#[test]
fn bidirectional_attention_and_padding_mask_are_distinct() {
    let z4 = [0.0; 4];
    let z2 = [0.0; 2];
    let one = [1.0; 2];
    let id = [1.0, 0.0, 0.0, 1.0];
    let w = weights(&z4, &z2, &one, &id);
    let x = [-1.0, 1.0, 1.0, -1.0];
    let all = run(&x, 2, 2, 1, 2, 2, &w).unwrap();
    let masked = run(&x, 2, 2, 1, 2, 1, &w).unwrap();
    assert!(all.attention[0].abs() < 1e-6);
    assert!(masked.attention[0] < -0.99);
    assert!(
        masked.attention[2] < -0.99,
        "padded queries still attend to valid keys"
    );
    assert!(
        (all.output[0] - masked.output[0]).abs() > 0.9,
        "future key affects first query"
    );
    assert_eq!(all.output, all.residual);
}

#[test]
fn second_ffn_uses_erf_gelu_and_both_residuals() {
    let z4 = [0.0; 4];
    let z2 = [0.0; 2];
    let one = [1.0; 2];
    let id = [1.0, 0.0, 0.0, 1.0];
    let mut w = weights(&z4, &z2, &one, &id);
    w.v_weight = &z4;
    let ffn_bias = [-1.0, 1.0];
    w.fc1_bias = &ffn_bias;
    w.fc2_weight = &id;
    let out = run(&[0.25, -0.25], 1, 2, 2, 2, 1, &w).unwrap();
    assert!((out.activated[0] - (-0.158_655_26)).abs() < 1e-6);
    assert!((out.activated[1] - 0.841_344_7).abs() < 1e-6);
    assert!((out.output[0] - (0.25 - 0.158_655_26)).abs() < 1e-6);
    assert!((out.output[1] - (-0.25 + 0.841_344_7)).abs() < 1e-6);
}

#[test]
fn refuses_invalid_geometry_shapes_and_nonfinite_before_work() {
    let z4 = [0.0; 4];
    let z2 = [0.0; 2];
    let one = [1.0; 2];
    let id = [1.0, 0.0, 0.0, 1.0];
    let w = weights(&z4, &z2, &one, &id);
    assert!(run(&z2, 1, 2, 0, 2, 1, &w).is_err());
    assert!(run(&z2, 1, 2, 3, 2, 1, &w).is_err());
    assert!(run(&z2, 1, 2, 1, 2, 0, &w).is_err());
    assert!(run(&z2, 1, 2, 1, 2, 2, &w).is_err());
    assert!(run(&z2, usize::MAX, 2, 1, 2, 1, &w).is_err());
    assert!(run(&z2[..1], 1, 2, 1, 2, 1, &w).is_err());
    assert!(run(&[f32::NAN, 0.0], 1, 2, 1, 2, 1, &w).is_err());
    let mut bad = w;
    let invalid_bias = [f32::INFINITY, 0.0];
    bad.q_bias = &invalid_bias;
    assert!(run(&z2, 1, 2, 1, 2, 1, &bad).is_err());
}
