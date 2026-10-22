use super::*;

fn tiny_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 6,
        temporal_patch_size: 1,
        num_position_embeddings: 16,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    }
}

fn make_test_merger_weights(cfg: &VisionModelConfig) -> VisualMergerWeights {
    let hidden = cfg.hidden_size;
    let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;
    let out_hidden = cfg.out_hidden_size;

    let mut state = 0x9e37_79b9_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    let mut v = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();

    VisualMergerWeights {
        fc1_weight: v(merge_in * merge_in),
        fc1_bias: v(merge_in),
        fc2_weight: v(out_hidden * merge_in),
        fc2_bias: v(out_hidden),
        norm_weight: vec![1.0; hidden],
        norm_bias: vec![0.0; hidden],
    }
}

#[test]
fn erf_matches_known_values() {
    // Reference values from a standard erf table.
    assert!((erf(0.0) - 0.0).abs() < 1e-6);
    assert!((erf(1.0) - 0.842_700_79).abs() < 1e-6);
    assert!((erf(-1.0) + 0.842_700_79).abs() < 1e-6);
    assert!((erf(2.0) - 0.995_322_3).abs() < 1e-6);
}

#[test]
fn gelu_exact_matches_known_values() {
    // gelu_exact(0) = 0; gelu_exact is odd-ish but not exactly odd.
    assert!((gelu_exact(0.0)).abs() < 1e-6);
    // gelu_exact(1.0) ~= 0.8413447
    assert!((gelu_exact(1.0) - 0.841_344_7).abs() < 1e-4);
    // gelu_exact(-1.0) ~= -0.1586553
    assert!((gelu_exact(-1.0) + 0.158_655_3).abs() < 1e-4);
}

#[test]
fn merger_forward_output_shape_and_finite() {
    let cfg = tiny_cfg();
    let weights = make_test_merger_weights(&cfg);
    let n = 8; // must be a multiple of spatial_merge_size^2 (4)
    let pre_merger = vec![0.1f32; n * cfg.hidden_size];

    let out = qwen35_merger_forward(&weights, &cfg, &pre_merger).expect("merger forward");
    let merge_sq = cfg.spatial_merge_size * cfg.spatial_merge_size;
    assert_eq!(out.len(), (n / merge_sq) * cfg.out_hidden_size);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn merger_forward_rejects_hidden_size_mismatch() {
    let cfg = tiny_cfg();
    let weights = make_test_merger_weights(&cfg);
    let bad = vec![0.0f32; 3]; // not a multiple of hidden_size(8)
    let err = qwen35_merger_forward(&weights, &cfg, &bad).unwrap_err();
    assert!(matches!(err, VisionError::ShapeMismatch { .. }));
}

#[test]
fn merger_forward_rejects_patch_count_not_multiple_of_merge_sq() {
    let cfg = tiny_cfg();
    let weights = make_test_merger_weights(&cfg);
    // 3 patches * hidden(8) -- 3 is not a multiple of merge_sq(4).
    let bad = vec![0.0f32; 3 * cfg.hidden_size];
    let err = qwen35_merger_forward(&weights, &cfg, &bad).unwrap_err();
    assert!(matches!(err, VisionError::ShapeMismatch { .. }));
}

#[test]
fn merger_forward_cancels_before_shape_validation() {
    let cfg = tiny_cfg();
    let weights = make_test_merger_weights(&cfg);
    let result = qwen35_merger_forward_with_cancel(&weights, &cfg, &[0.0; 3], &mut || true)
        .expect("cancellation is not a merger failure");
    assert!(result.is_none());
}

#[test]
fn merger_forward_cancels_after_work_started() {
    let cfg = tiny_cfg();
    let weights = make_test_merger_weights(&cfg);
    let pre_merger = vec![0.1f32; 8 * cfg.hidden_size];
    let mut polls = 0;
    let result = qwen35_merger_forward_with_cancel(&weights, &cfg, &pre_merger, &mut || {
        polls += 1;
        polls == 12
    })
    .expect("cancellation is not a merger failure");
    assert!(result.is_none());
    assert_eq!(polls, 12);
}

#[test]
fn merger_forward_is_deterministic() {
    let cfg = tiny_cfg();
    let weights = make_test_merger_weights(&cfg);
    let n = 8;
    let mut state = 0x1234_5678_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    let pre_merger: Vec<f32> = (0..n * cfg.hidden_size).map(|_| next()).collect();

    let out1 = qwen35_merger_forward(&weights, &cfg, &pre_merger).expect("forward 1");
    let out2 = qwen35_merger_forward(&weights, &cfg, &pre_merger).expect("forward 2");
    assert_eq!(out1, out2);
}

/// Perturbing a single loaded merger weight must change the output --
/// proves this forward pass is actually sensitive to the weights it's
/// handed (the same mutation-sensitivity property the committed golden
/// gate in `tests/vision_s4_merger_test.rs` relies on).
#[test]
fn merger_forward_is_sensitive_to_weight_mutation() {
    let cfg = tiny_cfg();
    let mut weights = make_test_merger_weights(&cfg);
    let n = 8;
    let pre_merger = vec![0.05f32; n * cfg.hidden_size];

    let baseline = qwen35_merger_forward(&weights, &cfg, &pre_merger).expect("forward");
    weights.fc1_weight[0] += 5.0;
    let mutated = qwen35_merger_forward(&weights, &cfg, &pre_merger).expect("forward");

    assert_ne!(
        baseline, mutated,
        "weight mutation had no effect on merger output"
    );
}
