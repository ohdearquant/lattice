use super::*;

fn tiny_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 8,
        temporal_patch_size: 1,
        num_position_embeddings: 16, // side = 4
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    }
}

fn make_test_png(w: u32, h: u32) -> Vec<u8> {
    use image::RgbImage;
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = ((x + y) % 256) as u8;
            img.put_pixel(x, y, image::Rgb([v, v, v]));
        }
    }
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

#[test]
fn preprocess_rejects_misaligned_image() {
    let cfg = tiny_cfg(); // factor = patch_size(2) * merge(2) = 4
    let png = make_test_png(6, 4); // 6 not a multiple of 4
    let err = preprocess_qwen35_image(&png, &cfg).unwrap_err();
    assert!(matches!(err, VisionError::InvalidConfig(_)));
}

#[test]
fn preprocess_produces_expected_shape_and_grid() {
    let cfg = tiny_cfg();
    let png = make_test_png(8, 8); // grid 4x4 patches of size 2
    let (patches, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");
    assert_eq!(grid, GridThw { t: 1, h: 4, w: 4 });
    let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    assert_eq!(patches.len(), grid.num_patches() * patch_len);
    assert!(patches.iter().all(|v| v.is_finite()));
}

#[test]
fn serving_preprocess_enforces_patch_budget_without_narrowing_embedding_path() {
    // Exercises the default budget via `preprocess_qwen35_image_inner`
    // with an explicit `Some(MAX_SERVE_VISION_PATCHES)` rather than
    // through the public `preprocess_qwen35_image_for_serve` wrapper,
    // which reads `LATTICE_VISION_MAX_PATCHES` from the real process
    // environment: a locally set override above 288 would silently
    // widen the budget this test asserts against and make the "over"
    // case below spuriously pass.
    let cfg = tiny_cfg();
    let boundary = make_test_png(32, 32); // 16 * 16 = 256 patches
    assert!(preprocess_qwen35_image_inner(&boundary, &cfg, Some(MAX_SERVE_VISION_PATCHES)).is_ok());

    let over = make_test_png(36, 32); // 18 * 16 = 288 patches
    let err =
        preprocess_qwen35_image_inner(&over, &cfg, Some(MAX_SERVE_VISION_PATCHES)).unwrap_err();
    assert!(
        matches!(err, VisionError::DimensionsExceeded(message) if message.contains("serving maximum is 256"))
    );
    assert!(
        preprocess_qwen35_image(&over, &cfg).is_ok(),
        "the serving-only latency budget must not narrow the embedding API"
    );
}

#[test]
fn serving_preprocess_rejects_declared_pixel_dimensions_over_the_hard_cap() {
    // A thin (1px tall) image keeps the PNG fixture tiny while still
    // tripping `image::Limits::check_dimensions` on width alone, before
    // any pixel data is decoded.
    let cfg = tiny_cfg();
    let over_width = make_test_png(MAX_IMAGE_DIMENSION_PIXELS + 1, 1);
    let err = preprocess_qwen35_image_for_serve(&over_width, &cfg).unwrap_err();
    assert!(
        matches!(&err, VisionError::DimensionsExceeded(message) if message.contains("2048px")),
        "expected DimensionsExceeded naming the 2048px cap, got: {err}"
    );
}

#[test]
fn resolve_serve_max_patches_falls_back_to_default_for_every_invalid_shape() {
    assert_eq!(resolve_serve_max_patches(None), MAX_SERVE_VISION_PATCHES);
    assert_eq!(
        resolve_serve_max_patches(Some("")),
        MAX_SERVE_VISION_PATCHES
    );
    assert_eq!(
        resolve_serve_max_patches(Some("not-a-number")),
        MAX_SERVE_VISION_PATCHES
    );
    assert_eq!(
        resolve_serve_max_patches(Some("0")),
        MAX_SERVE_VISION_PATCHES
    );
    assert_eq!(
        resolve_serve_max_patches(Some("-4")),
        MAX_SERVE_VISION_PATCHES
    );
}

#[test]
fn resolve_serve_max_patches_honors_a_positive_override() {
    assert_eq!(resolve_serve_max_patches(Some("64")), 64);
    assert_eq!(resolve_serve_max_patches(Some(" 64 ")), 64);
}

#[test]
fn serving_preprocess_inner_honors_a_lower_override_below_the_default_boundary() {
    let cfg = tiny_cfg();
    let boundary = make_test_png(32, 32); // 16 * 16 = 256 patches, accepted at the default
    assert!(preprocess_qwen35_image_inner(&boundary, &cfg, Some(256)).is_ok());
    let err = preprocess_qwen35_image_inner(&boundary, &cfg, Some(200)).unwrap_err();
    assert!(
        matches!(err, VisionError::DimensionsExceeded(message) if message.contains("serving maximum is 200"))
    );
}

#[test]
fn serving_preprocess_caps_output_allocation_for_pathological_config() {
    let mut cfg = tiny_cfg();
    cfg.patch_size = 8;
    cfg.spatial_merge_size = 1;
    cfg.temporal_patch_size = 128;
    let image = make_test_png(128, 128);
    let err = preprocess_qwen35_image_for_serve(&image, &cfg).unwrap_err();
    assert!(
        matches!(err, VisionError::DimensionsExceeded(message) if message.contains("preprocessing requires"))
    );
}

fn make_test_weights(cfg: &VisionModelConfig) -> Qwen35VisionWeights {
    use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
    let hidden = cfg.hidden_size;
    let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let mlp_dim = 2 * hidden;
    let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;

    // Deterministic pseudo-random weights (not all-zero/identity) so the
    // forward pass actually exercises every op, via a tiny xorshift LCG.
    let mut state = 0x1234_5678_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    let mut v = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();

    let block = VisualBlockWeights {
        qkv_weight: v(3 * hidden * hidden),
        qkv_bias: v(3 * hidden),
        proj_weight: v(hidden * hidden),
        proj_bias: v(hidden),
        fc1_weight: v(mlp_dim * hidden),
        fc1_bias: v(mlp_dim),
        fc2_weight: v(hidden * mlp_dim),
        fc2_bias: v(hidden),
        norm1_weight: vec![1.0; hidden],
        norm1_bias: vec![0.0; hidden],
        norm2_weight: vec![1.0; hidden],
        norm2_bias: vec![0.0; hidden],
    };

    Qwen35VisionWeights {
        patch_embed_weight: v(hidden * patch_len),
        patch_embed_weight_shape: vec![
            hidden,
            cfg.in_channels,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        ],
        patch_embed_bias: v(hidden),
        pos_embed: v(cfg.num_position_embeddings * hidden),
        blocks: vec![block],
        merger: VisualMergerWeights {
            fc1_weight: v(merge_in * merge_in),
            fc1_bias: v(merge_in),
            fc2_weight: v(cfg.out_hidden_size * merge_in),
            fc2_bias: v(cfg.out_hidden_size),
            norm_weight: vec![1.0; hidden],
            norm_bias: vec![0.0; hidden],
        },
    }
}

#[test]
fn vit_forward_output_shape_and_finite() {
    let cfg = tiny_cfg();
    let weights = make_test_weights(&cfg);
    let png = make_test_png(8, 8);
    let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

    let out = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward");
    assert_eq!(out.len(), grid.num_patches() * cfg.hidden_size);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn vit_forward_rejects_pixel_length_mismatch() {
    let cfg = tiny_cfg();
    let weights = make_test_weights(&cfg);
    let grid = GridThw { t: 1, h: 4, w: 4 };
    let bad_pixels = vec![0.0f32; 3]; // way too short
    let err = qwen35_vit_forward(&weights, &cfg, &bad_pixels, grid).unwrap_err();
    assert!(matches!(err, VisionError::ShapeMismatch { .. }));
}

#[test]
fn vit_forward_is_deterministic() {
    let cfg = tiny_cfg();
    let weights = make_test_weights(&cfg);
    let png = make_test_png(8, 8);
    let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

    let out1 = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward 1");
    let out2 = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward 2");
    assert_eq!(out1, out2);
}

/// Perturbing a single loaded weight must change the output — proves this
/// forward pass is actually sensitive to the weights it's handed (the
/// same mutation-sensitivity property the committed golden gate in
/// `tests/vision_s3_vit_forward_test.rs` relies on).
#[test]
fn vit_forward_is_sensitive_to_weight_mutation() {
    let cfg = tiny_cfg();
    let mut weights = make_test_weights(&cfg);
    let png = make_test_png(8, 8);
    let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

    let baseline = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward");
    weights.blocks[0].qkv_weight[0] += 5.0;
    let mutated = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward");

    assert_ne!(
        baseline, mutated,
        "weight mutation had no effect on ViT output"
    );
}
