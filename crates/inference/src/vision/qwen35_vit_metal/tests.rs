use super::*;
use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
use crate::vision::qwen35_vit::{preprocess_qwen35_image, qwen35_vit_forward};

fn tiny_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 8,
        temporal_patch_size: 1,
        num_position_embeddings: 16,
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

fn make_test_weights(cfg: &VisionModelConfig) -> Qwen35VisionWeights {
    let hidden = cfg.hidden_size;
    let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let mlp_dim = 2 * hidden;
    let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;

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

/// Below-GPU-dispatch-threshold shapes exercise the CPU-fallback path in
/// `gemm_bt`/`gemm_nn` on every platform (no Metal device required),
/// proving the fallback math matches the CPU reference exactly.
#[test]
fn metal_forward_matches_cpu_reference_small_shapes() {
    let cfg = tiny_cfg();
    let weights = make_test_weights(&cfg);
    let png = make_test_png(8, 8);
    let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

    let cpu_out = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("cpu forward");
    let metal_out =
        qwen35_vit_forward_metal(&weights, &cfg, &pixel_values, grid).expect("metal forward");
    let mut never_cancel = || false;
    let observed = qwen35_vit_forward_metal_with_cancel(
        &weights,
        &cfg,
        &pixel_values,
        grid,
        &mut never_cancel,
    )
    .expect("observed Metal forward")
    .expect("no cancellation requested");

    assert_eq!(cpu_out.len(), metal_out.len());
    assert_eq!(observed.hidden_states, metal_out);
    assert_eq!(observed.gemm_calls, 9);
    assert_eq!(
        observed.metal_dispatches, 0,
        "tiny shapes must exercise the documented CPU fallback"
    );
    for (a, b) in cpu_out.iter().zip(metal_out.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "cpu={a} metal={b} diverge beyond fallback-path tolerance"
        );
    }
}

#[test]
fn metal_forward_cancels_after_work_started() {
    let cfg = tiny_cfg();
    let weights = make_test_weights(&cfg);
    let png = make_test_png(8, 8);
    let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");
    let mut polls = 0;
    let result =
        qwen35_vit_forward_metal_with_cancel(&weights, &cfg, &pixel_values, grid, &mut || {
            polls += 1;
            polls == 90
        })
        .expect("cancellation is not a vision failure");
    assert!(result.is_none());
    assert_eq!(polls, 90);
}

#[test]
fn metal_forward_rejects_pixel_length_mismatch() {
    let cfg = tiny_cfg();
    let weights = make_test_weights(&cfg);
    let grid = GridThw { t: 1, h: 4, w: 4 };
    let bad_pixels = vec![0.0f32; 3];
    let err = qwen35_vit_forward_metal(&weights, &cfg, &bad_pixels, grid).unwrap_err();
    assert!(matches!(err, VisionError::ShapeMismatch { .. }));
}
