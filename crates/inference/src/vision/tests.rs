use super::*;
use merger::MlpMergerWeights;
use vit::{AttentionWeights, MlpWeights, ViTBlockWeights};

/// Build a test `VisionEncoder` with zero/identity weights for shape testing.
fn test_encoder(cfg: VisionConfig) -> VisionEncoder {
    let d = cfg.d_model;
    let patch_len = (cfg.patch_size as usize).pow(2) * 3;
    let d_mlp_vit = cfg.d_mlp;
    let d_decoder = cfg.d_decoder;
    let merge = cfg.spatial_merge_size;

    let make_block = || ViTBlockWeights {
        ln1_weight: vec![1.0f32; d],
        ln1_bias: vec![0.0f32; d],
        attn: AttentionWeights {
            q_proj: identity_w(d),
            k_proj: identity_w(d),
            v_proj: identity_w(d),
            o_proj: identity_w(d),
            q_norm_weight: vec![1.0f32; d],
            k_norm_weight: vec![1.0f32; d],
        },
        ln2_weight: vec![1.0f32; d],
        ln2_bias: vec![0.0f32; d],
        mlp: MlpWeights {
            gate_proj: vec![0.0f32; d_mlp_vit * d],
            up_proj: vec![0.0f32; d_mlp_vit * d],
            down_proj: vec![0.0f32; d * d_mlp_vit],
        },
    };

    let vit_weights = vit::VisionWeights {
        patch_embed_weight: rect_identity(d, patch_len),
        patch_embed_bias: vec![0.0f32; d],
        norm_weight: vec![1.0f32; d],
        norm_bias: vec![0.0f32; d],
        blocks: (0..cfg.n_layers).map(|_| make_block()).collect(),
    };

    let d_in_mlp = d * merge * merge;
    let d_hidden_mlp = d * 2;
    let mlp_weights = MlpMergerWeights {
        w1: vec![0.0f32; d_hidden_mlp * d_in_mlp],
        b1: vec![0.0f32; d_hidden_mlp],
        w2: vec![0.0f32; d_decoder * d_hidden_mlp],
        b2: vec![0.0f32; d_decoder],
    };

    VisionEncoder::new(vit_weights, mlp_weights, cfg, d_hidden_mlp)
        .expect("test encoder construction")
}

fn identity_w(n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        w[i * n + i] = 1.0;
    }
    w
}

fn rect_identity(rows: usize, cols: usize) -> Vec<f32> {
    let min_dim = rows.min(cols);
    let mut w = vec![0.0f32; rows * cols];
    for i in 0..min_dim {
        w[i * cols + i] = 1.0;
    }
    w
}

fn tiny_cfg() -> VisionConfig {
    let image_size = 8u32;
    let patch_size = 4u32;
    let n_patches = ((image_size / patch_size) as usize).pow(2); // 4
    let d_model = 8usize;
    VisionConfig {
        image_size,
        patch_size,
        n_patches,
        d_model,
        n_heads: 2,
        n_layers: 1,
        spatial_merge_size: 2,
        global_attn_every: 1,
        window_size: 2,
        mlp_ratio: 2,
        use_gelu: true,
        d_decoder: 16,
        d_mlp: d_model * 2,
    }
}

fn make_test_png(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    use image::RgbImage;
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

#[test]
fn vision_encoder_output_shape() {
    let cfg = tiny_cfg();
    let enc = test_encoder(cfg.clone());
    let png = make_test_png(16, 16, 100, 150, 200);
    let out = enc.encode(&png).expect("encode");
    assert_eq!(out.raw_patches, cfg.n_patches);
    assert_eq!(out.visual_tokens, cfg.visual_tokens());
    assert_eq!(out.d_model, cfg.d_decoder);
    assert_eq!(
        out.patch_embeddings.len(),
        cfg.visual_tokens() * cfg.d_decoder
    );
}

#[test]
fn vision_encoder_output_finite() {
    let cfg = tiny_cfg();
    let enc = test_encoder(cfg);
    let png = make_test_png(16, 16, 50, 100, 150);
    let out = enc.encode(&png).expect("encode");
    for &v in &out.patch_embeddings {
        assert!(v.is_finite(), "VisionOutput contains non-finite: {v}");
    }
}

#[test]
fn vision_encoder_bad_image_rejected() {
    let cfg = tiny_cfg();
    let enc = test_encoder(cfg);
    let result = enc.encode(b"not an image");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), VisionError::ImageDecode(_)));
}

#[test]
fn multimodal_input_construction_from_vision_output() {
    let cfg = tiny_cfg();
    let enc = test_encoder(cfg.clone());
    let png = make_test_png(16, 16, 0, 0, 0);
    let out = enc.encode(&png).expect("encode");

    let mm = MultimodalInput {
        patch_embeddings: out.patch_embeddings.clone(),
        raw_patches: out.raw_patches,
        visual_tokens: out.visual_tokens,
        d_model: out.d_model,
        text_tokens: vec![100, 200, 300],
    };
    mm.validate().expect("MultimodalInput valid");
    assert_eq!(mm.total_sequence_len(), out.visual_tokens + 3);
}

#[test]
fn vision_error_display_coverage() {
    let e1 = VisionError::ImageDecode("bad bytes".into());
    assert!(e1.to_string().contains("bad bytes"));

    let e2 = VisionError::ShapeMismatch {
        expected: 10,
        actual: 5,
        context: "test ctx".into(),
    };
    assert!(e2.to_string().contains("10"));
    assert!(e2.to_string().contains("5"));

    let e3 = VisionError::InvalidConfig("zero dim".into());
    assert!(e3.to_string().contains("zero dim"));
}
