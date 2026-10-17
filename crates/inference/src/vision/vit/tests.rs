use super::*;
use crate::vision::config::VisionConfig;

/// Tiny config for fast unit tests: 4 patches, d_model=8, 1 layer.
fn tiny_cfg() -> VisionConfig {
    let image_size = 8u32;
    let patch_size = 4u32;
    let n_patches = ((image_size / patch_size) as usize).pow(2); // 4
    let d_model = 8usize;
    let mlp_ratio = 2usize;
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
        mlp_ratio,
        use_gelu: true,
        d_decoder: 16,
        d_mlp: d_model * mlp_ratio,
    }
}

#[test]
fn vit_forward_output_shape() {
    let cfg = tiny_cfg();
    let vit = make_test_vit(&cfg);
    let n_patches = cfg.n_patches;
    let patch_len = (cfg.patch_size as usize).pow(2) * 3;
    let img = crate::vision::preprocess::ImageTensor {
        patches: vec![0.1f32; n_patches * patch_len],
        n_patches,
        patch_hw: cfg.patch_size as usize,
    };
    let out = vit.forward(&img).expect("ViT forward");
    assert_eq!(out.len(), n_patches * cfg.d_model);
}

#[test]
fn vit_forward_values_are_finite() {
    let cfg = tiny_cfg();
    let vit = make_test_vit(&cfg);
    let n_patches = cfg.n_patches;
    let patch_len = (cfg.patch_size as usize).pow(2) * 3;
    let img = crate::vision::preprocess::ImageTensor {
        patches: vec![0.5f32; n_patches * patch_len],
        n_patches,
        patch_hw: cfg.patch_size as usize,
    };
    let out = vit.forward(&img).expect("ViT forward");
    for &v in &out {
        assert!(v.is_finite(), "ViT output contained non-finite: {v}");
    }
}

// ADR-080 C1 / #741: `softmax_inplace` previously had NO fail-closed guard
// at all — a NaN or `+inf` logit divided every element by a NaN/Inf `sum`,
// propagating NaN through the whole row instead of zeroing it. RED before
// the fix: these assertions failed with a NaN-populated row.
#[test]
fn softmax_inplace_nan_fails_closed() {
    let mut x = vec![1.0f32, f32::NAN, 2.0, 0.5];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_pos_inf_fails_closed() {
    let mut x = vec![1.0f32, f32::INFINITY, 2.0];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_all_neg_inf_fails_closed() {
    let mut x = vec![f32::NEG_INFINITY; 4];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_normal_row_still_normalizes() {
    let mut x = vec![1.0f32, 2.0, 3.0];
    softmax_inplace(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "row must sum to ~1: {x:?}");
    assert!(x.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn vit_construction_wrong_block_count() {
    let cfg = tiny_cfg(); // n_layers = 1
    let d = cfg.d_model;
    let patch_len = (cfg.patch_size as usize).pow(2) * 3;
    let d_mlp = cfg.d_mlp;

    let dummy_block = ViTBlockWeights {
        ln1_weight: vec![1.0f32; d],
        ln1_bias: vec![0.0f32; d],
        attn: AttentionWeights {
            q_proj: vec![0.0f32; d * d],
            k_proj: vec![0.0f32; d * d],
            v_proj: vec![0.0f32; d * d],
            o_proj: vec![0.0f32; d * d],
            q_norm_weight: vec![1.0f32; d],
            k_norm_weight: vec![1.0f32; d],
        },
        ln2_weight: vec![1.0f32; d],
        ln2_bias: vec![0.0f32; d],
        mlp: MlpWeights {
            gate_proj: vec![0.0f32; d_mlp * d],
            up_proj: vec![0.0f32; d_mlp * d],
            down_proj: vec![0.0f32; d * d_mlp],
        },
    };

    // Providing 2 blocks when n_layers=1 should fail
    let weights = VisionWeights {
        patch_embed_weight: vec![0.0f32; d * patch_len],
        patch_embed_bias: vec![0.0f32; d],
        norm_weight: vec![1.0f32; d],
        norm_bias: vec![0.0f32; d],
        blocks: vec![dummy_block.clone(), dummy_block],
    };

    let err = ViT::new(weights, cfg);
    assert!(err.is_err());
}

#[test]
fn vit_forward_shape_mismatch_rejected() {
    let cfg = tiny_cfg();
    let vit = make_test_vit(&cfg);
    // Wrong n_patches (cfg expects 4, we provide 9)
    let patch_len = (cfg.patch_size as usize).pow(2) * 3;
    let img = crate::vision::preprocess::ImageTensor {
        patches: vec![0.0f32; 9 * patch_len],
        n_patches: 9,
        patch_hw: cfg.patch_size as usize,
    };
    assert!(vit.forward(&img).is_err());
}

// ADR-080 C4 (#796 round 1 finding 3): `matvec`/`batch_matvec` were unregistered safe CPU
// GEMV helpers with exact-match `assert_eq!` shape checks (no overflow-first `checked_mul`
// guard). Now routed through the shared `validate_gemm_nn`/`validate_gemm_bt`.

#[test]
#[should_panic(expected = "a too short for m*k")]
fn vit_matvec_rejects_short_a() {
    let a = [0.0f32; 3]; // needs rows*cols = 4
    let x = [0.0f32; 2];
    matvec(&a, &x, 2, 2);
}

#[test]
#[should_panic(expected = "b too short for k*n")]
fn vit_matvec_rejects_short_x() {
    let a = [0.0f32; 4];
    let x = [0.0f32; 1]; // needs cols = 2
    matvec(&a, &x, 2, 2);
}

#[test]
fn vit_matvec_accepts_oversized_x() {
    // Oversized-scratch-prefix allow-list: x longer than cols must not panic.
    let a = [1.0f32, 1.0, 1.0, 1.0];
    let x = [1.0f32, 1.0, 1.0, 99.0]; // cols=2, x has 4 elements
    let y = matvec(&a, &x, 2, 2);
    assert_eq!(y, vec![2.0, 2.0]);
}

#[test]
#[should_panic(expected = "a too short for m*k")]
fn vit_batch_matvec_rejects_short_x() {
    let a = [0.0f32; 4]; // rows*cols = 4
    let x = [0.0f32; 3]; // needs n*cols = 4
    batch_matvec(&a, &x, 2, 2, 2);
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn vit_batch_matvec_rejects_short_a() {
    let a = [0.0f32; 3]; // needs rows*cols = 4
    let x = [0.0f32; 4];
    batch_matvec(&a, &x, 2, 2, 2);
}

#[test]
#[should_panic(expected = "vit_batch_matvec: output shape overflow: n*rows")]
fn vit_batch_matvec_rejects_output_overflow() {
    let a = [0.0f32; 4];
    let x = [0.0f32; 4];
    batch_matvec(&a, &x, usize::MAX, 2, 2);
}
