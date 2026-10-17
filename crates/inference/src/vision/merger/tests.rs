use super::*;

/// Build a minimal merger with identity-like weights for testing.
fn test_merger(d_vit: usize, d_hidden: usize, d_model: usize, merge: usize) -> MlpMerger {
    let d_in = d_vit * merge * merge;
    // w1: identity-ish [d_hidden, d_in] (zeros — MLP is identity-zero)
    let w1 = vec![0.0f32; d_hidden * d_in];
    let b1 = vec![0.0f32; d_hidden];
    let w2 = vec![0.0f32; d_model * d_hidden];
    let b2 = vec![0.0f32; d_model];
    let weights = MlpMergerWeights { w1, b1, w2, b2 };
    MlpMerger::new(weights, d_vit, d_hidden, d_model, merge).expect("test merger")
}

#[test]
fn merger_output_shape_4_patches_merge2() {
    // 4 raw patches (2×2 grid), merge_size=2 → 1 merged patch
    let d_vit = 8usize;
    let d_hidden = 16usize;
    let d_model = 4usize;
    let merger = test_merger(d_vit, d_hidden, d_model, 2);

    let vit_out = vec![1.0f32; 4 * d_vit];
    let out = merger.merge_and_project(&vit_out, 4).expect("merge");
    // 4 / 4 = 1 merged patch
    assert_eq!(out.len(), d_model);
}

#[test]
fn merger_output_shape_784_patches_merge2() {
    // 784 raw patches (28×28), merge=2 → 196 merged patches (14×14)
    let d_vit = 4usize;
    let d_hidden = 8usize;
    let d_model = 6usize;
    let merger = test_merger(d_vit, d_hidden, d_model, 2);

    let vit_out = vec![0.5f32; 784 * d_vit];
    let out = merger.merge_and_project(&vit_out, 784).expect("merge");
    assert_eq!(out.len(), 196 * d_model);
}

#[test]
fn merger_rejects_non_square_patch_count() {
    let merger = test_merger(4, 8, 6, 2);
    // 6 patches is not a perfect square
    let vit_out = vec![0.0f32; 6 * 4];
    let result = merger.merge_and_project(&vit_out, 6);
    assert!(result.is_err());
}

#[test]
fn merger_rejects_non_divisible_patches() {
    // merge_size=2 → need multiple of 4; use 9 (not divisible by 4)
    let merger = test_merger(4, 8, 6, 2);
    let vit_out = vec![0.0f32; 9 * 4];
    let result = merger.merge_and_project(&vit_out, 9);
    assert!(result.is_err());
}

#[test]
fn merger_rejects_wrong_vit_out_len() {
    let merger = test_merger(4, 8, 6, 2);
    // raw_patches=4 but provide wrong length
    let vit_out = vec![0.0f32; 3 * 4]; // should be 4 * 4 = 16
    let result = merger.merge_and_project(&vit_out, 4);
    assert!(result.is_err());
}

#[test]
fn merger_output_is_finite() {
    let d_vit = 8usize;
    let d_hidden = 16usize;
    let d_model = 6usize;
    let merger = test_merger(d_vit, d_hidden, d_model, 2);
    let vit_out: Vec<f32> = (0..4 * d_vit).map(|i| (i as f32) * 0.01).collect();
    let out = merger.merge_and_project(&vit_out, 4).expect("merge");
    for &v in &out {
        assert!(v.is_finite(), "merger output non-finite: {v}");
    }
}

#[test]
fn merger_construction_wrong_w1_size() {
    let d_vit = 4usize;
    let d_hidden = 8usize;
    let d_model = 6usize;
    let merge = 2usize;
    let d_in = d_vit * merge * merge; // 16

    // Provide wrong w1 size
    let weights = MlpMergerWeights {
        w1: vec![0.0f32; d_hidden * d_in + 1], // off by 1
        b1: vec![0.0f32; d_hidden],
        w2: vec![0.0f32; d_model * d_hidden],
        b2: vec![0.0f32; d_model],
    };
    let result = MlpMerger::new(weights, d_vit, d_hidden, d_model, merge);
    assert!(result.is_err());
}
