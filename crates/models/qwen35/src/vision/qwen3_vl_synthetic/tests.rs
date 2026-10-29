use super::*;
use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};

fn weights() -> Qwen35VisionWeights {
    let block = VisualBlockWeights {
        qkv_weight: vec![0.0; 3 * HIDDEN * HIDDEN],
        qkv_bias: vec![0.0; 3 * HIDDEN],
        proj_weight: vec![0.0; HIDDEN * HIDDEN],
        proj_bias: vec![0.0; HIDDEN],
        fc1_weight: vec![0.0; FFN * HIDDEN],
        fc1_bias: vec![0.0; FFN],
        fc2_weight: vec![0.0; HIDDEN * FFN],
        fc2_bias: vec![0.0; HIDDEN],
        norm1_weight: vec![1.0; HIDDEN],
        norm1_bias: vec![0.0; HIDDEN],
        norm2_weight: vec![1.0; HIDDEN],
        norm2_bias: vec![0.0; HIDDEN],
    };
    let mut merger = VisualMergerWeights {
        fc1_weight: vec![0.0; 16 * HIDDEN * HIDDEN],
        fc1_bias: vec![0.0; 4 * HIDDEN],
        fc2_weight: vec![0.0; OUTPUT * 4 * HIDDEN],
        fc2_bias: vec![0.0; OUTPUT],
        norm_weight: vec![1.0; HIDDEN],
        norm_bias: vec![0.0; HIDDEN],
    };
    merger.fc1_bias[0] = 1.0;
    merger.fc2_weight[0] = 1.0;
    merger.fc2_bias[0] = 0.2;
    let mut pos = Vec::with_capacity(POSITIONS * HIDDEN);
    for _ in 0..POSITIONS {
        pos.extend((0..HIDDEN).map(|i| i as f32 * 0.1));
    }
    Qwen35VisionWeights {
        patch_embed_weight: vec![0.0; HIDDEN * PATCH_LEN],
        patch_embed_weight_shape: vec![HIDDEN, 3, 2, PATCH, PATCH],
        patch_embed_bias: vec![0.0; HIDDEN],
        pos_embed: pos,
        blocks: vec![block],
        merger,
    }
}
fn image() -> Vec<f32> {
    (0..3 * SIDE * SIDE)
        .map(|i| ((i % 251) as f32 / 125.0) - 1.0)
        .collect()
}
#[test]
fn normalized_still_image_is_packed_in_merge_block_order() {
    let input = image();
    let out = run(&input, &weights()).unwrap();
    assert_eq!(out.patches.len(), COUNT * PATCH_LEN);
    for (row, py, px) in [
        (0, 0, 0),
        (1, 0, 16),
        (2, 16, 0),
        (3, 16, 16),
        (4, 0, 32),
        (195, 208, 208),
    ] {
        for c in 0..3 {
            for t in 0..2 {
                for (dy, dx) in [(0, 0), (7, 9), (15, 15)] {
                    let k = (((c * 2 + t) * PATCH + dy) * PATCH) + dx;
                    assert_eq!(
                        out.patches[row * PATCH_LEN + k],
                        input[c * SIDE * SIDE + (py + dy) * SIDE + px + dx]
                    );
                }
            }
        }
    }
}
#[test]
fn block_and_merger_preserve_position_and_apply_exact_gelu() {
    let out = run(&image(), &weights()).unwrap();
    assert_eq!(out.patch_embedding.len(), COUNT * HIDDEN);
    assert!(out.patch_embedding.iter().all(|v| *v == 0.0));
    assert_eq!(out.block.len(), COUNT * HIDDEN);
    for row in 0..COUNT {
        for h in 0..HIDDEN {
            assert!((out.block[row * HIDDEN + h] - h as f32 * 0.1).abs() < 1e-6);
        }
    }
    assert_eq!(out.merged.len(), 49 * OUTPUT);
    for row in out.merged.chunks_exact(OUTPUT) {
        assert!((row[0] - 1.041_344_8).abs() < 1e-6);
        assert!(row[1..].iter().all(|v| *v == 0.0));
    }
}
#[test]
fn rejects_invalid_image_and_weight_geometry_or_values() {
    let input = image();
    let w = weights();
    assert!(run(&input[..input.len() - 1], &w).is_err());
    let mut invalid_image = input.clone();
    invalid_image[10] = f32::NAN;
    assert!(run(&invalid_image, &w).is_err());
    invalid_image[10] = 1.1;
    assert!(run(&invalid_image, &w).is_err());
    let mut bad = weights();
    bad.blocks[0].qkv_weight.pop();
    assert!(run(&input, &bad).is_err());
    let mut bad = weights();
    bad.merger.fc1_weight[0] = f32::INFINITY;
    assert!(run(&input, &bad).is_err());
    let mut bad = weights();
    bad.blocks.clear();
    assert!(run(&input, &bad).is_err());
    let mut bad = weights();
    bad.patch_embed_weight_shape[2] = 1;
    assert!(run(&input, &bad).is_err());
}
