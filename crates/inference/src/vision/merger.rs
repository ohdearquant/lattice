//! MLP merger: spatial pooling + projection from ViT dim → decoder dim.
//!
//! The Qwen3-VL merger groups `spatial_merge_size^2` spatially adjacent patches,
//! concatenates their ViT feature vectors, then projects the concatenated vector
//! through a two-layer MLP to the decoder hidden dimension.
//!
//! ## Spatial layout
//!
//! Patches are in raster order: patch (py, px) has index `py * patches_per_side + px`.
//! For `spatial_merge_size=2` and `patches_per_side=28` (448/16):
//! - Patch group (gy, gx) combines indices:
//!   `[(2gy)*28 + (2gx), (2gy)*28 + (2gx+1), (2gy+1)*28 + (2gx), (2gy+1)*28 + (2gx+1)]`
//! - Total groups: 14 × 14 = 196.
//!
//! ## MLP architecture
//!
//! Input: concatenated `[spatial_merge_size^2 * d_vit]` vector.
//! Layer 1: Linear(d_vit * merge^2 → d_hidden) + GELU
//! Layer 2: Linear(d_hidden → d_model)
//!
//! No bias is stored for the MLP in v0; callers can set bias to `vec![0.0]` if the
//! checkpoint has no bias, or provide the actual bias from the checkpoint.

use super::VisionError;

/// Weights for the two-layer MLP merger.
#[derive(Debug, Clone)]
pub struct MlpMergerWeights {
    /// First linear weight: [d_hidden, d_in] where d_in = d_vit * merge_size^2
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// Second linear weight: [d_model, d_hidden]
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

/// MLP merger that combines spatially adjacent ViT patches and projects to decoder dim.
pub struct MlpMerger {
    weights: MlpMergerWeights,
    /// d_vit: ViT hidden dimension (before merge)
    d_vit: usize,
    /// d_hidden: intermediate MLP dimension
    d_hidden: usize,
    /// d_model: output / decoder hidden dimension
    d_model: usize,
    /// spatial_merge_size: number of patches merged per side (total = merge^2)
    merge_size: usize,
}

impl MlpMerger {
    /// Construct the merger.
    ///
    /// - `d_vit`: ViT hidden dimension
    /// - `d_hidden`: MLP intermediate dimension
    /// - `d_model`: decoder hidden dimension (final output)
    /// - `merge_size`: spatial merge factor (Qwen3-VL: 2)
    pub fn new(
        weights: MlpMergerWeights,
        d_vit: usize,
        d_hidden: usize,
        d_model: usize,
        merge_size: usize,
    ) -> Result<Self, VisionError> {
        if merge_size == 0 {
            return Err(VisionError::InvalidConfig("merge_size must be > 0".into()));
        }
        let d_in = d_vit * merge_size * merge_size;
        let expected_w1 = d_hidden * d_in;
        let expected_b1 = d_hidden;
        let expected_w2 = d_model * d_hidden;
        let expected_b2 = d_model;

        if weights.w1.len() != expected_w1 {
            return Err(VisionError::ShapeMismatch {
                expected: expected_w1,
                actual: weights.w1.len(),
                context: "MlpMerger w1".into(),
            });
        }
        if weights.b1.len() != expected_b1 {
            return Err(VisionError::ShapeMismatch {
                expected: expected_b1,
                actual: weights.b1.len(),
                context: "MlpMerger b1".into(),
            });
        }
        if weights.w2.len() != expected_w2 {
            return Err(VisionError::ShapeMismatch {
                expected: expected_w2,
                actual: weights.w2.len(),
                context: "MlpMerger w2".into(),
            });
        }
        if weights.b2.len() != expected_b2 {
            return Err(VisionError::ShapeMismatch {
                expected: expected_b2,
                actual: weights.b2.len(),
                context: "MlpMerger b2".into(),
            });
        }

        Ok(Self {
            weights,
            d_vit,
            d_hidden,
            d_model,
            merge_size,
        })
    }

    /// Merge and project ViT output to decoder hidden dimension.
    ///
    /// `vit_out`: flat f32 slice `[raw_patches, d_vit]` in raster order.
    ///
    /// Returns `Vec<f32>` of shape `[merged_patches, d_model]` where
    /// `merged_patches = raw_patches / merge_size^2`.
    ///
    /// # Errors
    ///
    /// Returns `ShapeMismatch` if `raw_patches` is not divisible by `merge_size^2`,
    /// or if `vit_out.len()` is inconsistent with `raw_patches * d_vit`.
    pub fn merge_and_project(
        &self,
        vit_out: &[f32],
        raw_patches: usize,
    ) -> Result<Vec<f32>, VisionError> {
        let merge_sq = self.merge_size * self.merge_size;
        if raw_patches % merge_sq != 0 {
            return Err(VisionError::ShapeMismatch {
                expected: 0, // divisible by merge_sq
                actual: raw_patches % merge_sq,
                context: format!(
                    "raw_patches {raw_patches} must be divisible by merge_size^2={merge_sq}"
                ),
            });
        }
        let expected_len = raw_patches * self.d_vit;
        if vit_out.len() != expected_len {
            return Err(VisionError::ShapeMismatch {
                expected: expected_len,
                actual: vit_out.len(),
                context: "vit_out length".into(),
            });
        }

        let merged_patches = raw_patches / merge_sq;
        let patches_per_side = (raw_patches as f64).sqrt() as usize;
        // Verify square grid
        if patches_per_side * patches_per_side != raw_patches {
            return Err(VisionError::InvalidConfig(format!(
                "raw_patches {raw_patches} must be a perfect square for spatial merge"
            )));
        }
        let merged_per_side = patches_per_side / self.merge_size;
        let d_in = self.d_vit * merge_sq;

        let mut output = vec![0.0f32; merged_patches * self.d_model];

        for gy in 0..merged_per_side {
            for gx in 0..merged_per_side {
                let group_idx = gy * merged_per_side + gx;

                // Concatenate merge_size^2 patch vectors into a single d_in vector.
                let mut concat = vec![0.0f32; d_in];
                let mut concat_pos = 0usize;
                for dy in 0..self.merge_size {
                    for dx in 0..self.merge_size {
                        let py = gy * self.merge_size + dy;
                        let px = gx * self.merge_size + dx;
                        let patch_idx = py * patches_per_side + px;
                        let src = &vit_out[patch_idx * self.d_vit..(patch_idx + 1) * self.d_vit];
                        concat[concat_pos..concat_pos + self.d_vit].copy_from_slice(src);
                        concat_pos += self.d_vit;
                    }
                }

                // Layer 1: Linear + GELU
                let mut h1 = vec![0.0f32; self.d_hidden];
                for r in 0..self.d_hidden {
                    let row = &self.weights.w1[r * d_in..(r + 1) * d_in];
                    let mut acc = self.weights.b1[r];
                    for (a, c) in row.iter().zip(concat.iter()) {
                        acc += a * c;
                    }
                    h1[r] = gelu(acc);
                }

                // Layer 2: Linear (no activation)
                let out_slice =
                    &mut output[group_idx * self.d_model..(group_idx + 1) * self.d_model];
                for r in 0..self.d_model {
                    let row = &self.weights.w2[r * self.d_hidden..(r + 1) * self.d_hidden];
                    let mut acc = self.weights.b2[r];
                    for (a, h) in row.iter().zip(h1.iter()) {
                        acc += a * h;
                    }
                    out_slice[r] = acc;
                }
            }
        }

        Ok(output)
    }
}

/// GELU activation (tanh approximation, matches PyTorch default).
#[inline(always)]
fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
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
}
