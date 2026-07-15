//! Real Qwen3.5-0.8B patch-merger / projection stage (ADR-069 S4): maps the
//! S3a pre-merger hidden states `[num_patches, hidden_size]` to the
//! post-merger projected visual embeddings `[num_patches / spatial_merge_size^2,
//! out_hidden_size]` — the exact tensor injected into the decoder's residual
//! stream at image-token positions.
//!
//! Verified against the pinned HF reference implementation
//! (`transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionPatchMerger`,
//! `transformers==5.12.1`, same pin as S3a): `Qwen3_5VisionModel` always
//! constructs its merger with `use_postshuffle_norm=False`, so the op order
//! is `LayerNorm(hidden_size) -> fold merge_size^2 consecutive patches into
//! one row -> Linear(merge_in, merge_in) -> GELU(exact/erf) -> Linear(merge_in,
//! out_hidden_size)`. The fold is a plain reshape with no data movement
//! because `qwen35_vit_forward`'s input patches are already produced in
//! block-major (spatial-merge-block-outer) order by
//! `preprocess_qwen35_image` — the same convention this module's S3a sibling
//! relies on for its own position/RoPE indexing.
//!
//! The merger's activation is `nn.GELU()` (exact, erf-based) — NOT
//! `ACT2FN[config.hidden_act]` ("gelu_pytorch_tanh", the tanh approximation
//! [`super::vit::gelu`] implements for the ViT block MLPs). This is a
//! hardcoded HF implementation detail independent of `hidden_act`; using the
//! tanh approximation here would be a convention mismatch, not just added
//! imprecision.
//!
//! The committed gate is `tests/vision_s4_merger_test.rs`.

use super::VisionError;
use super::checkpoint::VisualMergerWeights;
use super::vit::{batch_matvec, layer_norm};
use crate::model::qwen35_config::VisionModelConfig;

/// Error-function approximation (Abramowitz & Stegun 7.1.26, max absolute
/// error ~1.5e-7), computed in f64 for headroom above f32 output precision.
/// `std` has no stable `erf`, and exact (non-tanh-approximated) GELU is the
/// merger's HF-pinned activation (see module docs), so it is implemented
/// here rather than reusing [`super::vit::gelu`].
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))` — matches PyTorch's
/// `nn.GELU()` default (`approximate="none"`), which is what
/// `Qwen3_5VisionPatchMerger.act_fn` hardcodes.
#[inline(always)]
fn gelu_exact(x: f32) -> f32 {
    let xd = x as f64;
    (0.5 * xd * (1.0 + erf(xd / std::f64::consts::SQRT_2))) as f32
}

/// Run the real Qwen3.5 patch-merger forward pass over one image's S3a
/// pre-merger hidden states, producing the post-merger projected visual
/// embeddings `[num_patches / spatial_merge_size^2, out_hidden_size]`
/// (row-major flat) — matching HF's `Qwen3_5VisionModel.forward(...).pooler_output`
/// exactly.
///
/// # Errors
///
/// [`VisionError::ShapeMismatch`] if `pre_merger_hidden.len()` is not a
/// multiple of `hidden_size`, or the patch count is not a multiple of
/// `spatial_merge_size^2`.
pub fn qwen35_merger_forward(
    weights: &VisualMergerWeights,
    cfg: &VisionModelConfig,
    pre_merger_hidden: &[f32],
) -> Result<Vec<f32>, VisionError> {
    let hidden = cfg.hidden_size;
    if hidden == 0 || !pre_merger_hidden.len().is_multiple_of(hidden) {
        return Err(VisionError::ShapeMismatch {
            expected: 0,
            actual: pre_merger_hidden.len(),
            context: "qwen35_merger_forward: pre_merger_hidden length must be a multiple of \
                      hidden_size"
                .into(),
        });
    }
    let n = pre_merger_hidden.len() / hidden;
    let merge_sq = cfg.spatial_merge_size * cfg.spatial_merge_size;
    if merge_sq == 0 || !n.is_multiple_of(merge_sq) {
        return Err(VisionError::ShapeMismatch {
            expected: 0,
            actual: n,
            context: "qwen35_merger_forward: patch count must be a multiple of \
                      spatial_merge_size^2"
                .into(),
        });
    }

    // ---- LayerNorm(hidden_size), applied per-patch BEFORE the spatial fold
    // (use_postshuffle_norm=False in the HF reference: the norm's own
    // normalized_shape is hidden_size, not merge_in). ----
    let mut normed = pre_merger_hidden.to_vec();
    for i in 0..n {
        layer_norm(
            &mut normed[i * hidden..(i + 1) * hidden],
            &weights.norm_weight,
            &weights.norm_bias,
            1e-6,
        );
    }

    // ---- Spatial fold: `x.view(-1, merge_in)` in the HF reference. Purely a
    // reinterpretation of the flat row-major buffer -- every merge_sq
    // consecutive patch rows already sit contiguously in block-major order
    // (see module docs), so no permutation is needed here. ----
    let merge_in = merge_sq * hidden;
    let num_visual_tokens = n / merge_sq;

    // ---- linear_fc1 (merge_in -> merge_in) + exact GELU + linear_fc2
    // (merge_in -> out_hidden_size). ----
    let mut fc1_out = batch_matvec(
        &weights.fc1_weight,
        &normed,
        num_visual_tokens,
        merge_in,
        merge_in,
    );
    for i in 0..num_visual_tokens {
        for j in 0..merge_in {
            let idx = i * merge_in + j;
            fc1_out[idx] = gelu_exact(fc1_out[idx] + weights.fc1_bias[j]);
        }
    }

    let out_hidden = cfg.out_hidden_size;
    let mut out = batch_matvec(
        &weights.fc2_weight,
        &fc1_out,
        num_visual_tokens,
        out_hidden,
        merge_in,
    );
    for i in 0..num_visual_tokens {
        for j in 0..out_hidden {
            out[i * out_hidden + j] += weights.fc2_bias[j];
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
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
            in_channels: 1,
            deepstack_visual_indexes: vec![],
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
}
