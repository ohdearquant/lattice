//! Block influence scoring for layer pruning (ADR-060 Phase 1).
//!
//! Implements the ShortGPT angular distance metric: for each transformer
//! layer, measure cos(θ) between the input and output hidden states.
//! Layers where θ ≈ 0 (input ≈ output) contribute little and are pruning
//! candidates.
//!
//! Reference: Men et al., "ShortGPT: Layers in Large Language Models are
//! More Redundant Than You Expect" (arXiv:2403.03853).

use crate::metrics::{LayerMetrics, l2_norm};

/// Angular distance between two vectors, returned as cosine similarity.
///
/// cos(θ) = (a · b) / (‖a‖ · ‖b‖)
///
/// Returns `1.0` (identical direction) when either vector is zero,
/// matching ShortGPT's convention that a zero-output layer is "no change."
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        return 1.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Per-layer block influence score.
#[derive(Debug, Clone)]
pub struct BlockInfluence {
    /// Layer index.
    pub layer_idx: usize,
    /// Cosine similarity between layer input and output hidden states.
    /// Values near 1.0 mean the layer barely transforms its input.
    pub cosine_sim: f32,
    /// Angular distance in radians: θ = arccos(cosine_sim).
    /// Values near 0.0 mean low influence (pruning candidate).
    pub angular_distance: f32,
    /// Block influence score: 1.0 - cos(θ).
    /// Higher = more influential (less prunable).
    pub influence: f32,
}

impl BlockInfluence {
    /// Compute block influence from raw hidden state vectors.
    pub fn from_vectors(layer_idx: usize, input: &[f32], output: &[f32]) -> Self {
        let cosine_sim = cosine_similarity(input, output);
        let angular_distance = cosine_sim.acos();
        let influence = 1.0 - cosine_sim;
        Self {
            layer_idx,
            cosine_sim,
            angular_distance,
            influence,
        }
    }
}

/// Compute block influence scores for all layers from pre-collected norms.
///
/// This variant works with `LayerMetrics` that have `input_norm` and
/// `output_norm` already computed. It estimates the angular distance
/// using a simplified bound: if the residual ‖output - input‖ is small
/// relative to the norms, the angle is small.
///
/// For exact scoring, use [`score_from_hidden_states`] with full vectors.
pub fn score_from_norms(layers: &[LayerMetrics]) -> Vec<BlockInfluence> {
    layers
        .iter()
        .map(|lm| {
            let norm_product = lm.input_norm * lm.output_norm;
            let cosine_sim = if norm_product < f32::EPSILON {
                1.0
            } else {
                // Upper bound: cos(θ) ≥ 1 - ‖Δ‖²/(2·‖a‖·‖b‖)
                // Without the actual dot product, we use the norm ratio as a proxy.
                // ratio near 1.0 with both norms large → likely small angle.
                let ratio = lm.input_norm.min(lm.output_norm) / lm.input_norm.max(lm.output_norm);
                ratio.clamp(0.0, 1.0)
            };
            let angular_distance = cosine_sim.acos();
            BlockInfluence {
                layer_idx: lm.layer_idx,
                cosine_sim,
                angular_distance,
                influence: 1.0 - cosine_sim,
            }
        })
        .collect()
}

/// Compute exact block influence scores from full hidden state vectors.
///
/// `hidden_states` should contain `n_layers + 1` entries: the input to
/// layer 0, then the output of each layer. Each entry is `[hidden_dim]`.
pub fn score_from_hidden_states(hidden_states: &[Vec<f32>]) -> Vec<BlockInfluence> {
    if hidden_states.len() < 2 {
        return Vec::new();
    }
    (0..hidden_states.len() - 1)
        .map(|i| BlockInfluence::from_vectors(i, &hidden_states[i], &hidden_states[i + 1]))
        .collect()
}

/// Rank layers by pruning priority (least influential first).
///
/// Returns layer indices sorted by ascending influence score.
/// The first elements are the best pruning candidates.
pub fn pruning_rank(scores: &[BlockInfluence]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> =
        scores.iter().map(|s| (s.layer_idx, s.influence)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().map(|(idx, _)| idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn assert_close(a: f32, b: f32, msg: &str) {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1e-8);
        assert!(
            diff / scale < TOL,
            "{msg}: got {a}, expected {b}, diff={diff}"
        );
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_close(cosine_similarity(&v, &v), 1.0, "identical vectors");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_close(cosine_similarity(&a, &b), 0.0, "orthogonal vectors");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert_close(cosine_similarity(&a, &b), -1.0, "opposite vectors");
    }

    #[test]
    fn test_cosine_known_angle() {
        // 45-degree angle: cos(π/4) = √2/2 ≈ 0.7071
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let expected = 1.0 / 2.0_f32.sqrt();
        assert_close(cosine_similarity(&a, &b), expected, "45-degree angle");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0, "zero vector → 1.0");
    }

    #[test]
    fn test_cosine_both_zero() {
        let z = vec![0.0; 4];
        assert_eq!(cosine_similarity(&z, &z), 1.0, "both zero → 1.0");
    }

    #[test]
    fn test_cosine_high_dim() {
        // All-ones vectors in high dim should give cos = 1.0.
        let v = vec![1.0_f32; 896];
        assert_close(cosine_similarity(&v, &v), 1.0, "896-dim identical");
    }

    #[test]
    fn test_block_influence_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let bi = BlockInfluence::from_vectors(0, &v, &v);
        assert_close(bi.cosine_sim, 1.0, "identical cosine");
        assert!(
            bi.angular_distance < 1e-3,
            "identical angle should be near 0, got {}",
            bi.angular_distance
        );
        assert!(
            bi.influence < 1e-5,
            "identical influence should be near 0, got {}",
            bi.influence
        );
    }

    #[test]
    fn test_block_influence_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let bi = BlockInfluence::from_vectors(5, &a, &b);
        assert_close(bi.cosine_sim, 0.0, "orthogonal cosine");
        assert_close(
            bi.angular_distance,
            std::f32::consts::FRAC_PI_2,
            "orthogonal angle",
        );
        assert_close(bi.influence, 1.0, "orthogonal influence");
    }

    #[test]
    fn test_score_from_hidden_states() {
        let states = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0], // layer 0: no change → influence ≈ 0
            vec![0.0, 1.0, 0.0], // layer 1: orthogonal → influence ≈ 1
            vec![0.0, 0.7, 0.7], // layer 2: 45° rotation → influence ≈ 0.29
        ];
        let scores = score_from_hidden_states(&states);
        assert_eq!(scores.len(), 3);

        assert_close(scores[0].influence, 0.0, "layer 0 no change");
        assert_close(scores[1].influence, 1.0, "layer 1 orthogonal");

        // Layer 2: cos between (0,1,0) and (0,0.7,0.7)
        let expected_cos = 0.7 / (0.7_f32 * 0.7 + 0.7 * 0.7).sqrt();
        assert_close(scores[2].cosine_sim, expected_cos, "layer 2 cosine");
    }

    #[test]
    fn test_score_from_hidden_states_single() {
        let states = vec![vec![1.0, 2.0]];
        assert!(score_from_hidden_states(&states).is_empty());
    }

    #[test]
    fn test_pruning_rank_order() {
        let states = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0], // layer 0: influence ≈ 0 (most prunable)
            vec![0.0, 1.0, 0.0], // layer 1: influence ≈ 1 (least prunable)
            vec![0.0, 0.7, 0.7], // layer 2: intermediate
        ];
        let scores = score_from_hidden_states(&states);
        let rank = pruning_rank(&scores);

        assert_eq!(rank[0], 0, "most prunable first");
        assert_eq!(rank[2], 1, "least prunable last");
    }

    #[test]
    fn test_score_from_norms() {
        let layers = vec![
            LayerMetrics {
                layer_idx: 0,
                input_norm: 10.0,
                output_norm: 10.0,
                ..Default::default()
            },
            LayerMetrics {
                layer_idx: 1,
                input_norm: 10.0,
                output_norm: 1.0,
                ..Default::default()
            },
        ];
        let scores = score_from_norms(&layers);
        assert_eq!(scores.len(), 2);
        // Equal norms → ratio = 1.0 → low influence
        assert_close(scores[0].cosine_sim, 1.0, "equal norms");
        // 10:1 ratio → ratio = 0.1 → high influence estimate
        assert_close(scores[1].cosine_sim, 0.1, "unequal norms");
    }

    #[test]
    fn test_score_from_norms_zero() {
        let layers = vec![LayerMetrics {
            layer_idx: 0,
            input_norm: 0.0,
            output_norm: 0.0,
            ..Default::default()
        }];
        let scores = score_from_norms(&layers);
        assert_close(scores[0].cosine_sim, 1.0, "zero norms");
    }

    #[test]
    fn test_cosine_numerical_stability() {
        // Near-identical large vectors should give cos ≈ 1.0, not > 1.0.
        let a = vec![1e6_f32; 1024];
        let mut b = a.clone();
        b[0] += 1e-2;
        let c = cosine_similarity(&a, &b);
        assert!(c <= 1.0, "cosine should be clamped to ≤ 1.0, got {c}");
        assert!(
            c > 0.999,
            "near-identical vectors should have cos > 0.999, got {c}"
        );
    }
}
