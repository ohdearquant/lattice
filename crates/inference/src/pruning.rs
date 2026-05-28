//! Block influence scoring for layer pruning (ADR-060 Phase 1).
//!
//! Implements the ShortGPT angular distance metric: for each transformer
//! layer, measure cos(θ) between the input and output hidden states.
//! Layers where θ ≈ 0 (input ≈ output) contribute little and are pruning
//! candidates.
//!
//! Reference: Men et al., "ShortGPT: Layers in Large Language Models are
//! More Redundant Than You Expect" (arXiv:2403.03853).
//!
//! # Calibration workflow
//!
//! For real calibration data with multiple tokens per layer, use
//! [`BlockInfluenceAccumulator`]. Feed one token at a time via [`update`],
//! then call [`finalize`] to get the averaged [`BlockInfluence`] score.
//!
//! [`update`]: BlockInfluenceAccumulator::update
//! [`finalize`]: BlockInfluenceAccumulator::finalize

use crate::metrics::l2_norm;

/// Angular distance between two vectors, returned as cosine similarity.
///
/// cos(θ) = (a · b) / (‖a‖ · ‖b‖)
///
/// Returns `None` when either vector has zero norm, so that erased
/// residual streams do not spuriously appear as "identical to input."
/// Callers should skip `None` observations during calibration.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`. This check is always-on (not
/// debug-only) because mismatched inputs silently truncate via zip,
/// producing wrong scores during offline calibration.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        return None;
    }
    Some((dot / denom).clamp(-1.0, 1.0))
}

/// Per-layer block influence score.
#[derive(Debug, Clone)]
pub struct BlockInfluence {
    /// Layer index.
    pub layer_idx: usize,
    /// Cosine similarity between layer input and output hidden states,
    /// averaged over the calibration tokens.
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
    /// Compute block influence from a single pair of hidden state vectors.
    ///
    /// Returns `None` if either vector is zero-norm (token is skipped
    /// by calibration code rather than polluting the average).
    pub fn from_vectors(layer_idx: usize, input: &[f32], output: &[f32]) -> Option<Self> {
        let cosine_sim = cosine_similarity(input, output)?;
        let angular_distance = cosine_sim.acos();
        let influence = 1.0 - cosine_sim;
        Some(Self {
            layer_idx,
            cosine_sim,
            angular_distance,
            influence,
        })
    }
}

/// Online accumulator for block influence scores over a calibration dataset.
///
/// ShortGPT defines BI_i = 1 − E[cos(X_{i,t}, X_{i+1,t})] averaged over
/// tokens T. This accumulator ingests one (input, output) token pair at a
/// time and computes the average cosine similarity at [`finalize`].
///
/// Zero-norm tokens are silently skipped; they do not contribute to the count.
///
/// # Example
///
/// ```
/// use lattice_inference::pruning::BlockInfluenceAccumulator;
///
/// let mut acc = BlockInfluenceAccumulator::new(0);
/// for t in 0..128 {
///     let input: Vec<f32> = (0..896).map(|i| (i + t) as f32).collect();
///     let output: Vec<f32> = (0..896).map(|i| (i + t + 1) as f32).collect();
///     acc.update(&input, &output);
/// }
/// if let Some(bi) = acc.finalize() {
///     println!("layer {} influence: {:.4}", bi.layer_idx, bi.influence);
/// }
/// ```
///
/// [`finalize`]: BlockInfluenceAccumulator::finalize
#[derive(Debug, Clone)]
pub struct BlockInfluenceAccumulator {
    /// Layer index this accumulator tracks.
    layer_idx: usize,
    /// Running sum of cosine similarity values across valid tokens.
    cosine_sum: f64,
    /// Number of valid (non-zero-norm) tokens accumulated so far.
    count: usize,
}

impl BlockInfluenceAccumulator {
    /// Create a new accumulator for the given layer.
    pub fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            cosine_sum: 0.0,
            count: 0,
        }
    }

    /// Feed one token's (input, output) hidden state pair into the accumulator.
    ///
    /// Zero-norm tokens are skipped and do not increment the count.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != output.len()`.
    pub fn update(&mut self, input: &[f32], output: &[f32]) {
        if let Some(cos) = cosine_similarity(input, output) {
            self.cosine_sum += cos as f64;
            self.count += 1;
        }
    }

    /// Finalize and return the averaged [`BlockInfluence`] score.
    ///
    /// Returns `None` if no valid tokens were accumulated (all were zero-norm
    /// or `update` was never called).
    pub fn finalize(&self) -> Option<BlockInfluence> {
        if self.count == 0 {
            return None;
        }
        let cosine_sim = (self.cosine_sum / self.count as f64) as f32;
        let angular_distance = cosine_sim.acos();
        let influence = 1.0 - cosine_sim;
        Some(BlockInfluence {
            layer_idx: self.layer_idx,
            cosine_sim,
            angular_distance,
            influence,
        })
    }

    /// Number of valid tokens accumulated so far.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Compute exact block influence scores from full hidden state vectors.
///
/// `hidden_states` should contain `n_layers + 1` entries: the input to
/// layer 0, then the output of each layer. Each entry is a single vector
/// of shape `[hidden_dim]` — one representative token per layer.
///
/// For real calibration over multiple tokens per layer, use
/// [`BlockInfluenceAccumulator`] instead.
///
/// Layers where `cosine_similarity` returns `None` (zero-norm input or
/// output) are omitted from the result.
pub fn score_from_hidden_states(hidden_states: &[Vec<f32>]) -> Vec<BlockInfluence> {
    if hidden_states.len() < 2 {
        return Vec::new();
    }
    (0..hidden_states.len() - 1)
        .filter_map(|i| BlockInfluence::from_vectors(i, &hidden_states[i], &hidden_states[i + 1]))
        .collect()
}

/// Rank layers by pruning priority (least influential first).
///
/// Returns layer indices sorted by ascending influence score.
/// The first elements are the best pruning candidates.
pub fn pruning_rank(scores: &[BlockInfluence]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> =
        scores.iter().map(|s| (s.layer_idx, s.influence)).collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
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

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_close(cosine_similarity(&v, &v).unwrap(), 1.0, "identical vectors");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_close(
            cosine_similarity(&a, &b).unwrap(),
            0.0,
            "orthogonal vectors",
        );
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert_close(cosine_similarity(&a, &b).unwrap(), -1.0, "opposite vectors");
    }

    #[test]
    fn test_cosine_known_angle() {
        // 45-degree angle: cos(π/4) = √2/2 ≈ 0.7071
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let expected = 1.0 / 2.0_f32.sqrt();
        assert_close(
            cosine_similarity(&a, &b).unwrap(),
            expected,
            "45-degree angle",
        );
    }

    #[test]
    fn test_cosine_zero_vector_returns_none() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), None, "zero input vector → None");
    }

    #[test]
    fn test_cosine_both_zero_returns_none() {
        let z = vec![0.0; 4];
        assert_eq!(cosine_similarity(&z, &z), None, "both zero → None");
    }

    #[test]
    fn test_cosine_high_dim() {
        // All-ones vectors in high dim should give cos = 1.0.
        let v = vec![1.0_f32; 896];
        assert_close(cosine_similarity(&v, &v).unwrap(), 1.0, "896-dim identical");
    }

    #[test]
    fn test_cosine_numerical_stability() {
        // Near-identical large vectors should give cos ≈ 1.0, not > 1.0.
        let a = vec![1e6_f32; 1024];
        let mut b = a.clone();
        b[0] += 1e-2;
        let c = cosine_similarity(&a, &b).unwrap();
        assert!(c <= 1.0, "cosine should be clamped to ≤ 1.0, got {c}");
        assert!(
            c > 0.999,
            "near-identical vectors should have cos > 0.999, got {c}"
        );
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn test_cosine_length_mismatch_panics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let _ = cosine_similarity(&a, &b);
    }

    // ── BlockInfluence::from_vectors ──────────────────────────────────────────

    #[test]
    fn test_block_influence_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let bi = BlockInfluence::from_vectors(0, &v, &v).unwrap();
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
        let bi = BlockInfluence::from_vectors(5, &a, &b).unwrap();
        assert_close(bi.cosine_sim, 0.0, "orthogonal cosine");
        assert_close(
            bi.angular_distance,
            std::f32::consts::FRAC_PI_2,
            "orthogonal angle",
        );
        assert_close(bi.influence, 1.0, "orthogonal influence");
    }

    #[test]
    fn test_block_influence_zero_vector_returns_none() {
        let zero = vec![0.0, 0.0, 0.0];
        let nonzero = vec![1.0, 0.0, 0.0];
        assert!(
            BlockInfluence::from_vectors(0, &zero, &nonzero).is_none(),
            "zero input → None"
        );
        assert!(
            BlockInfluence::from_vectors(0, &nonzero, &zero).is_none(),
            "zero output → None"
        );
    }

    // ── score_from_hidden_states ───────────────────────────────────────────────

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
    fn test_score_from_hidden_states_zero_layer_omitted() {
        // A layer with zero input is omitted; later layers are still scored.
        let states = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0], // zero output → layer 0 omitted
            vec![0.0, 1.0, 0.0], // layer 1 has nonzero input and output
        ];
        let scores = score_from_hidden_states(&states);
        // Layer 0 omitted, layer 1 has nonzero input [0,0,0] → also None
        // (zero input from previous zero-output)
        assert_eq!(scores.len(), 0, "both layers omitted due to zero norms");
    }

    // ── pruning_rank ───────────────────────────────────────────────────────────

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

    // ── BlockInfluenceAccumulator ──────────────────────────────────────────────

    #[test]
    fn test_accumulator_single_token_matches_from_vectors() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![2.0_f32, 3.0, 4.0];
        let direct = BlockInfluence::from_vectors(2, &a, &b).unwrap();

        let mut acc = BlockInfluenceAccumulator::new(2);
        acc.update(&a, &b);
        let averaged = acc.finalize().unwrap();

        assert_close(
            averaged.cosine_sim,
            direct.cosine_sim,
            "single-token acc == from_vectors",
        );
        assert_eq!(averaged.layer_idx, 2);
        assert_eq!(acc.count(), 1);
    }

    #[test]
    fn test_accumulator_average_over_tokens() {
        // Two tokens: one identical pair (cos=1) and one orthogonal pair (cos=0).
        // Expected average cosine = 0.5, influence = 0.5.
        let identical_a = vec![1.0_f32, 0.0, 0.0];
        let orthogonal_b = vec![0.0_f32, 1.0, 0.0];

        let mut acc = BlockInfluenceAccumulator::new(0);
        acc.update(&identical_a, &identical_a); // cos = 1.0
        acc.update(&identical_a, &orthogonal_b); // cos = 0.0
        let bi = acc.finalize().unwrap();

        assert_eq!(acc.count(), 2);
        assert_close(bi.cosine_sim, 0.5, "average of 1.0 and 0.0");
        assert_close(bi.influence, 0.5, "influence = 1 - 0.5");
    }

    #[test]
    fn test_accumulator_skips_zero_norm_tokens() {
        let zero = vec![0.0_f32; 4];
        let nonzero = vec![1.0_f32, 0.0, 0.0, 0.0];

        let mut acc = BlockInfluenceAccumulator::new(1);
        acc.update(&zero, &nonzero); // skipped
        acc.update(&nonzero, &zero); // skipped
        acc.update(&nonzero, &nonzero); // cos = 1.0, accepted

        assert_eq!(acc.count(), 1, "only non-zero-norm token counted");
        let bi = acc.finalize().unwrap();
        assert_close(bi.cosine_sim, 1.0, "only valid token was identical");
    }

    #[test]
    fn test_accumulator_all_zero_returns_none() {
        let zero = vec![0.0_f32; 4];
        let mut acc = BlockInfluenceAccumulator::new(0);
        acc.update(&zero, &zero);
        assert_eq!(acc.count(), 0);
        assert!(acc.finalize().is_none(), "all-zero tokens → None");
    }

    #[test]
    fn test_accumulator_empty_returns_none() {
        let acc = BlockInfluenceAccumulator::new(3);
        assert!(acc.finalize().is_none(), "no updates → None");
    }

    #[test]
    fn test_accumulator_many_tokens_identical() {
        // 512 identical tokens → influence ≈ 0 regardless of count.
        let v = vec![1.0_f32; 896];
        let mut acc = BlockInfluenceAccumulator::new(0);
        for _ in 0..512 {
            acc.update(&v, &v);
        }
        let bi = acc.finalize().unwrap();
        assert_eq!(acc.count(), 512);
        assert_close(bi.cosine_sim, 1.0, "512 identical tokens");
        assert!(
            bi.influence < 1e-5,
            "influence near 0 for 512 identical, got {}",
            bi.influence
        );
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn test_accumulator_length_mismatch_panics() {
        let mut acc = BlockInfluenceAccumulator::new(0);
        acc.update(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
    }
}
