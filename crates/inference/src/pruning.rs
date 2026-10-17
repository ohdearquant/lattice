//! Standalone ShortGPT block influence scorer (ADR-060 D2 seed).
//!
//! Implements the ShortGPT block-influence (BI) metric: for each transformer
//! layer, compute `BI_i = 1 − E_t[cos(X_{i,t}, X_{i+1,t})]` where X_{i,t}
//! is the residual hidden state at the input of layer i for token t and
//! X_{i+1,t} is the output of the full block. Layers where BI ≈ 0 (input ≈
//! output) contribute little and are removal candidates.
//!
//! **This is a standalone scorer utility, not the full ADR-060 calibration
//! pipeline.** The `CalibrationObserver` trait and `ForwardCtx` hooks that
//! feed real activation captures are ADR-060 D1/P0 work (future PR). This
//! module exposes the math primitives; callers provide the hidden-state slices.
//!
//! Reference: Men et al., "ShortGPT: Layers in Large Language Models are
//! More Redundant Than You Expect" (arXiv:2403.03853, ACL Findings 2025).
//!
//! # Calibration workflow
//!
//! For real calibration data with multiple tokens per layer, use
//! `BlockInfluenceAccumulator`. Feed one token at a time via `update`,
//! then call `finalize` to get the averaged `BlockInfluence` score.

/// Cosine similarity between two vectors.
///
/// cos(θ) = (a · b) / (‖a‖ · ‖b‖)
///
/// Dot product and norms are accumulated in `f64` to avoid overflow for
/// large-magnitude inputs: f32 squares overflow at ~1e19, while f64 is
/// safe to ~1e154.
///
/// Returns `None` when either vector is exactly zero-norm, so that erased
/// residual streams do not spuriously appear as "identical to input."
/// Callers should skip `None` observations during calibration.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`. This check is always-on (not
/// debug-only) because mismatched inputs silently truncate via zip,
/// producing wrong scores during offline calibration.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let aa: f64 = a.iter().map(|&x| x as f64 * x as f64).sum();
    let bb: f64 = b.iter().map(|&x| x as f64 * x as f64).sum();
    if aa == 0.0 || bb == 0.0 {
        return None;
    }
    let cos = dot / (aa.sqrt() * bb.sqrt());
    Some(cos.clamp(-1.0, 1.0) as f32)
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
    /// Diagnostic only — pruning decisions use `influence`, not this field.
    pub angular_distance: f32,
    /// Block influence score: 1.0 − cosine_sim (ShortGPT BI formula).
    /// Higher = more influential (less prunable); lower = more removable.
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
    #[inline]
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

/// Compute block influence scores from a single representative hidden-state
/// vector per layer.
///
/// `hidden_states` should contain `n_layers + 1` entries: the input to
/// layer 0, then the output of each layer. Each entry is a single vector
/// of shape `[hidden_dim]` representing one token.
///
/// For multi-token calibration (more accurate BI estimates), use
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
mod tests;
