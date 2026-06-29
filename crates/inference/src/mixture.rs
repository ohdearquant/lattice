//! Lightweight adapter routing: select and weight a subset of available adapters
//! for a single inference request.
//!
//! # Design rationale
//!
//! Adapter selection happens once per request on the CPU, before any GPU work
//! begins.  The gate is a small fann network whose forward pass costs well under
//! 1 ms — negligible compared to prefill.  The selected adapters are then blended
//! on the CPU (see `crates/tune/src/lora/blend.rs`) and loaded into the Metal
//! path through the existing single-slot adapter API.
//!
//! Mixture weights in PR1 are constant `1/k` (top-k uniform).  The gate network
//! output is used only to rank adapters; the weights themselves are not learned
//! until PR3.

use lattice_fann::{FannError, Network};

/// Opaque identifier for a LoRA adapter.
///
/// Callers may store any string key here (path, UUID, name).  The router treats
/// it as an opaque token and returns the same strings in its output.
pub type AdapterId = String;

/// Error type for routing operations.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    /// The fann network forward pass failed.
    #[error("gate network error: {0}")]
    Gate(#[from] FannError),

    /// The requested `k` exceeds the number of available adapters.
    #[error("k={k} exceeds available adapter count {available}")]
    KTooLarge {
        /// Requested top-k
        k: usize,
        /// Number of available adapters
        available: usize,
    },

    /// `k` must be at least 1.
    #[error("k must be >= 1, got {k}")]
    InvalidK {
        /// The offending k value
        k: usize,
    },

    /// The context vector has the wrong length for this gate.
    #[error("context vector length {got} does not match gate input size {expected}")]
    InputSizeMismatch {
        /// Expected size
        expected: usize,
        /// Received size
        got: usize,
    },
}

/// Routes a context vector to a top-k subset of available adapters with
/// constant equal mixture weights.
///
/// The gate network produces one score per available adapter.  The `k` adapters
/// with the highest scores are selected; each receives weight `1/k`.
///
/// # Mixture weight semantics
///
/// Weights are constant `1/k` in PR1 (ReMix constant-weight constraint).
/// The gate network trains in PR3; until then scores are random initialisation
/// and only the rank ordering matters.
pub struct AdapterRouter {
    gate: Network,
}

impl AdapterRouter {
    /// Create a router backed by an existing fann `Network`.
    ///
    /// The network's output dimension must be `≥` the maximum number of
    /// adapters that will ever be passed to `route`.  Callers that wish to
    /// support a dynamic adapter pool should size the output to the maximum
    /// expected pool size.
    pub fn new(gate: Network) -> Self {
        Self { gate }
    }

    /// Select the top-`k` adapters for the given context and assign each
    /// a constant mixture weight of `1/k`.
    ///
    /// # Arguments
    ///
    /// * `context_vector` — embedding of the current request context; must
    ///   match the gate network's input dimension.
    /// * `available` — ordered list of candidate adapter IDs.  The gate
    ///   output index `i` maps to `available[i]`.
    /// * `k` — number of adapters to select.
    ///
    /// # Returns
    ///
    /// A `Vec` of `(AdapterId, weight)` pairs, length `k`, sorted by
    /// descending gate score.  Each weight is `1.0 / k as f32`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `k == 0`
    /// - `k > available.len()`
    /// - the context vector length does not match the gate input size
    /// - the gate forward pass fails
    pub fn route(
        &mut self,
        context_vector: &[f32],
        available: &[AdapterId],
        k: usize,
    ) -> Result<Vec<(AdapterId, f32)>, RouterError> {
        if k == 0 {
            return Err(RouterError::InvalidK { k });
        }
        if k > available.len() {
            return Err(RouterError::KTooLarge {
                k,
                available: available.len(),
            });
        }
        let expected_input = self.gate.num_inputs();
        if context_vector.len() != expected_input {
            return Err(RouterError::InputSizeMismatch {
                expected: expected_input,
                got: context_vector.len(),
            });
        }

        // Run gate network: returns a score per output unit.
        let scores = self.gate.forward(context_vector)?;

        // Top-k by score, descending.  Only the first `available.len()` outputs
        // are meaningful; ignore extra outputs if the network is wider.
        let n = available.len().min(scores.len());
        let mut indexed: Vec<(usize, f32)> = scores[..n].iter().copied().enumerate().collect();
        // Partial-sort: move the top-k elements to the front.
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let weight = 1.0 / k as f32;
        let mut selected: Vec<(AdapterId, f32)> = indexed[..k]
            .iter()
            .map(|(idx, _)| (available[*idx].clone(), weight))
            .collect();
        // Sort by descending score for deterministic output ordering.
        selected.sort_by(|a, b| {
            let sa = scores[available.iter().position(|x| x == &a.0).unwrap_or(0)];
            let sb = scores[available.iter().position(|x| x == &b.0).unwrap_or(0)];
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(selected)
    }

    /// Return the number of inputs the gate network expects.
    pub fn input_size(&self) -> usize {
        self.gate.num_inputs()
    }

    /// Return the number of outputs (maximum supported adapter pool size).
    pub fn output_size(&self) -> usize {
        self.gate.num_outputs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_fann::{Activation, NetworkBuilder};

    fn make_router(inputs: usize, outputs: usize) -> AdapterRouter {
        let net = NetworkBuilder::new()
            .input(inputs)
            .output(outputs, Activation::Linear)
            .build()
            .unwrap();
        AdapterRouter::new(net)
    }

    #[test]
    fn route_returns_k_entries() {
        let mut router = make_router(4, 6);
        let available: Vec<AdapterId> = (0..6).map(|i| format!("adapter-{i}")).collect();
        let ctx = vec![1.0f32; 4];
        let result = router.route(&ctx, &available, 3).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn route_weights_sum_to_one() {
        let mut router = make_router(4, 4);
        let available: Vec<AdapterId> = (0..4).map(|i| format!("a{i}")).collect();
        let ctx = vec![1.0f32; 4];
        let result = router.route(&ctx, &available, 4).unwrap();
        let weight_sum: f32 = result.iter().map(|(_, w)| w).sum();
        assert!((weight_sum - 1.0).abs() < 1e-6, "weights must sum to 1.0");
    }

    #[test]
    fn route_uniform_weight() {
        let k = 3usize;
        let mut router = make_router(2, 5);
        let available: Vec<AdapterId> = (0..5).map(|i| format!("a{i}")).collect();
        let ctx = vec![0.5f32; 2];
        let result = router.route(&ctx, &available, k).unwrap();
        let expected_w = 1.0 / k as f32;
        for (_, w) in &result {
            assert!(
                (w - expected_w).abs() < 1e-6,
                "each weight must be 1/k={expected_w}"
            );
        }
    }

    #[test]
    fn route_k_zero_errors() {
        let mut router = make_router(2, 3);
        let available: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
        assert!(router.route(&[1.0, 2.0], &available, 0).is_err());
    }

    #[test]
    fn route_k_exceeds_available_errors() {
        let mut router = make_router(2, 2);
        let available: Vec<AdapterId> = vec!["a".into()];
        assert!(router.route(&[1.0, 2.0], &available, 2).is_err());
    }

    #[test]
    fn route_wrong_input_size_errors() {
        let mut router = make_router(4, 2);
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        // supply 3 floats instead of 4
        assert!(router.route(&[1.0, 2.0, 3.0], &available, 1).is_err());
    }
}
