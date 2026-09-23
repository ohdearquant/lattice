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
//! # Mixture weights
//!
//! [`AdapterRouter::route`] assigns each selected adapter a weight under a
//! configured [`WeightPolicy`]. The default and only fully-supported policy
//! is `Uniform` (`1/k` for every selected adapter), matching the router's
//! original behaviour. [`WeightPolicy::Softmax`] draws weights from the
//! gate's own scores instead: a learnable-softmax router can collapse to
//! effectively one adapter under sparse, noisy reward, so a temperature and
//! an `epsilon` floor bound the result (ADR-091; see
//! [`AdapterRouter::set_weight_policy`] and [`AdapterRouter::set_epsilon`]).
//!
//! ADR-091 Decision 3 names three collapse guards, all required before
//! `Softmax` should be preferred in production. Two are implemented here: a
//! floor below which [`AdapterRouter::route`] refuses `tau` rather than
//! letting it fall toward an argmax (see [`AdapterRouter::set_tau_floor`]
//! and [`RouterError::TauBelowFloor`]), and an entropy floor on a refit
//! round's produced weight vectors that rejects the refit — leaving the live
//! gate unchanged — after enough consecutive collapsed rounds (see
//! [`AdapterRouter::submit_refit`]). The third, load-balance and z-loss
//! terms applied to the weight distribution, is not implemented: the ADR
//! text is ambiguous between at least two architecturally different readings
//! (a monitoring signal evaluated on `route`'s selected/floored output vs.
//! restructuring the refit gradient itself to train against that output
//! rather than the gate's full logit vector), and no other document in this
//! repository resolves it. See `submit_refit`'s doc for the reasoning; an
//! answer invented here would be a spec change wearing an implementation's
//! clothes. `Softmax` should not be preferred in production ahead of it.

use lattice_fann::{FannError, Network};

/// Opaque identifier for a LoRA adapter.
///
/// Callers may store any string key here (path, UUID, name).  The router treats
/// it as an opaque token and returns the same strings in its output.
pub type AdapterId = String;

/// Error type for routing operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RouterError {
    /// The fann network parsing or forward pass failed.
    #[error("gate network error: {0}")]
    Gate(#[from] FannError),

    /// The replacement gate dimensions differ from the live gate.
    #[error(
        "replacement gate dimensions {got_inputs} -> {got_outputs} do not match live gate {expected_inputs} -> {expected_outputs}"
    )]
    GateDimensionMismatch {
        /// Live gate input width.
        expected_inputs: usize,
        /// Live gate output width.
        expected_outputs: usize,
        /// Replacement gate input width.
        got_inputs: usize,
        /// Replacement gate output width.
        got_outputs: usize,
    },

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

    /// `k` exceeds the number of adapters the gate can actually score.
    ///
    /// This happens when the gate network has fewer outputs than `available`
    /// adapters and `k` is larger than that narrower output count.
    #[error(
        "router: k={k} exceeds usable adapter count {usable} \
         (gate produced fewer scores than adapters)"
    )]
    GateTooNarrow {
        /// Requested top-k
        k: usize,
        /// Usable count: `min(available.len(), gate.num_outputs())`
        usable: usize,
    },

    /// The `available` set contains a repeated adapter ID.
    ///
    /// Duplicate IDs would cause one adapter to be selected and weighted twice,
    /// silently doubling its contribution.  The router rejects this to fail closed.
    #[error("duplicate adapter id in available set: {id}")]
    DuplicateAdapterId {
        /// The repeated adapter ID
        id: String,
    },

    /// The softmax temperature `tau` was not usable: it must be finite and
    /// strictly positive. A temperature free to reach zero collapses the
    /// distribution to an argmax with extra steps; a negative or non-finite
    /// temperature has no defined softmax at all.
    #[error("softmax temperature tau must be finite and > 0.0, got {tau}")]
    InvalidTau {
        /// The offending temperature.
        tau: f32,
    },

    /// `tau` was finite and strictly positive (so it passed the basic
    /// validity check) but fell below the configured collapse-guard floor
    /// (ADR-091 Decision 3, [`AdapterRouter::set_tau_floor`]).
    #[error("softmax temperature tau={tau} is below the configured floor {floor}")]
    TauBelowFloor {
        /// The offending (too-low but otherwise valid) temperature.
        tau: f32,
        /// The configured floor it fell below.
        floor: f32,
    },

    /// [`AdapterRouter::submit_refit`] was called with no weight vectors for
    /// the round: there is nothing to measure entropy over.
    #[error("refit round has no weight vectors to measure entropy over")]
    EmptyRefitRound,
}

/// Named mixture weight policy for [`AdapterRouter::route`] (ADR-091).
///
/// `Uniform` is the default and reproduces the router's original behaviour.
/// Reaching uniform weights is a matter of naming this variant, never a
/// matter of choosing a large `tau` — see [`AdapterRouter::set_weight_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WeightPolicy {
    /// Every selected adapter receives `1.0 / k`.
    #[default]
    Uniform,
    /// Softmax over the selected adapters' own gate scores at temperature
    /// `tau`, normalised to sum to 1 across the selected set. `tau` must be
    /// finite and strictly positive.
    Softmax {
        /// Softmax temperature. Lower sharpens the distribution toward the
        /// top score; higher flattens it toward uniform.
        tau: f32,
    },
}

/// Outcome of applying the `epsilon` floor to a weight vector.
///
/// Distinguishes an adapter that was never selected at all from one that was
/// selected (or explicitly supplied) and then removed for falling below
/// `epsilon`: the former appears in neither field, the latter appears only
/// in `dropped`.
#[derive(Debug, Clone, PartialEq)]
pub struct FloorOutcome {
    /// Surviving adapters and their final weight.
    pub weights: Vec<(AdapterId, f32)>,
    /// Adapters that were present before the floor and removed for falling
    /// below `epsilon`.
    pub dropped: Vec<AdapterId>,
}

/// Default floor on the softmax temperature `tau` (ADR-091 Decision 3,
/// [`AdapterRouter::set_tau_floor`]).
///
/// Below this, `exp((score - max_score) / tau)` underflows toward zero in
/// `f32` for any score gap of a couple of units, so the softmax is already
/// an argmax in floating point, not merely "practically" one. `1e-2` is a
/// round, conservative number rather than one derived from a specific score
/// scale — gate scores have no calibrated magnitude (see the module doc) —
/// chosen far enough above numerical underflow that it catches near-collapse
/// well before floating point would. It is a policy default, not a proof:
/// the Decision 5 evidence run is what should move it.
pub const DEFAULT_TAU_FLOOR: f32 = 1e-2;

/// Default entropy floor, in nats, below which a refit round's mean
/// weight-entropy counts as collapsed (ADR-091 Decision 3,
/// [`AdapterRouter::submit_refit`]).
///
/// At `k = 2` (the smallest, most common selection width) this is roughly a
/// 97.7 / 2.3 split — sharp enough to already read as collapse rather than a
/// confident, healthy preference. Entropy's scale depends on `k`, and this
/// guard compares every round against one static floor regardless of `k`;
/// a deployment mixing very different `k` across rounds should treat this as
/// a known limitation, not a tuned-away one. As the ADR states: this floor
/// cannot tell collapse from a correctly sharp distribution — a router that
/// has genuinely learned to prefer one adapter produces the same low-entropy
/// weight vector as one that has collapsed under sparse reward. It is a
/// guard on refits, not a guard on truth: it decides which gate gets
/// reloaded, not whether the resulting policy is good. A rejected refit
/// whose held-out metric (Decision 5) was improving is the stated falsifier
/// for the floor value — that observation, not intuition about the number,
/// is what should move it.
pub const DEFAULT_ENTROPY_FLOOR: f32 = 0.1;

/// Default number of consecutive collapsed rounds tolerated before
/// [`AdapterRouter::submit_refit`] rejects a refit (ADR-091 Decision 3).
///
/// `3` tolerates a single noisy round — one round at or above the floor
/// resets the count — while still catching a sustained collapse within a
/// small, bounded number of refit cycles. Policy, not derivation; see
/// [`DEFAULT_ENTROPY_FLOOR`].
pub const DEFAULT_MAX_CONSECUTIVE_COLLAPSED: usize = 3;

/// Routes a context vector to a top-k subset of available adapters and
/// assigns each a mixture weight under the router's [`WeightPolicy`].
///
/// The gate network produces one score per available adapter.  The `k`
/// adapters with the highest scores are selected; the selected scores are
/// then turned into weights by [`route`](Self::route)'s configured policy
/// and `epsilon` floor (see [`set_weight_policy`](Self::set_weight_policy)
/// and [`set_epsilon`](Self::set_epsilon)).
///
/// # Mixture weight semantics
///
/// The default policy is [`WeightPolicy::Uniform`]: constant `1/k`,
/// unaffected by the gate scores' magnitude — only the rank ordering
/// matters, which is why gate scores being random at initialisation is
/// harmless. [`WeightPolicy::Softmax`] draws weights from those same scores
/// instead; a learnable-softmax router can collapse to effectively one
/// adapter under sparse, noisy reward, so a temperature floor, a
/// load-balance/z-loss penalty applied during refit, and refit rejection are
/// required before this policy is safe to prefer in production (ADR-091
/// Decision 3). This module implements the temperature floor
/// ([`set_tau_floor`](Self::set_tau_floor)) and refit rejection on an
/// entropy floor ([`submit_refit`](Self::submit_refit)); the load-balance/
/// z-loss guard is not implemented pending a spec clarification — see the
/// module doc and `submit_refit`'s doc for what is ambiguous.
pub struct AdapterRouter {
    gate: Network,
    weight_policy: WeightPolicy,
    epsilon: f32,
    last_dropped: Vec<AdapterId>,
    tau_floor: f32,
    entropy_floor: f32,
    max_consecutive_collapsed: usize,
    consecutive_collapsed_rounds: usize,
}

impl AdapterRouter {
    /// Create a router backed by an existing fann `Network`.
    ///
    /// The network's output dimension must be `≥` the maximum number of
    /// adapters that will ever be passed to `route`.  Callers that wish to
    /// support a dynamic adapter pool should size the output to the maximum
    /// expected pool size.
    pub fn new(gate: Network) -> Self {
        Self {
            gate,
            weight_policy: WeightPolicy::default(),
            epsilon: 0.0,
            last_dropped: Vec::new(),
            tau_floor: DEFAULT_TAU_FLOOR,
            entropy_floor: DEFAULT_ENTROPY_FLOOR,
            max_consecutive_collapsed: DEFAULT_MAX_CONSECUTIVE_COLLAPSED,
            consecutive_collapsed_rounds: 0,
        }
    }

    /// Replace the gate from a complete [`Network::to_bytes`] blob.
    ///
    /// Parsing and both dimensions must validate before any mutation: a failed
    /// reload leaves the live gate intact so a bad refit fails once, rather than
    /// disrupting subsequent requests.
    pub fn reload(&mut self, gate_bytes: &[u8]) -> Result<(), RouterError> {
        let gate = Network::from_bytes(gate_bytes)?;
        let expected_inputs = self.gate.num_inputs();
        let expected_outputs = self.gate.num_outputs();
        let got_inputs = gate.num_inputs();
        let got_outputs = gate.num_outputs();
        if got_inputs != expected_inputs || got_outputs != expected_outputs {
            return Err(RouterError::GateDimensionMismatch {
                expected_inputs,
                expected_outputs,
                got_inputs,
                got_outputs,
            });
        }
        self.gate = gate;
        Ok(())
    }

    /// Set the mixture weight policy used by subsequent
    /// [`route`](Self::route) calls.
    ///
    /// The default, from [`AdapterRouter::new`], is [`WeightPolicy::Uniform`].
    pub fn set_weight_policy(&mut self, policy: WeightPolicy) {
        self.weight_policy = policy;
    }

    /// Set the `epsilon` floor used by subsequent [`route`](Self::route)
    /// calls.
    ///
    /// After weights are assigned, any selected adapter whose weight falls
    /// below `epsilon` is removed and the survivors are renormalised to sum
    /// to 1 again. The default, from [`AdapterRouter::new`], is `0.0`, which
    /// never drops anything, since every weight `route` can produce is
    /// `>= 0.0`. `epsilon` should be non-negative; a negative or NaN value
    /// disables the floor, since no weight then compares less than it.
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    /// Adapters dropped by the most recent [`route`](Self::route) call for
    /// falling below the `epsilon` floor.
    ///
    /// An adapter that never reached the top-`k` selection never appears
    /// here; only one that was selected and then removed does. Empty after
    /// a call that dropped nothing, and reset (not accumulated) on every
    /// call.
    pub fn last_dropped(&self) -> &[AdapterId] {
        &self.last_dropped
    }

    /// Set the floor below which [`route`](Self::route) refuses `tau` for
    /// [`WeightPolicy::Softmax`], returning [`RouterError::TauBelowFloor`]
    /// (ADR-091 Decision 3).
    ///
    /// Refusing rather than clamping keeps a misconfigured temperature
    /// visible to the caller instead of quietly changing what they asked
    /// for — the same reasoning behind dropping an under-`epsilon` weight
    /// instead of damping it: a silent substitution is not an approximation
    /// of the caller's request, it is a different request answered as if it
    /// were the one asked. The default, from [`AdapterRouter::new`], is
    /// [`DEFAULT_TAU_FLOOR`].
    pub fn set_tau_floor(&mut self, floor: f32) {
        self.tau_floor = floor;
    }

    /// Set the entropy floor (nats) used by [`submit_refit`](Self::submit_refit)
    /// to decide whether a refit round counts as collapsed (ADR-091
    /// Decision 3). The default, from [`AdapterRouter::new`], is
    /// [`DEFAULT_ENTROPY_FLOOR`].
    pub fn set_entropy_floor(&mut self, floor: f32) {
        self.entropy_floor = floor;
    }

    /// Set the number of consecutive collapsed rounds
    /// [`submit_refit`](Self::submit_refit) tolerates before rejecting a
    /// refit (ADR-091 Decision 3). The default, from [`AdapterRouter::new`],
    /// is [`DEFAULT_MAX_CONSECUTIVE_COLLAPSED`].
    pub fn set_max_consecutive_collapsed(&mut self, max_consecutive_collapsed: usize) {
        self.max_consecutive_collapsed = max_consecutive_collapsed;
    }

    /// The current consecutive-collapsed-round count tracked by
    /// [`submit_refit`](Self::submit_refit).
    ///
    /// Reset to `0` by any round whose mean weight-entropy is at or above
    /// the configured floor.
    pub fn consecutive_collapsed_rounds(&self) -> usize {
        self.consecutive_collapsed_rounds
    }

    /// Select the top-`k` adapters for the given context and assign each a
    /// mixture weight under the configured [`WeightPolicy`].
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
    /// A `Vec` of `(AdapterId, weight)` pairs, sorted by descending gate
    /// score, whose weights sum to 1 over the selected set — **length is at
    /// most `k`**, not always `k`: any selected adapter whose weight falls
    /// below the configured `epsilon` floor is removed rather than kept at a
    /// weight near zero, and the survivors are renormalised. Call
    /// [`last_dropped`](Self::last_dropped) after this returns to see which
    /// adapters, if any, were dropped that way.
    ///
    /// With the default policy ([`WeightPolicy::Uniform`]) and the default
    /// `epsilon` (`0.0`), this reproduces the router's original contract
    /// exactly: length `k`, every weight `1.0 / k as f32`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `k == 0`
    /// - `k > available.len()`
    /// - the context vector length does not match the gate input size
    /// - the gate forward pass fails
    /// - the policy is [`WeightPolicy::Softmax`] and `tau` is zero,
    ///   negative, or non-finite
    /// - the policy is [`WeightPolicy::Softmax`] and `tau` is otherwise valid
    ///   but below the configured floor (see
    ///   [`set_tau_floor`](Self::set_tau_floor))
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

        // Reject duplicate IDs before running the gate: a duplicate would cause
        // one adapter to be selected and weighted twice (fail-open).
        let mut seen = std::collections::HashSet::new();
        for id in available {
            if !seen.insert(id.as_str()) {
                return Err(RouterError::DuplicateAdapterId { id: id.clone() });
            }
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

        // `n` bounds the usable width in either direction of a size mismatch
        // between the gate and `available`:
        //   - gate wider than pool (scores.len() > available.len()): the extra
        //     gate outputs have no adapter to map to, and are ignored.
        //   - pool wider than gate (available.len() > scores.len()): adapters at
        //     index >= n are never scored by this call and can never be selected,
        //     for any context, as long as k stays within the check below. This is
        //     recorded as pre-existing, deliberate behaviour of `AdapterRouter`,
        //     not a bounds bug: a router that must cover a wider pool needs a
        //     wider gate, which is a new `AdapterRouter`, not a change to this
        //     function's contract. The width-agreement refusal for a live pool
        //     belongs one layer up, at router construction, where the resident
        //     adapter count is known before any request is scored.
        let n = available.len().min(scores.len());

        // Fail closed: if the gate has fewer outputs than adapters AND k exceeds
        // those usable outputs, returning an error beats a panic inside
        // select_nth_unstable (which would fire with "index out of bounds").
        if k > n {
            return Err(RouterError::GateTooNarrow { k, usable: n });
        }

        let mut indexed: Vec<(usize, f32)> = scores[..n].iter().copied().enumerate().collect();
        // Partial-sort in O(n) average: top-k elements land in indexed[..k].
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        // Sort the selected prefix by score descending for deterministic output.
        // We work directly on (orig_index, score) pairs so duplicate adapter IDs
        // cannot confuse a position() lookup.
        indexed[..k].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let raw_weights: Vec<(AdapterId, f32)> = match self.weight_policy {
            WeightPolicy::Uniform => {
                let weight = 1.0 / k as f32;
                indexed[..k]
                    .iter()
                    .map(|(idx, _)| (available[*idx].clone(), weight))
                    .collect()
            }
            WeightPolicy::Softmax { tau } => {
                if !tau.is_finite() || tau <= 0.0 {
                    return Err(RouterError::InvalidTau { tau });
                }
                // ADR-091 Decision 3: a valid-but-too-small tau is an argmax
                // with extra steps. Checked after the basic finite/positive
                // validity check above, never before it — a NaN or negative
                // tau must fail InvalidTau, not silently skip this floor
                // (NaN compares false against everything, including `<`).
                if tau < self.tau_floor {
                    return Err(RouterError::TauBelowFloor {
                        tau,
                        floor: self.tau_floor,
                    });
                }
                // Subtract the max selected score before exponentiating: the
                // max always maps to exp(0) = 1.0, so the sum of exponentials
                // is always >= 1.0 and this can never divide by zero — and it
                // keeps a wide selected score spread from overflowing f32,
                // which a raw `score / tau` exponent would not.
                let max_score = indexed[..k]
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = indexed[..k]
                    .iter()
                    .map(|(_, score)| ((*score - max_score) / tau).exp())
                    .collect();
                let sum: f32 = exp_scores.iter().sum();
                indexed[..k]
                    .iter()
                    .zip(exp_scores.iter())
                    .map(|((idx, _), exp_score)| (available[*idx].clone(), exp_score / sum))
                    .collect()
            }
        };

        let FloorOutcome { weights, dropped } = floor_and_renormalize(raw_weights, self.epsilon);
        self.last_dropped = dropped;
        Ok(weights)
    }

    /// Return the number of inputs the gate network expects.
    pub fn input_size(&self) -> usize {
        self.gate.num_inputs()
    }

    /// Return the number of outputs (maximum supported adapter pool size).
    pub fn output_size(&self) -> usize {
        self.gate.num_outputs()
    }

    /// Submit a refit's candidate gate for loading, gated by the entropy
    /// collapse guard (ADR-091 Decision 3).
    ///
    /// `round_weights` is the weight vector [`route`](Self::route) produced
    /// for every request in the refit round being evaluated — there is no
    /// driver anywhere in this repository that collects these across rounds
    /// yet (see the module doc); whatever eventually does is expected to
    /// call this instead of [`reload`](Self::reload) directly. That is a
    /// deliberate, and only, structural property of this method: routing the
    /// reload through the same call that performs the entropy check means a
    /// future driver's cheapest path to loading a refit is also the guarded
    /// one, rather than the guard being a second call a driver can build
    /// without and still compile. It does not, and cannot, stop a caller
    /// from calling `reload` directly instead — `reload` has to stay
    /// available on its own for non-refit gate swaps, such as initial
    /// provisioning, that have no round to measure entropy over.
    ///
    /// The round's mean [`weight_entropy`] is compared against the
    /// configured entropy floor (see
    /// [`set_entropy_floor`](Self::set_entropy_floor)): a round at or above
    /// the floor resets the consecutive-collapsed count to zero, and a round
    /// below it increments the count. While the count stays below the
    /// configured maximum (see
    /// [`set_max_consecutive_collapsed`](Self::set_max_consecutive_collapsed)),
    /// the refit is accepted and loaded — including individual collapsed
    /// rounds short of that count, since Decision 3 rejects on `M`
    /// *consecutive* collapsed rounds, not on any one of them. Once the
    /// count reaches the configured maximum, the refit is rejected:
    /// `reload` is never called, the live gate is unchanged, and it keeps
    /// serving its previous selection. `reload`'s own no-partial-mutation
    /// contract is what makes this rejection free — there is nothing to
    /// undo, because nothing was mutated.
    ///
    /// The floor cannot tell collapse from a correctly sharp distribution: a
    /// router that has genuinely learned to prefer one adapter produces the
    /// same low-entropy weight vector as one that has collapsed under
    /// sparse reward. This is a guard on refits, not a guard on truth — it
    /// decides which gate gets reloaded, not whether the resulting policy is
    /// good. A rejected refit whose held-out metric (Decision 5) was
    /// improving is the ADR's own stated falsifier for the floor value; that
    /// observation, not intuition about the number, is what should move it.
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::EmptyRefitRound`] if `round_weights` is empty
    /// — there is no request in the round to measure entropy over, and the
    /// consecutive-collapsed count is left unchanged. Otherwise, on
    /// acceptance, returns whatever [`reload`](Self::reload) itself returns.
    pub fn submit_refit(
        &mut self,
        gate_bytes: &[u8],
        round_weights: &[Vec<f32>],
    ) -> Result<RefitOutcome, RouterError> {
        if round_weights.is_empty() {
            return Err(RouterError::EmptyRefitRound);
        }
        let mean_entropy = round_weights
            .iter()
            .map(|weights| weight_entropy(weights))
            .sum::<f32>()
            / round_weights.len() as f32;
        if mean_entropy < self.entropy_floor {
            self.consecutive_collapsed_rounds += 1;
        } else {
            self.consecutive_collapsed_rounds = 0;
        }
        if self.consecutive_collapsed_rounds >= self.max_consecutive_collapsed {
            return Ok(RefitOutcome::Rejected {
                mean_entropy,
                consecutive_collapsed_rounds: self.consecutive_collapsed_rounds,
            });
        }
        self.reload(gate_bytes)?;
        Ok(RefitOutcome::Accepted { mean_entropy })
    }
}

/// Outcome of [`AdapterRouter::submit_refit`]: whether the candidate gate
/// was loaded or rejected by the entropy collapse guard (ADR-091 Decision 3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefitOutcome {
    /// The round's mean weight-entropy was at or above the floor, or below
    /// it for fewer than the configured consecutive-round limit. The
    /// candidate gate was loaded via [`AdapterRouter::reload`].
    Accepted {
        /// Mean weight-entropy (nats) measured over the accepted round.
        mean_entropy: f32,
    },
    /// The round's mean weight-entropy was below the floor for the
    /// configured number of consecutive rounds. `reload` was not called:
    /// the live gate is unchanged and continues to serve its previous
    /// selection.
    Rejected {
        /// Mean weight-entropy (nats) measured over the rejected round.
        mean_entropy: f32,
        /// The consecutive-collapsed-round count that triggered rejection.
        consecutive_collapsed_rounds: usize,
    },
}

/// Shannon entropy, in nats, of a weight vector: `-Σ wᵢ · ln(wᵢ)`.
///
/// Terms where `wᵢ <= 0.0` (including `NaN`, which compares false against
/// `0.0`) contribute `0.0` — the standard `0 · ln(0) := 0` convention,
/// extended defensively to non-positive input rather than propagating a
/// `NaN` or panicking. Does not require `weights` to sum to 1; a caller
/// comparing the result against a floor tuned for normalised weights should
/// pass a normalised vector, such as one [`AdapterRouter::route`] returned.
pub fn weight_entropy(weights: &[f32]) -> f32 {
    weights
        .iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| -w * w.ln())
        .sum()
}

/// Remove any weight strictly below `epsilon`, in original order, without
/// renormalising the survivors.
///
/// This is the gate-computed path's signed comparison. Both weight policies
/// produce non-negative weights, so comparing `w` and comparing `|w|` select
/// the same survivors here. The caller-supplied path, whose weights may be
/// negative, compares magnitude instead — see [`apply_caller_weights`].
fn apply_floor(weights: Vec<(AdapterId, f32)>, epsilon: f32) -> FloorOutcome {
    let mut survivors = Vec::with_capacity(weights.len());
    let mut dropped = Vec::new();
    for (id, weight) in weights {
        if weight < epsilon {
            dropped.push(id);
        } else {
            survivors.push((id, weight));
        }
    }
    FloorOutcome {
        weights: survivors,
        dropped,
    }
}

/// Apply the `epsilon` floor and renormalise the survivors back to summing
/// to 1.
///
/// This is the gate-computed ("learned") path's rule: [`AdapterRouter::route`]
/// always renormalises after a drop, whether its policy is
/// [`WeightPolicy::Uniform`] or [`WeightPolicy::Softmax`] — both are gate
/// output, as opposed to weights a caller supplies directly (see
/// [`apply_caller_weights`]). If the drop removes every survivor, the result
/// is an empty weight vector rather than a division by zero.
fn floor_and_renormalize(weights: Vec<(AdapterId, f32)>, epsilon: f32) -> FloorOutcome {
    let FloorOutcome {
        mut weights,
        dropped,
    } = apply_floor(weights, epsilon);
    if !dropped.is_empty() {
        let sum: f32 = weights.iter().map(|(_, weight)| *weight).sum();
        if sum > 0.0 {
            for (_, weight) in weights.iter_mut() {
                *weight /= sum;
            }
        }
    }
    FloorOutcome { weights, dropped }
}

/// Apply the `epsilon` floor to weights a caller supplied directly, without
/// renormalising the survivors (ADR-091).
///
/// A caller's weight is the caller's own request magnitude — naming one
/// adapter at `0.5` means half strength, and must still mean half strength
/// when a sibling adapter is dropped for falling below `epsilon`. This is
/// the one respect in which the caller-supplied path differs from
/// [`AdapterRouter::route`]'s gate-computed path, which renormalises: raw
/// gate scores have no calibrated magnitude of their own, so they only
/// become comparable proportions by being renormalised, while a caller's
/// scale already is one.
///
/// The floor compares `|w|`, not `w`. Decision 2's reason for dropping rather
/// than damping is a cost argument — an adapter below the floor pays its full
/// rank in the decode of every token and changes nothing — and that is a
/// statement about magnitude. On the gate-computed path the distinction is
/// invisible, since both weight policies produce non-negative weights and the
/// two comparisons agree. On this path they do not: the serving contract
/// accepts negative scales, and a caller naming an adapter at `-0.5` is asking
/// for a full-strength contribution in the other direction. Comparing the
/// signed value would drop it at every non-negative `epsilon` while keeping a
/// `+0.0005` adapter, which is the one the cost argument is actually about.
///
/// A surviving weight keeps its sign and its magnitude: the floor decides
/// presence only. `NaN` survives, here as under the signed comparison, because
/// every ordered comparison against it is false; the serving boundary rejects
/// non-finite scales before reaching this function, but the function is public
/// and says so rather than inheriting it silently.
pub fn apply_caller_weights(weights: Vec<(AdapterId, f32)>, epsilon: f32) -> FloorOutcome {
    let mut survivors = Vec::with_capacity(weights.len());
    let mut dropped = Vec::new();
    for (id, weight) in weights {
        if weight.abs() < epsilon {
            dropped.push(id);
        } else {
            survivors.push((id, weight));
        }
    }
    FloorOutcome {
        weights: survivors,
        dropped,
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

    fn fixed_gate(inputs: usize, outputs: usize, preferred: usize) -> Network {
        let mut layer = lattice_fann::Layer::zeros(inputs, outputs, Activation::Linear).unwrap();
        layer.biases_mut()[preferred] = 1.0;
        Network::new(vec![layer]).unwrap()
    }

    /// A gate whose forward pass returns exactly `scores`, for any input of
    /// the right length: the weight matrix is all zero and the activation is
    /// linear, so the output is the bias vector alone.
    fn scored_gate(inputs: usize, scores: &[f32]) -> Network {
        let mut layer =
            lattice_fann::Layer::zeros(inputs, scores.len(), Activation::Linear).unwrap();
        layer.biases_mut().copy_from_slice(scores);
        Network::new(vec![layer]).unwrap()
    }

    #[test]
    fn reload_changes_selection() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let pool = vec!["first".into(), "second".into()];
        let context = [1.0, 0.5];
        let before = router.route(&context, &pool, 1).unwrap();
        assert_eq!(before, vec![(pool[0].clone(), 1.0)]);
        router.reload(&fixed_gate(2, 2, 1).to_bytes()).unwrap();
        let after = router.route(&context, &pool, 1).unwrap();
        assert_ne!(before, after, "reload must replace the live gate");
        assert_eq!(after, vec![(pool[1].clone(), 1.0)]);
    }

    #[test]
    fn reload_wrong_dimensions_preserves_selection() {
        for (inputs, outputs) in [(3, 2), (2, 3), (3, 3)] {
            let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
            let pool = vec!["first".into(), "second".into()];
            let context = [1.0, 0.5];
            let before = router.route(&context, &pool, 1).unwrap();
            let error = router
                .reload(&fixed_gate(inputs, outputs, 1).to_bytes())
                .unwrap_err();
            assert!(matches!(
                error,
                RouterError::GateDimensionMismatch {
                    expected_inputs: 2,
                    expected_outputs: 2,
                    got_inputs,
                    got_outputs,
                } if got_inputs == inputs && got_outputs == outputs
            ));
            assert_eq!(router.route(&context, &pool, 1).ok(), Some(before));
            assert_eq!(router.input_size(), 2);
            assert_eq!(router.output_size(), 2);
        }
    }

    #[test]
    fn reload_malformed_bytes_preserves_selection() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let pool = vec!["first".into(), "second".into()];
        let context = [1.0, 0.5];
        let before = router.route(&context, &pool, 1).unwrap();
        assert!(matches!(
            router.reload(b"invalid"),
            Err(RouterError::Gate(_))
        ));
        assert_eq!(router.route(&context, &pool, 1).unwrap(), before);
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

    #[test]
    fn route_narrow_gate_returns_err() {
        // Gate has only 3 outputs; available has 5 adapters.
        // k=4 exceeds the 3 usable outputs → must return GateTooNarrow, not panic.
        let mut router = make_router(2, 3);
        let available: Vec<AdapterId> = (0..5).map(|i| format!("a{i}")).collect();
        let ctx = vec![1.0f32; 2];
        let result = router.route(&ctx, &available, 4);
        assert!(
            matches!(result, Err(RouterError::GateTooNarrow { k: 4, usable: 3 })),
            "expected GateTooNarrow {{k:4, usable:3}}, got {result:?}"
        );
    }

    /// A pool wider than the gate, with `k` small enough to stay inside
    /// `GateTooNarrow`, must not error and must never surface an adapter past
    /// the gate's own output width. This is deliberate, pre-existing
    /// `AdapterRouter` behaviour, not a bounds bug: a router that must cover a
    /// wider pool needs a wider gate (a new `AdapterRouter`, not a change to
    /// this function), and refusing a live width disagreement belongs one
    /// layer up, at router construction, not inside `route` itself.
    #[test]
    fn route_pool_wider_than_gate_never_selects_the_unscored_adapter() {
        // Two-output gate, three candidates. The gate scores favour index 1
        // ("second"), so the top-1 and top-2 selections both exercise a real
        // ranking rather than an arbitrary tie-break, and "third" has no
        // scored column at all -- there is no score assignment under which it
        // could win.
        let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 10.0]));
        let available: Vec<AdapterId> = vec!["first".into(), "second".into(), "third".into()];

        for k in [1usize, 2usize] {
            let result = router
                .route(&[0.0], &available, k)
                .unwrap_or_else(|e| panic!("k={k}: expected Ok, got {e:?}"));
            assert_eq!(
                result.len(),
                k,
                "k={k}: full k must be satisfiable from the 2 scored adapters"
            );
            assert!(
                result.iter().all(|(id, _)| id != "third"),
                "k={k}: an adapter beyond the gate's own output width must never be selected, got {result:?}"
            );
        }
    }

    #[test]
    fn route_duplicate_adapter_ids_returns_err() {
        // available contains "same" twice; k=2 would select it twice (fail-open).
        // The router must reject this with DuplicateAdapterId before running the gate.
        let mut router = make_router(2, 3);
        let available: Vec<AdapterId> = vec!["same".into(), "same".into(), "other".into()];
        let ctx = vec![1.0f32; 2];
        let result = router.route(&ctx, &available, 2);
        assert!(
            matches!(result, Err(RouterError::DuplicateAdapterId { .. })),
            "duplicate adapter id should return DuplicateAdapterId error, got {result:?}"
        );
    }

    #[test]
    fn softmax_weights_sum_to_one_nonuniform() {
        let mut router = AdapterRouter::new(scored_gate(1, &[3.0, 1.0, 0.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
        let available: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
        let result = router.route(&[0.0], &available, 3).unwrap();
        let sum: f32 = result.iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax weights must sum to 1, got {sum}"
        );
        let uniform = 1.0 / 3.0;
        assert!(
            result.iter().any(|(_, w)| (w - uniform).abs() > 1e-3),
            "softmax at tau=1.0 over distinct scores must not degenerate to uniform"
        );
    }

    #[test]
    fn uniform_default_ignores_score_magnitude() {
        // No set_weight_policy call: the default must be Uniform, not merely
        // a large tau. Scores are wildly different, which a softmax could
        // only make exactly uniform in the tau -> infinity limit.
        let mut router = AdapterRouter::new(scored_gate(1, &[1000.0, 0.0, -1000.0]));
        let available: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
        let result = router.route(&[0.0], &available, 3).unwrap();
        let expected = 1.0 / 3.0;
        for (_, w) in &result {
            assert!(
                (w - expected).abs() < 1e-6,
                "default policy must be exactly uniform regardless of score spread, got {w}"
            );
        }
    }

    #[test]
    fn route_drops_adapter_below_epsilon_and_shrinks_length() {
        let mut router = AdapterRouter::new(scored_gate(1, &[10.0, 0.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
        router.set_epsilon(1e-3);
        let available: Vec<AdapterId> = vec!["strong".into(), "weak".into()];
        let result = router.route(&[0.0], &available, 2).unwrap();
        assert_eq!(
            result.len(),
            1,
            "the below-epsilon adapter must be dropped, not damped, and length must shrink below k"
        );
        assert!(
            !result.iter().any(|(id, _)| id == "weak"),
            "a dropped adapter must be absent from the result, not present at a small weight"
        );
        assert_eq!(router.last_dropped(), &["weak".to_string()]);
    }

    #[test]
    fn floor_and_renormalize_drops_and_renormalizes_survivors() {
        let weights = vec![
            ("a".to_string(), 0.6),
            ("b".to_string(), 0.39),
            ("c".to_string(), 0.01),
        ];
        let outcome = floor_and_renormalize(weights, 0.05);
        assert_eq!(outcome.dropped, vec!["c".to_string()]);
        assert_eq!(outcome.weights.len(), 2);
        let sum: f32 = outcome.weights.iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "survivors must renormalise to sum 1, got {sum}"
        );
        let a_weight = outcome.weights.iter().find(|(id, _)| id == "a").unwrap().1;
        assert!(
            (a_weight - 0.6).abs() > 1e-6,
            "a surviving weight must be rescaled by the drop, not left as-is, got {a_weight}"
        );
    }

    #[test]
    fn caller_weights_not_renormalized_after_drop() {
        // The isolating arm for the two rules: a caller asking for 0.5 must
        // still get exactly 0.5 when a sibling is dropped, unlike the
        // gate-computed path, which renormalises.
        let weights = vec![("kept".to_string(), 0.5), ("dropped".to_string(), 0.01)];
        let outcome = apply_caller_weights(weights, 0.05);
        assert_eq!(outcome.dropped, vec!["dropped".to_string()]);
        assert_eq!(outcome.weights, vec![("kept".to_string(), 0.5)]);
    }

    /// ADR-091 Decision 4's floor compares `|w|`. This is the arm that
    /// separates that rule from the signed one, and it is load-bearing:
    /// reverting `apply_caller_weights` to `apply_floor` reddens it.
    ///
    /// On this one input the rules give different survivor sets. The signed
    /// rule drops both entries, since `-0.5 < 0.001` and `0.0005 < 0.001`.
    /// The magnitude rule keeps the full-strength negative adapter and drops
    /// only the one the cost argument is about. A fixture built from
    /// non-negative weights cannot express this, which is why both arms are
    /// asserted here on the same input.
    #[test]
    fn caller_floor_compares_magnitude_not_the_signed_weight() {
        let weights = vec![("negative".to_string(), -0.5), ("tiny".to_string(), 0.0005)];
        let outcome = apply_caller_weights(weights.clone(), 0.001);

        assert_eq!(
            outcome.weights,
            vec![("negative".to_string(), -0.5)],
            "a full-strength negative scale is above the floor in magnitude and must survive"
        );
        assert_eq!(
            outcome.dropped,
            vec!["tiny".to_string()],
            "only the adapter that pays its rank and changes nothing may be dropped"
        );

        // The same input under the gate-computed path's signed rule, so the
        // divergence is exhibited rather than asserted. This is what the
        // caller path must NOT do.
        let signed = apply_floor(weights, 0.001);
        assert_eq!(
            signed.dropped,
            vec!["negative".to_string(), "tiny".to_string()],
            "the signed rule drops the negative adapter too -- that is the defect this arm pins"
        );
    }

    /// At the default `epsilon` the caller path drops nothing, negative scales
    /// included, so wiring the floor into serving cannot change behaviour
    /// until an operator sets a positive floor.
    ///
    /// The consequence stated rather than left to be discovered: a
    /// `scale: 0.0` adapter — the very case Decision 2's cost argument names,
    /// since it pays its full rank for exactly zero output change — ALSO
    /// survives here, because `0.0 < 0.0` is false. The cost argument only
    /// bites once `epsilon` is positive, and ADR-091 leaves that number open
    /// on purpose.
    #[test]
    fn caller_floor_at_zero_epsilon_drops_nothing_including_zero_and_negative() {
        let weights = vec![
            ("negative".to_string(), -0.5),
            ("zero".to_string(), 0.0),
            ("tiny".to_string(), 1e-9),
            ("normal".to_string(), 1.0),
        ];
        let outcome = apply_caller_weights(weights.clone(), 0.0);

        assert!(
            outcome.dropped.is_empty(),
            "epsilon = 0.0 must be a no-op on the caller path, got dropped {:?}",
            outcome.dropped
        );
        assert_eq!(
            outcome.weights, weights,
            "every entry must survive unchanged, in original order"
        );
    }

    /// The floor decides presence only. A survivor's sign and magnitude are
    /// the caller's request and are returned untouched, including when a
    /// sibling is dropped -- the non-renormalisation property of Decision 4,
    /// exercised here on a negative weight where a stray `abs()` in the
    /// returned value would be invisible to the positive-only fixtures.
    #[test]
    fn caller_floor_preserves_a_surviving_negative_weight_exactly() {
        let weights = vec![("kept".to_string(), -0.75), ("dropped".to_string(), -0.002)];
        let outcome = apply_caller_weights(weights, 0.01);

        assert_eq!(outcome.dropped, vec!["dropped".to_string()]);
        assert_eq!(
            outcome.weights,
            vec![("kept".to_string(), -0.75)],
            "sign and magnitude of a survivor are the caller's request, not a normalised value"
        );
    }

    /// The confinement claim, tested rather than asserted: the gate-computed
    /// path keeps the signed comparison and is unchanged by this fix. Its
    /// weights are a softmax and therefore non-negative, which is exactly why
    /// the two comparisons agree there -- so this arm would stay green under
    /// either rule, and its job is to catch a fix that widened its blast
    /// radius into `route`.
    #[test]
    fn learned_path_floor_is_unchanged_by_the_caller_path_rule() {
        let mut router = AdapterRouter::new(scored_gate(1, &[4.0, 2.0, 2.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: 0.5 });
        router.set_epsilon(1e-3);
        let available: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
        let result = router.route(&[0.0], &available, 3).unwrap();

        for (id, w) in &result {
            assert!(
                *w >= 0.0,
                "gate-computed weights are non-negative by construction; {id} was {w}"
            );
        }
        let sum: f32 = result.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "the learned path still renormalises to 1 over the selected set, got {sum}"
        );
    }

    #[test]
    fn softmax_equal_scores_gives_uniform_weights() {
        let mut router = AdapterRouter::new(scored_gate(1, &[2.0, 2.0, 2.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: 0.7 });
        let available: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
        let result = router.route(&[0.0], &available, 3).unwrap();
        let expected = 1.0 / 3.0;
        for (_, w) in &result {
            assert!(
                w.is_finite(),
                "equal scores must not produce a NaN/inf weight"
            );
            assert!(
                (w - expected).abs() < 1e-6,
                "equal scores must softmax to uniform, got {w}"
            );
        }
    }

    #[test]
    fn softmax_large_scores_do_not_overflow() {
        // Without subtracting the max score first, exp(1000.0) overflows f32
        // to infinity and inf/inf yields NaN. This is the case the
        // max-subtraction exists for.
        let mut router = AdapterRouter::new(scored_gate(1, &[1000.0, 999.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        let result = router.route(&[0.0], &available, 2).unwrap();
        for (_, w) in &result {
            assert!(
                w.is_finite(),
                "large selected scores must not overflow to NaN/inf"
            );
        }
        let sum: f32 = result.iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights must still sum to 1, got {sum}"
        );
    }

    #[test]
    fn softmax_rejects_degenerate_tau() {
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        for tau in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 0.0]));
            router.set_weight_policy(WeightPolicy::Softmax { tau });
            let result = router.route(&[0.0], &available, 2);
            assert!(
                matches!(result, Err(RouterError::InvalidTau { .. })),
                "tau={tau} must be rejected as InvalidTau, got {result:?}"
            );
        }
    }

    #[test]
    fn route_last_dropped_distinguishes_floored_from_unselected() {
        // Three candidates; k=2 selects the top two by score ("strong" and
        // "weak"); "unselected" never reaches the top-k at all. "weak" is
        // then floored. last_dropped must report "weak" and nothing else.
        let mut router = AdapterRouter::new(scored_gate(1, &[10.0, 0.0, -100.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
        router.set_epsilon(1e-3);
        let available: Vec<AdapterId> = vec!["strong".into(), "weak".into(), "unselected".into()];
        let result = router.route(&[0.0], &available, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "strong");
        assert_eq!(router.last_dropped(), &["weak".to_string()]);
        assert!(
            !router.last_dropped().contains(&"unselected".to_string()),
            "an adapter that never reached the top-k must not appear in last_dropped"
        );
    }

    // ─── ADR-091 Decision 3, guard (a): tau floor ──────────────────────────

    #[test]
    fn softmax_rejects_tau_below_floor() {
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 0.0]));
        // A `const` block, so a future edit that lowers DEFAULT_TAU_FLOOR under
        // 1e-9 fails to COMPILE rather than silently turning this test into an
        // assertion about a tau at or above the floor. A runtime assert! here
        // is also what clippy::assertions_on_constants rejects.
        const {
            assert!(
                DEFAULT_TAU_FLOOR > 1e-9,
                "test assumes 1e-9 is below the default floor"
            )
        };
        router.set_weight_policy(WeightPolicy::Softmax { tau: 1e-9 });
        let result = router.route(&[0.0], &available, 2);
        assert!(
            matches!(
                result,
                Err(RouterError::TauBelowFloor { tau, floor })
                    if tau == 1e-9 && floor == DEFAULT_TAU_FLOOR
            ),
            "tau=1e-9 (finite, positive, below the default floor) must be \
             rejected as TauBelowFloor, got {result:?}"
        );
    }

    #[test]
    fn softmax_accepts_tau_immediately_above_floor() {
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 0.0]));
        let above_floor = DEFAULT_TAU_FLOOR * 1.01;
        router.set_weight_policy(WeightPolicy::Softmax { tau: above_floor });
        let result = router.route(&[0.0], &available, 2);
        assert!(
            result.is_ok(),
            "tau just above the floor must be accepted, got {result:?}"
        );
    }

    #[test]
    fn softmax_accepts_tau_exactly_at_floor() {
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 0.0]));
        router.set_weight_policy(WeightPolicy::Softmax {
            tau: DEFAULT_TAU_FLOOR,
        });
        let result = router.route(&[0.0], &available, 2);
        assert!(
            result.is_ok(),
            "tau exactly at the floor must be accepted (floor is inclusive), got {result:?}"
        );
    }

    #[test]
    fn set_tau_floor_is_respected() {
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 0.0]));
        router.set_tau_floor(0.5);
        router.set_weight_policy(WeightPolicy::Softmax { tau: 0.3 });
        let result = router.route(&[0.0], &available, 2);
        assert!(
            matches!(
                result,
                Err(RouterError::TauBelowFloor { tau, floor })
                    if tau == 0.3 && floor == 0.5
            ),
            "a raised floor must reject a tau the default floor would accept, got {result:?}"
        );
    }

    #[test]
    fn tau_floor_check_runs_after_basic_validity_check() {
        // A degenerate tau (here NaN) must still resolve to InvalidTau, never
        // to TauBelowFloor: `NaN < floor` is false, so an ordering bug that
        // ran the floor check first would let NaN fall through this branch
        // entirely and reach the softmax computation.
        let available: Vec<AdapterId> = vec!["a".into(), "b".into()];
        let mut router = AdapterRouter::new(scored_gate(1, &[1.0, 0.0]));
        router.set_weight_policy(WeightPolicy::Softmax { tau: f32::NAN });
        let result = router.route(&[0.0], &available, 2);
        assert!(
            matches!(result, Err(RouterError::InvalidTau { .. })),
            "NaN tau must be rejected as InvalidTau, not TauBelowFloor or Ok, got {result:?}"
        );
    }

    // ─── ADR-091 Decision 3, guard (c): entropy floor + refit rejection ────

    #[test]
    fn weight_entropy_flat_is_ln_k() {
        let k = 4;
        let flat = vec![1.0 / k as f32; k];
        let entropy = weight_entropy(&flat);
        let expected = (k as f32).ln();
        assert!(
            (entropy - expected).abs() < 1e-5,
            "flat k={k} entropy must be ln(k)={expected}, got {entropy}"
        );
    }

    #[test]
    fn weight_entropy_skewed_is_lower_than_flat() {
        let flat = vec![0.25, 0.25, 0.25, 0.25];
        let skewed = vec![0.97, 0.01, 0.01, 0.01];
        let flat_entropy = weight_entropy(&flat);
        let skewed_entropy = weight_entropy(&skewed);
        assert!(
            skewed_entropy < flat_entropy,
            "a skewed distribution must have lower entropy than the flat one: \
             skewed={skewed_entropy}, flat={flat_entropy}"
        );
    }

    #[test]
    fn weight_entropy_one_hot_is_zero() {
        let one_hot = vec![1.0, 0.0, 0.0];
        assert_eq!(weight_entropy(&one_hot), 0.0);
    }

    #[test]
    fn submit_refit_empty_round_is_rejected_as_error() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let result = router.submit_refit(&fixed_gate(2, 2, 1).to_bytes(), &[]);
        assert!(matches!(result, Err(RouterError::EmptyRefitRound)));
    }

    /// `M-1` consecutive collapsed rounds must not reject: each round is
    /// individually below the floor, but Decision 3 rejects on `M`
    /// *consecutive* collapsed rounds, not on any single one.
    #[test]
    fn submit_refit_m_minus_one_collapsed_rounds_does_not_reject() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let collapsed_round = vec![vec![1.0, 0.0]]; // one-hot: entropy 0.0, collapsed
        for _ in 0..DEFAULT_MAX_CONSECUTIVE_COLLAPSED - 1 {
            let outcome = router
                .submit_refit(&fixed_gate(2, 2, 1).to_bytes(), &collapsed_round)
                .unwrap();
            assert!(
                matches!(outcome, RefitOutcome::Accepted { .. }),
                "fewer than M consecutive collapsed rounds must still be accepted, got {outcome:?}"
            );
        }
        assert_eq!(
            router.consecutive_collapsed_rounds(),
            DEFAULT_MAX_CONSECUTIVE_COLLAPSED - 1
        );
    }

    /// The `M`-th consecutive collapsed round rejects, and the live gate
    /// still produces its previous selection afterwards — the same
    /// preserve-on-failure assertion style as `reload_malformed_bytes_preserves_selection`.
    #[test]
    fn submit_refit_mth_consecutive_collapsed_round_rejects_and_preserves_selection() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let pool = vec!["first".into(), "second".into()];
        let context = [1.0, 0.5];
        let collapsed_round = vec![vec![1.0, 0.0]];
        for _ in 0..DEFAULT_MAX_CONSECUTIVE_COLLAPSED - 1 {
            router
                .submit_refit(&fixed_gate(2, 2, 1).to_bytes(), &collapsed_round)
                .unwrap();
        }
        let before = router.route(&context, &pool, 1).unwrap();
        let outcome = router
            .submit_refit(&fixed_gate(2, 2, 1).to_bytes(), &collapsed_round)
            .unwrap();
        assert!(
            matches!(
                outcome,
                RefitOutcome::Rejected {
                    consecutive_collapsed_rounds,
                    ..
                } if consecutive_collapsed_rounds == DEFAULT_MAX_CONSECUTIVE_COLLAPSED
            ),
            "the Mth consecutive collapsed round must reject, got {outcome:?}"
        );
        let after = router.route(&context, &pool, 1).unwrap();
        assert_eq!(
            before, after,
            "a rejected refit must never call reload; live gate must be unchanged"
        );
    }

    /// One good round in the middle of a collapsed streak resets the count:
    /// collapsed, collapsed, good, collapsed, collapsed must not reject
    /// (default M=3), because the streak never reaches 3 consecutive.
    #[test]
    fn submit_refit_good_round_in_middle_resets_count() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let collapsed_round = vec![vec![1.0, 0.0]];
        let good_round = vec![vec![0.5, 0.5]]; // flat over k=2: entropy ln(2) > default floor
        let sequence = [
            &collapsed_round,
            &collapsed_round,
            &good_round,
            &collapsed_round,
            &collapsed_round,
        ];
        for round in sequence {
            let outcome = router
                .submit_refit(&fixed_gate(2, 2, 1).to_bytes(), round)
                .unwrap();
            assert!(
                matches!(outcome, RefitOutcome::Accepted { .. }),
                "a good round mid-streak must prevent the M-consecutive threshold \
                 from ever being reached, got {outcome:?}"
            );
        }
        assert_eq!(router.consecutive_collapsed_rounds(), 2);
    }

    #[test]
    fn submit_refit_accepted_round_reports_mean_entropy_and_reloads() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        let pool = vec!["first".into(), "second".into()];
        let context = [1.0, 0.5];
        let before = router.route(&context, &pool, 1).unwrap();
        let good_round = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let outcome = router
            .submit_refit(&fixed_gate(2, 2, 1).to_bytes(), &good_round)
            .unwrap();
        let expected_entropy = 2.0f32.ln();
        assert!(
            matches!(
                outcome,
                RefitOutcome::Accepted { mean_entropy }
                    if (mean_entropy - expected_entropy).abs() < 1e-5
            ),
            "expected Accepted with mean_entropy={expected_entropy}, got {outcome:?}"
        );
        let after = router.route(&context, &pool, 1).unwrap();
        assert_ne!(
            before, after,
            "an accepted refit must actually reload the gate"
        );
    }

    #[test]
    fn set_entropy_floor_and_set_max_consecutive_collapsed_are_respected() {
        let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
        // Raise the entropy floor above ln(2) so even the flat k=2
        // distribution counts as collapsed, and lower M to 1 so a single
        // collapsed round rejects immediately.
        router.set_entropy_floor(10.0);
        router.set_max_consecutive_collapsed(1);
        let flat_round = vec![vec![0.5, 0.5]];
        let outcome = router
            .submit_refit(&fixed_gate(2, 2, 1).to_bytes(), &flat_round)
            .unwrap();
        assert!(
            matches!(outcome, RefitOutcome::Rejected { .. }),
            "a raised floor and M=1 must reject on the first round, got {outcome:?}"
        );
    }
}
