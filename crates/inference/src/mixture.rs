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
//!
//! [`AdapterRouter::route`]: crate::mixture::AdapterRouter::route
//! [`WeightPolicy`]: crate::mixture::WeightPolicy
//! [`WeightPolicy::Softmax`]: crate::mixture::WeightPolicy::Softmax
//! [`AdapterRouter::set_weight_policy`]: crate::mixture::AdapterRouter::set_weight_policy
//! [`AdapterRouter::set_epsilon`]: crate::mixture::AdapterRouter::set_epsilon
//! [`AdapterRouter::set_tau_floor`]: crate::mixture::AdapterRouter::set_tau_floor
//! [`RouterError::TauBelowFloor`]: crate::mixture::RouterError::TauBelowFloor
//! [`AdapterRouter::submit_refit`]: crate::mixture::AdapterRouter::submit_refit

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

    /// The exploration scale was not usable: it must be finite and strictly
    /// positive. The draw is Gaussian on the raw logits, `z' = z + e` with
    /// `e ~ N(0, s^2 I)`, so a zero scale is no perturbation at all while
    /// claiming to record one, a negative scale is a distribution with the
    /// same density (the square removes the sign) that reads as a different
    /// policy, and a non-finite scale has no density to score.
    #[error("exploration scale must be finite and > 0.0, got {scale}")]
    InvalidExplorationScale {
        /// The offending scale.
        scale: f32,
    },

    /// The exploration scale was finite and strictly positive (so it passed
    /// the basic validity check) but fell below [`MIN_EXPLORATION_SCALE`].
    #[error("exploration scale {scale} is below the minimum {floor}")]
    ExplorationScaleBelowFloor {
        /// The offending (too-small but otherwise valid) scale.
        scale: f32,
        /// The minimum it fell below. Widened to the artifact's own type so
        /// the router and the file it loads compare the same number.
        floor: f64,
    },

    /// A draw was requested in a configuration where it cannot reach the
    /// action: [`WeightPolicy::Uniform`] over every candidate the gate scores.
    ///
    /// The selected set is then all of them whatever the perturbation does,
    /// and every weight is `1/k` whatever the scores are, so the served
    /// mixture is the same for any gate and any seed. The draw would still
    /// record a scale, a seed and a set of sampled logits -- known variation
    /// in the event log that the action never had.
    #[error(
        "a draw cannot change this decision: the uniform policy over all {candidates} scored \
         candidate(s) with k={k} selects the whole set and weights it 1/k regardless of the \
         gate's scores; narrow k, or name a policy that reads the scores"
    )]
    ExplorationCannotChangeTheAction {
        /// The requested top-k, equal to `candidates`.
        k: usize,
        /// The number of candidates the gate can score.
        candidates: usize,
    },
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

/// Everything a routing decision was made from, returned by
/// [`AdapterRouter::route_detailed`] and [`AdapterRouter::route_explored`].
///
/// [`AdapterRouter::route`] returns only [`RouteOutcome::weights`]; this
/// carries the sampled logits, the pre-floor weights and the exploration draw
/// so a learner reading a decision can see what the policy proposed as well as
/// what was served. The floor can drop an adapter entirely, so the applied
/// weights alone cannot say what the policy proposed.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteOutcome {
    /// The weights the request ran with, after the floor and renormalisation.
    pub weights: Vec<(AdapterId, f32)>,
    /// The weights the policy produced, before the floor.
    pub pre_floor_weights: Vec<(AdapterId, f32)>,
    /// The gate scores `z` of the selected columns, in selection order.
    pub selected_scores: Vec<(AdapterId, f32)>,
    /// The logits the selection and the weight policy actually consumed, for
    /// the selected columns, in selection order: `z` on the deterministic
    /// path and `z'` under a draw.
    pub sampled_scores: Vec<(AdapterId, f32)>,
    /// Adapters the floor removed. Also readable from
    /// [`AdapterRouter::last_dropped`], which this does not replace.
    pub dropped: Vec<AdapterId>,
    /// The exploration draw that produced this decision, or `None` when the
    /// decision was the gate's deterministic one. `selected_scores` holds the
    /// gate's own scores either way, so a learner reading an outcome can tell
    /// what the policy computed from what the draw served without having to
    /// know which mode the request ran in.
    pub exploration: Option<ExplorationOutcome>,
}

/// Default floor on the softmax temperature `tau` (ADR-091 Decision 3,
/// [`AdapterRouter::set_tau_floor`]).
///
/// Re-exported rather than defined here: see
/// [`crate::router_state::DEFAULT_TAU_FLOOR`] for the value and why it is
/// enforced at both the artifact and the router.
pub use crate::router_state::DEFAULT_TAU_FLOOR;

/// Default entropy floor, in nats, below which a refit round's mean
/// weight-entropy counts as collapsed (ADR-091 Decision 3,
/// [`AdapterRouter::submit_refit`]).
///
/// Re-exported: see [`crate::router_state::DEFAULT_ENTROPY_FLOOR`].
pub use crate::router_state::DEFAULT_ENTROPY_FLOOR;

/// Default number of consecutive collapsed rounds tolerated before
/// [`AdapterRouter::submit_refit`] rejects a refit (ADR-091 Decision 3).
///
/// Re-exported: see [`crate::router_state::DEFAULT_MAX_CONSECUTIVE_COLLAPSED`].
pub use crate::router_state::DEFAULT_MAX_CONSECUTIVE_COLLAPSED;

/// Smallest exploration scale [`AdapterRouter::route_explored`] accepts.
///
/// Re-exported rather than defined here, because the artifact validates
/// against the same number and two constants with one meaning drift: see
/// [`crate::router_state::MIN_EXPLORATION_SCALE`] for what the value is and
/// why it refuses instead of clamping.
pub use crate::router_state::MIN_EXPLORATION_SCALE;

/// The policy epoch a freshly constructed [`AdapterRouter`] serves under
/// (ADR-096 decision 3).
///
/// A router that has never been reconfigured and has never reloaded a gate is
/// still serving the policy it was constructed with, and that is an epoch of
/// its own rather than an absence of one: an episode recorded under it is an
/// on-policy sample for it.
pub const INITIAL_POLICY_EPOCH: u64 = 0;

/// A request to draw the routing decision instead of computing it
/// (ADR-096 decision 3).
///
/// Exploration is per-call rather than router state on purpose: the scale is
/// recorded on every event it produced, so a learner can weight an event by
/// the scale that actually drew it. A scale held on the router would be a
/// second place for the served configuration to live, and the recorded events
/// would be the copy that drifts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exploration {
    /// Standard deviation of the Gaussian perturbation added to the gate's
    /// raw logits. Must be finite, strictly positive, and at least
    /// [`MIN_EXPLORATION_SCALE`].
    pub scale: f32,
    /// Seed for this decision's draw. The caller owns seed derivation, so a
    /// server can tie the seed to the completion id and reproduce the exact
    /// mixture a request ran with from the recorded event alone.
    pub seed: u64,
}

/// What an exploration draw did, recorded beside the decision it produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplorationOutcome {
    /// The scale the draw used.
    pub scale: f32,
    /// The seed the draw used.
    pub seed: u64,
    /// The sampled logits `z'` the selection and the weight policy consumed:
    /// one value per gate output, in gate column order, including outputs the
    /// selection or the floor removed. These are what the density below is
    /// scored against; [`RouteOutcome::selected_scores`] holds the
    /// deterministic gate scores `z` of the selected adapters, and
    /// [`RouteOutcome::sampled_scores`] holds `z'` for those same adapters.
    pub sampled_logits: Vec<f32>,
    /// The density of `z'` given `z` under `N(z, s^2 I)`, as a natural log:
    /// `-||z' - z||^2 / (2 s^2) - (n / 2) ln(2 pi s^2)`, with `n` the number
    /// of gate outputs. This is the quantity the ADR scores, chosen over the
    /// density of the applied weights because ADR-091's floor puts the
    /// weight distribution on a lower-dimensional face with no Lebesgue
    /// density. Recorded for audit; a learner takes its gradient,
    /// `(z' - z) / s^2`, which needs no derivative of the floor or the
    /// softmax.
    pub log_density: f64,
}

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
    policy_epoch: u64,
}

impl AdapterRouter {
    #[cfg(feature = "router-learning")]
    pub(crate) fn replay_recorded_logits(
        &self,
        logits: &[f32],
        available: &[AdapterId],
        exploration: Exploration,
    ) -> Result<RouteOutcome, RouterError> {
        let mut layer =
            lattice_fann::Layer::zeros(1, logits.len(), lattice_fann::Activation::Linear)?;
        layer.biases_mut().copy_from_slice(logits);
        let mut replay = self.refit_snapshot();
        replay.gate = Network::new(vec![layer])?;
        replay.route_explored(&[0.0], available, available.len(), Some(exploration))
    }

    #[cfg(all(feature = "router-learning", feature = "serve"))]
    pub(crate) fn gate_bytes(&self) -> Vec<u8> {
        self.gate.to_bytes()
    }

    #[cfg(feature = "router-learning")]
    pub(crate) fn refit_snapshot(&self) -> Self {
        Self {
            gate: self.gate.clone(),
            weight_policy: self.weight_policy,
            epsilon: self.epsilon,
            last_dropped: self.last_dropped.clone(),
            tau_floor: self.tau_floor,
            entropy_floor: self.entropy_floor,
            max_consecutive_collapsed: self.max_consecutive_collapsed,
            consecutive_collapsed_rounds: self.consecutive_collapsed_rounds,
            policy_epoch: self.policy_epoch,
        }
    }

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
            policy_epoch: INITIAL_POLICY_EPOCH,
        }
    }

    /// Replace the gate from a complete [`Network::to_bytes`] blob.
    ///
    /// Parsing and both dimensions must validate before any mutation: a failed
    /// reload leaves the live gate intact so a bad refit fails once, rather than
    /// disrupting subsequent requests. A failed reload also leaves
    /// [`policy_epoch`](Self::policy_epoch) alone, because nothing about what
    /// the next request runs has changed.
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
        self.policy_epoch = self.policy_epoch.saturating_add(1);
        Ok(())
    }

    /// Set the mixture weight policy used by subsequent
    /// [`route`](Self::route) calls.
    ///
    /// The default, from [`AdapterRouter::new`], is [`WeightPolicy::Uniform`].
    ///
    /// A change to the policy advances [`policy_epoch`](Self::policy_epoch);
    /// re-applying the policy already in force does not.
    pub fn set_weight_policy(&mut self, policy: WeightPolicy) {
        if self.weight_policy != policy {
            self.weight_policy = policy;
            self.policy_epoch = self.policy_epoch.saturating_add(1);
        }
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
    ///
    /// A change advances [`policy_epoch`](Self::policy_epoch); re-applying the
    /// same value does not. Values are compared bit for bit rather than by
    /// `==`, and that is what makes "the same value" true for a NaN: a NaN
    /// compares false against everything, itself included, so `!=` would read
    /// every re-application of the same disabling NaN as a policy change and
    /// advance the epoch on each call -- ending the epoch of the samples a
    /// server that re-applies its configuration per request is trying to
    /// accumulate. `to_bits` compares the payload, so the first NaN is a
    /// change and every identical one after it is not.
    pub fn set_epsilon(&mut self, epsilon: f32) {
        if self.epsilon.to_bits() != epsilon.to_bits() {
            self.epsilon = epsilon;
            self.policy_epoch = self.policy_epoch.saturating_add(1);
        }
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

    /// The mixture weight policy this router currently serves under
    /// (ADR-091 decision 1).
    ///
    /// Reported so a caller can record the temperature that produced a
    /// decision: ADR-096 decision 3 requires an event to carry "the `tau` that
    /// produced them", and a temperature read from anywhere other than the
    /// policy in force at the decision is a different claim. `Uniform` carries
    /// no temperature, which is why the served temperature is an `Option`.
    pub fn weight_policy(&self) -> WeightPolicy {
        self.weight_policy
    }

    /// The `epsilon` floor this router currently applies, for the same reason
    /// [`weight_policy`](Self::weight_policy) is reported: a surface that says
    /// what a request runs under has to read the value in force rather than
    /// the value a file records.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
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
    ///
    /// A change advances [`policy_epoch`](Self::policy_epoch); re-applying the
    /// same value does not. Compared by bit pattern for the same reason
    /// [`set_epsilon`](Self::set_epsilon) is: a NaN floor is accepted and
    /// disables the check, and `!=` would charge an epoch bump to every
    /// identical re-application of it.
    pub fn set_tau_floor(&mut self, floor: f32) {
        if self.tau_floor.to_bits() != floor.to_bits() {
            self.tau_floor = floor;
            self.policy_epoch = self.policy_epoch.saturating_add(1);
        }
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

    /// The policy epoch this router currently serves under (ADR-096
    /// decision 3).
    ///
    /// Starts at [`INITIAL_POLICY_EPOCH`] and advances whenever a mutation
    /// changes what a request would actually run with: a successful gate
    /// [`reload`](Self::reload) — which is also how an accepted
    /// [`submit_refit`](Self::submit_refit) advances it — or a change to the
    /// weight policy, the `epsilon` floor, or the `tau` floor. A feedback
    /// event recorded under one epoch is not an on-policy sample for a later
    /// one.
    ///
    /// It deliberately does **not** advance for the refit guards
    /// ([`set_entropy_floor`](Self::set_entropy_floor),
    /// [`set_max_consecutive_collapsed`](Self::set_max_consecutive_collapsed)):
    /// those decide which future gate is accepted and change no request's
    /// action distribution, so bumping on them would discard on-policy samples
    /// that are still on-policy. The falsifier is one of those guards starting
    /// to influence a served decision; then it belongs in the list above.
    ///
    /// The setters compare before they bump, so re-applying an identical
    /// configuration is not a policy change. A server that re-applies its
    /// config per request would otherwise never accumulate a single on-policy
    /// sample and would report that as normal operation.
    pub fn policy_epoch(&self) -> u64 {
        self.policy_epoch
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
        self.route_detailed(context_vector, available, k)
            .map(|outcome| outcome.weights)
    }

    /// Route, and return everything the decision was made from.
    ///
    /// The same computation as [`route`](Self::route), which is a wrapper over
    /// this. One implementation, deliberately: a second body that produced the
    /// weights alongside their provenance would be a copy of the routing rule,
    /// and the copy that drifts is the one the audit trail is built on -- so
    /// the recorded logits would stop being the logits that were served
    /// without anything failing.
    ///
    /// ADR-096 decision 3 requires a feedback event to carry the sampled
    /// logits and the pre-floor weights alongside the applied ones. The floor
    /// can drop an adapter entirely, so the applied weights alone cannot say
    /// what the policy proposed, and a learner reading only them would credit
    /// a mixture that was never formed.
    pub fn route_detailed(
        &mut self,
        context_vector: &[f32],
        available: &[AdapterId],
        k: usize,
    ) -> Result<RouteOutcome, RouterError> {
        self.route_explored(context_vector, available, k, None)
    }

    /// Route, optionally drawing the decision instead of computing it.
    ///
    /// The single routing body: [`route`](Self::route) and
    /// [`route_detailed`](Self::route_detailed) are both wrappers over this,
    /// and `exploration: None` reproduces them exactly, bit for bit. ADR-096
    /// decision 3 requires the served mixture to be the learned action, which
    /// means the exploration has to happen inside the decision rather than
    /// beside it -- a sampler that picked a mixture and then asked the router
    /// to weight it would record logits the request never ran with.
    ///
    /// # Exploration
    ///
    /// Under `Some(exploration)` the gate's raw logits `z` gain independent
    /// Gaussian noise at the given scale, `z' = z + e` with `e ~ N(0, s^2 I)`,
    /// and the top-`k` selection and the weight policy both consume `z'`
    /// exactly as the deterministic path consumes `z`, followed by the same
    /// floor. This is an explicit amendment to ADR-091 decision 1, stated in
    /// ADR-096 decision 3: during enabled learning the logits are not solely
    /// the deterministic scores the gate computes. The argument is
    /// identifiability -- one scalar judgement of one deterministic mixture
    /// contains neither a target column nor any observation of an alternative
    /// mixture, and recorded randomisation supplies known variation without
    /// inventing either.
    ///
    /// What is scored later is the density of `z'` given `z`, recorded here as
    /// [`ExplorationOutcome::log_density`], not the density of the applied
    /// weights: ADR-091's floor clamps a set of coordinates and renormalises
    /// the rest, so the weight distribution has no density with respect to
    /// Lebesgue measure in exactly the region the floor produces.
    ///
    /// # Errors
    ///
    /// Everything [`route`](Self::route) returns, plus
    /// [`RouterError::InvalidExplorationScale`] and
    /// [`RouterError::ExplorationScaleBelowFloor`] for an unusable scale, and
    /// [`RouterError::ExplorationCannotChangeTheAction`] for a draw over a
    /// decision the perturbation cannot move.
    pub fn route_explored(
        &mut self,
        context_vector: &[f32],
        available: &[AdapterId],
        k: usize,
        exploration: Option<Exploration>,
    ) -> Result<RouteOutcome, RouterError> {
        if k == 0 {
            return Err(RouterError::InvalidK { k });
        }
        if k > available.len() {
            return Err(RouterError::KTooLarge {
                k,
                available: available.len(),
            });
        }

        // Validate the draw with the other arguments, before the gate runs: an
        // unusable scale is a caller error of the same class as `k == 0`, and
        // it is the pair this function carries forward, so validating here is
        // also what lets the sampling path below hold a plain `(f32, u64)`
        // rather than re-checking an option it already proved.
        let draw = match exploration {
            None => None,
            Some(Exploration { scale, seed }) => {
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(RouterError::InvalidExplorationScale { scale });
                }
                // Checked after the finite/positive check, never before it: a
                // NaN scale must fail InvalidExplorationScale rather than
                // silently skip this floor, since NaN compares false against
                // `<` as it does against everything else.
                if f64::from(scale) < MIN_EXPLORATION_SCALE {
                    return Err(RouterError::ExplorationScaleBelowFloor {
                        scale,
                        floor: MIN_EXPLORATION_SCALE,
                    });
                }
                Some((scale, seed))
            }
        };

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

        // Refused here rather than at the caller, because this is the only
        // place both halves are visible: the policy is router state and `k`
        // is per-call. ADR-096 decision 4 makes the point for the serving
        // path -- evidence that a learned policy works has to show the request
        // coefficients change -- and a draw over a decision whose coefficients
        // cannot change is that failure with a randomiser attached.
        //
        // Uniform over every candidate gives `1/k` whatever the scores are, so
        // a draw over it records variation the action never had. A SINGLE
        // candidate is the other door into the same defect: it takes weight
        // 1.0 under every policy, softmax included, so the draw would describe
        // a decision that had exactly one outcome. `n == 1` implies `k == 1`
        // implies `k == n`, so it needs no `k` term of its own.
        if draw.is_some() && (n == 1 || (k == n && self.weight_policy == WeightPolicy::Uniform)) {
            return Err(RouterError::ExplorationCannotChangeTheAction { k, candidates: n });
        }

        // What the selection and the weight policy consume: the gate's own
        // logits, or those logits perturbed by this request's draw. `scores`
        // itself is left alone -- the deterministic values are what
        // `selected_scores` reports, and a learner needs both.
        let effective: Vec<f32> = match draw {
            None => scores[..n].to_vec(),
            Some((scale, seed)) => {
                let mut state = seed;
                scores[..n]
                    .iter()
                    .map(|score| score + scale * normal_from(&mut state) as f32)
                    .collect()
            }
        };

        let mut indexed: Vec<(usize, f32)> = effective.iter().copied().enumerate().collect();
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

        // The gate's own scores for the selected adapters, and the logits the
        // decision actually consumed. Under exploration these are two
        // different claims, and the deterministic column is the one that is a
        // function of the input alone.
        let selected_scores: Vec<(AdapterId, f32)> = indexed[..k]
            .iter()
            .map(|(idx, _)| (available[*idx].clone(), scores[*idx]))
            .collect();
        let sampled_scores: Vec<(AdapterId, f32)> = indexed[..k]
            .iter()
            .map(|(idx, logit)| (available[*idx].clone(), *logit))
            .collect();
        let pre_floor_weights = raw_weights.clone();

        let exploration_outcome = draw.map(|(scale, seed)| {
            // The draw is `z' = z + e`; `effective` holds `z'` for the first
            // `n` columns, so the difference recovers `e` and the density is a
            // closed form over all gate outputs, including any the selection
            // or the floor removed.
            let squared: f64 = effective
                .iter()
                .zip(scores[..n].iter())
                .map(|(drawn, own)| {
                    let difference = f64::from(*drawn) - f64::from(*own);
                    difference * difference
                })
                .sum();
            let scale64 = f64::from(scale);
            let n64 = n as f64;
            let log_density = -squared / (2.0 * scale64 * scale64)
                - (n64 / 2.0) * (2.0 * std::f64::consts::PI * scale64 * scale64).ln();
            ExplorationOutcome {
                scale,
                seed,
                sampled_logits: effective.clone(),
                log_density,
            }
        });

        let FloorOutcome { weights, dropped } = floor_and_renormalize(raw_weights, self.epsilon);
        self.last_dropped = dropped.clone();
        Ok(RouteOutcome {
            weights,
            pre_floor_weights,
            selected_scores,
            sampled_scores,
            dropped,
            exploration: exploration_outcome,
        })
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

/// Derive the exploration seed a completion draws with, from its id.
///
/// The seed is part of the recorded evidence, so it has to be reproducible by
/// anyone holding the episode: given the completion id, this function and the
/// artifact, the draw replays exactly. That rules out a randomly generated
/// seed, which would be recorded truthfully and reproducible by nobody, and it
/// rules out `DefaultHasher`, whose output Rust explicitly does not promise to
/// keep stable across releases -- a recorded seed that a later toolchain maps
/// differently makes every stored episode unreplayable with no error anywhere.
///
/// FNV-1a over the id's bytes, then one `splitmix64` round to spread the low
/// bits FNV leaves correlated for inputs sharing a suffix. The three constants
/// here are as much part of the format as the field names are: changing any of
/// them invalidates replay for every episode already recorded.
pub fn seed_from_completion_id(completion_id: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in completion_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // `splitmix64` advances its argument and returns the mixed value, so the
    // call site reads oddly here: the mixed output is what we want, and the
    // advanced state is discarded.
    let mut state = hash;
    splitmix64(&mut state)
}

/// Advance `state` and return one 64-bit value: the SplitMix64 finalizer.
///
/// A recorded seed exists to reproduce a draw from a stored event, possibly
/// across dependency bumps, so the generator has to be a fixed integer
/// recurrence rather than a crate's default RNG: a stream free to change
/// between minor releases cannot carry a reproducibility contract. The
/// constants are SplitMix64's own.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw one standard normal `N(0, 1)` variate, advancing `state`.
///
/// Box-Muller from two uniforms. The generator is SplitMix64 with the
/// dependency-free style the routing module already uses for recorded seeds,
/// because a recorded seed exists to reproduce a draw from a stored event,
/// possibly across dependency bumps, and a crate that documents its generators
/// as not value-stable between minor releases cannot carry that contract.
///
/// The uniform is drawn from `(0, 1]`, never `[0, 1)`: `ln(u)` is the first
/// step, and `u == 0.0` sends it to `-inf`. Drawing 53 random bits and adding
/// one before scaling puts `u` at `1.0` for the all-ones draw and at `2^-53`
/// for the all-zeros one, both strictly inside `(0, 1]`, so `ln(u)` is finite
/// and `sqrt(-2 ln(u))` is finite. The magnitude is therefore bounded by the
/// largest finite `sqrt(-2 ln u)` at `u = 2^-53`, roughly `8.6`, and the
/// result is finite for every reachable `u`. The second uniform is
/// `cos(2 pi v)` with `v` in `(0, 1]`, which is finite for every draw as well,
/// so no reachable draw can produce a non-finite variate.
fn normal_from(state: &mut u64) -> f64 {
    let u = uniform_from(state);
    let v = uniform_from(state);
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// Draw one uniform from `(0, 1]`, advancing `state`.
///
/// The `+ 1` before scaling is what keeps `u` off `0.0`: a plain
/// `bits / 2^53` reaches `0.0` with probability `2^-53`, which is the one value
/// `ln` cannot take. The scale is `2^-53`, so the reachable values are the
/// `2^53` points `k * 2^-53` for `k` in `1..=2^53`, of which the largest is
/// exactly `1.0`.
fn uniform_from(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64 + 1.0) * (1.0 / 9_007_199_254_740_992.0)
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
mod tests;
