//! REINFORCE with Leave-One-Out baseline (RLOO) policy-gradient trainer.
//!
//! Trains a selector gate from rewards over its raw output logits: a single
//! index for the discrete-selector path, and the whole served mixture for the
//! explored path. Its output layer must use `Activation::Linear`; this module
//! applies softmax itself. EWC integration remains an independent call-site
//! concern.
//!
//! See `docs/training.md` for reward semantics, RLOO sampling, and loss terms.

use crate::activation::Activation;
use crate::error::{FannError, FannResult, validate_allocation_size};
use crate::network::Network;

use rand::Rng;
use rand::SeedableRng;

/// Rate at which one routing decision is folded into the frequency EMA.
///
/// Trainer-private on purpose: `RlooConfig` is `pub` with `pub` fields and so is
/// externally constructible, which makes any added field a major-version break
/// (`constructible_struct_adds_field`). The load-balance coefficient that scales
/// this term is already exposed as `RlooConfig::aux_loss_coeff`, and the EMA rate
/// only rescales the same gradient, so nothing is lost by fixing it here.
const ROUTE_FREQ_DECAY: f32 = 0.01;

/// Hyperparameters for one policy-gradient refit step.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RlooConfig {
    /// Step size applied per policy-gradient update.
    pub learning_rate: f32,
    /// Coefficient for the load-balance auxiliary loss (prevents routing collapse).
    pub aux_loss_coeff: f32,
    /// Coefficient for the router z-loss (discourages logit explosion).
    pub z_loss_coeff: f32,
}

impl Default for RlooConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            aux_loss_coeff: 0.01,
            z_loss_coeff: 0.001,
        }
    }
}

/// Per-call override for the load-balance term, carrying the learned mixture's
/// projected weight vector and inverse temperature (ADR-091 Decision 3, amended
/// 2026-09-20).
///
/// Private: it exists only so [`RlooTrainer::step`] and
/// [`RlooTrainer::step_with_learned_weights`] can share one implementation
/// without the two ever computing the load-balance term differently by
/// accident — a second copy of this wiring is exactly the failure the
/// 2026-09-19 amendment records (a documented guard sitting uncalled while the
/// trainer optimised something else).
struct WeightedAux<'a> {
    /// `softmax(selected logits / tau)` embedded into the gate's full output
    /// space (`w_full[selected[i]] = w[i]`, zero elsewhere), taken **before**
    /// Decision 2's `epsilon` floor drop and renormalisation. Length must equal
    /// `gate.num_outputs()`.
    w_full: &'a [f32],
    /// The temperature that produced `w_full`. The load-balance gradient this
    /// module computes is `dL/dz` for `p = softmax(z)`; the mixture weights are
    /// `softmax(z / tau)`, so the update path scales the returned gradient by
    /// `1 / tau` to reach the gradient with respect to the logits it updates.
    tau: f32,
}

/// One recorded explored action, as
/// [`RlooTrainer::step_with_explored_action`] consumes it.
///
/// The learned action is the whole served mixture, and the randomisation that
/// makes it attributable is Gaussian noise added to the gate's raw logits at a
/// recorded scale (ADR-096 decision 3, amended 2026-09-21). `sampled_logits`
/// is the `z'` the server served; the gate recomputes `z` from `context`.
/// Grouped rather than passed as fields because they are facts about ONE
/// decision and only make sense together: the draw that produced `sampled_logits`
/// ran at `exploration_scale`, and the mixture it produced was formed at `tau`.
/// A call site that could pass them from different episodes is a call site
/// that will.
pub struct ExploredAction<'a> {
    /// The sampled logits `z'` the server served from: `z + e`, one per gate
    /// output, with `e ~ N(0, s^2 I)` and `s` equal to `exploration_scale`.
    /// The density scored is the density of this vector, not of the weights.
    pub sampled_logits: &'a [f32],
    /// The exploration scale `s` the draw used. From the episode, not from the
    /// server's current configuration: the gradient has to be taken at the
    /// policy that produced the action.
    pub exploration_scale: f32,
    /// `softmax(sampled logits / tau)` embedded into the gate's full output
    /// space, taken BEFORE the `epsilon` floor's drop and renormalisation.
    /// Same meaning as in [`RlooTrainer::step_with_learned_weights`].
    pub w_full: &'a [f32],
    /// The temperature that produced `w_full`.
    pub tau: f32,
}

impl ExploredAction<'_> {
    /// The sampled logits and draw scale, in the form the shared update path
    /// takes.
    fn as_action(&self) -> Action<'_> {
        Action::Explored {
            sampled_logits: self.sampled_logits,
            inv_scale_sq: 1.0 / (self.exploration_scale * self.exploration_scale),
        }
    }
}

/// One judged episode, as [`RlooTrainer::step_batch_with_explored_actions`]
/// consumes it.
///
/// Deliberately plain data with no knowledge of where an episode is stored or
/// how a judgement reached it: this crate holds the objective, and the server
/// that recorded the provenance owns the conversion into this shape.
pub struct MixtureSample<'a> {
    /// The context vector the gate routed on, as the server captured it.
    pub context: &'a [f32],
    /// The mixture action that was actually served.
    pub action: ExploredAction<'a>,
    /// The judgement, signed. Missing feedback is absent from the round
    /// rather than present here as a zero.
    pub reward: f32,
}

/// What one round of [`RlooTrainer::step_batch_with_explored_actions`] did.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BatchRefitOutcome {
    /// Episodes in the round.
    pub samples: usize,
    /// Episodes whose leave-one-out advantage was non-zero, the count of
    /// non-zero advantages. `0` means the gate was not touched.
    pub informative: usize,
    /// Mean raw reward over the whole round, before any baseline. The
    /// operator-facing number: it says what the feedback was, not what the
    /// update did with it.
    pub mean_reward: f32,
    /// Mean over the whole round of the advantage-weighted per-sample loss,
    /// the same `1 / n` the update was scaled by: each sample's loss is taken
    /// at its leave-one-out advantage, not at its raw reward. `0.0` when no
    /// episode contributed, where the round is a no-op on the gate.
    pub mean_loss: f32,
}

/// Leave-one-out baseline advantages: `r_i - mean(r_j : j != i)`.
///
/// Returns all zeros for a round of one, where the baseline is undefined, and
/// EXACTLY zero for a round whose rewards are all equal, where the baseline is
/// defined and equals every reward. The second is the one that matters: a
/// residue of a few ULPs there would let a round carrying no comparison move
/// the gate, and it would do it in whichever direction the rounding fell.
///
/// The accumulator is `f64` for that reason and not for range. An `f32`
/// reward has 24 significand bits and `f64` has 53, so `m` copies of one
/// reward sum exactly for any `m` below `2^29`; `(total - r)` is then exactly
/// `(m - 1) * r`, and dividing it by `m - 1` is a correctly-rounded division
/// whose exact quotient `r` is representable, so it returns `r` itself. The
/// advantage is `r - r`. This was first written with an explicit all-equal
/// early return beside it; mutation testing found that branch unreachable,
/// which is the property above stated a second way, so the branch is gone and
/// the property is pinned by
/// `an_all_equal_round_is_exactly_zero_without_a_special_case`.
///
/// A round of more than `2^29` episodes is outside that argument. Nothing in
/// this crate produces one, and a caller that did would be handing a single
/// refit half a billion judgements.
pub fn leave_one_out_advantages(rewards: &[f32]) -> Vec<f32> {
    let m = rewards.len();
    if m < 2 {
        return vec![0.0; m];
    }
    let total: f64 = rewards.iter().map(|r| f64::from(*r)).sum();
    let denom = (m - 1) as f64;
    rewards
        .iter()
        .map(|r| {
            let r = f64::from(*r);
            (r - (total - r) / denom) as f32
        })
        .collect()
}

/// Which action a step is updating toward.
///
/// One enum rather than two `step_impl` bodies, for the reason `WeightedAux`
/// is one: the policy term is the thing most likely to drift between two
/// copies, and it is the term whose drift is silent -- both copies keep
/// training, toward different objectives.
enum Action<'a> {
    /// A single output index, the discrete-selector case.
    Single(usize),
    /// The served logits of a mixture decision, the case ADR-096 decision 3
    /// defines: the learned action is the whole mixture.
    Explored {
        /// The sampled logits `z'` the server served from, one per output.
        sampled_logits: &'a [f32],
        /// `1 / s^2`, the factor the Gaussian score carries. Computed once
        /// here so the policy term does not divide per component.
        inv_scale_sq: f32,
    },
}

/// A policy-gradient trainer for selector gates with linear logits.
///
/// It applies softmax in the loss and leaves EWC composition to the caller.
/// See [`docs/training.md`](../../docs/training.md#reinforce-with-leave-one-out-rloo) for the gate contract and loss terms.
#[derive(Clone)]
pub struct RlooTrainer {
    config: RlooConfig,
    /// RNG used for Gumbel-max sampling in the Phase-2 multi-sample path.
    rng: rand::rngs::SmallRng,
    /// EMA of the routing frequency `f_i`: the share of recent decisions that
    /// selected expert `i`. Empty until the first step sizes it from the gate's
    /// output width, then initialised uniform so that a trainer with no history
    /// applies no load-balance pressure at all.
    route_freqs: Vec<f32>,
}

/// Per-layer weight and bias gradients for one backward pass.
struct GateGradients {
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
}

impl GateGradients {
    /// Add another sample's gradient in place.
    ///
    /// Shapes come from the same gate within one round, so a mismatch is a
    /// programming error rather than a caller error; `zip` would silently
    /// truncate, which is why the lengths are asserted instead.
    fn accumulate(&mut self, other: &GateGradients) {
        debug_assert_eq!(self.weights.len(), other.weights.len());
        debug_assert_eq!(self.biases.len(), other.biases.len());
        for (dst, src) in self.weights.iter_mut().zip(other.weights.iter()) {
            debug_assert_eq!(dst.len(), src.len());
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
        for (dst, src) in self.biases.iter_mut().zip(other.biases.iter()) {
            debug_assert_eq!(dst.len(), src.len());
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
    }

    /// Divide through by a sample count, so that a round's step size does not
    /// grow with the number of episodes it happened to include.
    fn mean_over(&mut self, count: usize) {
        let inv = 1.0 / count as f32;
        for row in self.weights.iter_mut().chain(self.biases.iter_mut()) {
            for g in row.iter_mut() {
                *g *= inv;
            }
        }
    }
}

/// What one sample contributes to a round, before anything is applied.
struct SampleUpdate {
    /// The output-layer delta: policy term plus both guard terms.
    output_deltas: Vec<f32>,
    /// The traffic this decision represents, to fold into the frequency EMA
    /// after the round has been applied.
    mass: Vec<f32>,
    /// `-reward * log pi(action)`, for caller logging.
    loss: f32,
}

/// What a learned-mixture weight vector and its temperature must satisfy
/// before the load-balance term reads them.
///
/// Shared by [`validate_explored_action`] and
/// [`RlooTrainer::step_with_learned_weights`] so both paths refuse the same
/// malformed inputs with the same words: `w_full` of the wrong width, any
/// non-finite `w_full` entry, a `tau` that is not finite and strictly positive,
/// and a `tau` whose `1 / tau` is not finite. `label` names the entry point in
/// the message, so a refusal says which call refused it.
///
/// `1 / tau` is the quantity the update path actually multiplies the
/// load-balance gradient by, and it overflows `f32` for `tau` below roughly
/// `2.94e-39`, where a finite positive `tau` would otherwise pass. Checking
/// only `tau > 0` therefore lets an infinite factor reach the gate.
fn validate_weighted_aux(label: &str, gate: &Network, w_full: &[f32], tau: f32) -> FannResult<()> {
    let num_outputs = gate.num_outputs();
    if w_full.len() != num_outputs {
        return Err(FannError::InputSizeMismatch {
            expected: num_outputs,
            actual: w_full.len(),
        });
    }
    for (j, &w) in w_full.iter().enumerate() {
        if !w.is_finite() {
            return Err(FannError::TrainingError(format!(
                "{label}: w_full[{j}] must be finite, got {w}"
            )));
        }
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(FannError::TrainingError(format!(
            "{label}: tau must be finite and strictly positive, got {tau}"
        )));
    }
    if !(1.0 / tau).is_finite() {
        return Err(FannError::TrainingError(format!(
            "{label}: 1 / tau must be finite, got {} for tau = {tau}",
            1.0 / tau
        )));
    }
    Ok(())
}

/// What an [`ExploredAction`] must satisfy before its columns are read.
///
/// Separate from [`validate_action`] because these are facts about the
/// recorded decision rather than about the update, and because a batch has
/// to check both for every sample before applying any of them.
///
/// This function refuses a non-finite `sampled_logits` entry or `w_full` entry
/// and the listed domain errors (a negative or non-finite `exploration_scale`,
/// a `tau` that is not finite and strictly positive, and a `tau` whose `1 / tau`
/// is not finite), before any work; each refusal names the field and the value
/// it saw. A non-finite `context` entry is not a fact about the recorded action,
/// so it is refused by [`validate_action`] instead, which every caller runs
/// before this one. `w_full` and `tau` go through [`validate_weighted_aux`],
/// which is also what [`RlooTrainer::step_with_learned_weights`] uses, so the
/// two paths cannot drift about what a usable weight vector is. The scale is
/// checked here as well as through the reciprocal the update path takes,
/// because `1.0 / 0.0` is infinite rather than zero: a scale of `0.0` would
/// arrive there as `inf`, which the finite check catches while naming a number
/// the caller never supplied.
///
/// These checks do not make the whole call finite-input-safe: a finite input
/// can still overflow the update arithmetic (the policy and load-balance
/// products can sit at the `f32` ceiling). One that would is refused at the
/// apply, by [`RlooTrainer::apply_gradients`], before anything is written, so no
/// non-finite value reaches the gate either way. A negative `exploration_scale`
/// and a non-positive `tau` are refused even though their own arithmetic is
/// finite, because they are outside the domain the estimates are defined on.
///
/// Beyond these domain checks the learner imposes no floor on a served
/// `exploration_scale`. A floor — the smallest scale a policy is willing to
/// serve, not the smallest this update can survive — belongs to whoever records
/// the policy, and is deliberately not imposed here.
fn validate_explored_action(gate: &Network, action: &ExploredAction<'_>) -> FannResult<()> {
    let num_outputs = gate.num_outputs();
    if action.sampled_logits.len() != num_outputs {
        return Err(FannError::InputSizeMismatch {
            expected: num_outputs,
            actual: action.sampled_logits.len(),
        });
    }
    for (j, &z) in action.sampled_logits.iter().enumerate() {
        if !z.is_finite() {
            return Err(FannError::TrainingError(format!(
                "explored action: sampled_logits[{j}] must be finite, got {z}"
            )));
        }
    }
    validate_weighted_aux("explored action", gate, action.w_full, action.tau)?;
    if !action.exploration_scale.is_finite() || action.exploration_scale <= 0.0 {
        return Err(FannError::TrainingError(format!(
            "explored action: exploration_scale must be finite and strictly positive, got {}",
            action.exploration_scale
        )));
    }
    Ok(())
}

/// Everything [`RlooTrainer::step_impl`] refuses before it touches the gate.
///
/// Lifted out of `step_impl` so a batch can refuse a whole round *before*
/// applying any of it. A batch that validated as it went would leave the gate
/// half-updated at the first bad sample, in a state no caller named and no
/// epoch describes; and a second copy of these rules written for the batch is
/// a place for the two to disagree about what an action is.
fn validate_action(gate: &Network, context: &[f32], action: &Action<'_>) -> FannResult<()> {
    let num_inputs = gate.num_inputs();
    let num_outputs = gate.num_outputs();
    let num_layers = gate.num_layers();

    if context.len() != num_inputs {
        return Err(FannError::InputSizeMismatch {
            expected: num_inputs,
            actual: context.len(),
        });
    }
    for (j, &x) in context.iter().enumerate() {
        if !x.is_finite() {
            return Err(FannError::TrainingError(format!(
                "context[{j}] must be finite, got {x}"
            )));
        }
    }

    match &action {
        Action::Single(idx) => {
            if *idx >= num_outputs {
                return Err(FannError::InputSizeMismatch {
                    expected: num_outputs,
                    actual: idx.saturating_add(1),
                });
            }
        }
        Action::Explored {
            sampled_logits,
            inv_scale_sq,
        } => {
            // The served vector is one value per output. A caller reaching the
            // update path without going through [`validate_explored_action`]
            // still gets the refusal, because the policy term subtracts this
            // vector from the gate's output component by component.
            if sampled_logits.len() != num_outputs {
                return Err(FannError::InputSizeMismatch {
                    expected: num_outputs,
                    actual: sampled_logits.len(),
                });
            }
            for (j, &z) in sampled_logits.iter().enumerate() {
                if !z.is_finite() {
                    return Err(FannError::TrainingError(format!(
                        "explored action: sampled_logits[{j}] must be finite, got {z}"
                    )));
                }
            }
            if !inv_scale_sq.is_finite() || *inv_scale_sq <= 0.0 {
                return Err(FannError::TrainingError(format!(
                    "explored action: 1 / s^2 must be finite and strictly positive, got {inv_scale_sq}"
                )));
            }
        }
    }

    // Gate output layer must be Linear (raw logits, softmax applied here).
    {
        let layers = gate.layers();
        if !matches!(layers[num_layers - 1].activation(), Activation::Linear) {
            return Err(FannError::TrainingError(
                "rloo gate output layer must be Linear (logits)".into(),
            ));
        }
    }

    Ok(())
}

impl RlooTrainer {
    /// Create a new trainer, seeding the RNG from system entropy.
    pub fn new(config: RlooConfig) -> Self {
        Self {
            config,
            rng: rand::rngs::SmallRng::from_entropy(),
            route_freqs: Vec::new(),
        }
    }

    /// Create a new trainer with a fixed RNG seed for deterministic behaviour.
    ///
    /// Prefer this constructor in tests to guarantee reproducible results.
    pub fn with_seed(config: RlooConfig, seed: u64) -> Self {
        Self {
            config,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            route_freqs: Vec::new(),
        }
    }

    /// Applies one REINFORCE update for `action_idx` from a signed reward.
    ///
    /// The reward sign controls policy direction and its magnitude controls update strength.
    /// Returns the scalar policy loss for logging.
    /// See [`docs/training.md`](../../docs/training.md#phase-1-single-sample-reinforce-step-the-active-path) for reward semantics and the loss formula.
    ///
    /// The load-balance term here runs on this gate's own full-width selection
    /// distribution. For the learned mixture-weight path, where ADR-091
    /// Decision 3 (amended 2026-09-20) requires that term to run on the
    /// caller's projected weight vector at `1 / tau` instead, see
    /// [`Self::step_with_learned_weights`].
    ///
    /// A non-finite context entry or a non-finite reward is refused before
    /// anything is applied, exactly as before. The one change in behaviour is
    /// the refusal of a finite input whose update would still overflow: that is
    /// now refused at the apply, before anything is written, and the inputs it
    /// newly refuses are exactly the ones that previously wrote a non-finite
    /// value into the gate. Every other input is bit-identical to its previous
    /// form.
    pub fn step(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        action_idx: usize,
        reward: f32,
    ) -> FannResult<f32> {
        self.step_impl(gate, context, Action::Single(action_idx), reward, None)
    }

    /// Applies one REINFORCE update whose load-balance term runs on the learned
    /// mixture's projected weight vector instead of this gate's own selection
    /// distribution (ADR-091 Decision 3, amended 2026-09-20).
    ///
    /// `w_full` is the served mixture `softmax(z' / tau)` projected into this
    /// gate's full output space — the selected columns carry their weight and
    /// the rest are zero — taken **before** the `epsilon` floor's drop and
    /// renormalisation: an adapter the floor is about to drop must still be
    /// present and carry gradient when this runs, which is exactly the event a
    /// post-floor vector would already have zeroed out. `tau` is the
    /// temperature that produced it; the
    /// load-balance gradient this module computes is `dL/dz` for
    /// `p = softmax(z)`, and the weights are `softmax(z / tau)`, so this call
    /// scales the returned gradient by `1 / tau` to reach the gradient with
    /// respect to the logits it updates. The floor on `tau` is guard (a)'s job
    /// in the router; this call enforces only the arithmetic precondition that
    /// it be finite and strictly positive, and does not reimplement that floor.
    ///
    /// The policy-gradient and z-loss terms, and the `action_idx`/`reward`
    /// arguments that drive them, are unchanged from [`Self::step`]: only the
    /// load-balance term's input and the frequency EMA's fold differ.
    ///
    /// Returns `FannError::InputSizeMismatch` if `w_full.len() !=
    /// gate.num_outputs()`, and `FannError::TrainingError` if any `w_full`
    /// entry is not finite, if `tau` is not finite and strictly positive, if
    /// `1 / tau` is not finite (below roughly `2.94e-39` a finite positive
    /// `tau` still has an infinite reciprocal), or if a non-finite
    /// context entry or reward is supplied. The floor on `tau` is guard (a)'s
    /// job in the router; this call enforces only the arithmetic precondition
    /// that the update path's `1 / tau` factor be finite, and does not
    /// reimplement that floor.
    ///
    /// A non-finite context entry, a non-finite reward, a non-finite `w_full`
    /// entry, a `tau` that is not finite and strictly positive, and a `tau`
    /// whose `1 / tau` is not finite are refused before anything is applied,
    /// exactly as before. The one change in behaviour is the refusal of a
    /// finite input whose update would still overflow: that is now refused at
    /// the apply, before anything is written, and the inputs it newly refuses
    /// are exactly the ones that previously wrote a non-finite value into the
    /// gate. Every other input is bit-identical to its previous form.
    pub fn step_with_learned_weights(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        action_idx: usize,
        reward: f32,
        w_full: &[f32],
        tau: f32,
    ) -> FannResult<f32> {
        validate_weighted_aux("step_with_learned_weights", gate, w_full, tau)?;

        self.step_impl(
            gate,
            context,
            Action::Single(action_idx),
            reward,
            Some(WeightedAux { w_full, tau }),
        )
    }

    /// Applies one REINFORCE update toward the served logits of an explored
    /// mixture: the whole-mixture action of ADR-096 decision 3.
    ///
    /// The existing entries take a single `action_idx` and apply a
    /// discrete-action loss over it. A completion produced by a weighted
    /// mixture does not identify such an index, and choosing the
    /// largest-weight adapter or rewarding every adapter equally manufactures
    /// credit the feedback did not supply. So this is a different entry rather
    /// than a wrapper that picks one.
    ///
    /// The gate's raw logits gained Gaussian noise at the recorded scale `s`,
    /// so the served logits are `z' = z + e` with `e ~ N(0, s^2 I)`. The
    /// density scored is the density of `z'` given `z`: the per-sample
    /// gradient is `reward * (z' - z) / s^2` on the gate's pre-activation
    /// output, and the distance is taken from the gate's forward output on
    /// `context` at the gate being updated. The gradient of the scored
    /// density is not taken through the weight transform, and `w_full` and
    /// `tau` therefore feed only the load-balance term and the traffic mass,
    /// with the same meaning as in [`Self::step_with_learned_weights`]. That
    /// split is deliberate and the ADR states why: the applied weights are
    /// `floor(softmax(z'))`, and the floor puts positive mass on a
    /// lower-dimensional face, so the distribution of the weights has no
    /// density to differentiate and `z'` is the only quantity that does.
    ///
    /// `sampled_logits` and `exploration_scale` come off the recorded episode
    /// rather than from the server's current configuration: the gradient has
    /// to be taken at the policy that produced the action, and an episode
    /// whose epoch has ended is not an on-policy sample at all.
    ///
    /// # This is the per-sample primitive, not the ADR estimator
    ///
    /// This entry applies the raw reward with **no baseline**. The estimator
    /// ADR-096 decision 3 defines is the leave-one-out round,
    /// [`Self::step_batch_with_explored_actions`], and this is the single
    /// sample it is built from. A caller holding one judgement should know what
    /// that means: one completion rated helpful says the mixture was good, not
    /// that it was better than whatever else the gate would have served, so the
    /// update here moves on the reward's sign alone. Reach for the batch entry
    /// when a round of judgements exists; a loop over this one is not the
    /// estimator and, because each call moves the gate before the next reads
    /// it, is not on-policy either.
    ///
    /// # Refusals
    ///
    /// Returns `FannError::InputSizeMismatch` if `sampled_logits.len()`,
    /// `w_full.len()` or `context.len()` differs from the gate's input or
    /// output count, and `FannError::TrainingError` if any context entry,
    /// sampled logit or `w_full` entry is not finite, if `reward` is not
    /// finite, if `tau` is not finite and strictly positive or its `1 / tau`
    /// is not finite, or if `exploration_scale` is not finite and strictly
    /// positive. Every up-front refusal names the field and the value it saw.
    /// A finite input whose update would still overflow is refused at the
    /// apply, before anything is written, naming the layer and the parameter.
    pub fn step_with_explored_action(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        action: &ExploredAction<'_>,
        reward: f32,
    ) -> FannResult<f32> {
        validate_explored_action(gate, action)?;
        self.step_impl(
            gate,
            context,
            action.as_action(),
            reward,
            Some(WeightedAux {
                w_full: action.w_full,
                tau: action.tau,
            }),
        )
    }

    /// Applies ONE update from a round of recorded mixture actions, with a
    /// leave-one-out baseline across the round (ADR-096 decision 3).
    ///
    /// # Why this is not a loop over [`Self::step_with_explored_action`]
    ///
    /// Two reasons, and the second is the one that makes a loop wrong rather
    /// than merely noisier.
    ///
    /// A single judgement carries no baseline. One completion rated helpful
    /// says the mixture was good, not that it was better than the alternative
    /// the gate would otherwise have served, and a policy gradient taken
    /// against no baseline moves the gate on the reward's sign alone. The
    /// baseline here is the mean of the OTHER rewards in the round, which is
    /// independent of the sample's own action and so leaves the estimator
    /// unbiased. It does not control for how hard each context was: rewards
    /// that vary more across contexts than across actions get little variance
    /// reduction out of it, and that limitation is the round's, not a bug to
    /// be tuned away.
    ///
    /// And every gradient in a round is taken at the gate that produced the
    /// actions. A loop would apply the first sample's update and then compute
    /// the second's gradient at a gate that did not produce the second
    /// action, while the epoch bookkeeping that admitted both still called
    /// them on-policy. The accumulated form keeps the code's meaning of
    /// on-policy and the episode store's meaning the same thing.
    ///
    /// # A round with nothing to compare is a no-op, not a small step
    ///
    /// Rewards that are all equal carry no comparative information: every
    /// leave-one-out advantage is zero, no sample contributes, and the gate is
    /// left byte-for-byte unchanged. That includes a round of one.
    /// The load-balance and z-loss terms do not fill the gap, and deliberately:
    /// they exist to shape a policy update, not to constitute one, and a gate
    /// that drifted toward balance because a refit was attempted would change
    /// served behaviour with no feedback behind it.
    ///
    /// # The step is the mean over the round, not over the contributors
    ///
    /// In a round with at least one non-zero advantage the update is
    /// `(1 / n) * sum_i [A_i * grad log pi_i + aux_i + zloss_i]` over every
    /// sample, as ADR-096 decision 3 writes it: the mean of the
    /// advantage-weighted gradients over events `i = 1..n`. Dividing by the
    /// contributor count instead would make the step discontinuous — a reward
    /// one ULP off a tie moves a sample between the two denominators and
    /// rescales the whole round — and would drop the regularisers of the
    /// zero-advantage episodes even though their traffic still folds. Averaging
    /// over `n` keeps the policy term continuous in the rewards and averages
    /// each regulariser over the round it belongs to. `informative` still
    /// reports the count of non-zero advantages, which is the number that says
    /// how much of the round carried signal.
    ///
    /// The frequency EMA still sees every validated sample. Those completions
    /// were served traffic whether or not the round could learn from them.
    ///
    /// # Refusals
    ///
    /// An empty round, a non-finite reward, a non-finite leave-one-out
    /// advantage, and anything [`Self::step_with_explored_action`] refuses
    /// about a single action (a non-finite context entry, sampled logit, or
    /// `w_full` entry; a `tau` or `exploration_scale` that is not finite and
    /// strictly positive; a `tau` whose `1 / tau` is not finite). Every sample
    /// is checked before any is applied, and a finite round whose update would
    /// still overflow is refused at the apply before anything is written, so a
    /// refused round leaves the gate and the traffic EMA untouched.
    pub fn step_batch_with_explored_actions(
        &mut self,
        gate: &mut Network,
        samples: &[MixtureSample<'_>],
    ) -> FannResult<BatchRefitOutcome> {
        if samples.is_empty() {
            return Err(FannError::TrainingError(
                "a batch refit needs at least one recorded action; an empty round is not a \
                 refit that changed nothing, it is a caller with no data"
                    .into(),
            ));
        }

        for sample in samples {
            if !sample.reward.is_finite() {
                return Err(FannError::TrainingError(format!(
                    "batch refit: reward must be finite, got {}",
                    sample.reward
                )));
            }
            validate_explored_action(gate, &sample.action)?;
            validate_action(gate, sample.context, &sample.action.as_action())?;
        }

        let rewards: Vec<f32> = samples.iter().map(|s| s.reward).collect();
        let advantages = leave_one_out_advantages(&rewards);
        for (i, &a) in advantages.iter().enumerate() {
            if !a.is_finite() {
                return Err(FannError::TrainingError(format!(
                    "batch refit: leave-one-out advantage[{i}] must be finite, got {a}"
                )));
            }
        }

        // Read once, before the round: every sample's load-balance term sees
        // the traffic history that preceded the whole round.
        let route_freqs = self.route_freqs_or_uniform(gate.num_outputs());
        let num_layers = gate.num_layers();

        let mut accumulated: Option<GateGradients> = None;
        let mut masses: Vec<Vec<f32>> = Vec::with_capacity(samples.len());
        let mut informative = 0_usize;
        let mut loss_total = 0.0_f32;

        for (sample, advantage) in samples.iter().zip(advantages.iter()) {
            let weighted = WeightedAux {
                w_full: sample.action.w_full,
                tau: sample.action.tau,
            };
            // The forward pass inside this call is what leaves the activation
            // buffers the backward pass below reads, so the two stay adjacent.
            let update = self.sample_update(
                gate,
                sample.context,
                &sample.action.as_action(),
                *advantage,
                Some(&weighted),
                &route_freqs,
            )?;
            masses.push(update.mass);
            loss_total += update.loss;
            if *advantage != 0.0 {
                informative += 1;
            }

            let gradients =
                self.backprop_gradients(gate, sample.context, &update.output_deltas, num_layers)?;
            match accumulated.as_mut() {
                Some(total) => total.accumulate(&gradients),
                None => accumulated = Some(gradients),
            }
        }

        // The update is the mean over ALL n samples, not over the contributors:
        // ADR-096 decision 3 writes it as the mean of the advantage-weighted
        // gradients over events `i = 1..n`, so the policy term stays continuous
        // in the rewards (a reward a hair off a tie cannot rescale the whole
        // step) and the regularisers are averaged over the round they belong to.
        // A round whose every advantage is zero is the one exception, and it is
        // a no-op on the gate: there is no policy update for the regularisers to
        // shape, and applying them alone would move served behaviour on feedback
        // that carried no comparison.
        if informative > 0
            && let Some(mut gradients) = accumulated
        {
            gradients.mean_over(samples.len());
            self.apply_gradients(gate, &gradients)?;
        }

        for mass in &masses {
            self.observe_routing(mass);
        }

        let count = samples.len() as f32;
        Ok(BatchRefitOutcome {
            samples: samples.len(),
            informative,
            mean_reward: rewards.iter().sum::<f32>() / count,
            mean_loss: if informative == 0 {
                0.0
            } else {
                loss_total / count
            },
        })
    }

    /// Shared body for [`Self::step`] and [`Self::step_with_learned_weights`].
    ///
    /// One implementation behind two public entries, so the load-balance
    /// term's two call shapes cannot drift apart the way the per-context and
    /// batch objectives once did (ADR-091's 2026-09-19 amendment). `weighted`
    /// is `None` for the base selection-distribution path and `Some` for the
    /// learned mixture-weight path; it changes exactly the load-balance term's
    /// second argument and scale, and the vector folded into the frequency
    /// EMA — the policy-gradient and z-loss terms never see it.
    fn step_impl(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        action: Action<'_>,
        reward: f32,
        weighted: Option<WeightedAux<'_>>,
    ) -> FannResult<f32> {
        let num_layers = gate.num_layers();

        validate_action(gate, context, &action)?;
        if !reward.is_finite() {
            return Err(FannError::TrainingError(format!(
                "reward must be finite, got {reward}"
            )));
        }

        let route_freqs = self.route_freqs_or_uniform(gate.num_outputs());
        let update = self.sample_update(
            gate,
            context,
            &action,
            reward,
            weighted.as_ref(),
            &route_freqs,
        )?;

        self.backprop_and_apply(gate, context, &update.output_deltas, num_layers)?;
        self.observe_routing(&update.mass);
        Ok(update.loss)
    }

    /// The output-layer delta one sample contributes, with the traffic mass
    /// to fold and the scalar loss to report.
    ///
    /// Split out of [`Self::step_impl`] so the batch entry can compute every
    /// sample's delta at ONE gate before any of them is applied. `route_freqs`
    /// is passed in rather than read from `self` for the same reason: within a
    /// batch every sample's load-balance term must see the same traffic
    /// history, which is the history that preceded the whole round.
    ///
    /// Applies nothing and folds nothing; the caller owns both.
    fn sample_update(
        &self,
        gate: &mut Network,
        context: &[f32],
        action: &Action<'_>,
        reward: f32,
        weighted: Option<&WeightedAux<'_>>,
        route_freqs: &[f32],
    ) -> FannResult<SampleUpdate> {
        // Forward pass (populates activation buffers).
        let logits: Vec<f32> = gate.forward(context)?.to_vec();

        let probs = softmax(&logits);

        let k = gate.num_outputs();
        // Both guard terms are applied by calling the functions that define them,
        // never by re-deriving their gradients here: a second copy is a place for
        // the two to disagree silently, and it is what let the documented guard
        // sit uncalled while the trainer optimised something else.

        // The load-balance term runs on this gate's own selection distribution
        // by default, or on the learned mixture's projected weight vector at
        // `1 / tau` when `weighted` is supplied (ADR-091 Decision 3, amended
        // 2026-09-20). `aux_scale = 1.0` in the default case is exact for
        // every finite `f32` (`x * 1.0 == x` bit-for-bit), so this branch
        // changes no bit of `step`'s existing output.
        let (aux_source, aux_scale): (&[f32], f32) = match weighted {
            Some(w) => (w.w_full, 1.0 / w.tau),
            None => (probs.as_slice(), 1.0),
        };
        let aux_grad = load_balance_aux_gradient(route_freqs, aux_source)?;
        let z_grad = router_z_gradient(&logits, &probs);

        // The policy term is `-reward * dlog(pi)/dz`. For a single index that
        // derivative is `onehot - p`, so the expression below is written in
        // its original `reward * (p - onehot)` form rather than through the
        // general path: identical arithmetic, and no argument needed about
        // whether two ways of spelling it agree bit-for-bit.
        //
        // For an explored mixture the scored density is the Gaussian of `z'`
        // given `z` at scale `s`, whose score is `(z' - z) / s^2`. The term is
        // `-reward` times that score, `reward * (z - z') / s^2` component by
        // component, so that the SGD update `w -= lr * delta` moves the gate
        // toward `z'` on a positive reward. `z` is this gate's forward output
        // on `context`, taken at the gate being updated.
        //
        // Linear output makes this the pre-activation error; preserve reward polarity — see docs/training.md.
        let policy_term: Vec<f32> = match action {
            Action::Single(idx) => probs
                .iter()
                .enumerate()
                .map(|(j, &pj)| {
                    let onehot_j = if j == *idx { 1.0_f32 } else { 0.0_f32 };
                    reward * (pj - onehot_j)
                })
                .collect(),
            Action::Explored {
                sampled_logits,
                inv_scale_sq,
            } => logits
                .iter()
                .zip(sampled_logits.iter())
                .map(|(z, sampled)| reward * (*z - *sampled) * *inv_scale_sq)
                .collect(),
        };
        let output_deltas: Vec<f32> = policy_term
            .iter()
            .enumerate()
            .map(|(j, policy)| {
                let aux = self.config.aux_loss_coeff * aux_scale * aux_grad[j];
                let zloss = self.config.z_loss_coeff * z_grad[j];
                policy + aux + zloss
            })
            .collect();
        // The decision actually taken, for the caller to fold into the
        // frequency EMA once the update has been applied -- so that the
        // gradient above saw the traffic that preceded it and not its own.
        // The learned path reports the same projected vector the aux term
        // above just used, so `f` tracks weighted traffic rather than
        // selection counts: the comparison a weight-collapse guard needs.
        let mass: Vec<f32> = match (weighted, action) {
            (Some(w), _) => w.w_full.to_vec(),
            (None, Action::Single(idx)) => {
                let mut m = vec![0.0_f32; k];
                m[*idx] = 1.0;
                m
            }
            // The explored mixture's traffic is the vector it actually served,
            // so a call without one has nothing to fold. Refused rather than
            // folded as zero: a zero mass would drag the EMA off the traffic
            // it is meant to describe.
            (None, Action::Explored { .. }) => {
                return Err(FannError::TrainingError(
                    "explored action requires the recorded projected weight vector w_full".into(),
                ));
            }
        };
        // Scalar policy loss for caller logging. For the explored mixture that
        // is the reward-weighted negative log density of `z'` without its
        // constant, `reward * ||z' - z||^2 / (2 s^2)`.
        let loss = match action {
            Action::Single(idx) => -reward * probs[*idx].max(1e-9).ln(),
            Action::Explored {
                sampled_logits,
                inv_scale_sq,
            } => {
                let sq_dist: f32 = logits
                    .iter()
                    .zip(sampled_logits.iter())
                    .map(|(z, sampled)| {
                        let d = *sampled - *z;
                        d * d
                    })
                    .sum();
                reward * sq_dist * 0.5 * *inv_scale_sq
            }
        };
        Ok(SampleUpdate {
            output_deltas,
            mass,
            loss,
        })
    }

    /// Runs a Gumbel-top-`k` RLOO update with a leave-one-out baseline.
    ///
    /// Pair its positive events with [`Self::step`] negative feedback; a positive-only stream is invalid.
    /// Returns the mean sampled reward or a validation error. A finite update
    /// whose arithmetic would still overflow is refused at the apply, before
    /// anything is written, and returns that error in place of the reward.
    /// See [`docs/training.md`](../../docs/training.md#phase-2-multi-sample-rloo-rloo_step-not-the-default-path) for sampling and update details.
    pub fn rloo_step(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        preferred_idx: usize,
        k: usize,
        m_samples: usize,
    ) -> FannResult<f32> {
        let num_inputs = gate.num_inputs();
        let num_outputs = gate.num_outputs();
        let num_layers = gate.num_layers();

        if context.len() != num_inputs {
            return Err(FannError::InputSizeMismatch {
                expected: num_inputs,
                actual: context.len(),
            });
        }

        if preferred_idx >= num_outputs {
            return Err(FannError::InputSizeMismatch {
                expected: num_outputs,
                actual: preferred_idx.saturating_add(1),
            });
        }

        if k < 1 || k > num_outputs {
            return Err(FannError::TrainingError(format!(
                "rloo_step: k={k} out of range [1, {num_outputs}]"
            )));
        }

        if m_samples < 1 {
            return Err(FannError::TrainingError(
                "rloo_step: m_samples must be >= 1".into(),
            ));
        }

        // Bound both caller-controlled capacities before allocation; checked multiplication prevents overflow.
        validate_allocation_size(m_samples)?;
        let subset_storage = m_samples.checked_mul(k).ok_or_else(|| {
            FannError::TrainingError(format!(
                "rloo_step: m_samples ({m_samples}) * k ({k}) overflows usize"
            ))
        })?;
        validate_allocation_size(subset_storage)?;

        {
            let layers = gate.layers();
            if !matches!(layers[num_layers - 1].activation(), Activation::Linear) {
                return Err(FannError::TrainingError(
                    "rloo gate output layer must be Linear (logits)".into(),
                ));
            }
        }

        let logits: Vec<f32> = gate.forward(context)?.to_vec();
        let probs = softmax(&logits);

        // Draw `m_samples` Gumbel-top-`k` subsets — see docs/training.md.
        let mut rewards: Vec<f32> = Vec::with_capacity(m_samples);
        let mut subsets: Vec<Vec<usize>> = Vec::with_capacity(m_samples);

        for _ in 0..m_samples {
            // Perturb each logit with independent Gumbel(0,1) noise.
            let mut perturbed: Vec<(f32, usize)> = logits
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let u: f32 = {
                        let raw: f32 = self.rng.r#gen::<f32>();
                        // Clamp to open (0,1) to avoid ln(0).
                        raw.clamp(1e-38_f32, 1.0 - f32::EPSILON)
                    };
                    let gumbel_noise = -(-u.ln()).ln();
                    (s + gumbel_noise, i)
                })
                .collect();

            // Select top-k by perturbed logit (descending).
            perturbed.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let subset: Vec<usize> = perturbed.iter().take(k).map(|&(_, i)| i).collect();

            let r = if subset.contains(&preferred_idx) {
                1.0_f32
            } else {
                -1.0_f32
            };
            rewards.push(r);
            subsets.push(subset);
        }

        // Build output-layer gradient with leave-one-out baseline.
        let reward_sum: f32 = rewards.iter().sum();
        let ko = num_outputs;

        let mut output_deltas = vec![0.0_f32; ko];

        for (r_m, subset_m) in rewards.iter().zip(subsets.iter()) {
            // Leave-one-out baseline: (ΣR − R_m) / (M − 1); 0 if M == 1.
            let baseline = if m_samples > 1 {
                (reward_sum - r_m) / (m_samples - 1) as f32
            } else {
                0.0
            };
            let advantage = r_m - baseline;

            // g[j] += -(1/M) * advantage * (count(j ∈ subset) − k · p[j])
            for j in 0..ko {
                let in_subset: f32 = subset_m
                    .iter()
                    .map(|&i| if i == j { 1.0_f32 } else { 0.0_f32 })
                    .sum();
                output_deltas[j] +=
                    -(1.0 / m_samples as f32) * advantage * (in_subset - probs[j] * k as f32);
            }
        }

        // Aux and z-loss terms, through the same named guards step() calls.
        let route_freqs = self.route_freqs_or_uniform(ko);
        let aux_grad = load_balance_aux_gradient(&route_freqs, &probs)?;
        let z_grad = router_z_gradient(&logits, &probs);
        for j in 0..ko {
            output_deltas[j] +=
                self.config.aux_loss_coeff * aux_grad[j] + self.config.z_loss_coeff * z_grad[j];
        }

        self.backprop_and_apply(gate, context, &output_deltas, num_layers)?;

        // Attribute this decision across the sampled subsets: each of the
        // `m_samples` subsets contributes `1/k` to each expert it selected, so
        // the mass sums to one exactly as the single-sample one-hot does.
        let mut mass = vec![0.0_f32; ko];
        let per_pick = 1.0 / (m_samples as f32 * k as f32);
        for subset in &subsets {
            for &i in subset {
                mass[i] += per_pick;
            }
        }
        self.observe_routing(&mass);

        Ok(reward_sum / m_samples as f32)
    }

    /// The routing-frequency EMA, sized to `k` and initialised uniform on first
    /// use.
    ///
    /// Uniform initialisation is what makes the guard silent on a fresh trainer:
    /// at `f_i = 1/k` for every `i` the load-balance gradient is identically zero
    /// regardless of how sharp the gate is, so the term contributes nothing until
    /// observed routing actually drifts away from balance. That is the intended
    /// semantics, not an accident of initialisation.
    ///
    /// Reads only: it returns the uniform vector for an unsized field without
    /// storing it. The field is sized by [`Self::observe_routing`] when the
    /// first decision is actually folded, so an update refused before it
    /// applies leaves the field exactly as it found it — an uninitialised
    /// trainer included.
    fn route_freqs_or_uniform(&self, k: usize) -> Vec<f32> {
        if self.route_freqs.len() != k {
            vec![1.0 / k as f32; k]
        } else {
            self.route_freqs.clone()
        }
    }

    /// Fold one routing decision into the frequency EMA, sizing it uniform on
    /// first use (or after a width change) before the fold.
    ///
    /// `mass` is the share of this decision attributed to each expert and sums to
    /// one: a one-hot vector for the single-sample path, and the empirical
    /// selection frequency across the sampled subsets for the multi-sample path.
    /// This is the only writer of the field, so it is reached only by a call
    /// whose update was applied (or, for an all-equal round, deliberately
    /// skipped); a refusal never gets here.
    fn observe_routing(&mut self, mass: &[f32]) {
        if self.route_freqs.len() != mass.len() {
            self.route_freqs = vec![1.0 / mass.len() as f32; mass.len()];
        }
        for (f, &m) in self.route_freqs.iter_mut().zip(mass.iter()) {
            *f = (1.0 - ROUTE_FREQ_DECAY) * *f + ROUTE_FREQ_DECAY * m;
        }
    }

    /// Backpropagate the output-layer delta through hidden layers and apply
    /// a plain SGD update (no momentum, no weight decay).
    ///
    /// Mirrors `backprop.rs::compute_gradients` lines 94–152 (hidden-layer
    /// backprop) and `apply_gradients` lines 170–191 (simplified, batch_size=1).
    fn backprop_and_apply(
        &self,
        gate: &mut Network,
        input: &[f32],
        output_deltas: &[f32],
        num_layers: usize,
    ) -> FannResult<()> {
        let gradients = self.backprop_gradients(gate, input, output_deltas, num_layers)?;
        self.apply_gradients(gate, &gradients)?;
        Ok(())
    }

    /// Phase 1 alone: the weight and bias gradients this sample produces,
    /// returned instead of applied.
    ///
    /// A batch needs every sample's gradient taken at the SAME gate, because
    /// the epoch bookkeeping that admits those samples says they were all
    /// drawn from one policy. Applying as it went would evaluate every
    /// gradient after the first at a gate that did not produce its action,
    /// while the epoch still called them on-policy.
    fn backprop_gradients(
        &self,
        gate: &Network,
        input: &[f32],
        output_deltas: &[f32],
        num_layers: usize,
    ) -> FannResult<GateGradients> {
        // Shared borrows end before the caller's mutable update phase.
        let (weight_grads, bias_grads) = {
            // Start with the output-layer delta (from caller).
            let mut deltas: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
            deltas.push(output_deltas.to_vec());

            // Mirror backprop.rs:94-121: propagate deltas backward through hidden layers.
            let layers = gate.layers(); // shared borrow: released at end of this block
            for layer_idx in (0..num_layers - 1).rev() {
                let layer_activation = layers[layer_idx].activation();
                let layer_num_outputs = layers[layer_idx].num_outputs();
                let next_num_inputs = layers[layer_idx + 1].num_inputs();
                let next_num_outputs = layers[layer_idx + 1].num_outputs();
                let next_weights = layers[layer_idx + 1].weights();

                let prev_deltas = deltas.last().ok_or_else(|| {
                    FannError::TrainingError("empty deltas during backpropagation".to_string())
                })?;

                // gate.activations() is also a &self borrow — OK alongside layers.
                let layer_activations = gate.activations(layer_idx).ok_or_else(|| {
                    FannError::TrainingError(format!("missing activations for layer {layer_idx}"))
                })?;

                let mut layer_deltas = vec![0.0_f32; layer_num_outputs];
                for i in 0..layer_num_outputs {
                    let mut sum = 0.0_f32;
                    for j in 0..next_num_outputs {
                        // Weight layout: row-major, row j, column i.
                        let weight = next_weights[j * next_num_inputs + i];
                        sum += weight * prev_deltas[j];
                    }
                    let deriv = layer_activation.derivative(layer_activations[i]);
                    layer_deltas[i] = sum * deriv;
                }

                deltas.push(layer_deltas);
            }
            // Mirror backprop.rs:123-124: reverse to forward layer order.
            deltas.reverse();

            // Allocate gradient buffers (mirror backprop.rs:126-152).
            let mut weight_grads: Vec<Vec<f32>> = layers
                .iter()
                .map(|l| vec![0.0_f32; l.weights().len()])
                .collect();
            let mut bias_grads: Vec<Vec<f32>> = layers
                .iter()
                .map(|l| vec![0.0_f32; l.biases().len()])
                .collect();

            for (layer_idx, delta) in deltas.iter().enumerate() {
                let num_i = layers[layer_idx].num_inputs();
                let num_o = layers[layer_idx].num_outputs();

                let layer_input: &[f32] = if layer_idx == 0 {
                    input
                } else {
                    gate.activations(layer_idx - 1).ok_or_else(|| {
                        FannError::TrainingError(format!(
                            "missing activations for layer {}",
                            layer_idx - 1
                        ))
                    })?
                };

                // dW[i,j] = delta[i] * input[j]
                for (i, &d) in delta.iter().enumerate().take(num_o) {
                    for (j, &inp) in layer_input.iter().enumerate().take(num_i) {
                        weight_grads[layer_idx][i * num_i + j] += d * inp;
                    }
                }

                // dB[i] = delta[i]
                for (i, &d) in delta.iter().enumerate().take(num_o) {
                    bias_grads[layer_idx][i] += d;
                }
            }

            (weight_grads, bias_grads)
            // `layers` (shared borrow) drops here — gate is free for mutation.
        };

        Ok(GateGradients {
            weights: weight_grads,
            biases: bias_grads,
        })
    }

    /// Phase 2 alone: plain SGD from an already-computed gradient
    /// (mirror backprop.rs:170-191, no momentum).
    ///
    /// Two-phase and fallible. The first pass computes every candidate
    /// `w - lr * g` and `b - lr * g` and writes nothing; if any gradient or
    /// candidate is not finite it returns an error naming the layer, the
    /// parameter kind and index, and the value, and the gate is left
    /// byte-for-byte as it was. Only when every candidate is finite does the
    /// second pass write them. That matters because a finite input can still
    /// overflow here — the policy and load-balance products upstream can sit
    /// at the `f32` ceiling — and a half-applied gate is a state no caller
    /// named. For an update whose candidates are all finite, the arithmetic,
    /// the order and the written bytes are exactly the plain `w -= lr * g`
    /// form this replaces.
    fn apply_gradients(&self, gate: &mut Network, gradients: &GateGradients) -> FannResult<()> {
        let num_layers = gate.num_layers();
        let lr = self.config.learning_rate;

        // Pass 1: compute every candidate, writing nothing.
        let mut candidate_weights: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut candidate_biases: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let Some(layer) = gate.layer(layer_idx) else {
                candidate_weights.push(Vec::new());
                candidate_biases.push(Vec::new());
                continue;
            };

            let mut weights = Vec::with_capacity(layer.weights().len());
            for (index, (&w, &g)) in layer
                .weights()
                .iter()
                .zip(gradients.weights[layer_idx].iter())
                .enumerate()
            {
                weights.push(finite_step(layer_idx, "weight", index, w, g, lr)?);
            }

            let mut biases = Vec::with_capacity(layer.biases().len());
            for (index, (&b, &g)) in layer
                .biases()
                .iter()
                .zip(gradients.biases[layer_idx].iter())
                .enumerate()
            {
                biases.push(finite_step(layer_idx, "bias", index, b, g, lr)?);
            }

            candidate_weights.push(weights);
            candidate_biases.push(biases);
        }

        // Pass 2: every candidate is finite, so write them.
        for layer_idx in 0..num_layers {
            let Some(layer) = gate.layer_mut(layer_idx) else {
                continue;
            };

            for (w, &c) in layer
                .weights_mut()
                .iter_mut()
                .zip(candidate_weights[layer_idx].iter())
            {
                *w = c;
            }

            for (b, &c) in layer
                .biases_mut()
                .iter_mut()
                .zip(candidate_biases[layer_idx].iter())
            {
                *b = c;
            }
        }

        Ok(())
    }
}

/// One `current - lr * gradient` candidate for [`RlooTrainer::apply_gradients`].
///
/// Returns the candidate only when both the gradient and the candidate are
/// finite; otherwise the layer, the parameter kind, the index and the value are
/// named in the error, so a refused update says exactly which parameter it
/// would have poisoned.
fn finite_step(
    layer: usize,
    kind: &str,
    index: usize,
    current: f32,
    gradient: f32,
    lr: f32,
) -> FannResult<f32> {
    let candidate = current - lr * gradient;
    if !candidate.is_finite() || !gradient.is_finite() {
        return Err(FannError::TrainingError(format!(
            "refusing to write layer {layer} {kind}[{index}]: \
             {current} - {lr} * {gradient} = {candidate}"
        )));
    }
    Ok(candidate)
}

/// Per-context load-balance loss: `(1/K) Σ_i (p_i − 1/K)²`. **Superseded.**
///
/// This is a pull toward uniform on a *single* context, so it penalises a gate
/// that is confidently and correctly routing one request. Load balance is a
/// property of the traffic, not of any one decision: see
/// [`load_balance_aux_loss_batch`], which is what the trainer optimises.
///
/// Kept, deprecated rather than deleted, because removing a `pub` item is a
/// major-version break. The deprecation is the machine-readable form of the
/// warning: correcting the objective *here* changes no behaviour, because the
/// trainer does not call this function.
#[deprecated(
    note = "per-context pull to uniform taxes correct sharpness; use load_balance_aux_loss_batch"
)]
pub fn load_balance_aux_loss(probs: &[f32]) -> f32 {
    if probs.is_empty() {
        return 0.0;
    }
    let k = probs.len() as f32;
    let uniform = 1.0 / k;
    probs
        .iter()
        .map(|&p| (p - uniform) * (p - uniform))
        .sum::<f32>()
        / k
}

/// Load-balance loss over observed traffic: `K * Σ_i f_i * P_i`.
///
/// `route_freqs` is `f`, the share of recent decisions that selected each expert;
/// `probs` is this context's gate distribution, standing in for `P`. This is the
/// Switch / ST-MoE form, and the property that matters is what it does *not*
/// penalise: at balanced `f` its gradient is identically zero however sharp
/// `probs` is, so it bounds collapse without taxing correct confidence.
///
/// This function states the objective; [`load_balance_aux_gradient`] is what the
/// trainer applies. `rloo::tests::aux_gradient_matches_finite_difference_of_the_batch_loss`
/// is what keeps the two from drifting apart.
pub fn load_balance_aux_loss_batch(route_freqs: &[f32], probs: &[f32]) -> FannResult<f32> {
    if route_freqs.len() != probs.len() {
        return Err(FannError::InputSizeMismatch {
            expected: probs.len(),
            actual: route_freqs.len(),
        });
    }
    if probs.is_empty() {
        return Ok(0.0);
    }
    let k = probs.len() as f32;
    Ok(k * route_freqs
        .iter()
        .zip(probs.iter())
        .map(|(&f, &p)| f * p)
        .sum::<f32>())
}

/// Gradient of [`load_balance_aux_loss_batch`] with respect to the gate logits.
///
/// `K * p_j * (f_j − Σ_i f_i p_i)`, which is positive for an over-used expert and
/// negative for an under-used one, so gradient descent moves mass off the former.
/// Scale it by `RlooConfig::aux_loss_coeff` at the call site.
pub fn load_balance_aux_gradient(route_freqs: &[f32], probs: &[f32]) -> FannResult<Vec<f32>> {
    if route_freqs.len() != probs.len() {
        return Err(FannError::InputSizeMismatch {
            expected: probs.len(),
            actual: route_freqs.len(),
        });
    }
    if probs.is_empty() {
        return Ok(Vec::new());
    }
    let k = probs.len() as f32;
    let f_dot_p: f32 = route_freqs
        .iter()
        .zip(probs.iter())
        .map(|(&f, &p)| f * p)
        .sum();
    Ok(probs
        .iter()
        .zip(route_freqs.iter())
        .map(|(&p, &f)| k * p * (f - f_dot_p))
        .collect())
}

/// Gradient of [`router_z_loss`] with respect to the gate logits: `2 * lse * p_j`.
///
/// Scale it by `RlooConfig::z_loss_coeff` at the call site.
pub fn router_z_gradient(logits: &[f32], probs: &[f32]) -> Vec<f32> {
    let lse = log_sum_exp(logits);
    probs.iter().map(|&p| 2.0 * lse * p).collect()
}

/// Router z-loss: `(log Σ_i exp(s_i))²`.
///
/// Discourages logit explosion by penalising large log-sum-exp values.
/// `logits` should be the raw gate output (pre-softmax).
pub fn router_z_loss(logits: &[f32]) -> f32 {
    let lse = log_sum_exp(logits);
    lse * lse
}

// --- Private helpers ---

/// Numerically stable softmax: subtract max before exp to prevent overflow.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Degenerate case: return uniform distribution.
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Numerically stable log-sum-exp: `max(s) + log Σ_i exp(s_i − max(s))`.
fn log_sum_exp(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&s| (s - max_val).exp()).sum();
    max_val + sum.ln()
}

#[cfg(test)]
mod tests;
