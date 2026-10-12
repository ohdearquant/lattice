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
    /// `1 / tau`. The load-balance gradient this module computes is `dL/dz`
    /// for `p = softmax(z)`; the mixture weights are `softmax(z / tau)`, so
    /// this factor rescales the returned gradient to the logits the trainer
    /// actually updates.
    inv_tau: f32,
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
    /// Episodes whose leave-one-out advantage was non-zero, which is to say
    /// the ones that contributed anything. `0` means the gate was not
    /// touched.
    pub informative: usize,
    /// Mean raw reward over the whole round, before any baseline. The
    /// operator-facing number: it says what the feedback was, not what the
    /// update did with it.
    pub mean_reward: f32,
    /// Mean advantage-weighted policy loss over the contributing episodes,
    /// `0.0` when none contributed.
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

/// What an [`ExploredAction`] must satisfy before its columns are read.
///
/// Separate from [`validate_action`] because these are facts about the
/// recorded decision rather than about the update, and because a batch has
/// to check both for every sample before applying any of them.
///
/// The scale is checked here as well as through the reciprocal the update
/// path takes, because `1.0 / 0.0` is infinite rather than zero: a scale of
/// `0.0` would arrive there as `inf`, which the finite check catches while
/// naming a number the caller never supplied.
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
    if action.w_full.len() != num_outputs {
        return Err(FannError::InputSizeMismatch {
            expected: num_outputs,
            actual: action.w_full.len(),
        });
    }
    if !action.tau.is_finite() || action.tau <= 0.0 {
        return Err(FannError::TrainingError(format!(
            "explored action: tau must be finite and strictly positive, got {}",
            action.tau
        )));
    }
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
    /// gate.num_outputs()`, and `FannError::TrainingError` if `tau` is not
    /// finite and strictly positive.
    pub fn step_with_learned_weights(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        action_idx: usize,
        reward: f32,
        w_full: &[f32],
        tau: f32,
    ) -> FannResult<f32> {
        let num_outputs = gate.num_outputs();
        if w_full.len() != num_outputs {
            return Err(FannError::InputSizeMismatch {
                expected: num_outputs,
                actual: w_full.len(),
            });
        }
        if !tau.is_finite() || tau <= 0.0 {
            return Err(FannError::TrainingError(format!(
                "step_with_learned_weights: tau must be finite and strictly positive, got {tau}"
            )));
        }

        self.step_impl(
            gate,
            context,
            Action::Single(action_idx),
            reward,
            Some(WeightedAux {
                w_full,
                inv_tau: 1.0 / tau,
            }),
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
    /// Returns `FannError::TrainingError` if any sampled logit is not finite
    /// or `exploration_scale` is not finite and strictly positive, and
    /// `FannError::InputSizeMismatch` if any length differs from the gate's
    /// output count.
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
                inv_tau: 1.0 / action.tau,
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
    /// leave-one-out advantage is zero, no sample contributes, and the gate
    /// is left byte-for-byte unchanged. That includes a round of one. The
    /// load-balance and z-loss terms do not fill the gap, and deliberately:
    /// they exist to shape a policy update, not to constitute one, and a gate
    /// that drifted toward balance because a refit was attempted would change
    /// served behaviour with no feedback behind it.
    ///
    /// The frequency EMA still sees every validated sample. Those completions
    /// were served traffic whether or not the round could learn from them.
    ///
    /// # Refusals
    ///
    /// An empty round, a non-finite reward, and anything
    /// [`Self::step_with_explored_action`] refuses about a single action. Every
    /// sample is checked before any is applied, so a refused round leaves the
    /// gate untouched.
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
                inv_tau: 1.0 / sample.action.tau,
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

            if *advantage == 0.0 {
                continue;
            }
            informative += 1;
            loss_total += update.loss;

            let gradients =
                self.backprop_gradients(gate, sample.context, &update.output_deltas, num_layers)?;
            match accumulated.as_mut() {
                Some(total) => total.accumulate(&gradients),
                None => accumulated = Some(gradients),
            }
        }

        if let Some(mut gradients) = accumulated {
            // Averaged over the samples that contributed, so a round's step
            // size is not diluted by episodes the baseline found nothing to
            // say about.
            gradients.mean_over(informative);
            self.apply_gradients(gate, &gradients);
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
                loss_total / informative as f32
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
            Some(w) => (w.w_full, w.inv_tau),
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
    /// Returns the mean sampled reward or a validation error.
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

    /// The routing-frequency EMA, sized to `k` and initialised uniform on first use.
    ///
    /// Uniform initialisation is what makes the guard silent on a fresh trainer:
    /// at `f_i = 1/k` for every `i` the load-balance gradient is identically zero
    /// regardless of how sharp the gate is, so the term contributes nothing until
    /// observed routing actually drifts away from balance. That is the intended
    /// semantics, not an accident of initialisation.
    fn route_freqs_or_uniform(&mut self, k: usize) -> Vec<f32> {
        if self.route_freqs.len() != k {
            self.route_freqs = vec![1.0 / k as f32; k];
        }
        self.route_freqs.clone()
    }

    /// Fold one routing decision into the frequency EMA.
    ///
    /// `mass` is the share of this decision attributed to each expert and sums to
    /// one: a one-hot vector for the single-sample path, and the empirical
    /// selection frequency across the sampled subsets for the multi-sample path.
    fn observe_routing(&mut self, mass: &[f32]) {
        if self.route_freqs.len() != mass.len() {
            return;
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
        self.apply_gradients(gate, &gradients);
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
    fn apply_gradients(&self, gate: &mut Network, gradients: &GateGradients) {
        let num_layers = gate.num_layers();
        let lr = self.config.learning_rate;
        for layer_idx in 0..num_layers {
            let Some(layer) = gate.layer_mut(layer_idx) else {
                continue;
            };

            let weights = layer.weights_mut();
            for (w, &g) in weights.iter_mut().zip(gradients.weights[layer_idx].iter()) {
                *w -= lr * g;
            }

            let biases = layer.biases_mut();
            for (b, &g) in biases.iter_mut().zip(gradients.biases[layer_idx].iter()) {
                *b -= lr * g;
            }
        }
    }
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
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::error::MAX_ALLOWED_ELEMENTS;
    use crate::network::NetworkBuilder;

    /// Build a deterministic 4-input → 8-hidden(ReLU) → 4-output(Linear) gate.
    fn test_gate() -> Network {
        NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .output(4, Activation::Linear)
            .build_with_seed(1)
            .unwrap()
    }

    const CTX: [f32; 4] = [0.1, -0.2, 0.3, 0.4];

    /// A deterministic 4-input → 4-output Linear gate with NO hidden layer.
    ///
    /// The output is an affine function of the input, so a plain SGD update to
    /// its weights and biases moves the forward output on a fixed context by
    /// exactly `-learning_rate * delta`, where `delta` is the output-layer
    /// delta the trainer computed. That makes the applied policy delta
    /// readable from the moved output without re-deriving the term.
    fn linear_gate() -> Network {
        NetworkBuilder::new()
            .input(4)
            .output(4, Activation::Linear)
            .build_with_seed(3)
            .unwrap()
    }

    /// `1 + ||CTX||^2`, the factor relating a single Linear layer's
    /// output-layer delta to the move it causes in that layer's forward output
    /// on `CTX`: each weight gets `-lr * delta_i * context_j`, so the output
    /// moves by `-lr * (1 + ||CTX||^2) * delta_i`.
    fn linear_output_gain() -> f32 {
        1.0 + CTX.iter().map(|x| x * x).sum::<f32>()
    }

    /// The output-layer delta a step applied, recovered from a Linear gate's
    /// forward output on `CTX` before and after the step.
    fn applied_delta(before: &[f32], after: &[f32], learning_rate: f32) -> Vec<f32> {
        let gain = learning_rate * linear_output_gain();
        before
            .iter()
            .zip(after.iter())
            .map(|(b, a)| (b - a) / gain)
            .collect()
    }

    /// A positive reward on action 2 must increase the logit for action 2.
    ///
    /// Mutation that fails this test: setting learning_rate to 0 or zeroing
    /// the policy gradient term in `step`.
    #[test]
    fn rloo_positive_reward_increases_action_score() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);

        let before = gate.forward(&CTX).unwrap()[2];
        trainer.step(&mut gate, &CTX, 2, 1.0).unwrap();
        let after = gate.forward(&CTX).unwrap()[2];

        assert!(
            after > before,
            "positive reward must increase action score: before={before}, after={after}"
        );
    }

    /// A negative reward on action 1 must DECREASE the logit for action 1.
    ///
    /// This is the critical polarity test. Mutation that fails: dropping the
    /// `reward *` factor (treating −1.0 as +1.0) or routing negative reward
    /// through a cross-entropy call toward a different preferred index.
    #[test]
    fn rloo_negative_reward_decreases_action_score() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);

        let before = gate.forward(&CTX).unwrap()[1];
        trainer.step(&mut gate, &CTX, 1, -1.0).unwrap();
        let after = gate.forward(&CTX).unwrap()[1];

        assert!(
            after < before,
            "negative reward must decrease action score (polarity test): before={before}, after={after}"
        );
    }

    /// Load-balance loss must be nonzero for a peaked distribution and near
    /// zero for a uniform distribution.
    ///
    /// Mutation that fails: returning 0.0 unconditionally.
    #[test]
    #[allow(deprecated)]
    fn load_balance_aux_loss_nonzero_on_skewed() {
        let skewed_logits = [10.0_f32, -10.0, -10.0, -10.0];
        let probs = softmax(&skewed_logits);
        let loss = load_balance_aux_loss(&probs);
        assert!(
            loss > 1e-6,
            "load-balance loss must be nonzero on skewed distribution, got {loss}"
        );

        let uniform = [0.25_f32; 4];
        let uniform_loss = load_balance_aux_loss(&uniform);
        assert!(
            uniform_loss < 1e-6,
            "load-balance loss must be near zero for uniform distribution, got {uniform_loss}"
        );
    }

    /// Router z-loss must be strictly larger for large logits than for zero logits.
    ///
    /// Mutation that fails: returning 0.0 unconditionally.
    #[test]
    fn router_z_loss_nonzero_on_large_logits() {
        let large = [100.0_f32; 4];
        let small = [0.0_f32; 4];
        let z_large = router_z_loss(&large);
        let z_small = router_z_loss(&small);
        assert!(
            z_large > z_small,
            "z-loss must be larger for large logits: z_large={z_large}, z_small={z_small}"
        );
        assert!(
            z_large > 1e-6,
            "z-loss must be nonzero for large logits, got {z_large}"
        );
    }

    /// Wrong context dimension must return Err, not panic.
    #[test]
    fn rloo_wrong_context_dim_errors() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
        let wrong_ctx = [1.0_f32; 5]; // gate expects 4
        let result = trainer.step(&mut gate, &wrong_ctx, 0, 1.0);
        assert!(
            result.is_err(),
            "wrong context dimension must return Err, not panic"
        );
    }

    /// Explicit reward (+1.0) must move the action logit more than implicit (+0.5).
    ///
    /// Mutation that fails: computing the gradient without the `reward *` scale
    /// factor, or treating all rewards as +1.0.
    #[test]
    fn rloo_implicit_weaker_than_explicit() {
        let gate_base = test_gate();

        let mut gate_explicit = gate_base.clone();
        let mut gate_implicit = gate_base;

        let mut trainer_e = RlooTrainer::with_seed(RlooConfig::default(), 1);
        let mut trainer_i = RlooTrainer::with_seed(RlooConfig::default(), 1);

        let before_e = gate_explicit.forward(&CTX).unwrap()[2];
        let before_i = gate_implicit.forward(&CTX).unwrap()[2];

        trainer_e.step(&mut gate_explicit, &CTX, 2, 1.0).unwrap();
        trainer_i.step(&mut gate_implicit, &CTX, 2, 0.5).unwrap();

        let after_e = gate_explicit.forward(&CTX).unwrap()[2];
        let after_i = gate_implicit.forward(&CTX).unwrap()[2];

        let delta_e = (after_e - before_e).abs();
        let delta_i = (after_i - before_i).abs();

        assert!(
            delta_e > delta_i,
            "explicit reward (+1.0) must move action score more than implicit (+0.5): \
             delta_e={delta_e}, delta_i={delta_i}"
        );
    }

    // ---- Allocation-bound guard tests ---------------------------------------

    /// rloo_step with m_samples > MAX_ALLOWED_ELEMENTS must return Err, not panic.
    ///
    /// Mutation that breaks this: removing the `validate_allocation_size(m_samples)?` call.
    #[test]
    fn rloo_step_m_samples_too_large_returns_err() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
        let result = trainer.rloo_step(&mut gate, &CTX, 0, 1, MAX_ALLOWED_ELEMENTS + 1);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "expected ShapeTooLarge error for m_samples > MAX, got {result:?}"
        );
    }

    /// A per-call m_samples that is itself within bounds must still be rejected
    /// when m_samples * k (the aggregate retained-subset storage) exceeds the cap.
    /// Here m_samples = 30M passes the standalone m_samples guard (< 100M) but
    /// 30M * 4 = 120M exceeds it, so only the product guard can produce the error.
    ///
    /// Mutation that breaks this: removing the `validate_allocation_size(subset_storage)?`
    /// product check (the standalone m_samples guard alone admits this input).
    #[test]
    fn rloo_step_m_samples_times_k_product_too_large_returns_err() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
        let k = gate.num_outputs(); // 4
        let result = trainer.rloo_step(&mut gate, &CTX, 0, k, 30_000_000);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "expected ShapeTooLarge error for m_samples*k > MAX, got {result:?}"
        );
    }

    /// The property the per-context form lacked: at balanced traffic the guard is
    /// silent no matter how confident the gate is.
    ///
    /// The second half is the control. It asserts that the superseded per-context
    /// form is NOT silent on the same input, so this test discriminates between
    /// the two objectives rather than passing for both.
    #[test]
    #[allow(deprecated)]
    fn balanced_route_freqs_give_zero_aux_gradient_even_for_a_sharp_gate() {
        let k = 4usize;
        let freqs = vec![1.0 / k as f32; k];
        let probs = softmax(&[10.0, 0.0, 0.0, 0.0]);
        assert!(
            probs[0] > 0.99,
            "fixture must be a sharp gate, got {probs:?}"
        );

        let grad = load_balance_aux_gradient(&freqs, &probs).unwrap();
        for (j, &g) in grad.iter().enumerate() {
            assert!(
                g.abs() < 1e-6,
                "balanced traffic must produce no load-balance pressure at j={j}, got {g}"
            );
        }

        let per_context = load_balance_aux_loss(&probs);
        assert!(
            per_context > 1e-3,
            "control: the superseded per-context form must be NONZERO here, else this \
             test would pass for both objectives and prove nothing (got {per_context})"
        );
    }

    /// Collapsed traffic must push probability mass off the over-used expert.
    ///
    /// Deltas are dL/dz and the trainer descends, so a POSITIVE gradient lowers
    /// that logit. The zero-sum assertion is the shift-invariance any softmax
    /// gradient must satisfy; it fails for most ways of getting the formula wrong.
    #[test]
    fn collapsed_route_freqs_push_mass_off_the_overused_expert() {
        let freqs = [0.9_f32, 0.05, 0.05];
        let probs = [1.0 / 3.0_f32; 3];

        let grad = load_balance_aux_gradient(&freqs, &probs).unwrap();
        assert!(
            grad[0] > 0.0,
            "over-used expert must get a positive gradient (descent lowers its logit), got {}",
            grad[0]
        );
        assert!(
            grad[1] < 0.0 && grad[2] < 0.0,
            "under-used experts must be pushed up, got {:?}",
            &grad[1..]
        );
        let sum: f32 = grad.iter().sum();
        assert!(
            sum.abs() < 1e-5,
            "a softmax gradient must be zero-sum, got {sum}"
        );
    }

    /// Ties the applied gradient to the stated loss, so the two cannot drift.
    ///
    /// Named in `load_balance_aux_loss_batch`'s doc comment: that function exists to
    /// state the objective, and this is what makes it load-bearing rather than
    /// decorative. A scalar nobody consumes is exactly the defect this change fixes.
    #[test]
    fn aux_gradient_matches_finite_difference_of_the_batch_loss() {
        let freqs = [0.5_f32, 0.2, 0.2, 0.1];
        let logits = [0.3_f32, -0.1, 0.7, 0.2];
        let probs = softmax(&logits);
        let analytic = load_balance_aux_gradient(&freqs, &probs).unwrap();

        let h = 1e-3_f32;
        for j in 0..logits.len() {
            let mut up = logits;
            let mut dn = logits;
            up[j] += h;
            dn[j] -= h;
            let l_up = load_balance_aux_loss_batch(&freqs, &softmax(&up)).unwrap();
            let l_dn = load_balance_aux_loss_batch(&freqs, &softmax(&dn)).unwrap();
            let numeric = (l_up - l_dn) / (2.0 * h);
            assert!(
                (analytic[j] - numeric).abs() < 1e-2,
                "analytic gradient must match the loss it claims to differentiate at j={j}: \
                 analytic={}, numeric={numeric}",
                analytic[j]
            );
        }
    }

    /// The one-copy assertion: this reddens if `step` stops routing through the
    /// named guard, or stops folding the decision into the frequency EMA.
    ///
    /// Reward 0 kills the policy term and `z_loss_coeff` 0 kills the z term, so the
    /// load-balance guard is the ONLY thing left that can move a weight.
    #[test]
    fn step_applies_the_named_load_balance_guard() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.0,
        };
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(config, 1);

        // A fresh trainer is at uniform f, so the guard is silent and nothing at
        // all may move. This arm also pins the documented initialisation.
        let before = gate.forward(&CTX).unwrap().to_vec();
        trainer.step(&mut gate, &CTX, 0, 0.0).unwrap();
        let after_first = gate.forward(&CTX).unwrap().to_vec();
        assert_eq!(
            before, after_first,
            "at uniform f with no policy and no z term, the guard must contribute nothing"
        );

        // Send every decision to expert 0 so observed traffic drifts off balance.
        for _ in 0..200 {
            trainer.step(&mut gate, &CTX, 0, 0.0).unwrap();
        }
        let after_drift = gate.forward(&CTX).unwrap().to_vec();

        assert_ne!(
            after_first, after_drift,
            "once traffic has collapsed onto one expert the guard must act; if this \
             passes, step() is no longer applying the named function"
        );
        assert!(
            after_drift[0] < after_first[0],
            "the over-used expert's logit must fall: before={}, after={}",
            after_first[0],
            after_drift[0]
        );
    }

    /// A frequency vector of the wrong width is a caller error, not a shape to
    /// silently paper over with a uniform default.
    #[test]
    fn aux_gradient_rejects_a_frequency_vector_of_the_wrong_width() {
        let err = load_balance_aux_gradient(&[0.5, 0.5], &[0.3, 0.3, 0.4]).unwrap_err();
        assert!(
            matches!(err, FannError::InputSizeMismatch { .. }),
            "expected InputSizeMismatch, got {err:?}"
        );
    }

    // ---- ADR-091 Decision 3 (amended 2026-09-20): the learned-weight path ---

    /// Repeats [`test_gate`]'s drift preamble against two independent clones
    /// of the same base gate/trainer, so a caller can vary exactly one
    /// argument to `step_with_learned_weights` and attribute any difference in
    /// the result to that argument alone.
    fn drifted_off_uniform(gate_base: &Network, config: RlooConfig) -> (Network, RlooTrainer) {
        let mut gate = gate_base.clone();
        let mut trainer = RlooTrainer::with_seed(config, 1);
        for _ in 0..50 {
            trainer.step(&mut gate, &CTX, 0, 0.0).unwrap();
        }
        (gate, trainer)
    }

    /// Control 1 — the projection changes the number: the caller's projected
    /// mixture-weight vector and a plausible full-width selection distribution
    /// must give different load-balance gradients, or a test that swaps one
    /// for the other could never tell them apart.
    ///
    /// Mutation this guards: `step_with_learned_weights` silently ignoring
    /// `w_full` and falling back to its own internally computed `probs` for
    /// the load-balance term.
    #[test]
    fn step_with_learned_weights_moves_the_gate_by_w_full_not_by_probs() {
        // Non-uniform on purpose: at uniform route_freqs, `load_balance_aux_gradient`
        // is identically zero for ANY probability vector (see
        // `balanced_route_freqs_give_zero_aux_gradient_even_for_a_sharp_gate`),
        // which would make this fixture uninformative.
        let route_freqs = [0.1_f32, 0.3, 0.3, 0.3];
        let logits = [1.0_f32, 0.0, 3.0, 0.0];
        let probs = softmax(&logits);

        let tau = 0.5_f32;
        let w = softmax(&[logits[0] / tau, logits[2] / tau]);
        let w_full = [w[0], 0.0, w[1], 0.0];

        let grad_probs = load_balance_aux_gradient(&route_freqs, &probs).unwrap();
        let grad_w_full = load_balance_aux_gradient(&route_freqs, &w_full).unwrap();
        assert!(
            (grad_probs[0] - grad_w_full[0]).abs() > 1e-2,
            "fixture must make the two vectors diverge: grad(probs)[0]={}, grad(w_full)[0]={}",
            grad_probs[0],
            grad_w_full[0]
        );

        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.0,
        };
        let gate_base = test_gate();

        let (mut gate_a, mut trainer_a) = drifted_off_uniform(&gate_base, config.clone());
        trainer_a
            .step_with_learned_weights(&mut gate_a, &CTX, 0, 0.0, &w_full, tau)
            .unwrap();
        let after_w_full = gate_a.forward(&CTX).unwrap().to_vec();

        let (mut gate_b, mut trainer_b) = drifted_off_uniform(&gate_base, config);
        trainer_b
            .step_with_learned_weights(&mut gate_b, &CTX, 0, 0.0, &probs, tau)
            .unwrap();
        let after_probs = gate_b.forward(&CTX).unwrap().to_vec();

        assert_ne!(
            after_w_full, after_probs,
            "step_with_learned_weights must move the gate by w_full, not by a \
             different vector of the same shape: w_full={w_full:?}, probs={probs:?}"
        );
    }

    /// Control 2 — the `1 / tau` factor: with `w_full` and everything else
    /// held fixed, two calls that differ only in `tau` (chosen well away from
    /// 1.0) must move the gate by different amounts.
    ///
    /// Mutation this guards: dropping the `1 / tau` scale on the load-balance
    /// gradient. Predicted before running: with the factor dropped, `tau` has
    /// no effect on the applied gradient at all, so the two calls below would
    /// produce identical states and this test would redden.
    #[test]
    fn step_with_learned_weights_scales_the_aux_gradient_by_inverse_tau() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.0,
        };
        let gate_base = test_gate();
        let w_full = [0.7_f32, 0.0, 0.3, 0.0];

        let (mut gate_a, mut trainer_a) = drifted_off_uniform(&gate_base, config.clone());
        trainer_a
            .step_with_learned_weights(&mut gate_a, &CTX, 0, 0.0, &w_full, 1.0)
            .unwrap();
        let after_tau_1 = gate_a.forward(&CTX).unwrap().to_vec();

        let (mut gate_b, mut trainer_b) = drifted_off_uniform(&gate_base, config);
        trainer_b
            .step_with_learned_weights(&mut gate_b, &CTX, 0, 0.0, &w_full, 0.1)
            .unwrap();
        let after_tau_p1 = gate_b.forward(&CTX).unwrap().to_vec();

        let max_diff = after_tau_1
            .iter()
            .zip(after_tau_p1.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "tau=1.0 and tau=0.1 must move the gate by measurably different \
             amounts; got after_tau_1.0={after_tau_1:?}, after_tau_0.1={after_tau_p1:?}"
        );
    }

    /// Control 3 — the EMA fold is reached: `observe_routing`'s silent
    /// length-mismatch early return must not be swallowing the weighted path.
    /// `route_freqs` (private; visible here because `tests` is a descendant
    /// module of the one that defines it) must move toward `w_full`'s shape —
    /// split across indices 0 and 2 — not toward a one-hot on `action_idx`
    /// (which is 0 here, so a one-hot fold would push index 2 DOWN, not up).
    #[test]
    fn step_with_learned_weights_folds_w_full_into_route_freqs() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
        let w_full = [0.5_f32, 0.0, 0.5, 0.0];

        for _ in 0..50 {
            trainer
                .step_with_learned_weights(&mut gate, &CTX, 0, 0.0, &w_full, 1.0)
                .unwrap();
        }

        let f = &trainer.route_freqs;
        assert!(
            (f[0] - 0.25).abs() > 1e-3,
            "f must have moved off uniform (1/4) at index 0, got {}",
            f[0]
        );
        assert!(
            f[2] > 0.25 + 1e-3,
            "f must have moved toward the weighted vector at index 2, which \
             carries half the mass despite action_idx=0 — a one-hot(action_idx) \
             fold would have pushed this DOWN instead: got {}",
            f[2]
        );
        assert!(
            f[1] < 0.25 && f[3] < 0.25,
            "indices absent from w_full must have moved down, got f={f:?}"
        );
    }

    /// Control 4 — `step`'s existing behaviour is pinned across this change:
    /// one full `step` call from a fresh, seeded gate must produce the exact
    /// output it produced before `step` was rewritten to delegate to
    /// `step_impl`. The refactor moves code; it must not move a float.
    #[test]
    fn step_output_is_pinned_across_the_step_impl_refactor() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
        trainer.step(&mut gate, &CTX, 2, 1.0).unwrap();
        let after = gate.forward(&CTX).unwrap().to_vec();

        // Captured from this exact fixture (test_gate() seed 1, CTX,
        // action_idx=2, reward=1.0, RlooConfig::default()) by running the
        // pre-refactor `step` at base commit 0d4507889a, before `step_impl`
        // existed, then asserted tight against drift.
        //
        // This fixture CANNOT see the load-balance term at all, and that is a
        // property of the fixture rather than of the code it pins: a fresh
        // trainer's `route_freqs` are uniform, and `load_balance_aux_gradient`
        // is identically zero at uniform freqs (`K * p_j * (f_j - f-bar)` with
        // every `f_j` equal), whatever `probs` is. Mutating the aux term's
        // scale leaves this assertion green. It pins the policy and z-loss
        // arms; the arm that pins the load-balance one is
        // `step_output_is_pinned_on_a_fixture_that_can_see_the_aux_term`
        // below, which drifts the frequencies first. Kept because the
        // fresh-trainer path is also worth pinning, and labelled because an
        // unlabelled pin that cannot express the defect it names is worse
        // than no pin.
        let expected: [f32; 4] = [0.09232578, 0.0055880453, -0.08243938, -0.02555845];
        for (i, (&got, &want)) in after.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "step()'s output must be pinned at i={i}: got={got}, want={want}"
            );
        }
    }

    /// `step`'s output is BIT-identical across the `step_impl` refactor on a
    /// fixture where the load-balance term is non-zero, which is the arm the
    /// fresh-trainer pin above cannot provide.
    ///
    /// The refactor routes `step` through `aux_scale = 1.0`. That is exact for
    /// every finite `f32`, so the claim is bit-identity rather than tolerance,
    /// and the assertion compares raw bit patterns: a tolerance here would
    /// accept precisely the drift the claim denies.
    ///
    /// The expected bits were captured by running this same fixture against
    /// the PRE-refactor `step` at base commit `0d4507889a` and read back as
    /// `to_bits()`, not transcribed from a decimal rendering. The fixture's
    /// sensitivity was then established in the same pass by mutating the
    /// `None` branch's scale from `1.0` to `2.0`, which moved every one of the
    /// four words.
    #[test]
    fn step_output_is_pinned_on_a_fixture_that_can_see_the_aux_term() {
        let (mut gate, mut trainer) = drifted_off_uniform(&test_gate(), RlooConfig::default());
        trainer.step(&mut gate, &CTX, 2, 1.0).unwrap();
        let bits: Vec<u32> = gate
            .forward(&CTX)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();

        assert_eq!(
            bits,
            vec![1035781587, 1001842595, 3181957232, 3167838379],
            "step() must be bit-identical to its pre-refactor form on a \
             fixture with non-uniform route frequencies"
        );
    }
    // ── The explored-mixture policy term (ADR-096 decision 3) ───────────────

    // Each arm below states, in its own doc, what would make it fail. The
    // mutation controls that redden them are in the handoff report, not here:
    // a control list in the same file as the arms is a second copy of the arms.

    /// The applied score is the analytic gradient of the Gaussian log density
    /// `log N(z'; z, s^2 I)` with respect to `z`, checked against a central
    /// finite difference of that density taken in `f64`.
    ///
    /// Read from the applied output-layer delta on a Linear gate, so the arm
    /// exercises the source term rather than restating it. Fails if the scored
    /// quantity is not the density of `z'`, or if the scale appears with the
    /// wrong power: a term that is not `(z' - z) / s^2` (dropped or linear
    /// factor, or the sign of the density's own gradient) is a different
    /// number, and a central difference of the density is the one thing that
    /// can tell.
    #[test]
    fn the_applied_score_matches_a_finite_difference_of_the_gaussian_log_density() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.4_f32, 0.0, 0.6, 0.0];
        let gate = linear_gate();
        let z = gate.clone().forward(&CTX).unwrap().to_vec();
        let offset = [3.7_f32, -0.7, 1.1, -0.2];
        let sampled: Vec<f32> = z.iter().zip(offset.iter()).map(|(zj, d)| zj + d).collect();

        for scale in [0.5_f32, 1.7] {
            let s = f64::from(scale);
            // `log N(z'; z, s^2 I) = -||z' - z||^2 / (2 s^2) + const` as a
            // function of `z`. The difference is taken in f64 on the f64
            // image of the gate's output, so the step `h` is exact and the
            // comparison isolates the term rather than f32 rounding of `z +- h`.
            let log_density = |z: &[f64]| -> f64 {
                let sq: f64 = z
                    .iter()
                    .zip(sampled.iter())
                    .map(|(zj, sj)| {
                        let d = f64::from(*sj) - zj;
                        d * d
                    })
                    .sum();
                -sq / (2.0 * s * s)
            };
            let base: Vec<f64> = z.iter().map(|v| f64::from(*v)).collect();

            let mut g = gate.clone();
            let before = g.forward(&CTX).unwrap().to_vec();
            RlooTrainer::new(config.clone())
                .step_with_explored_action(
                    &mut g,
                    &CTX,
                    &ExploredAction {
                        sampled_logits: &sampled,
                        exploration_scale: scale,
                        w_full: &w_full,
                        tau: 1.0,
                    },
                    1.0,
                )
                .unwrap();
            let after = g.forward(&CTX).unwrap().to_vec();
            let delta = applied_delta(&before, &after, config.learning_rate);

            let h = 1e-3_f64;
            for j in 0..base.len() {
                let mut up = base.clone();
                let mut down = base.clone();
                up[j] += h;
                down[j] -= h;
                let numeric = (log_density(&up) - log_density(&down)) / (2.0 * h);
                // The applied delta is `-reward` times the density's gradient
                // at reward 1.0.
                let analytic = -f64::from(delta[j]);
                assert!(
                    (analytic - numeric).abs() < 1e-4,
                    "scale {scale} component {j}: applied {analytic} vs finite difference {numeric}"
                );
            }
        }
    }

    /// A positive reward moves the gate's output on the sample's context
    /// TOWARD the logits that were served, and a negative reward moves it
    /// away; on a Linear gate the applied delta is computable in closed form.
    ///
    /// Fails on a sign slip: `reward * (z' - z) / s^2` in place of
    /// `reward * (z - z') / s^2` reverses both inequalities.
    #[test]
    fn reward_polarity_moves_the_gate_toward_the_served_logits() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.4_f32, 0.0, 0.6, 0.0];
        let gate_base = linear_gate();
        let z = gate_base.clone().forward(&CTX).unwrap().to_vec();
        let offset = [0.4_f32, -0.3, 0.5, -0.2];
        let sampled: Vec<f32> = z.iter().zip(offset.iter()).map(|(zj, d)| zj + d).collect();

        let delta_for = |reward: f32| -> Vec<f32> {
            let mut gate = gate_base.clone();
            let before = gate.forward(&CTX).unwrap().to_vec();
            RlooTrainer::new(config.clone())
                .step_with_explored_action(
                    &mut gate,
                    &CTX,
                    &ExploredAction {
                        sampled_logits: &sampled,
                        exploration_scale: 1.0,
                        w_full: &w_full,
                        tau: 1.0,
                    },
                    reward,
                )
                .unwrap();
            let after = gate.forward(&CTX).unwrap().to_vec();
            applied_delta(&before, &after, config.learning_rate)
        };

        let towards = delta_for(1.0);
        let away = delta_for(-1.0);
        for ((&actual, &d), &back) in towards.iter().zip(offset.iter()).zip(away.iter()) {
            // The applied delta is `reward * (z - z') / s^2 = -reward * offset`
            // at `s = 1`.
            assert!(
                (actual + d).abs() < 1e-5,
                "positive reward must move toward the served logits: got {actual}, want {}",
                -d
            );
            assert!(
                (back - d).abs() < 1e-5,
                "negative reward must move away: got {back}, want {d}"
            );
        }
    }

    /// With both guards off, an action whose served logits equal the gate's
    /// own output carries no signal: `z' - z` is exactly zero, the score is
    /// zero, and the gate must be left byte-for-byte unchanged.
    ///
    /// Fails if the term acquires any component not proportional to
    /// `z' - z` -- a softmax or weight-derivative term would move the gate
    /// even when the served logits are the unperturbed output.
    #[test]
    fn a_sample_with_no_exploration_leaves_the_gate_byte_for_byte_unchanged() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.25_f32; 4];
        let mut gate = test_gate();
        let z = gate.forward(&CTX).unwrap().to_vec();
        let before = gate.to_bytes();

        let loss = RlooTrainer::new(config)
            .step_with_explored_action(
                &mut gate,
                &CTX,
                &ExploredAction {
                    sampled_logits: &z,
                    exploration_scale: 0.7,
                    w_full: &w_full,
                    tau: 1.0,
                },
                1.0,
            )
            .unwrap();

        assert_eq!(loss, 0.0, "a zero offset has zero loss");
        assert_eq!(gate.to_bytes(), before, "z' == z must not move the gate");
    }

    /// The applied policy delta scales as `1 / s^2`: doubling `s` with the
    /// same `z' - z` quarters it. On the Linear gate the applied delta is
    /// `reward * (z' - z) / s^2` exactly.
    ///
    /// Fails if the term drops the `1 / s^2` factor (ratio becomes 1), uses
    /// `1 / s` instead (ratio becomes 1/2), or flips the sign.
    #[test]
    fn doubling_the_scale_quarters_the_policy_delta() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.4_f32, 0.0, 0.6, 0.0];
        let gate = linear_gate();
        let z = gate.clone().forward(&CTX).unwrap().to_vec();
        let offset = [0.4_f32, -0.7, 1.1, -0.2];
        let sampled: Vec<f32> = z.iter().zip(offset.iter()).map(|(zj, d)| zj + d).collect();

        let run = |scale: f32| -> Vec<f32> {
            let mut g = gate.clone();
            let before = g.forward(&CTX).unwrap().to_vec();
            let loss = RlooTrainer::new(config.clone())
                .step_with_explored_action(
                    &mut g,
                    &CTX,
                    &ExploredAction {
                        sampled_logits: &sampled,
                        exploration_scale: scale,
                        w_full: &w_full,
                        tau: 1.0,
                    },
                    1.0,
                )
                .unwrap();
            // The logged loss is `reward * ||z' - z||^2 / (2 s^2)`, divided by
            // the same `s^2` as the applied delta.
            let sq: f32 = offset.iter().map(|d| d * d).sum();
            let expected = sq * 0.5 / (scale * scale);
            assert!(
                (loss - expected).abs() < 1e-4,
                "logged loss at s={scale} should be {expected}, got {loss}"
            );
            let after = g.forward(&CTX).unwrap().to_vec();
            applied_delta(&before, &after, config.learning_rate)
        };

        let narrow = run(1.0);
        let wide = run(2.0);
        for ((&n, &w), &d) in narrow.iter().zip(wide.iter()).zip(offset.iter()) {
            assert!(d.abs() > 1e-3, "fixture must produce a non-zero delta");
            assert!(
                w.abs() > 1e-6,
                "the wide-scale delta must be non-zero to take a ratio"
            );
            let ratio = n / w;
            assert!(
                (ratio - 4.0).abs() < 1e-3,
                "doubling s must quarter the delta, got ratio {ratio} (narrow {n}, wide {w})"
            );
        }
    }

    /// The single-index path is untouched by the explored-mixture branch: the
    /// pinned outputs above pin `step`, and this arm pins that an explored
    /// action with zero offset is also a no-op for the batch entry.
    ///
    /// Fails if the branch applies a term the `Single` path does not, which
    /// the byte comparison to an all-equal (uninformative) round would show.
    #[test]
    fn a_zero_offset_batch_round_leaves_the_gate_untouched() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.4_f32, 0.0, 0.6, 0.0];
        let base = test_gate();

        let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
        // The served logits must be THIS gate's own output, or `z' - z` is the
        // drift the setup applied and not the zero offset this arm is about.
        let z = gate.forward(&CTX).unwrap().to_vec();
        let before = gate.to_bytes();
        let samples = [
            MixtureSample {
                context: &CTX,
                action: ExploredAction {
                    sampled_logits: &z,
                    exploration_scale: 0.8,
                    w_full: &w_full,
                    tau: 0.5,
                },
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: ExploredAction {
                    sampled_logits: &z,
                    exploration_scale: 0.8,
                    w_full: &w_full,
                    tau: 0.5,
                },
                reward: -1.0,
            },
        ];
        let outcome = trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();

        assert_eq!(
            gate.to_bytes(),
            before,
            "a round whose every action served the gate's own output moved the gate"
        );
        assert_eq!(
            outcome.informative, 2,
            "both episodes carry a non-zero advantage; only the policy term is zero"
        );
    }

    /// The batch entry refuses a sampled logit that is not finite, before any
    /// of the round is applied. Fails if the finiteness check is dropped: the
    /// NaN would reach the forward pass and corrupt every parameter.
    #[test]
    fn a_malformed_explored_action_leaves_the_whole_round_unapplied() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = test_gate();
        let good = [0.1_f32, -0.2, 0.3, 0.4];

        let bad: [(&str, &[f32], f32); 3] = [
            ("nan sampled logit", &[0.1, f32::NAN, 0.3, 0.4], 1.0),
            (
                "infinite sampled logit",
                &[0.1, f32::INFINITY, 0.3, 0.4],
                1.0,
            ),
            ("wrong sampled width", &[0.1, 0.2], 1.0),
        ];

        for (label, sampled, scale) in bad {
            let (mut gate, mut trainer) = drifted_off_uniform(&base, config.clone());
            let before = gate.to_bytes();
            let samples = [
                MixtureSample {
                    context: &CTX,
                    action: ExploredAction {
                        sampled_logits: &good,
                        exploration_scale: 1.0,
                        w_full: &w_full,
                        tau: 1.0,
                    },
                    reward: 1.0,
                },
                MixtureSample {
                    context: &CTX,
                    action: ExploredAction {
                        sampled_logits: sampled,
                        exploration_scale: scale,
                        w_full: &w_full,
                        tau: 1.0,
                    },
                    reward: -1.0,
                },
            ];
            let err = trainer
                .step_batch_with_explored_actions(&mut gate, &samples)
                .unwrap_err();
            assert_eq!(gate.to_bytes(), before, "{label}: {err} moved the gate");
        }

        // The control, in the same test: the same first sample paired with a
        // VALID second one does reach the gate.
        let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
        let before = gate.to_bytes();
        let other = [0.4_f32, 0.3, -0.2, 0.1];
        let samples = [
            MixtureSample {
                context: &CTX,
                action: ExploredAction {
                    sampled_logits: &good,
                    exploration_scale: 1.0,
                    w_full: &w_full,
                    tau: 1.0,
                },
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: ExploredAction {
                    sampled_logits: &other,
                    exploration_scale: 1.0,
                    w_full: &w_full,
                    tau: 1.0,
                },
                reward: -1.0,
            },
        ];
        trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        assert_ne!(gate.to_bytes(), before);
    }

    /// Every refusal, and each names the field it saw: a zero or negative or
    /// non-finite scale, a non-finite tau, and a `w_full` of the wrong width.
    ///
    /// Fails if any of the checks is dropped; the accepted shape is the
    /// control in the same test.
    #[test]
    fn the_explored_entry_refuses_every_shape_that_is_not_an_action() {
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        };
        let w_full = [0.5_f32, 0.0, 0.5, 0.0];
        let sampled = [0.1_f32, -0.2, 0.3, 0.4];
        let mut trainer = RlooTrainer::new(config);
        let mut gate = test_gate();

        // (label, scale, tau, w_full length)
        let cases: [(&str, f32, f32, usize); 5] = [
            ("zero scale", 0.0, 1.0, 4),
            ("negative scale", -1.0, 1.0, 4),
            ("nan scale", f32::NAN, 1.0, 4),
            ("zero tau", 1.0, 0.0, 4),
            ("wrong w_full length", 1.0, 1.0, 3),
        ];
        for (name, scale, tau, w_len) in cases {
            let w: Vec<f32> = w_full.iter().copied().take(w_len).collect();
            let err = trainer
                .step_with_explored_action(
                    &mut gate,
                    &CTX,
                    &ExploredAction {
                        sampled_logits: &sampled,
                        exploration_scale: scale,
                        w_full: &w,
                        tau,
                    },
                    1.0,
                )
                .unwrap_err();
            // The scale and tau failures are `TrainingError` and name the field;
            // the width failure is the `InputSizeMismatch` that
            // `step_with_learned_weights` already returns for a `w_full` of the
            // wrong width.
            if matches!(err, FannError::TrainingError(_)) {
                assert!(
                    err.to_string().contains("explored action"),
                    "{name} must be refused and name the field, got {err}"
                );
            } else {
                assert!(
                    w_len != 4,
                    "{name} was refused as {err} but is not a width failure"
                );
            }
        }

        // The control: the shape all of these differ from is accepted.
        trainer
            .step_with_explored_action(
                &mut gate,
                &CTX,
                &ExploredAction {
                    sampled_logits: &sampled,
                    exploration_scale: 1.0,
                    w_full: &w_full,
                    tau: 1.0,
                },
                1.0,
            )
            .expect("the valid shape must be accepted");
    }

    // ── Batch refit over recorded explored actions (ADR-096 decision 3) ─────

    /// Build an explored action from the served logits and the recorded
    /// mixture at scale `1.0`. The context is the same across the arms below,
    /// so any per-sample difference has to come from the action.
    fn batch_action<'a>(sampled_logits: &'a [f32], w_full: &'a [f32]) -> ExploredAction<'a> {
        ExploredAction {
            sampled_logits,
            exploration_scale: 1.0,
            w_full,
            tau: 0.5,
        }
    }

    /// The leave-one-out baseline is the mean of the OTHER rewards, checked
    /// against arithmetic done by hand rather than by the same expression.
    ///
    /// Fails if the baseline uses the sample's own reward: for `r_0` in
    /// `(1, -1, 1)` the others have mean 0, not 1.
    #[test]
    fn the_leave_one_out_advantage_is_the_reward_minus_the_mean_of_the_others() {
        // total 1.0; for r_0 the others are (-1, 1) with mean 0, so adv 1.0.
        // For r_1 the others are (1, 1) with mean 1, so adv -2.0.
        let advantages = leave_one_out_advantages(&[1.0, -1.0, 1.0]);
        assert_eq!(advantages, vec![1.0, -2.0, 1.0]);

        // A round of one has no other reward to stand as a baseline. Zero
        // rather than the raw reward: the alternative reads one judgement as
        // "better than nothing", which is not what one judgement says.
        assert_eq!(leave_one_out_advantages(&[1.0]), vec![0.0]);
        assert_eq!(leave_one_out_advantages(&[]), Vec::<f32>::new());
    }

    /// A round whose rewards agree gives EXACTLY zero, with no special case
    /// in the code to make it so.
    ///
    /// Fails if the `f64` accumulator is downgraded to `f32`: a residue of a
    /// few ULPs then makes the all-equal advantages non-zero in whichever
    /// direction the rounding fell. The `f64` accumulator is what buys this
    /// (see [`leave_one_out_advantages`]). The rewards below are awkward on
    /// purpose: `0.1` has no exact binary form, and the round sizes are not
    /// powers of two, so a residue would show if the argument were wrong.
    #[test]
    fn an_all_equal_round_is_exactly_zero_without_a_special_case() {
        for rewards in [
            vec![0.1_f32, 0.1, 0.1],
            vec![1.0_f32; 7],
            vec![-1.0_f32, -1.0],
            vec![0.7_f32; 13],
            vec![-33.333_f32; 5],
        ] {
            let advantages = leave_one_out_advantages(&rewards);
            assert!(
                advantages.iter().all(|a| *a == 0.0),
                "all-equal rewards {rewards:?} produced advantages {advantages:?}"
            );
        }

        // The control, in the same test: the zero is exact equality and not a
        // tolerance, so a round differing by ONE ULP still carries signal.
        let nudged = vec![0.1_f32, 0.1, f32::from_bits(0.1_f32.to_bits() + 1)];
        assert!(
            leave_one_out_advantages(&nudged).iter().any(|a| *a != 0.0),
            "a one-ULP difference is a difference"
        );
    }

    /// The gradient is taken against the ADVANTAGE, not the raw reward.
    ///
    /// `(+1, -1)` and `(+2, 0)` are different feedback and the same pair of
    /// advantages, so the two rounds must land in the same place. Fails if the
    /// policy term is scaled by the raw reward: the all-equal no-op is
    /// protected by the idle skip rather than by which number reaches the
    /// policy term, so a learner that optimised raw reward would still look
    /// correct there while optimising the wrong objective on every round that
    /// learns anything.
    #[test]
    fn the_round_optimises_the_advantage_and_not_the_raw_reward() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = test_gate();
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.1,
        };
        assert_eq!(
            leave_one_out_advantages(&[1.0, -1.0]),
            leave_one_out_advantages(&[2.0, 0.0]),
            "fixture assumption: two reward pairs, one pair of advantages"
        );

        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let first: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let second: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();

        let run = |rewards: [f32; 2]| {
            let (mut gate, mut trainer) = drifted_off_uniform(&base, config.clone());
            let samples = [
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&first, &w_full),
                    reward: rewards[0],
                },
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&second, &w_full),
                    reward: rewards[1],
                },
            ];
            trainer
                .step_batch_with_explored_actions(&mut gate, &samples)
                .unwrap();
            gate.to_bytes()
        };

        let shifted = run([2.0, 0.0]);
        assert_eq!(run([1.0, -1.0]), shifted);

        // Control: a round whose advantages really do differ lands elsewhere.
        assert_ne!(run([3.0, -1.0]), shifted);
    }

    /// A round whose rewards agree carries no comparison, so it is a no-op
    /// byte-for-byte -- the load-balance and z-loss terms do not move the gate
    /// on their own.
    ///
    /// Fails if the idle skip is dropped, so a round with zero advantage still
    /// accumulates the guard terms. A learner that stepped here would change
    /// served behaviour on feedback that said only "these were all fine".
    #[test]
    fn a_round_with_nothing_to_compare_leaves_the_gate_byte_for_byte_unchanged() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = test_gate();
        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();

        for rewards in [vec![1.0_f32, 1.0, 1.0], vec![-1.0_f32, -1.0], vec![1.0_f32]] {
            let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
            let before = gate.to_bytes();
            let samples: Vec<MixtureSample<'_>> = rewards
                .iter()
                .map(|r| MixtureSample {
                    context: &CTX,
                    action: batch_action(&served, &w_full),
                    reward: *r,
                })
                .collect();

            let outcome = trainer
                .step_batch_with_explored_actions(&mut gate, &samples)
                .unwrap();

            assert_eq!(outcome.informative, 0, "rewards {rewards:?}");
            assert_eq!(outcome.samples, rewards.len());
            assert_eq!(outcome.mean_loss, 0.0);
            assert_eq!(
                gate.to_bytes(),
                before,
                "rewards {rewards:?} moved the gate"
            );
        }

        // The control, in the same test: the fixture CAN move under this
        // trainer, so the equalities above are about the rewards and not about
        // a round that never reached the gate.
        let other: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let before = gate.to_bytes();
        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&served, &w_full),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&other, &w_full),
                reward: -1.0,
            },
        ];
        let outcome = trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        assert_eq!(outcome.informative, 2);
        assert_ne!(gate.to_bytes(), before);
    }

    /// Two episodes that share a context and an action but carry opposite
    /// rewards cancel their policy gradient at one gate, so the round lands in
    /// very nearly the same place whatever the magnitude of the pair -- and
    /// NOT in the same place as the same two updates applied one at a time.
    ///
    /// Fails if the round applies each sample inside the loop instead of
    /// accumulating all gradients at one gate: the second gradient would then
    /// be taken at a gate the first update already moved.
    ///
    /// The policy term is linear in the advantage, and `(+r, -r)` gives
    /// advantages `(+2r, -2r)` whose terms are exact negations. They cancel
    /// only if both gradients are taken at the gate that produced the actions.
    /// The residue is float rounding: the two terms are added to the guard
    /// terms before they meet, so the cancellation loses low bits, and it
    /// loses more of them at magnitude 17 than at 1.
    ///
    /// That residue is why the tolerance here is not a typed constant. The
    /// test measures how far a LOOP lands from the round and requires the
    /// magnitude drift to be two orders of magnitude smaller than that, so the
    /// arm calibrates itself against the rival implementation rather than
    /// against a number chosen to pass.
    #[test]
    fn an_antisymmetric_pair_cancels_at_one_gate_but_not_under_a_loop() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = linear_gate();
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.1,
        };
        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();

        let round = |magnitude: f32| {
            let (mut gate, mut trainer) = drifted_off_uniform(&base, config.clone());
            let samples = [
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&served, &w_full),
                    reward: magnitude,
                },
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&served, &w_full),
                    reward: -magnitude,
                },
            ];
            trainer
                .step_batch_with_explored_actions(&mut gate, &samples)
                .unwrap();
            gate.forward(&CTX).unwrap().to_vec()
        };
        let spread = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        };

        let at_one = round(1.0);
        let drift = spread(&at_one, &round(17.0));

        // The rival: the same two advantage-weighted updates applied one at a
        // time, at half the step so that the total step size matches the
        // averaged round. The remaining difference is where the second
        // gradient was taken.
        let (mut gate, mut trainer) = drifted_off_uniform(
            &base,
            RlooConfig {
                learning_rate: config.learning_rate / 2.0,
                ..config.clone()
            },
        );
        for advantage in [2.0_f32, -2.0] {
            trainer
                .step_with_explored_action(
                    &mut gate,
                    &CTX,
                    &batch_action(&served, &w_full),
                    advantage,
                )
                .unwrap();
        }
        let separation = spread(&at_one, gate.forward(&CTX).unwrap());

        assert!(
            drift * 100.0 < separation,
            "magnitude drift {drift} is not negligible beside the {separation} a loop lands away"
        );

        // Control: the round moved the gate at all, so neither number is a
        // comparison between two untouched fixtures.
        let (mut untouched, _) = drifted_off_uniform(&base, config);
        assert!(
            spread(&at_one, untouched.forward(&CTX).unwrap()) > drift * 100.0,
            "the round must move the gate by more than the residue this test calls negligible"
        );
    }

    /// The round moves the gate's output in the direction that separates the
    /// better-rewarded served logits from the worse-rewarded ones.
    ///
    /// The action is the whole served logit vector, so the comparison is over
    /// the vector the reward preferred, not over a set of columns. Fails if
    /// the policy term's sign is flipped: the projection below would be
    /// negative.
    #[test]
    fn the_round_moves_the_gate_toward_the_better_rewarded_logits() {
        let w_full = [0.5_f32, 0.5, 0.0, 0.0];
        // A plain Linear gate, no drift: the projection below is about the
        // policy term, and both guards are off so nothing else moves the gate.
        let mut gate = linear_gate();
        let mut trainer = RlooTrainer::new(RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 0.0,
            z_loss_coeff: 0.0,
        });
        let z = gate.forward(&CTX).unwrap().to_vec();
        // A per-column separation with mixed signs, so "toward" is a direction
        // and not one uniform offset every column happens to share.
        let separation = [0.4_f32, -0.3, 0.5, -0.2];
        let preferred: Vec<f32> = z
            .iter()
            .zip(separation.iter())
            .map(|(zj, d)| zj + d)
            .collect();
        let dispreferred: Vec<f32> = z
            .iter()
            .zip(separation.iter())
            .map(|(zj, d)| zj - d)
            .collect();

        let before = gate.forward(&CTX).unwrap().to_vec();
        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&preferred, &w_full),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&dispreferred, &w_full),
                reward: -1.0,
            },
        ];
        trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        let after = gate.forward(&CTX).unwrap().to_vec();

        // Positive when the move has a component in the direction that
        // separates `preferred` from `dispreferred`.
        let projection: f32 = before
            .iter()
            .zip(after.iter())
            .zip(separation.iter())
            .map(|((b, a), d)| (a - b) * d)
            .sum();
        assert!(
            projection > 0.0,
            "the round must move the gate toward the better-rewarded logits: \
             projection {projection}, before={before:?} after={after:?}"
        );
    }

    /// A round is refused as a whole, before any of it is applied -- including
    /// when the bad sample is not the first one.
    ///
    /// Fails if validation happens inside the apply loop, because sample one
    /// would already have moved the gate before the refusal is reached.
    #[test]
    fn a_refused_sample_leaves_the_whole_round_unapplied() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = test_gate();
        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let good: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let short_w = [0.5_f32, 0.5];
        let bad_logits = [0.1_f32, f32::NAN, 0.3, 0.4];

        let bad: [(&str, ExploredAction<'_>); 2] = [
            ("nan sampled logit", batch_action(&bad_logits, &w_full)),
            ("w_full of the wrong width", batch_action(&good, &short_w)),
        ];

        for (label, action) in bad {
            let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
            let before = gate.to_bytes();
            let samples = [
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&good, &w_full),
                    reward: 1.0,
                },
                MixtureSample {
                    context: &CTX,
                    action,
                    reward: -1.0,
                },
            ];
            let err = trainer
                .step_batch_with_explored_actions(&mut gate, &samples)
                .unwrap_err();
            assert_eq!(gate.to_bytes(), before, "{label}: {err} moved the gate");
        }

        // The control, in the same test: the same first sample paired with a
        // VALID second one does reach the gate, so the equalities above are
        // about the refusal.
        let other: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let before = gate.to_bytes();
        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&good, &w_full),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&other, &w_full),
                reward: -1.0,
            },
        ];
        trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        assert_ne!(gate.to_bytes(), before);
    }

    /// An empty round is a caller with no data, not a refit that changed
    /// nothing. A non-finite reward is refused for the same reason the single
    /// entry refuses a non-finite scale.
    ///
    /// Fails if either guard is dropped: an empty round would silently return
    /// an outcome, and a NaN reward would reach the policy term.
    #[test]
    fn an_empty_round_and_a_non_finite_reward_are_refused() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let mut gate = test_gate();
        let z = gate.clone().forward(&CTX).unwrap().to_vec();
        let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let other: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);

        assert!(
            trainer
                .step_batch_with_explored_actions(&mut gate, &[])
                .is_err()
        );

        for reward in [f32::NAN, f32::INFINITY] {
            let samples = [
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&served, &w_full),
                    reward,
                },
                MixtureSample {
                    context: &CTX,
                    action: batch_action(&other, &w_full),
                    reward: 1.0,
                },
            ];
            assert!(
                trainer
                    .step_batch_with_explored_actions(&mut gate, &samples)
                    .is_err(),
                "reward {reward} was accepted"
            );
        }
    }

    /// Every validated sample is traffic, whether or not the round could learn
    /// from it: the frequency EMA sees an all-equal round that moved no
    /// weight, and a later step feels the difference through the load-balance
    /// term.
    ///
    /// Fails if the idle samples are skipped before their mass is folded, or
    /// if the mass fold is dropped altogether.
    #[test]
    fn a_round_that_learns_nothing_still_counts_as_traffic() {
        let w_full = [1.0_f32, 0.0, 0.0, 0.0];
        let base = test_gate();
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.0,
        };
        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();

        // Arm A: an all-equal round (no weight change), then one ordinary step.
        let (mut gate_a, mut trainer_a) = drifted_off_uniform(&base, config.clone());
        let before_round = gate_a.to_bytes();
        let samples: Vec<MixtureSample<'_>> = (0..8)
            .map(|_| MixtureSample {
                context: &CTX,
                action: batch_action(&served, &w_full),
                reward: 1.0,
            })
            .collect();
        trainer_a
            .step_batch_with_explored_actions(&mut gate_a, &samples)
            .unwrap();
        assert_eq!(
            gate_a.to_bytes(),
            before_round,
            "the round must not have moved the gate, or this test measures the wrong thing"
        );
        trainer_a.step(&mut gate_a, &CTX, 2, 1.0).unwrap();

        // Arm B: the same step with no round before it.
        let (mut gate_b, mut trainer_b) = drifted_off_uniform(&base, config);
        trainer_b.step(&mut gate_b, &CTX, 2, 1.0).unwrap();

        assert_ne!(
            gate_a.to_bytes(),
            gate_b.to_bytes(),
            "the uninformative round left no trace in the traffic history"
        );
    }

    /// A round can hold episodes the baseline has nothing to say about beside
    /// ones it does, and the step is averaged over the CONTRIBUTORS.
    ///
    /// Fails if the divisor is the round size rather than the contributing
    /// count: the idle third episode would then shrink the step.
    ///
    /// Rewards `(+1, -1, 0)` put the third episode exactly on its own
    /// leave-one-out baseline. Dividing by the round size instead would shrink
    /// the step because an uninformative episode happened to be in the batch,
    /// which is a step size that depends on what the round did NOT learn from.
    #[test]
    fn the_step_is_averaged_over_the_episodes_that_contributed() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = test_gate();
        let config = RlooConfig {
            learning_rate: 0.5,
            aux_loss_coeff: 1.0,
            z_loss_coeff: 0.1,
        };
        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let first: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let second: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let advantages = leave_one_out_advantages(&[1.0, -1.0, 0.0]);
        assert_eq!(advantages, vec![1.5, -1.5, 0.0], "fixture assumption");

        let run = |rewards: &[f32]| {
            let (mut gate, mut trainer) = drifted_off_uniform(&base, config.clone());
            let samples: Vec<MixtureSample<'_>> = rewards
                .iter()
                .enumerate()
                .map(|(i, r)| MixtureSample {
                    context: &CTX,
                    action: batch_action(if i == 1 { &second } else { &first }, &w_full),
                    reward: *r,
                })
                .collect();
            let outcome = trainer
                .step_batch_with_explored_actions(&mut gate, &samples)
                .unwrap();
            (outcome, gate.to_bytes())
        };

        let (three, with_idle) = run(&[1.0, -1.0, 0.0]);
        assert_eq!(three.samples, 3);
        assert_eq!(three.informative, 2);

        // The same two contributing episodes, alone, at the rewards that
        // reproduce those advantages: a two-episode round has denominator 1,
        // so `(+0.75, -0.75)` gives `(+1.5, -1.5)` again. Same advantages,
        // same gradients, same divisor -- the idle third episode must not have
        // changed where the round landed.
        assert_eq!(leave_one_out_advantages(&[0.75, -0.75]), vec![1.5, -1.5]);
        let (two, without_idle) = run(&[0.75, -0.75]);
        assert_eq!(two.informative, 2);
        assert_eq!(
            with_idle, without_idle,
            "an episode the baseline found nothing to say about changed the step size"
        );
    }

    /// The outcome reports the raw feedback and the size of the population
    /// that contributed, separately.
    ///
    /// Fails if `mean_reward` is taken over the contributors rather than the
    /// round: the two below have the same contributing count but different
    /// round means.
    #[test]
    fn the_outcome_separates_what_the_feedback_said_from_what_contributed() {
        let w_full = [0.6_f32, 0.0, 0.4, 0.0];
        let base = test_gate();
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let z = base.clone().forward(&CTX).unwrap().to_vec();
        let first: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let second: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let third: Vec<f32> = z.iter().map(|zj| zj + 0.25).collect();
        let fourth: Vec<f32> = z.iter().map(|zj| zj - 0.25).collect();

        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&first, &w_full),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&second, &w_full),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&third, &w_full),
                reward: -1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&fourth, &w_full),
                reward: -1.0,
            },
        ];
        let outcome = trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();

        assert_eq!(outcome.samples, 4);
        assert_eq!(outcome.informative, 4);
        assert_eq!(outcome.mean_reward, 0.0);
        assert!(
            outcome.mean_loss.is_finite(),
            "loss {} is not a number",
            outcome.mean_loss
        );
    }
}
