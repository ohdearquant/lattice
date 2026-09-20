//! REINFORCE with Leave-One-Out baseline (RLOO) policy-gradient trainer.
//!
//! Trains a selector gate from rewards over its raw output logits. Its output
//! layer must use `Activation::Linear`; this module applies softmax itself.
//! EWC integration remains an independent call-site concern.
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
        self.step_impl(gate, context, action_idx, reward, None)
    }

    /// Applies one REINFORCE update whose load-balance term runs on the learned
    /// mixture's projected weight vector instead of this gate's own selection
    /// distribution (ADR-091 Decision 3, amended 2026-09-20).
    ///
    /// `w_full` is `softmax(selected logits / tau)` embedded into this gate's
    /// full output space — `w_full[selected[i]] = w[i]`, zero elsewhere — taken
    /// **before** the `epsilon` floor's drop and renormalisation: an adapter
    /// the floor is about to drop must still be present and carry gradient
    /// when this runs, which is exactly the event a post-floor vector would
    /// already have zeroed out. `tau` is the temperature that produced it; the
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
            action_idx,
            reward,
            Some(WeightedAux {
                w_full,
                inv_tau: 1.0 / tau,
            }),
        )
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
        action_idx: usize,
        reward: f32,
        weighted: Option<WeightedAux<'_>>,
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

        if action_idx >= num_outputs {
            return Err(FannError::InputSizeMismatch {
                expected: num_outputs,
                actual: action_idx.saturating_add(1),
            });
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

        // Forward pass (populates activation buffers).
        let logits: Vec<f32> = gate.forward(context)?.to_vec();

        let probs = softmax(&logits);

        let k = num_outputs;
        // Both guard terms are applied by calling the functions that define them,
        // never by re-deriving their gradients here: a second copy is a place for
        // the two to disagree silently, and it is what let the documented guard
        // sit uncalled while the trainer optimised something else.
        let route_freqs = self.route_freqs_or_uniform(k);

        // The load-balance term runs on this gate's own selection distribution
        // by default, or on the learned mixture's projected weight vector at
        // `1 / tau` when `weighted` is supplied (ADR-091 Decision 3, amended
        // 2026-09-20). `aux_scale = 1.0` in the default case is exact for
        // every finite `f32` (`x * 1.0 == x` bit-for-bit), so this branch
        // changes no bit of `step`'s existing output.
        let (aux_source, aux_scale): (&[f32], f32) = match &weighted {
            Some(w) => (w.w_full, w.inv_tau),
            None => (probs.as_slice(), 1.0),
        };
        let aux_grad = load_balance_aux_gradient(&route_freqs, aux_source)?;
        let z_grad = router_z_gradient(&logits, &probs);

        // Linear output makes this the pre-activation error; preserve reward polarity — see docs/training.md.
        let output_deltas: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(j, &pj)| {
                let onehot_j = if j == action_idx { 1.0_f32 } else { 0.0_f32 };
                let policy = reward * (pj - onehot_j);
                let aux = self.config.aux_loss_coeff * aux_scale * aux_grad[j];
                let zloss = self.config.z_loss_coeff * z_grad[j];
                policy + aux + zloss
            })
            .collect();

        self.backprop_and_apply(gate, context, &output_deltas, num_layers)?;

        // Fold the decision actually taken into the frequency EMA, after the
        // update, so the gradient above saw the traffic that preceded it. The
        // learned path folds the same projected vector the aux term above
        // just used, so `f` tracks weighted traffic rather than selection
        // counts — the comparison a weight-collapse guard needs.
        let mass: Vec<f32> = match &weighted {
            Some(w) => w.w_full.to_vec(),
            None => {
                let mut m = vec![0.0_f32; k];
                m[action_idx] = 1.0;
                m
            }
        };
        self.observe_routing(&mass);

        // Scalar policy loss for caller logging.
        let loss = -reward * probs[action_idx].max(1e-9).ln();
        Ok(loss)
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
        // Shared borrows end before the mutable update phase.
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

        // --- Phase 2: apply plain SGD (mirror backprop.rs:170-191, no momentum) ---
        let lr = self.config.learning_rate;
        for layer_idx in 0..num_layers {
            let Some(layer) = gate.layer_mut(layer_idx) else {
                continue;
            };

            let weights = layer.weights_mut();
            for (w, &g) in weights.iter_mut().zip(weight_grads[layer_idx].iter()) {
                *w -= lr * g;
            }

            let biases = layer.biases_mut();
            for (b, &g) in biases.iter_mut().zip(bias_grads[layer_idx].iter()) {
                *b -= lr * g;
            }
        }

        Ok(())
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

    /// `w_full` of the wrong width is a caller error, not a shape to silently
    /// paper over.
    #[test]
    fn step_with_learned_weights_rejects_wrong_length_w_full() {
        let mut gate = test_gate();
        let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
        let wrong_w_full = [0.5_f32, 0.5]; // gate has 4 outputs
        let err = trainer
            .step_with_learned_weights(&mut gate, &CTX, 0, 0.0, &wrong_w_full, 1.0)
            .unwrap_err();
        assert!(
            matches!(err, FannError::InputSizeMismatch { .. }),
            "expected InputSizeMismatch, got {err:?}"
        );
    }

    /// `tau` must be finite and strictly positive; this is the arithmetic
    /// precondition only — the collapse floor on `tau` is guard (a)'s job in
    /// the router, not reimplemented here.
    #[test]
    fn step_with_learned_weights_rejects_non_finite_or_nonpositive_tau() {
        let w_full = [0.25_f32; 4];
        for bad_tau in [0.0_f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut gate = test_gate();
            let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
            let err = trainer
                .step_with_learned_weights(&mut gate, &CTX, 0, 0.0, &w_full, bad_tau)
                .unwrap_err();
            assert!(
                matches!(err, FannError::TrainingError(_)),
                "tau={bad_tau} must be rejected as TrainingError, got {err:?}"
            );
        }
    }
}
