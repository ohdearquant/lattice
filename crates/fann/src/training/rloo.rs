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

/// Single-sample policy-gradient trainer for a selector gate.
///
/// The gate is a `Network` whose output layer activation is `Linear` (it
/// returns raw logits; softmax is applied here in the loss, not in the
/// network).
///
/// EWC forgetting-guard integration is intentionally outside this struct —
/// compose `DiagonalFisher` at the call site to apply penalty gradients on
/// top of the policy-gradient delta independently.
pub struct RlooTrainer {
    config: RlooConfig,
    /// RNG used for Gumbel-max sampling in the Phase-2 multi-sample path.
    rng: rand::rngs::SmallRng,
}

impl RlooTrainer {
    /// Create a new trainer, seeding the RNG from system entropy.
    pub fn new(config: RlooConfig) -> Self {
        Self {
            config,
            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }

    /// Create a new trainer with a fixed RNG seed for deterministic behaviour.
    ///
    /// Prefer this constructor in tests to guarantee reproducible results.
    pub fn with_seed(config: RlooConfig, seed: u64) -> Self {
        Self {
            config,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
        }
    }

    /// Phase-1 single-sample REINFORCE step (M=1), the active path.
    ///
    /// `action_idx` is the adapter index the reward is about:
    ///   - positive reward  → `action_idx` = the preferred adapter
    ///   - negative reward  → `action_idx` = the selected adapter (push it down)
    ///
    /// `reward` carries polarity AND strength: `+1.0` / `−1.0` explicit,
    /// `+0.5` / `−0.5` implicit. There is exactly one code path for ±reward —
    /// the sign is embedded in the gradient formula; no special-casing for
    /// negative reward.
    ///
    /// Returns the scalar policy loss `−reward · log p[action_idx]` for logging.
    pub fn step(
        &mut self,
        gate: &mut Network,
        context: &[f32],
        action_idx: usize,
        reward: f32,
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
        let lse = log_sum_exp(&logits);

        let k = num_outputs;
        let sum_p_sq: f32 = probs.iter().map(|&p| p * p).sum();
        // Scalar used in the load-balance gradient (computed once).
        let c = sum_p_sq - 1.0 / k as f32;

        // Output-layer error vector g (length K).
        //
        // Because the output activation is Linear (derivative = 1), g is the
        // pre-activation error directly — no additional derivative multiply.
        //
        // g[j] = reward * (p[j] − onehot(action)[j])          (1) policy term
        //      + aux_coeff * (2/K) * p[j] * (p[j] − 1/K − c) (2) load-balance
        //      + z_coeff   * 2 * LSE * p[j]                   (3) z-loss
        //
        // For reward > 0 the policy term is the CE gradient pulling q(action) UP.
        // For reward < 0 the sign flips and pushes q(action) DOWN.
        // One code path handles both — DO NOT special-case negative reward.
        let output_deltas: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(j, &pj)| {
                let onehot_j = if j == action_idx { 1.0_f32 } else { 0.0_f32 };
                let policy = reward * (pj - onehot_j);
                let aux =
                    self.config.aux_loss_coeff * (2.0 / k as f32) * pj * (pj - 1.0 / k as f32 - c);
                let zloss = self.config.z_loss_coeff * 2.0 * lse * pj;
                policy + aux + zloss
            })
            .collect();

        self.backprop_and_apply(gate, context, &output_deltas, num_layers)?;

        // Scalar policy loss for caller logging.
        let loss = -reward * probs[action_idx].max(1e-9).ln();
        Ok(loss)
    }

    /// Phase-2 full RLOO multi-sample step (M > 1).
    ///
    /// Draws `m_samples` k-subsets via Gumbel-max sampling and uses a
    /// leave-one-out baseline to reduce gradient variance. Returns the mean
    /// reward across samples.
    ///
    /// This is not the default path — activate via the bench harness when
    /// comparing against Phase-1.
    ///
    /// Invariant for any future activation: this path consumes only
    /// preferred-known (positive) events, so it MUST be paired with the M=1
    /// [`Self::step`] negative path; never run it positive-only. The
    /// convergence bench's positive-only arm collapsed to chance (policy
    /// entropy toward zero, mass on a single output) because dropping
    /// negative feedback removes the signal that pushes a wrongly-selected
    /// output down. Carrying both polarities is a correctness requirement,
    /// not a tuning choice.
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

        // Reject caller-supplied sizes that would cause Vec::with_capacity to OOM.
        // m_samples sizes the rewards/subsets outer vecs directly; the retained
        // subsets additionally hold m_samples * k indices in aggregate. k alone is
        // range-checked, but the product can still overflow usize or exceed the
        // allocation cap, so bound the product before any allocation runs.
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
        let lse = log_sum_exp(&logits);

        // Gumbel-max sampling: draw m_samples k-subsets.
        let mut rewards: Vec<f32> = Vec::with_capacity(m_samples);
        let mut subsets: Vec<Vec<usize>> = Vec::with_capacity(m_samples);

        for _ in 0..m_samples {
            // Perturb each logit with independent Gumbel(0,1) noise:
            //   g_i = s_i + (−ln(−ln(u_i))),  u_i ~ Uniform(0,1) open interval.
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
        let sum_p_sq: f32 = probs.iter().map(|&p| p * p).sum();
        let c = sum_p_sq - 1.0 / ko as f32;

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

        // Aux and z-loss terms (identical to step).
        for (j, &pj) in probs.iter().enumerate() {
            let aux =
                self.config.aux_loss_coeff * (2.0 / ko as f32) * pj * (pj - 1.0 / ko as f32 - c);
            let zloss = self.config.z_loss_coeff * 2.0 * lse * pj;
            output_deltas[j] += aux + zloss;
        }

        self.backprop_and_apply(gate, context, &output_deltas, num_layers)?;

        Ok(reward_sum / m_samples as f32)
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
        // --- Phase 1: compute all layer deltas and accumulate gradients ---
        // Both gate.layers() and gate.activations() are &self borrows and may
        // coexist simultaneously; they are released before the mutable apply phase.
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

/// Load-balance auxiliary loss: `(1/K) Σ_i (p_i − 1/K)²`.
///
/// Penalises routing collapse where one adapter receives most probability mass.
/// `probs` should be the softmax of the gate logits.
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
}
