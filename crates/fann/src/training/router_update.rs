//! Online refit of an adapter-selection gate from preference feedback.
//!
//! Each signed reward takes the same RLOO update path: positive feedback raises
//! the selected adapter's score and negative feedback lowers it. v1 protects
//! prior routing behavior with Fisher delta projection; `ewc_lambda` is
//! validated but intentionally inactive until the penalty-gradient path ships.
//! This module requires the `online-router` feature.
//! See `docs/lora-router.md` in this crate.

use std::collections::VecDeque;

use crate::{
    Network,
    training::{RlooConfig, RlooTrainer},
};

use crate::error::{FannError, FannResult};

// Re-export so callers can import DiagonalFisher from this module without
// a direct dependency on this crate's training module.
pub use crate::training::DiagonalFisher;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Polarity and strength of a preference signal from a completed request.
///
/// Explicit signals (thumbs / binary) carry full reward magnitude.
/// Implicit signals (dwell time, follow-up rate) carry half magnitude.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PreferenceSignal {
    /// Explicit positive: the selected adapter produced a useful response.
    Positive,
    /// Explicit negative: the selected adapter was unhelpful or wrong.
    Negative,
    /// Implicit positive: weak evidence of a helpful response (e.g., long dwell).
    ImplicitPositive,
    /// Implicit negative: weak evidence of an unhelpful response (e.g., immediate
    /// re-query).
    ImplicitNegative,
}

impl PreferenceSignal {
    /// Signed reward scalar forwarded to the policy-gradient step.
    ///
    /// Positive signals pull the target adapter's probability up; negative
    /// signals push it down.  The fann RLOO trainer handles both polarities
    /// through a single code path — do not special-case negative reward.
    #[inline]
    pub fn reward(&self) -> f32 {
        match self {
            Self::Positive => 1.0,
            Self::ImplicitPositive => 0.5,
            Self::ImplicitNegative => -0.5,
            Self::Negative => -1.0,
        }
    }

    /// Returns `true` for signals with positive reward (useful for replay
    /// buffer filtering: only preferred adapters belong in replay storage).
    #[inline]
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Positive | Self::ImplicitPositive)
    }
}

/// A single recorded preference signal from a completed request.
///
/// Callers supply the pre-computed context vector that was passed to `route()`
/// and the adapter index the feedback is about.  The `signal` field encodes
/// polarity and strength; a single code path handles both positive and negative
/// cases (see [`PreferenceSignal::reward`]).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeedbackEvent {
    /// Embedding of the request context; must match the gate's input dimension.
    ///
    /// Dimension is determined at gate construction time; validated at entry
    /// to [`update_router`] against `Network::num_inputs()`.
    pub context_vector: Vec<f32>,
    /// Index into the adapter pool that this feedback is about.
    ///
    /// For positive signals: the adapter that was correctly selected.
    /// For negative signals: the adapter that was incorrectly selected and
    /// should be pushed down.  Must be within `[0, gate.num_outputs())`.
    pub preferred_adapter_idx: usize,
    /// Opaque adapter identifier (mirrors `AdapterId = String` in the router).
    ///
    /// Stored for audit / tracing only; not used in gradient computation.
    pub adapter_id: String,
    /// Polarity and strength of the preference signal.
    pub signal: PreferenceSignal,
}

/// Hyperparameters for one `update_router` refit call.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RouterUpdateConfig {
    /// Learning rate applied inside [`RlooTrainer::step`].
    pub learning_rate: f32,
    /// Load-balancing auxiliary loss coefficient (prevents routing collapse).
    /// Set to `0.0` to disable.
    pub aux_loss_coeff: f32,
    /// Router z-loss coefficient (discourages logit explosion).
    /// Set to `0.0` to disable.
    pub z_loss_coeff: f32,
    /// Training epochs over the combined feedback + replay batch per call.
    pub epochs: usize,
    /// EWC++ Fisher regularisation strength.
    /// Reserved for the inactive anchor-penalty path; v1 uses Fisher projection instead.
    /// See [`docs/lora-router.md`](../../docs/lora-router.md#routerupdateconfigewc_lambda) for the phase boundary.
    pub ewc_lambda: f32,
    /// Fraction of the replay buffer to include in each refit epoch.
    ///
    /// `0.0` disables replay; `1.0` includes all buffered events.
    /// Must be finite and within `[0.0, 1.0]`; `update_router` rejects
    /// values outside that range with `FannError::Validation`.
    pub replay_mix_fraction: f32,
}

impl Default for RouterUpdateConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            aux_loss_coeff: 0.01,
            z_loss_coeff: 0.001,
            epochs: 5,
            ewc_lambda: 0.1,
            replay_mix_fraction: 0.5,
        }
    }
}

/// Result of one `update_router` call: the retrained gate ready to replace the
/// prior one.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RouterDelta {
    /// Serialised updated gate in FANN binary format.
    ///
    /// This is a complete network blob produced by `Network::to_bytes()`,
    /// not a parameter diff.  Load with `Network::from_bytes()` or pass as
    /// the `gate_bytes` argument to the next `update_router` call.
    pub network_bytes: Vec<u8>,
    /// Number of feedback events consumed in this refit.
    pub events_consumed: usize,
    /// Gate top-1 accuracy on the replay buffer after refit.
    ///
    /// Fraction of buffered entries where the gate's argmax equals the stored
    /// preferred adapter index.  `None` when the replay buffer is empty.
    pub replay_accuracy: Option<f32>,
}

/// Bounded FIFO of positive `(context_vector, adapter index)` feedback.
/// Refit replays its entries with reward `+1.0` and evicts the oldest at capacity.
/// See [`docs/lora-router.md`](../../docs/lora-router.md#replaybuffer) for replay-polarity rationale.
#[derive(Clone, Debug, Default)]
pub struct ReplayBuffer {
    inner: VecDeque<ReplayEntry>,
    max_size: usize,
}

impl ReplayBuffer {
    /// Create an empty replay buffer with the given capacity.
    ///
    /// `max_size` is the maximum number of entries retained; once full, the
    /// oldest entry is evicted on each new push.  A `max_size` of `0` accepts
    /// no entries.
    pub fn new(max_size: usize) -> Self {
        Self {
            inner: VecDeque::new(),
            max_size,
        }
    }

    /// Current number of stored entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when the buffer holds no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Add a new entry, evicting the oldest if the buffer is at capacity.
    ///
    /// Entries pushed beyond `max_size` silently replace the front (oldest)
    /// entry.  Pushing to a zero-capacity buffer is a no-op.
    fn push(&mut self, entry: ReplayEntry) {
        if self.max_size == 0 {
            return;
        }
        if self.inner.len() >= self.max_size {
            self.inner.pop_front();
        }
        self.inner.push_back(entry);
    }

    /// Iterate over all buffered entries in insertion order (oldest first).
    fn entries(&self) -> impl Iterator<Item = &ReplayEntry> {
        self.inner.iter()
    }

    /// Return up to `n` entries sampled uniformly across the buffer.
    ///
    /// Steps through the buffer at a uniform interval so the sample is drawn
    /// from across the full history rather than only the oldest `n` events.
    fn sample_n(&self, n: usize) -> Vec<&ReplayEntry> {
        let len = self.inner.len();
        if n == 0 || len == 0 {
            return Vec::new();
        }
        let n = n.min(len);
        // Step size: ceil(len / n) so we space out across the buffer.
        let step = len.div_ceil(n).max(1);
        self.inner.iter().step_by(step).take(n).collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Private storage type
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct ReplayEntry {
    context_vector: Vec<f32>,
    preferred_adapter_idx: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Core function
// ────────────────────────────────────────────────────────────────────────────

/// Refit a serialized adapter-selector gate from a non-empty feedback batch.
/// Updates `replay` and `fisher` in place, then returns complete replacement gate bytes.
/// Returns validation errors for invalid input/state and training errors for gate or Fisher failures.
/// See [`docs/lora-router.md`](../../docs/lora-router.md#update_router) for the refit algorithm and invariants.
pub fn update_router(
    gate_bytes: &[u8],
    events: &[FeedbackEvent],
    replay: &mut ReplayBuffer,
    fisher: &mut DiagonalFisher,
    config: &RouterUpdateConfig,
) -> FannResult<RouterDelta> {
    // ── 1. Guard: non-empty batch ─────────────────────────────────────────
    if events.is_empty() {
        return Err(FannError::Validation(
            "feedback events must not be empty; provide at least one event per refit call"
                .to_owned(),
        ));
    }

    // ── 2. Load gate from FANN binary ────────────────────────────────────
    let mut gate = Network::from_bytes(gate_bytes)
        .map_err(|e| FannError::TrainingError(format!("gate deserialisation failed: {e}")))?;

    let num_inputs = gate.num_inputs();
    let num_outputs = gate.num_outputs();
    let total_params = gate.total_params();

    // ── 3. Validate all events (fail-closed: reject the whole batch) ──────
    for (idx, ev) in events.iter().enumerate() {
        if ev.context_vector.len() != num_inputs {
            return Err(FannError::Validation(format!(
                "event {idx}: context_vector length {} != gate input dimension {num_inputs}",
                ev.context_vector.len()
            )));
        }
        if ev.preferred_adapter_idx >= num_outputs {
            return Err(FannError::Validation(format!(
                "event {idx}: preferred_adapter_idx {} >= gate output count {num_outputs}",
                ev.preferred_adapter_idx
            )));
        }
        // Reward values are derived from PreferenceSignal::reward(), which always
        // returns a finite f32; only the caller-supplied context vector needs this
        // non-finite guard.
        if ev.context_vector.iter().any(|&v| !v.is_finite()) {
            return Err(FannError::Validation(format!(
                "event {idx}: context_vector contains a non-finite value (NaN or Inf)"
            )));
        }
    }

    // ── 3.5. Validate config hyperparameters ─────────────────────────────
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(FannError::Validation(format!(
            "learning_rate must be finite and > 0, got {}",
            config.learning_rate
        )));
    }
    if !config.aux_loss_coeff.is_finite() || config.aux_loss_coeff < 0.0 {
        return Err(FannError::Validation(format!(
            "aux_loss_coeff must be finite and >= 0, got {}",
            config.aux_loss_coeff
        )));
    }
    if !config.z_loss_coeff.is_finite() || config.z_loss_coeff < 0.0 {
        return Err(FannError::Validation(format!(
            "z_loss_coeff must be finite and >= 0, got {}",
            config.z_loss_coeff
        )));
    }
    if !config.ewc_lambda.is_finite() || config.ewc_lambda < 0.0 {
        return Err(FannError::Validation(format!(
            "ewc_lambda must be finite and >= 0, got {}",
            config.ewc_lambda
        )));
    }
    if !config.replay_mix_fraction.is_finite()
        || config.replay_mix_fraction < 0.0
        || config.replay_mix_fraction > 1.0
    {
        return Err(FannError::Validation(format!(
            "replay_mix_fraction must be finite and in [0.0, 1.0], got {}",
            config.replay_mix_fraction
        )));
    }
    if config.epochs == 0 {
        return Err(FannError::Validation(
            "epochs must be > 0; a refit with zero training epochs has no effect".to_owned(),
        ));
    }
    // Fisher decay must be in the open interval (0, 1) on both the empty
    // auto-init path and the non-empty reuse path. Mirror DiagonalFisher::new.
    if !fisher.decay.is_finite() || fisher.decay <= 0.0 || fisher.decay >= 1.0 {
        return Err(FannError::Validation(format!(
            "DiagonalFisher decay must be finite and in the open interval (0, 1), got {}",
            fisher.decay
        )));
    }

    // ── 4. Initialise or validate DiagonalFisher ──────────────────────────
    // Auto-initialise when caller passes a default-constructed (empty) Fisher.
    if fisher.values.is_empty() {
        fisher.values = vec![0.0_f32; total_params];
        fisher.anchor = vec![0.0_f32; total_params];
        // `fisher.decay` remains as set by the caller.
    } else {
        // Non-empty Fisher: validate dimensions and numeric integrity before
        // any mutation so we fail closed before touching the training state.
        if fisher.values.len() != total_params {
            return Err(FannError::Validation(format!(
                "DiagonalFisher size {} does not match gate parameter count {total_params}",
                fisher.values.len()
            )));
        }
        // Fisher diagonal entries are squared-gradient EMAs — they must be
        // finite and non-negative by construction.
        if fisher.values.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(FannError::Validation(
                "DiagonalFisher values must all be finite and >= 0 \
                 (Fisher diagonal entries are squared-gradient EMAs)"
                    .to_owned(),
            ));
        }
        if fisher.anchor.len() != total_params {
            return Err(FannError::Validation(format!(
                "DiagonalFisher anchor length {} does not match gate parameter count \
                 {total_params}",
                fisher.anchor.len()
            )));
        }
        if fisher.anchor.iter().any(|&v| !v.is_finite()) {
            return Err(FannError::Validation(
                "DiagonalFisher anchor contains a non-finite value".to_owned(),
            ));
        }
    }

    // ── 5. Build RLOO trainer ─────────────────────────────────────────────
    let rloo_config = RlooConfig {
        learning_rate: config.learning_rate,
        aux_loss_coeff: config.aux_loss_coeff,
        z_loss_coeff: config.z_loss_coeff,
    };
    let mut trainer = RlooTrainer::new(rloo_config);

    // ── 6. Sample replay entries before mutating the buffer ───────────────
    let replay_count =
        (replay.len() as f32 * config.replay_mix_fraction.clamp(0.0, 1.0)).ceil() as usize;
    // Clone replay entries so the buffer is free to be updated below.
    let replay_entries: Vec<ReplayEntry> = replay
        .sample_n(replay_count)
        .into_iter()
        .map(|e| ReplayEntry {
            context_vector: e.context_vector.clone(),
            preferred_adapter_idx: e.preferred_adapter_idx,
        })
        .collect();

    // ── 7. Training loop ──────────────────────────────────────────────────
    for _ in 0..config.epochs {
        // New events — reward carries the original signal polarity.
        // Positive reward pulls the target adapter UP; negative pushes it DOWN.
        // One code path handles both; DO NOT special-case negative reward.
        for ev in events {
            one_gradient_step(
                &mut gate,
                &mut trainer,
                fisher,
                &ev.context_vector,
                ev.preferred_adapter_idx,
                ev.signal.reward(),
                config.learning_rate,
            )?;
        }

        // Replay entries — always positive (only preferred adapters are stored).
        for entry in &replay_entries {
            one_gradient_step(
                &mut gate,
                &mut trainer,
                fisher,
                &entry.context_vector,
                entry.preferred_adapter_idx,
                1.0_f32,
                config.learning_rate,
            )?;
        }
    }

    // ── 8. Fix Fisher anchor at post-refit parameters ─────────────────────
    let post_params = collect_params(&gate);
    fisher
        .set_anchor(&post_params)
        .map_err(|e| FannError::TrainingError(format!("fisher anchor update failed: {e}")))?;

    // ── 9. Add positive new events to the replay buffer ───────────────────
    for ev in events {
        if ev.signal.is_positive() {
            replay.push(ReplayEntry {
                context_vector: ev.context_vector.clone(),
                preferred_adapter_idx: ev.preferred_adapter_idx,
            });
        }
    }

    // ── 10. Compute accuracy on the updated replay buffer ─────────────────
    let replay_accuracy = compute_replay_accuracy(&mut gate, replay);

    // ── 11. Serialise updated gate ────────────────────────────────────────
    let network_bytes = gate.to_bytes();

    Ok(RouterDelta {
        network_bytes,
        events_consumed: events.len(),
        replay_accuracy,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Private helpers
// ────────────────────────────────────────────────────────────────────────────

/// Collect gate parameters as row-major weights then biases for each layer.
fn collect_params(gate: &Network) -> Vec<f32> {
    let mut params = Vec::with_capacity(gate.total_params());
    for layer in gate.layers() {
        params.extend_from_slice(layer.weights());
        params.extend_from_slice(layer.biases());
    }
    params
}

/// Restore parameters collected from this gate with [`collect_params`].
fn set_params(gate: &mut Network, params: &[f32]) {
    let mut offset = 0_usize;
    for layer in gate.layers_mut() {
        // Compute lengths from layer dimensions to avoid a double-borrow on `layer`.
        let nw = layer.num_inputs() * layer.num_outputs();
        let nb = layer.num_outputs();
        layer
            .weights_mut()
            .copy_from_slice(&params[offset..offset + nw]);
        offset += nw;
        layer
            .biases_mut()
            .copy_from_slice(&params[offset..offset + nb]);
        offset += nb;
    }
    debug_assert_eq!(
        offset,
        params.len(),
        "param vector length mismatch in set_params"
    );
}

/// Apply one RLOO step and project its parameter delta through the Fisher state.
fn one_gradient_step(
    gate: &mut Network,
    trainer: &mut RlooTrainer,
    fisher: &mut DiagonalFisher,
    context: &[f32],
    action_idx: usize,
    reward: f32,
    lr: f32,
) -> FannResult<()> {
    let before = collect_params(gate);

    // RLOO policy-gradient step: handles ±reward in one code path.
    trainer
        .step(gate, context, action_idx, reward)
        .map_err(|e| FannError::TrainingError(format!("policy-gradient step failed: {e}")))?;

    let after = collect_params(gate);

    // Raw delta: after - before = -lr * gradient (SGD update rule).
    // Approximate gradient = -(delta / lr).
    let raw_delta: Vec<f32> = after
        .iter()
        .zip(before.iter())
        .map(|(a, b)| a - b)
        .collect();

    // Update Fisher EMA with the approximate gradient.
    // `observe_gradient` squares internally: F_i ← decay·F_i + (1−decay)·g_i².
    let safe_lr = lr.abs().max(1e-10_f32);
    let approx_grad: Vec<f32> = raw_delta.iter().map(|&d| -d / safe_lr).collect();
    fisher.observe_gradient(&approx_grad).map_err(|e| {
        FannError::TrainingError(format!("Fisher gradient observation failed: {e}"))
    })?;

    // Preserve high-importance parameters through Fisher projection — see docs/lora-router.md.
    let mut projected_delta = raw_delta;
    fisher.project_delta(&mut projected_delta);

    // Apply projected delta: new_params = before + projected_delta.
    let projected: Vec<f32> = before
        .iter()
        .zip(projected_delta.iter())
        .map(|(b, d)| b + d)
        .collect();
    set_params(gate, &projected);

    Ok(())
}

/// Compute the gate's top-1 accuracy on the replay buffer.
///
/// For each buffered entry, the gate's `argmax` output is compared against
/// `preferred_adapter_idx`.  Returns `None` when the buffer is empty.
fn compute_replay_accuracy(gate: &mut Network, replay: &ReplayBuffer) -> Option<f32> {
    if replay.is_empty() {
        return None;
    }
    let total = replay.len();
    let correct = replay
        .entries()
        .filter(|entry| {
            gate.forward(&entry.context_vector)
                .ok()
                .and_then(|scores| {
                    scores
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                })
                .map(|top1| top1 == entry.preferred_adapter_idx)
                .unwrap_or(false)
        })
        .count();
    Some(correct as f32 / total as f32)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
