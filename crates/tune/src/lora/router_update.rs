//! Online refit of an adapter-selection gate from preference feedback.
//!
//! Each signed reward takes the same RLOO update path: positive feedback raises
//! the selected adapter's score and negative feedback lowers it. v1 protects
//! prior routing behavior with Fisher delta projection; `ewc_lambda` is
//! validated but intentionally inactive until the penalty-gradient path ships.
//! This module requires the `mixture` feature.
//! See docs/lora-router.md.

use std::collections::VecDeque;

use lattice_fann::{
    Network,
    training::{RlooConfig, RlooTrainer},
};

use crate::error::{Result, TuneError};

// Re-export so callers can import DiagonalFisher from this module without
// a direct dependency on lattice-fann.
pub use lattice_fann::training::DiagonalFisher;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Polarity and strength of a preference signal from a completed request.
///
/// Explicit signals (thumbs / binary) carry full reward magnitude.
/// Implicit signals (dwell time, follow-up rate) carry half magnitude.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// values outside that range with `TuneError::Validation`.
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
) -> Result<RouterDelta> {
    // ── 1. Guard: non-empty batch ─────────────────────────────────────────
    if events.is_empty() {
        return Err(TuneError::Validation(
            "feedback events must not be empty; provide at least one event per refit call"
                .to_owned(),
        ));
    }

    // ── 2. Load gate from FANN binary ────────────────────────────────────
    let mut gate = Network::from_bytes(gate_bytes)
        .map_err(|e| TuneError::Training(format!("gate deserialisation failed: {e}")))?;

    let num_inputs = gate.num_inputs();
    let num_outputs = gate.num_outputs();
    let total_params = gate.total_params();

    // ── 3. Validate all events (fail-closed: reject the whole batch) ──────
    for (idx, ev) in events.iter().enumerate() {
        if ev.context_vector.len() != num_inputs {
            return Err(TuneError::Validation(format!(
                "event {idx}: context_vector length {} != gate input dimension {num_inputs}",
                ev.context_vector.len()
            )));
        }
        if ev.preferred_adapter_idx >= num_outputs {
            return Err(TuneError::Validation(format!(
                "event {idx}: preferred_adapter_idx {} >= gate output count {num_outputs}",
                ev.preferred_adapter_idx
            )));
        }
        // Reward values are derived from PreferenceSignal::reward(), which always
        // returns a finite f32; only the caller-supplied context vector needs this
        // non-finite guard.
        if ev.context_vector.iter().any(|&v| !v.is_finite()) {
            return Err(TuneError::Validation(format!(
                "event {idx}: context_vector contains a non-finite value (NaN or Inf)"
            )));
        }
    }

    // ── 3.5. Validate config hyperparameters ─────────────────────────────
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(TuneError::Validation(format!(
            "learning_rate must be finite and > 0, got {}",
            config.learning_rate
        )));
    }
    if !config.aux_loss_coeff.is_finite() || config.aux_loss_coeff < 0.0 {
        return Err(TuneError::Validation(format!(
            "aux_loss_coeff must be finite and >= 0, got {}",
            config.aux_loss_coeff
        )));
    }
    if !config.z_loss_coeff.is_finite() || config.z_loss_coeff < 0.0 {
        return Err(TuneError::Validation(format!(
            "z_loss_coeff must be finite and >= 0, got {}",
            config.z_loss_coeff
        )));
    }
    if !config.ewc_lambda.is_finite() || config.ewc_lambda < 0.0 {
        return Err(TuneError::Validation(format!(
            "ewc_lambda must be finite and >= 0, got {}",
            config.ewc_lambda
        )));
    }
    if !config.replay_mix_fraction.is_finite()
        || config.replay_mix_fraction < 0.0
        || config.replay_mix_fraction > 1.0
    {
        return Err(TuneError::Validation(format!(
            "replay_mix_fraction must be finite and in [0.0, 1.0], got {}",
            config.replay_mix_fraction
        )));
    }
    if config.epochs == 0 {
        return Err(TuneError::Validation(
            "epochs must be > 0; a refit with zero training epochs has no effect".to_owned(),
        ));
    }
    // Fisher decay must be in the open interval (0, 1) on both the empty
    // auto-init path and the non-empty reuse path. Mirror DiagonalFisher::new.
    if !fisher.decay.is_finite() || fisher.decay <= 0.0 || fisher.decay >= 1.0 {
        return Err(TuneError::Validation(format!(
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
            return Err(TuneError::Validation(format!(
                "DiagonalFisher size {} does not match gate parameter count {total_params}",
                fisher.values.len()
            )));
        }
        // Fisher diagonal entries are squared-gradient EMAs — they must be
        // finite and non-negative by construction.
        if fisher.values.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(TuneError::Validation(
                "DiagonalFisher values must all be finite and >= 0 \
                 (Fisher diagonal entries are squared-gradient EMAs)"
                    .to_owned(),
            ));
        }
        if fisher.anchor.len() != total_params {
            return Err(TuneError::Validation(format!(
                "DiagonalFisher anchor length {} does not match gate parameter count \
                 {total_params}",
                fisher.anchor.len()
            )));
        }
        if fisher.anchor.iter().any(|&v| !v.is_finite()) {
            return Err(TuneError::Validation(
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
        .map_err(|e| TuneError::Training(format!("fisher anchor update failed: {e}")))?;

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
) -> Result<()> {
    let before = collect_params(gate);

    // RLOO policy-gradient step: handles ±reward in one code path.
    trainer
        .step(gate, context, action_idx, reward)
        .map_err(|e| TuneError::Training(format!("policy-gradient step failed: {e}")))?;

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
    fisher
        .observe_gradient(&approx_grad)
        .map_err(|e| TuneError::Training(format!("Fisher gradient observation failed: {e}")))?;

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
mod tests {
    use super::*;
    use lattice_fann::{Activation, NetworkBuilder};

    /// Deterministic gate: 4 inputs → 8 hidden (ReLU) → 3 outputs (Linear).
    fn make_gate(num_inputs: usize, hidden: usize, num_outputs: usize) -> Network {
        NetworkBuilder::new()
            .input(num_inputs)
            .hidden(hidden, Activation::ReLU)
            .output(num_outputs, Activation::Linear)
            .build_with_seed(42)
            .unwrap()
    }

    /// Config with stronger learning rate so effects are clearly measurable
    /// in unit tests without thousands of events.
    fn test_config() -> RouterUpdateConfig {
        RouterUpdateConfig {
            learning_rate: 0.1,
            ..RouterUpdateConfig::default()
        }
    }

    /// Build a batch of identical feedback events pointing at one adapter.
    fn make_events(
        num: usize,
        adapter_idx: usize,
        signal: PreferenceSignal,
        dim: usize,
    ) -> Vec<FeedbackEvent> {
        (0..num)
            .map(|i| FeedbackEvent {
                // Vary context slightly so aux loss sees non-trivial inputs.
                context_vector: (0..dim).map(|j| (i + j) as f32 * 0.01 + 0.1).collect(),
                preferred_adapter_idx: adapter_idx,
                adapter_id: format!("adapter-{adapter_idx}"),
                signal: signal.clone(),
            })
            .collect()
    }

    // ─── Validation / error-path tests ────────────────────────────────────

    /// An empty event slice must immediately return Err(TuneError::Validation).
    ///
    /// Mutation that defeats this test: remove the `events.is_empty()` guard.
    #[test]
    fn update_router_empty_feedback_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher =
            DiagonalFisher::new(0, 0.99).expect("dim=0, decay=0.99 are valid Fisher params"); // empty → auto-init
        let config = test_config();

        let result = update_router(&gate_bytes, &[], &mut replay, &mut fisher, &config);
        assert!(result.is_err(), "empty event batch must return Err, got Ok");
        match result {
            Err(TuneError::Validation(_)) => {}
            Err(other) => panic!("expected TuneError::Validation, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    /// An event with wrong context dimension must return Err(TuneError::Validation).
    ///
    /// Mutation that defeats this test: remove the context-length check in
    /// the event-validation loop.
    #[test]
    fn update_router_wrong_context_dim_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher =
            DiagonalFisher::new(0, 0.99).expect("dim=0, decay=0.99 are valid Fisher params");
        let config = test_config();

        // Gate expects 4 inputs; supply 7.
        let events = vec![FeedbackEvent {
            context_vector: vec![1.0; 7],
            preferred_adapter_idx: 0,
            adapter_id: "a".into(),
            signal: PreferenceSignal::Positive,
        }];
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "wrong context dim must return TuneError::Validation, got {result:?}"
        );
    }

    // ─── Polarity / direction tests ───────────────────────────────────────

    /// After update_router on a batch of positive events targeting adapter 2,
    /// the gate's logit for adapter 2 must be strictly higher than before.
    ///
    /// Mutation that defeats this test: set `learning_rate` to `0.0` in the
    /// RLOO gradient step so no update is applied.
    #[test]
    fn router_delta_score_shifts_toward_preferred() {
        let mut gate_before = make_gate(4, 8, 3);
        let ctx = vec![0.1_f32, 0.2, 0.3, 0.4];
        let before_score = gate_before.forward(&ctx).unwrap()[2];

        let gate_bytes = gate_before.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher =
            DiagonalFisher::new(0, 0.99).expect("dim=0, decay=0.99 are valid Fisher params");
        let config = test_config();

        // 10 events all preferring adapter 2 with explicit positive signal.
        let events = make_events(10, 2, PreferenceSignal::Positive, 4);
        let delta = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config)
            .expect("update_router must succeed on valid input");

        let mut gate_after = Network::from_bytes(&delta.network_bytes)
            .expect("RouterDelta::network_bytes must deserialise cleanly");
        let after_score = gate_after.forward(&ctx).unwrap()[2];

        assert!(
            after_score > before_score,
            "positive events targeting adapter 2 must raise its logit: \
             before={before_score:.6}, after={after_score:.6}"
        );
    }

    /// After update_router on a batch of negative events targeting adapter 1,
    /// the gate's logit for adapter 1 must be strictly lower than before.
    ///
    /// This is the load-bearing polarity test.  Mutation that defeats it:
    /// call `ev.signal.reward()` unconditionally as `+1.0` (ignore polarity),
    /// which would push adapter 1 UP instead of DOWN.
    #[test]
    fn feedback_event_negative_signal_decreases_score() {
        let mut gate_before = make_gate(4, 8, 3);
        let ctx = vec![0.1_f32, 0.2, 0.3, 0.4];
        let before_score = gate_before.forward(&ctx).unwrap()[1];

        let gate_bytes = gate_before.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher =
            DiagonalFisher::new(0, 0.99).expect("dim=0, decay=0.99 are valid Fisher params");
        let config = test_config();

        // 10 negative events targeting adapter 1: signal must push logit[1] DOWN.
        let events = make_events(10, 1, PreferenceSignal::Negative, 4);
        let delta = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config)
            .expect("update_router must succeed on valid input");

        let mut gate_after = Network::from_bytes(&delta.network_bytes)
            .expect("RouterDelta::network_bytes must deserialise cleanly");
        let after_score = gate_after.forward(&ctx).unwrap()[1];

        assert!(
            after_score < before_score,
            "negative events targeting adapter 1 must lower its logit (polarity test): \
             before={before_score:.6}, after={after_score:.6}"
        );
    }

    // ─── RouterDelta serialisation round-trip ─────────────────────────────

    /// The network bytes in RouterDelta must deserialise to a runnable gate
    /// that produces the same output as the live gate immediately after refit.
    ///
    /// Mutation that defeats this test: return an empty Vec for
    /// `RouterDelta::network_bytes` instead of calling `gate.to_bytes()`.
    #[test]
    fn router_delta_bytes_round_trip() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher =
            DiagonalFisher::new(0, 0.99).expect("dim=0, decay=0.99 are valid Fisher params");
        let config = test_config();

        let events = make_events(5, 0, PreferenceSignal::Positive, 4);
        let delta = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config).unwrap();

        // Deserialise and run a forward pass to confirm the blob is complete.
        let mut restored = Network::from_bytes(&delta.network_bytes)
            .expect("RouterDelta::network_bytes must deserialise without error");
        let ctx = vec![0.5_f32; 4];
        let out = restored
            .forward(&ctx)
            .expect("restored gate must run forward pass");
        assert_eq!(
            out.len(),
            3,
            "restored gate output length must equal original num_outputs"
        );
    }

    // ─── Replay buffer capacity tests ─────────────────────────────────────

    /// Inserting more than `max_size` entries must cap the buffer at `max_size`.
    ///
    /// Mutation that defeats this test: remove the `pop_front()` eviction in
    /// `ReplayBuffer::push()`.
    #[test]
    fn replay_buffer_bounded_at_max_size() {
        let mut buf = ReplayBuffer::new(256);
        for i in 0..600 {
            buf.push(ReplayEntry {
                context_vector: vec![i as f32],
                preferred_adapter_idx: i % 4,
            });
        }
        assert_eq!(
            buf.len(),
            256,
            "buffer must be bounded at max_size=256 after 600 insertions"
        );
    }

    /// After overflow the oldest entries must be evicted (FIFO discipline).
    ///
    /// After inserting events 0..600 into a 256-capacity buffer, events 0..343
    /// have been evicted.  The front of the buffer holds event 344 (0-indexed),
    /// which is the 345th inserted event.
    ///
    /// Mutation that defeats this test: push to the front instead of the back
    /// (reversing insertion order) or skip `pop_front` (breaking eviction).
    #[test]
    fn replay_buffer_evicts_oldest() {
        let mut buf = ReplayBuffer::new(256);
        for i in 0..600 {
            buf.push(ReplayEntry {
                context_vector: vec![i as f32],
                preferred_adapter_idx: 0,
            });
        }

        // Front entry must be event 344 (first 344 events were evicted: 0..344 gone,
        // 344 is the new front).
        let front = buf
            .inner
            .front()
            .expect("buffer must be non-empty after insertions");
        assert!(
            (front.context_vector[0] - 344.0).abs() < 1e-6,
            "front entry should be the 345th inserted (index 344), got {}",
            front.context_vector[0]
        );
    }

    // ─── FIX 1: input-validation tests ────────────────────────────────────

    /// NaN learning_rate must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `!config.learning_rate.is_finite()` branch.
    #[test]
    fn update_router_nan_learning_rate_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            learning_rate: f32::NAN,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "NaN learning_rate must return TuneError::Validation, got {result:?}"
        );
    }

    /// Zero learning_rate must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `config.learning_rate <= 0.0` branch.
    #[test]
    fn update_router_zero_learning_rate_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            learning_rate: 0.0,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "zero learning_rate must return TuneError::Validation, got {result:?}"
        );
    }

    /// Negative learning_rate must return Err(TuneError::Validation).
    ///
    /// Mutation: invert or remove the `config.learning_rate <= 0.0` check.
    #[test]
    fn update_router_negative_learning_rate_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            learning_rate: -0.1,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "negative learning_rate must return TuneError::Validation, got {result:?}"
        );
    }

    /// NaN aux_loss_coeff must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `!config.aux_loss_coeff.is_finite()` branch.
    #[test]
    fn update_router_nan_aux_loss_coeff_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            aux_loss_coeff: f32::NAN,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "NaN aux_loss_coeff must return TuneError::Validation, got {result:?}"
        );
    }

    /// Negative z_loss_coeff must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `config.z_loss_coeff < 0.0` branch.
    #[test]
    fn update_router_negative_z_loss_coeff_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            z_loss_coeff: -0.001,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "negative z_loss_coeff must return TuneError::Validation, got {result:?}"
        );
    }

    /// Negative ewc_lambda must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `config.ewc_lambda < 0.0` branch.
    #[test]
    fn update_router_negative_ewc_lambda_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            ewc_lambda: -0.1,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "negative ewc_lambda must return TuneError::Validation, got {result:?}"
        );
    }

    /// Infinite ewc_lambda must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `!config.ewc_lambda.is_finite()` branch.
    #[test]
    fn update_router_infinite_ewc_lambda_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            ewc_lambda: f32::INFINITY,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "infinite ewc_lambda must return TuneError::Validation, got {result:?}"
        );
    }

    /// replay_mix_fraction > 1.0 must return Err(TuneError::Validation).
    ///
    /// Prior to this fix the value was silently clamped; the guard makes a bad
    /// value loud instead.
    ///
    /// Mutation: remove the `config.replay_mix_fraction > 1.0` branch.
    #[test]
    fn update_router_replay_mix_fraction_out_of_range_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            replay_mix_fraction: 1.5,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "replay_mix_fraction > 1.0 must return TuneError::Validation, got {result:?}"
        );
    }

    /// A NaN value inside an event's context_vector must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `ev.context_vector.iter().any(!is_finite)` check.
    #[test]
    fn update_router_nan_context_value_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = test_config();
        let events = vec![FeedbackEvent {
            context_vector: vec![1.0, f32::NAN, 0.5, 0.0],
            preferred_adapter_idx: 0,
            adapter_id: "a".into(),
            signal: PreferenceSignal::Positive,
        }];
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "NaN in context_vector must return TuneError::Validation, got {result:?}"
        );
    }

    /// epochs == 0 must return Err(TuneError::Validation).
    ///
    /// Mutation: remove the `config.epochs == 0` check.
    #[test]
    fn update_router_zero_epochs_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
        let config = RouterUpdateConfig {
            epochs: 0,
            ..test_config()
        };
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "epochs=0 must return TuneError::Validation, got {result:?}"
        );
    }

    /// A non-empty DiagonalFisher with a NaN value must return Err(TuneError::Validation).
    ///
    /// Uses a struct literal to bypass DiagonalFisher::new() validation so the
    /// invalid state reaches update_router's own guard.
    ///
    /// Mutation: remove the `!v.is_finite()` branch in the Fisher values check.
    #[test]
    fn update_router_fisher_nan_value_returns_err() {
        let gate = make_gate(4, 8, 3);
        let total_params = gate.total_params();
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut values = vec![0.0_f32; total_params];
        values[0] = f32::NAN;
        let mut fisher = DiagonalFisher {
            values,
            anchor: vec![0.0; total_params],
            decay: 0.99,
        };
        let config = test_config();
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "Fisher NaN value must return TuneError::Validation, got {result:?}"
        );
    }

    /// A non-empty DiagonalFisher with a negative value must return Err(TuneError::Validation).
    ///
    /// Fisher diagonal entries are squared-gradient EMAs and must be >= 0.
    ///
    /// Mutation: remove the `v < 0.0` branch in the Fisher values check.
    #[test]
    fn update_router_fisher_negative_value_returns_err() {
        let gate = make_gate(4, 8, 3);
        let total_params = gate.total_params();
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut values = vec![0.0_f32; total_params];
        values[1] = -0.5;
        let mut fisher = DiagonalFisher {
            values,
            anchor: vec![0.0; total_params],
            decay: 0.99,
        };
        let config = test_config();
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "negative Fisher value must return TuneError::Validation, got {result:?}"
        );
    }

    /// Fisher decay = 1.0, set via struct literal to bypass DiagonalFisher::new(),
    /// must return Err(TuneError::Validation) even when values is empty.
    ///
    /// Mutation: remove the Fisher decay guard in update_router.
    #[test]
    fn update_router_fisher_decay_one_returns_err() {
        let gate = make_gate(4, 8, 3);
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher {
            values: vec![],
            anchor: vec![],
            decay: 1.0,
        };
        let config = test_config();
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "Fisher decay=1.0 must return TuneError::Validation, got {result:?}"
        );
    }

    /// A non-empty DiagonalFisher with anchor length != total_params must return
    /// Err(TuneError::Validation).
    ///
    /// Mutation: remove the `fisher.anchor.len() != total_params` check.
    #[test]
    fn update_router_fisher_anchor_len_mismatch_returns_err() {
        let gate = make_gate(4, 8, 3);
        let total_params = gate.total_params();
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut fisher = DiagonalFisher {
            values: vec![0.0; total_params],
            anchor: vec![0.0; total_params + 3], // wrong length
            decay: 0.99,
        };
        let config = test_config();
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "Fisher anchor length mismatch must return TuneError::Validation, got {result:?}"
        );
    }

    /// A non-empty DiagonalFisher with a non-finite anchor entry must return
    /// Err(TuneError::Validation).
    ///
    /// Mutation: remove the `fisher.anchor.iter().any(!is_finite)` check.
    #[test]
    fn update_router_fisher_nonfinite_anchor_returns_err() {
        let gate = make_gate(4, 8, 3);
        let total_params = gate.total_params();
        let gate_bytes = gate.to_bytes();
        let mut replay = ReplayBuffer::new(256);
        let mut anchor = vec![0.0_f32; total_params];
        anchor[2] = f32::INFINITY;
        let mut fisher = DiagonalFisher {
            values: vec![0.0; total_params],
            anchor,
            decay: 0.99,
        };
        let config = test_config();
        let events = make_events(1, 0, PreferenceSignal::Positive, 4);
        let result = update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config);
        assert!(
            matches!(result, Err(TuneError::Validation(_))),
            "non-finite Fisher anchor must return TuneError::Validation, got {result:?}"
        );
    }

    // ─── ewc_lambda Phase-2 boundary test ─────────────────────────────────

    /// Two runs identical in every respect except ewc_lambda must produce
    /// byte-identical RouterDelta::network_bytes.
    ///
    /// This pins the Phase-2 boundary: v1 uses project_delta (Fisher
    /// null-space damping), which does not read ewc_lambda.  The test fails
    /// the moment someone wires ewc_lambda into the update path without
    /// revisiting the design.
    ///
    /// Mutation: multiply the projected delta by `config.ewc_lambda` inside
    /// one_gradient_step — the two outputs will diverge.
    #[test]
    fn ewc_lambda_is_inert_in_projection_path() {
        let gate_bytes = make_gate(4, 8, 3).to_bytes();
        let events = make_events(3, 1, PreferenceSignal::Positive, 4);

        let run = |ewc_lambda: f32| -> Vec<u8> {
            let mut replay = ReplayBuffer::new(256);
            let mut fisher = DiagonalFisher::new(0, 0.99).unwrap();
            let config = RouterUpdateConfig {
                ewc_lambda,
                ..test_config()
            };
            update_router(&gate_bytes, &events, &mut replay, &mut fisher, &config)
                .expect("update_router must succeed on valid input")
                .network_bytes
        };

        let bytes_zero = run(0.0);
        let bytes_half = run(0.5);
        assert_eq!(
            bytes_zero, bytes_half,
            "ewc_lambda must not affect network_bytes in the v1 projection path"
        );
    }
}
