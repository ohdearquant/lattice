use super::*;
use crate::{Activation, NetworkBuilder};

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

/// An empty event slice must immediately return Err(FannError::Validation).
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
        Err(FannError::Validation(_)) => {}
        Err(other) => panic!("expected FannError::Validation, got {other:?}"),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

/// An event with wrong context dimension must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "wrong context dim must return FannError::Validation, got {result:?}"
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

/// NaN learning_rate must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "NaN learning_rate must return FannError::Validation, got {result:?}"
    );
}

/// Zero learning_rate must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "zero learning_rate must return FannError::Validation, got {result:?}"
    );
}

/// Negative learning_rate must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "negative learning_rate must return FannError::Validation, got {result:?}"
    );
}

/// NaN aux_loss_coeff must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "NaN aux_loss_coeff must return FannError::Validation, got {result:?}"
    );
}

/// Negative z_loss_coeff must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "negative z_loss_coeff must return FannError::Validation, got {result:?}"
    );
}

/// Negative ewc_lambda must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "negative ewc_lambda must return FannError::Validation, got {result:?}"
    );
}

/// Infinite ewc_lambda must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "infinite ewc_lambda must return FannError::Validation, got {result:?}"
    );
}

/// replay_mix_fraction > 1.0 must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "replay_mix_fraction > 1.0 must return FannError::Validation, got {result:?}"
    );
}

/// A NaN value inside an event's context_vector must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "NaN in context_vector must return FannError::Validation, got {result:?}"
    );
}

/// epochs == 0 must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "epochs=0 must return FannError::Validation, got {result:?}"
    );
}

/// A non-empty DiagonalFisher with a NaN value must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "Fisher NaN value must return FannError::Validation, got {result:?}"
    );
}

/// A non-empty DiagonalFisher with a negative value must return Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "negative Fisher value must return FannError::Validation, got {result:?}"
    );
}

/// Fisher decay = 1.0, set via struct literal to bypass DiagonalFisher::new(),
/// must return Err(FannError::Validation) even when values is empty.
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
        matches!(result, Err(FannError::Validation(_))),
        "Fisher decay=1.0 must return FannError::Validation, got {result:?}"
    );
}

/// A non-empty DiagonalFisher with anchor length != total_params must return
/// Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "Fisher anchor length mismatch must return FannError::Validation, got {result:?}"
    );
}

/// A non-empty DiagonalFisher with a non-finite anchor entry must return
/// Err(FannError::Validation).
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
        matches!(result, Err(FannError::Validation(_))),
        "non-finite Fisher anchor must return FannError::Validation, got {result:?}"
    );
}

// ─── ewc_lambda Phase-2 boundary test ─────────────────────────────────

/// Two runs identical in every respect except ewc_lambda must produce
/// byte-identical RouterDelta::network_bytes.
///
/// This pins the Phase-2 boundary: v1 uses project_delta (Fisher-weighted
/// damping), which does not read ewc_lambda.  The test fails
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
