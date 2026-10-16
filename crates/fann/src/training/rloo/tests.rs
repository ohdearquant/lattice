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

/// Control 2 — the `1 / tau` factor's VALUE: with reward 0, z-loss off and
/// the same `w_full`, the gate delta at `tau = 0.1` is 10× the delta at
/// `tau = 1.0`, because `1 / tau` is 10 there.
///
/// Mutation this guards: using `w.tau` itself as the load-balance scale
/// instead of `1.0 / w.tau`. That mutant gives scale 1.0 at `tau = 1.0` (so
/// the `tau = 1.0` arm is unchanged) and 0.1 instead of 10 at `tau = 0.1`;
/// the ratio it produces is `0.1`, not 10, and this assertion reddens. A
/// test that only asked the two states to differ would stay green under it,
/// which is why the ratio is pinned rather than the difference.
///
/// The tolerance is on the ratio, not the deltas: the gate is a fresh
/// seeded network and the two arms share `w_full` and the drift preamble,
/// so the only source of imprecision is f32 rounding in the two forward
/// passes, which moves the ratio by far less than the 1e-3 allowed here
/// (measured on this fixture: the ratio lands within 1e-5 of 10.0).
#[test]
fn step_with_learned_weights_scales_the_aux_gradient_by_inverse_tau() {
    let config = RlooConfig {
        learning_rate: 0.5,
        aux_loss_coeff: 1.0,
        z_loss_coeff: 0.0,
    };
    // A single Linear layer, so the forward output moves by exactly
    // `-lr * delta_j * (1 + ||CTX||^2)` and the raw output move is
    // proportional to the output-layer delta. On a gate with a hidden
    // layer the move would also carry a second-order term in the delta,
    // and the 10x ratio would be off by that much.
    let gate_base = linear_gate();
    let w_full = [0.7_f32, 0.0, 0.3, 0.0];

    let move_at_tau = |tau: f32| -> Vec<f32> {
        let (mut gate, mut trainer) = drifted_off_uniform(&gate_base, config.clone());
        let before = gate.forward(&CTX).unwrap().to_vec();
        trainer
            .step_with_learned_weights(&mut gate, &CTX, 0, 0.0, &w_full, tau)
            .unwrap();
        let after = gate.forward(&CTX).unwrap().to_vec();
        before
            .iter()
            .zip(after.iter())
            .map(|(b, a)| a - b)
            .collect()
    };

    let at_one = move_at_tau(1.0);
    let at_p1 = move_at_tau(0.1);
    // `w_full` is zero at indices 1 and 3, so the load-balance gradient
    // there is zero and there is no ratio to take; the non-zero columns are
    // where the factor is observable.
    let mut checked = 0_usize;
    for (j, (&one, &p1)) in at_one.iter().zip(at_p1.iter()).enumerate() {
        if one.abs() <= 1e-6 {
            continue;
        }
        let ratio = p1 / one;
        assert!(
            (ratio - 10.0).abs() < 1e-3,
            "index {j}: the aux scale must be 1/tau, so tau=0.1 must move \
                 the gate 10x the tau=1.0 move; got ratio {ratio} \
                 (at_one {one}, at_p1 {p1})"
        );
        checked += 1;
    }
    assert!(
        checked >= 2,
        "fixture must expose the factor at two columns, saw {checked}"
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

/// `tau` must be finite and strictly positive with a finite reciprocal;
/// this is the arithmetic precondition only — the collapse floor on `tau`
/// is guard (a)'s job in the router, not reimplemented here. The smallest
/// subnormal is included because a finite positive `tau` can still have an
/// infinite `1 / tau`, which is the value the update path actually
/// multiplies by; `f32::MIN_POSITIVE` is NOT in the list because its
/// reciprocal is finite and it is a legitimate temperature.
#[test]
fn step_with_learned_weights_rejects_non_finite_or_nonpositive_tau() {
    let w_full = [0.25_f32; 4];
    for bad_tau in [
        0.0_f32,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ] {
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
/// Fails if the branch applies a term the `Single` path does not: the two
/// samples carry opposite rewards, so the round IS informative and reaches
/// the apply, yet `z' == z` makes every policy delta zero. The gate bytes
/// are compared before and after the round, not to any other round.
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

/// Every refusal this entry makes on the action, and each names the field
/// it saw: a zero, negative or non-finite scale, a zero or non-finite tau,
/// and a `w_full` of the wrong width.
///
/// Fails if any of these checks is dropped; the accepted shape is the
/// control in the same test. (This arm covers only the refusals its `cases`
/// list actually passes — `tau = 0` and `scale = NaN` are here, `tau = NaN`
/// is not; the arms that pin the other refusals are
/// `step_with_learned_weights_rejects_non_finite_or_nonpositive_tau` and
/// `a_tau_that_cannot_be_inverted_is_refused`.)
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

/// A non-finite `w_full` entry is refused by both single-step entries
/// before the gate or the traffic EMA is touched.
///
/// Fails if the `w_full` finiteness check is dropped: one NaN mass makes
/// the load-balance gradient NaN, the applied weights NaN, and the EMA
/// NaN, and a NaN EMA stays NaN through every later fold, so the gate never
/// recovers. The refusal must name the field and the value.
#[test]
fn non_finite_w_full_is_refused_before_the_gate_moves() {
    let base = test_gate();
    let sampled = [0.1_f32, -0.2, 0.3, 0.4];

    for (label, bad) in [
        ("nan w_full", [0.5_f32, f32::NAN, 0.5, 0.0]),
        ("infinite w_full", [0.5_f32, f32::INFINITY, 0.5, 0.0]),
    ] {
        // step_with_learned_weights
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let before = gate.to_bytes();
        let freqs_before = trainer.route_freqs.clone();
        let err = trainer
            .step_with_learned_weights(&mut gate, &CTX, 0, 1.0, &bad, 1.0)
            .unwrap_err();
        assert!(
            err.to_string().contains("w_full"),
            "{label}: step_with_learned_weights must name w_full, got {err}"
        );
        assert_eq!(gate.to_bytes(), before, "{label}: the gate moved");
        assert_eq!(
            trainer.route_freqs, freqs_before,
            "{label}: the traffic EMA moved"
        );

        // step_with_explored_action
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let before = gate.to_bytes();
        let freqs_before = trainer.route_freqs.clone();
        let err = trainer
            .step_with_explored_action(
                &mut gate,
                &CTX,
                &ExploredAction {
                    sampled_logits: &sampled,
                    exploration_scale: 1.0,
                    w_full: &bad,
                    tau: 1.0,
                },
                1.0,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("w_full"),
            "{label}: step_with_explored_action must name w_full, got {err}"
        );
        assert_eq!(gate.to_bytes(), before, "{label}: the gate moved");
        assert_eq!(
            trainer.route_freqs, freqs_before,
            "{label}: the traffic EMA moved"
        );
    }

    // The control: a finite `w_full` of the same shape is accepted and does
    // move the gate, so the equalities above are about the refusal.
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let before = gate.to_bytes();
    trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &sampled,
                exploration_scale: 1.0,
                w_full: &[0.5, 0.0, 0.5, 0.0],
                tau: 1.0,
            },
            1.0,
        )
        .unwrap();
    assert_ne!(gate.to_bytes(), before);
}

/// A non-finite reward is refused by the single explored entry before
/// anything is applied.
///
/// Fails if the reward check on the single-step entry is dropped: the
/// policy term is `reward * (z - z') / s^2`, which is NaN for a NaN reward
/// even when `z == z'` (`NaN * 0.0` is NaN), and it is applied to the gate.
#[test]
fn a_non_finite_reward_is_refused_on_the_explored_entry() {
    let config = RlooConfig {
        learning_rate: 0.5,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
    };
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let base = test_gate();

    for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (mut gate, mut trainer) = drifted_off_uniform(&base, config.clone());
        let z = gate.forward(&CTX).unwrap().to_vec();
        let before = gate.to_bytes();
        let err = trainer
            .step_with_explored_action(
                &mut gate,
                &CTX,
                &ExploredAction {
                    // `z' == z`: the offset cannot be the source of the NaN.
                    sampled_logits: &z,
                    exploration_scale: 1.0,
                    w_full: &w_full,
                    tau: 1.0,
                },
                reward,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("reward"),
            "reward {reward} must be refused and named, got {err}"
        );
        assert_eq!(gate.to_bytes(), before, "reward {reward} moved the gate");
    }

    // The control: the same action with a finite reward is accepted.
    let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
    let z = gate.forward(&CTX).unwrap().to_vec();
    trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &z,
                exploration_scale: 1.0,
                w_full: &w_full,
                tau: 1.0,
            },
            1.0,
        )
        .unwrap();
}

/// A `tau` that is non-finite, or finite and positive but whose reciprocal
/// overflows `f32`, is refused on both single-step entries.
///
/// Fails if the `1 / tau` check is dropped or weakened to `tau > 0`: the
/// smallest subnormal passes `tau > 0`, its `1 / tau` is `inf`, and an
/// unselected entry's `0.0 * inf` then makes the load-balance term NaN.
/// Fails too if the `tau` finiteness check is dropped: a NaN `tau` passes
/// `tau <= 0.0` (the comparison is false) and poisons the gate the same way.
#[test]
fn a_tau_that_cannot_be_inverted_is_refused() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let sampled = [0.1_f32, -0.2, 0.3, 0.4];
    let bad_taus = [f32::from_bits(1), f32::NAN, f32::INFINITY];
    assert!(
        f32::from_bits(1).is_finite() && f32::from_bits(1) > 0.0,
        "fixture: the subnormal must be finite and positive"
    );
    assert!(
        !(1.0 / f32::from_bits(1)).is_finite(),
        "fixture: the subnormal's reciprocal must overflow"
    );

    let base = test_gate();
    for bad_tau in bad_taus {
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let before = gate.to_bytes();
        let err = trainer
            .step_with_learned_weights(&mut gate, &CTX, 0, 1.0, &w_full, bad_tau)
            .unwrap_err();
        assert!(
            err.to_string().contains("tau"),
            "step_with_learned_weights must name tau, got {err}"
        );
        assert_eq!(gate.to_bytes(), before, "tau={bad_tau} moved the gate");

        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let before = gate.to_bytes();
        let err = trainer
            .step_with_explored_action(
                &mut gate,
                &CTX,
                &ExploredAction {
                    sampled_logits: &sampled,
                    exploration_scale: 1.0,
                    w_full: &w_full,
                    tau: bad_tau,
                },
                1.0,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("tau"),
            "step_with_explored_action must name tau, got {err}"
        );
        assert_eq!(gate.to_bytes(), before, "tau={bad_tau} moved the gate");
    }

    // The control: a small but representable `tau` whose reciprocal is
    // finite is accepted.
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    trainer
        .step_with_learned_weights(&mut gate, &CTX, 0, 1.0, &w_full, 1e-38)
        .expect("a tau whose 1 / tau is finite must be accepted");
}

/// A non-finite context entry is refused before the gate is touched, on
/// both the single and the batch entry.
///
/// Fails if the context finiteness check is dropped: a NaN context makes
/// every forward activation NaN, so every weight gradient is NaN and the
/// gate is poisoned.
#[test]
fn a_non_finite_context_is_refused_before_the_gate_moves() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let sampled = [0.1_f32, -0.2, 0.3, 0.4];
    let bad_ctx = [0.1_f32, f32::NAN, 0.3, 0.4];
    let base = test_gate();

    // Single entry: step_with_explored_action.
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let before = gate.to_bytes();
    let err = trainer
        .step_with_explored_action(
            &mut gate,
            &bad_ctx,
            &ExploredAction {
                sampled_logits: &sampled,
                exploration_scale: 1.0,
                w_full: &w_full,
                tau: 1.0,
            },
            1.0,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("context"),
        "must name the context, got {err}"
    );
    assert_eq!(gate.to_bytes(), before);

    // Batch entry: the bad context is at a NON-zero index, and the gate and
    // the traffic EMA must be byte-identical after the refusal.
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
    let other: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
    let before = gate.to_bytes();
    let freqs_before = trainer.route_freqs.clone();
    let samples = [
        MixtureSample {
            context: &CTX,
            action: batch_action(&served, &w_full),
            reward: 1.0,
        },
        MixtureSample {
            context: &bad_ctx,
            action: batch_action(&other, &w_full),
            reward: -1.0,
        },
    ];
    let err = trainer
        .step_batch_with_explored_actions(&mut gate, &samples)
        .unwrap_err();
    assert!(
        err.to_string().contains("context"),
        "must name the context, got {err}"
    );
    assert_eq!(gate.to_bytes(), before, "the refused round moved the gate");
    assert_eq!(
        trainer.route_freqs, freqs_before,
        "the refused round folded traffic"
    );

    // The control: the same round with a finite context reaches the gate.
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
    trainer
        .step_batch_with_explored_actions(&mut gate, &samples)
        .unwrap();
    assert_ne!(gate.to_bytes(), before);
}
/// A finite reward whose policy term overflows the gate's arithmetic is
/// refused at the apply, before any weight or bias is written, and the
/// traffic EMA is left alone too.
///
/// `1e38` is finite, but `reward * (z - z') / s^2` at `s = 1` with
/// `z' = z + 4` is past the `f32` ceiling, so the output-layer delta is
/// `-inf`. The fixture asserts the overflow itself, so the test cannot pass
/// by never reaching it. Fails if the apply writes a non-finite candidate.
#[test]
fn a_finite_reward_that_overflows_the_policy_term_is_refused() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let base = test_gate();
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<f32> = z.iter().map(|zj| zj + 4.0).collect();

    let reward = 1e38_f32;
    assert!(reward.is_finite(), "fixture: the reward must be finite");
    let policy: Vec<f32> = z
        .iter()
        .zip(served.iter())
        .map(|(zj, sj)| reward * (zj - sj) * (1.0 / (1.0 * 1.0)))
        .collect();
    assert!(
        policy.iter().any(|p| !p.is_finite()),
        "fixture: the policy term must overflow, got {policy:?}"
    );

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &served,
                exploration_scale: 1.0,
                w_full: &w_full,
                tau: 1.0,
            },
            reward,
        )
        .unwrap_err();

    let message = err.to_string();
    assert!(message.contains("layer"), "must name the layer: {message}");
    assert!(
        message.contains("weight") || message.contains("bias"),
        "must name the parameter kind: {message}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
}

/// A finite `w_full` entry whose load-balance term overflows is refused at
/// the apply, with the gate and the traffic EMA left unchanged.
///
/// `w_full = [3e38, 0, 0, 0]` is finite of the right width and its `tau = 1`
/// inverts, so the up-front validation accepts it; but the load-balance
/// gradient `k * p * (f - f.p)` overflows at index 0 in the `k * p` product
/// alone: `4 * 3e38 = 1.2e39` is past `f32::MAX`, whatever `f - f.p` is.
/// The product is evaluated left to right, so the gradient is non-finite
/// before the difference is even reached. Fails if the apply writes a
/// non-finite candidate.
#[test]
fn a_finite_w_full_entry_that_overflows_the_load_balance_term_is_refused() {
    let w_full = [3e38_f32, 0.0, 0.0, 0.0];
    assert!(
        w_full.iter().all(|w| w.is_finite()),
        "fixture: every w_full entry must be finite"
    );
    let base = test_gate();
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());

    // Fixture: the load-balance gradient at index 0 overflows, computed
    // from the traffic the trainer will read.
    let freqs = trainer.route_freqs.clone();
    let f_dot_p: f32 = freqs.iter().zip(w_full.iter()).map(|(f, p)| f * p).sum();
    let grad0 = 4.0_f32 * w_full[0] * (freqs[0] - f_dot_p);
    assert!(
        !grad0.is_finite(),
        "fixture: the load-balance gradient must overflow, got {grad0}"
    );

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_learned_weights(&mut gate, &CTX, 0, 0.0, &w_full, 1.0)
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("layer"),
        "the refusal must name the layer: {message}"
    );
    assert!(
        message.contains("NaN") || message.contains("inf"),
        "the refusal must name the non-finite value it saw: {message}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
}

/// A finite `aux_loss_coeff` times a `tau` whose reciprocal is finite can
/// still overflow the aux product, and that is refused at the apply.
///
/// `tau = 1e-38` has `1 / tau = 1.0000000694e38`, finite, so the up-front
/// check accepts it; `aux_loss_coeff = 10` times that reciprocal is `inf`,
/// and an unselected `w_full` entry then multiplies it by `0.0` to `NaN`.
/// Fails if the apply writes a non-finite candidate.
#[test]
fn a_finite_aux_scale_that_overflows_the_aux_product_is_refused() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let config = RlooConfig {
        learning_rate: 0.5,
        aux_loss_coeff: 10.0,
        z_loss_coeff: 0.001,
    };
    let tau = 1e-38_f32;
    let inv_tau = 1.0 / tau;
    // Fixture: the reciprocal is finite, the scaled aux term is not, and an
    // unselected entry turns it into NaN.
    assert!(
        inv_tau.is_finite(),
        "fixture: 1/tau must be finite, got {inv_tau}"
    );
    let scaled = config.aux_loss_coeff * inv_tau;
    assert!(
        !scaled.is_finite(),
        "fixture: coeff * (1/tau) must overflow, got {scaled}"
    );
    assert!(
        (scaled * 0.0_f32).is_nan(),
        "fixture: an unselected entry must become NaN"
    );

    let base = test_gate();
    let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_learned_weights(&mut gate, &CTX, 0, 0.0, &w_full, tau)
        .unwrap_err();
    assert!(
        err.to_string().contains("layer"),
        "the refusal must name the layer: {err}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
}

/// A tiny but finite exploration scale whose reciprocal is finite can still
/// overflow the policy product, which is refused at the apply.
///
/// `s = 1e-19` has `1 / s^2 = 1.0000000694e38`, finite, so the up-front
/// check accepts it; with `reward = 1` and `z' = z + 10` the policy term is
/// `-10 * 1.0000000694e38`, past the `f32` ceiling. Fails if the apply
/// writes a non-finite candidate.
#[test]
fn a_tiny_finite_scale_that_overflows_the_policy_term_is_refused() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let scale = 1e-19_f32;
    let inv = 1.0 / (scale * scale);
    assert!(inv.is_finite(), "fixture: 1/s^2 must be finite, got {inv}");
    let base = test_gate();
    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<f32> = z.iter().map(|zj| zj + 10.0).collect();
    let policy: Vec<f32> = z
        .iter()
        .zip(served.iter())
        .map(|(zj, sj)| 1.0_f32 * (zj - sj) * inv)
        .collect();
    assert!(
        policy.iter().any(|p| !p.is_finite()),
        "fixture: the policy term must overflow, got {policy:?}"
    );

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &served,
                exploration_scale: scale,
                w_full: &w_full,
                tau: 1.0,
            },
            1.0,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("layer"),
        "the refusal must name the layer: {err}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
}

/// A finite gradient whose SGD step still overflows is refused, with the
/// gate and the traffic EMA left unchanged.
///
/// The output-layer delta is finite here — `reward = 1e38` against a
/// one-unit logit difference — and so is every gradient: this fixture is a
/// single Linear layer whose weights the context shrinks (every
/// `|context_j| < 0.5`) and whose bias gradients are the delta itself,
/// `-1e38`. The gradient check therefore passes while
/// `learning_rate * gradient` is past `f32::MAX`: at `learning_rate = 4`
/// the candidate `w - lr * g` is `inf`. This is the case the candidate arm
/// exists for, so removing that arm makes the write succeed and this
/// `unwrap_err` panic. Fails if the apply writes a non-finite candidate.
#[test]
fn a_finite_gradient_whose_step_overflows_is_refused() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let config = RlooConfig {
        learning_rate: 4.0,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
    };
    let lr = config.learning_rate;
    let base = linear_gate();
    let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<f32> = z.iter().map(|zj| zj + 1.0).collect();

    // Fixture: every gradient is finite and the candidate is not. The
    // gradients are `delta_i * context_j` for the weights and `delta_i` for
    // the biases, and the context entries are all below 0.5 in magnitude.
    let reward = 1e38_f32;
    assert!(reward.is_finite(), "fixture: the reward must be finite");
    let delta: Vec<f32> = z
        .iter()
        .zip(served.iter())
        .map(|(zj, sj)| reward * (zj - sj) * (1.0 / (1.0 * 1.0)))
        .collect();
    assert!(
        delta.iter().all(|d| d.is_finite()),
        "fixture: every output delta must be finite, got {delta:?}"
    );
    assert!(
        CTX.iter().all(|x| x.abs() < 0.5),
        "fixture: every context entry must shrink the weight gradient"
    );
    assert!(
        delta.iter().any(|d| !(lr * d).is_finite()),
        "fixture: some lr * delta must overflow, got {:?}",
        delta.iter().map(|d| lr * d).collect::<Vec<_>>()
    );

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &served,
                exploration_scale: 1.0,
                w_full: &w_full,
                tau: 1.0,
            },
            reward,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("layer"),
        "the refusal must name the layer: {err}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
}

/// A non-finite `learning_rate` is refused at the write and leaves the gate
/// unchanged, rather than writing NaN into every parameter.
///
/// The config fields are `pub` and unvalidated, so a NaN rate is a
/// caller's to supply; what the learner guarantees is that the refusal
/// happens before anything is written, on the single-step entry as well as
/// the batch. Fails if the write happens without the candidate check.
#[test]
fn a_non_finite_learning_rate_is_refused_at_the_write() {
    let config = RlooConfig {
        learning_rate: f32::NAN,
        aux_loss_coeff: 0.01,
        z_loss_coeff: 0.001,
    };
    let mut gate = test_gate();
    let mut trainer = RlooTrainer::with_seed(config, 1);

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer.step(&mut gate, &CTX, 0, 1.0).unwrap_err();
    assert!(
        err.to_string().contains("layer"),
        "the refusal must name the layer: {err}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
}

/// A refused update leaves an uninitialised trainer's traffic estimate
/// exactly as it was: empty, not filled with the uniform vector the next
/// successful call would start from.
///
/// The uniform fill happens when a decision is folded, and a refused
/// update never folds one, so the first call being refused leaves the
/// field unset. Fails if the fill moves back ahead of the apply.
#[test]
fn a_refused_first_update_leaves_the_traffic_estimate_unset() {
    let config = RlooConfig {
        learning_rate: 4.0,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
    };
    let mut gate = linear_gate();
    let mut trainer = RlooTrainer::with_seed(config, 1);
    assert!(
        trainer.route_freqs.is_empty(),
        "fixture: a fresh trainer must have no traffic estimate"
    );

    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<f32> = z.iter().map(|zj| zj + 1.0).collect();
    let reward = 1e38_f32;

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &served,
                exploration_scale: 1.0,
                w_full: &[0.5_f32, 0.0, 0.5, 0.0],
                tau: 1.0,
            },
            reward,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("layer"),
        "the refusal must name the layer: {err}"
    );
    assert!(
        trainer.route_freqs.is_empty(),
        "the refused update sized the traffic estimate: {:?}",
        trainer.route_freqs
    );
    assert_eq!((gate.to_bytes(), trainer.route_freqs.clone()), before);
}

/// A round whose first candidate is finite but a later one is not is
/// refused as a whole: the finite candidate is not written on the way to
/// discovering the non-finite one.
///
/// The output layer is `Linear`, so its weight gradients are
/// `delta_i * context_j` and its bias gradients are `delta_i`. Every
/// `delta_i` is `-1e38`, so the bias gradients are finite and their step
/// `lr * delta_i` is `-4e38`, past `f32::MAX`; the weight gradients shrink
/// by `|context_j| < 0.5` and their step stays finite. The apply visits
/// layer 0's weights (all 16 finite) before layer 0's biases, so
/// `bias[0]` is the first non-finite candidate and every candidate before
/// it is finite. Fails if the apply writes each candidate as it is checked:
/// the finite weight candidates would then already be in the gate.
#[test]
fn a_round_with_a_finite_candidate_before_a_non_finite_one_writes_nothing() {
    // A single Linear layer: 4 inputs, 4 outputs, no hidden layer.
    let config = RlooConfig {
        learning_rate: 4.0,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
    };
    let lr = config.learning_rate;
    let base = linear_gate();
    let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
    let z = gate.forward(&CTX).unwrap().to_vec();
    // The delta is `reward * (z - z')`; `z' - z` is exactly one for every
    // index, so all four entries are `-reward`.
    let served: Vec<f32> = z.iter().map(|zj| zj + 1.0).collect();
    let reward = 1e38_f32;

    let delta: Vec<f32> = z
        .iter()
        .zip(served.iter())
        .map(|(zj, sj)| reward * (zj - sj) * (1.0 / (1.0 * 1.0)))
        .collect();
    assert!(
        delta.iter().all(|d| d.is_finite()),
        "fixture: every delta must be finite, got {delta:?}"
    );
    // The weight gradient is `delta * context_j`; every `|context_j|` is
    // below 0.5, so the first weight candidate is finite. The apply forms
    // the step as `lr * (delta * context)`, not `(lr * delta) * context`.
    let first_weight_step = lr * (delta[0] * CTX[0]);
    assert!(
        first_weight_step.is_finite(),
        "fixture: the first weight step must be finite, got {first_weight_step}"
    );
    // The bias gradient is the delta itself; the step from it overflows.
    assert!(
        !(lr * delta[3]).is_finite(),
        "fixture: the bias step must overflow, got {}",
        lr * delta[3]
    );

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_with_explored_action(
            &mut gate,
            &CTX,
            &ExploredAction {
                sampled_logits: &served,
                exploration_scale: 1.0,
                w_full: &[0.5_f32, 0.0, 0.5, 0.0],
                tau: 1.0,
            },
            reward,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("bias") || err.to_string().contains("weight"),
        "the refusal must name the parameter: {err}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused update wrote something"
    );
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

mod batch_refit;
