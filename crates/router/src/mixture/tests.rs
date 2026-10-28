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
    let mut layer = lattice_fann::Layer::zeros(inputs, scores.len(), Activation::Linear).unwrap();
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

/// The deterministic path is bit-identical: `exploration: None` reproduces
/// `route_detailed`, and the outcome carries no draw. The existing `route`
/// tests pin the weights; this pins that the wrapper and the explicit
/// `None` agree on the whole outcome.
///
/// Would fail if `route_explored` perturbed anything when `draw` is `None`,
/// or if `route_detailed` stopped being the `None` arm.
#[test]
fn exploration_off_reproduces_the_deterministic_decision() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.9, 0.4, 0.1]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    let detailed = router.route_detailed(&context, &pool, 2).unwrap();
    let explicit = router.route_explored(&context, &pool, 2, None).unwrap();
    assert_eq!(detailed, explicit);
    assert!(detailed.exploration.is_none());
    assert_eq!(detailed.selected_scores, detailed.sampled_scores);
    assert_eq!(router.route(&context, &pool, 2).unwrap(), detailed.weights);
}

/// Same seed, same inputs: identical `z'`, weights and `log_density`.
/// A different seed moves `z'`.
///
/// Would fail if the generator stopped being a pure function of the seed,
/// or if the seed were ignored (the second half).
#[test]
fn a_draw_is_reproducible_from_its_recorded_seed() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.9, 0.8, 0.7]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    let draw = Some(Exploration {
        scale: 0.5,
        seed: 12_345,
    });
    let first = router.route_explored(&context, &pool, 2, draw).unwrap();
    let second = router.route_explored(&context, &pool, 2, draw).unwrap();
    assert_eq!(first, second, "the same seed must replay the same decision");

    let other = router
        .route_explored(
            &context,
            &pool,
            2,
            Some(Exploration {
                scale: 0.5,
                seed: 12_346,
            }),
        )
        .unwrap();
    let a = first.exploration.as_ref().unwrap();
    let b = other.exploration.as_ref().unwrap();
    assert_ne!(a.sampled_logits, b.sampled_logits, "a new seed must redraw");
    assert_ne!(a.log_density, b.log_density);
}

/// Over a fixed seed and 20,000 draws on one context, the per-output mean
/// of `z' - z` is within 4 standard errors of 0 and the per-output variance
/// is within a stated tolerance of `s^2`, for two scales.
///
/// The tolerance: the sample variance of a normal is itself random, with
/// relative standard error `sqrt(2 / (N - 1))`. At `N = 20,000` that is
/// about `1.0%`, so a 5% band is five standard errors of the estimator and
/// a real bias in the scale would have to be larger than 5% to hide. The
/// mean band is four standard errors of the mean, `4 * s / sqrt(N)`.
///
/// Would fail if the draw dropped the `s` factor (variance collapses to 1
/// and the two scales agree), if it added the noise before scaling, or if
/// the generator were biased.
#[test]
fn the_draw_is_gaussian_with_the_recorded_scale() {
    let context = [1.0, 0.5];
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let rounds = 20_000u64;
    for scale in [0.5_f32, 2.0_f32] {
        let mut router = AdapterRouter::new(scored_gate(2, &[0.3, 0.2, 0.1]));
        // f32 constants converted exactly as the gate's own scores are,
        // so the mean tests the draw and not the literal's precision.
        let gate_scores = [0.3_f32, 0.2, 0.1];
        let mut sums = [0.0_f64; 3];
        let mut squares = [0.0_f64; 3];
        for seed in 0..rounds {
            let outcome = router
                .route_explored(&context, &pool, 2, Some(Exploration { scale, seed }))
                .unwrap();
            let logits = &outcome.exploration.as_ref().unwrap().sampled_logits;
            assert_eq!(logits.len(), 3, "one value per gate output");
            for (i, drawn) in logits.iter().enumerate() {
                let difference = f64::from(*drawn) - f64::from(gate_scores[i]);
                sums[i] += difference;
                squares[i] += difference * difference;
            }
        }
        let n = rounds as f64;
        for i in 0..3 {
            let mean = sums[i] / n;
            let variance = squares[i] / n - mean * mean;
            let standard_error = f64::from(scale) / n.sqrt();
            assert!(
                mean.abs() < 4.0 * standard_error,
                "scale {scale}: output {i} mean {mean} outside 4 standard errors"
            );
            let expected = f64::from(scale) * f64::from(scale);
            assert!(
                (variance - expected).abs() / expected < 0.05,
                "scale {scale}: output {i} variance {variance} is not within 5% of {expected}"
            );
        }
    }
}

/// `log_density` equals the closed form computed independently in the test
/// from the recorded sampled logits and the deterministic gate scores.
///
/// Would fail if the density were computed from the selected subset rather
/// than all gate outputs, if the `n` in the normalising term were the
/// selected width rather than the output count, or if the sign of the
/// quadratic term were flipped.
#[test]
fn the_log_density_matches_the_closed_form() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.3, 0.2, 0.1]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    // The gate's own f32 scores, converted the way the source converts
    // them, so the closed form is compared bit for bit.
    let gate_scores = [0.3_f32, 0.2, 0.1];
    let scale = 0.75_f32;
    let outcome = router
        .route_explored(&context, &pool, 2, Some(Exploration { scale, seed: 99 }))
        .unwrap();
    let recorded = outcome.exploration.as_ref().unwrap();

    let squared: f64 = recorded
        .sampled_logits
        .iter()
        .zip(gate_scores.iter())
        .map(|(drawn, own)| {
            let difference = f64::from(*drawn) - f64::from(*own);
            difference * difference
        })
        .sum();
    let scale64 = f64::from(scale);
    let n = gate_scores.len() as f64;
    let expected = -squared / (2.0 * scale64 * scale64)
        - (n / 2.0) * (2.0 * std::f64::consts::PI * scale64 * scale64).ln();
    assert!(
        (recorded.log_density - expected).abs() < 1e-12,
        "recorded {} against closed form {expected}",
        recorded.log_density
    );
    // The density is of `z'` given `z`, so a zero perturbation would give
    // the density's maximum. The drawn value must be below it, and finite.
    assert!(recorded.log_density.is_finite());
    assert!(recorded.log_density < 0.0);
}

/// Every drawn value is finite over the same 20,000 draws, and the normal
/// sampler is driven directly with the boundary uniform value 1.0 and the
/// smallest value the generator can produce.
///
/// Would fail if the uniform reached `0.0` (the `ln` of it is `-inf`) or if
/// the half-ulp offset were removed, which is the one change that lets the
/// all-zeros draw reach `0.0`.
#[test]
fn every_drawn_variate_is_finite() {
    let mut state = 0u64;
    for _ in 0..200_000 {
        let variate = normal_from(&mut state);
        assert!(variate.is_finite(), "{variate}");
    }

    // The generator's own range, checked directly: every value is in
    // `(0, 1]`, so `ln` never sees `0.0`. This is what the `+ 1` before
    // scaling buys, and removing it is the mutation that reaches `0.0`.
    let mut range_state = 0u64;
    for _ in 0..200_000 {
        let u = uniform_from(&mut range_state);
        assert!(u > 0.0 && u <= 1.0, "uniform out of (0, 1]: {u}");
    }

    // The boundary uniforms, fed straight in: the generator's largest and
    // smallest reachable values, and 0.0, which the `+ 1` must make
    // unreachable. `uniform_from` maps the all-ones bits to exactly 1.0
    // and the all-zeros bits to the smallest positive it can produce; both
    // must keep the normal finite.
    for u in [1.0_f64, f64::MIN_POSITIVE, 1.0 / 9_007_199_254_740_992.0] {
        let magnitude = (-2.0 * u.ln()).sqrt();
        assert!(
            magnitude.is_finite(),
            "u = {u} gave a non-finite magnitude {magnitude}"
        );
        for v in [0.0_f64, 1.0] {
            let variate = magnitude * (2.0 * std::f64::consts::PI * v).cos();
            assert!(variate.is_finite(), "u = {u}, v = {v} gave {variate}");
        }
    }
}

/// Selection and weights are computed from `z'`: with `k` smaller than the
/// output count and a large `s`, the selected set differs from the
/// deterministic top-`k` for some seed; and with the same `z'` injected,
/// the draw's weights equal `route`'s weights on `z'`.
///
/// Would fail if the perturbation were applied after the selection, or if
/// the weight policy read the deterministic scores instead of `z'`.
#[test]
fn selection_and_weights_are_computed_from_the_sampled_logits() {
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    // Near-tied scores and a large scale: the deterministic top-1 is "a",
    // and a draw has to be able to pick another column.
    let mut router = AdapterRouter::new(scored_gate(2, &[0.30, 0.29, 0.28]));
    let deterministic = router.route_detailed(&context, &pool, 1).unwrap();
    let deterministic_choice = deterministic.weights[0].0.clone();
    let mut seen_different = false;
    for seed in 0..200u64 {
        let outcome = router
            .route_explored(&context, &pool, 1, Some(Exploration { scale: 3.0, seed }))
            .unwrap();
        if outcome.weights[0].0 != deterministic_choice {
            seen_different = true;
            break;
        }
    }
    assert!(
        seen_different,
        "a large-scale draw over near-tied scores never changed the top-1"
    );

    // With the same `z'` injected, the draw's weights equal the
    // deterministic route's weights on `z'`. A softmax router makes this a
    // real comparison rather than `1/k` on both sides.
    let mut drawn = AdapterRouter::new(scored_gate(2, &[0.30, 0.29, 0.28]));
    drawn.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    let outcome = drawn
        .route_explored(
            &context,
            &pool,
            2,
            Some(Exploration {
                scale: 1.0,
                seed: 4,
            }),
        )
        .unwrap();
    let logits = outcome.exploration.as_ref().unwrap().sampled_logits.clone();

    // A gate that serves exactly `z'` deterministically: the bias vector
    // is `z'`, so `route` on it consumes the same logits.
    let mut injected = AdapterRouter::new(scored_gate(2, &logits));
    injected.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    let direct = injected.route(&context, &pool, 2).unwrap();
    assert_eq!(
        outcome.weights, direct,
        "the draw's weights must equal route's weights on the sampled logits"
    );
}

/// A draw over a decision the perturbation cannot move is refused, naming
/// both escapes. Uniform with a narrower `k` is allowed, and a policy that
/// reads the scores is allowed.
///
/// Would fail if the refusal were dropped, if it fired for a narrower `k`,
/// or if it fired for a softmax policy.
#[test]
fn a_draw_that_cannot_change_the_action_is_refused() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.3, 0.2, 0.1]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    let draw = Some(Exploration {
        scale: 1.0,
        seed: 1,
    });

    let error = router.route_explored(&context, &pool, 3, draw).unwrap_err();
    let message = error.to_string();
    assert!(
        matches!(
            error,
            RouterError::ExplorationCannotChangeTheAction {
                k: 3,
                candidates: 3
            }
        ),
        "got {error}"
    );
    // Both escapes are named, so the refusal points at a fork rather than a
    // dead end.
    assert!(message.contains("narrow k"), "{message}");
    assert!(message.contains("reads the scores"), "{message}");

    // The deterministic path over the same shape is untouched: uniform
    // over everything is a legitimate way to serve, it is only a
    // dishonest way to explore.
    assert!(router.route_detailed(&context, &pool, 3).is_ok());

    // Both escapes, exercised: a narrower k lets the draw move the selected
    // set, and a policy that reads the scores lets it move the weights.
    assert!(router.route_explored(&context, &pool, 2, draw).is_ok());
    router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    assert!(router.route_explored(&context, &pool, 3, draw).is_ok());

    // The second door into the same defect: ONE candidate. Softmax is an
    // escape from the uniform case and not from this one, because a lone
    // column takes weight 1.0 under every policy there is.
    let mut single = AdapterRouter::new(scored_gate(2, &[0.3]));
    let one: Vec<AdapterId> = vec!["a".into()];
    single.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    let error = single.route_explored(&context, &one, 1, draw).unwrap_err();
    assert!(
        matches!(
            error,
            RouterError::ExplorationCannotChangeTheAction {
                k: 1,
                candidates: 1
            }
        ),
        "got {error}"
    );
    // And the control: the same single-column router serves fine without a
    // draw, so the refusal is about exploring it, not about routing it.
    assert!(single.route_detailed(&context, &one, 1).is_ok());
}

/// An unusable scale is refused, and the floor itself is admissible.
///
/// Would fail if a non-finite or non-positive scale fell through the
/// validity check, or if the floor refused the boundary value.
#[test]
fn an_unusable_exploration_scale_is_refused() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.9, 0.4]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into()];
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let error = router
            .route_explored(&[1.0, 0.5], &pool, 1, Some(Exploration { scale, seed: 1 }))
            .unwrap_err();
        assert!(
            matches!(error, RouterError::InvalidExplorationScale { .. }),
            "scale {scale} must fail the validity check, not fall through it: {error}"
        );
    }

    let error = router
        .route_explored(
            &[1.0, 0.5],
            &pool,
            1,
            Some(Exploration {
                scale: (MIN_EXPLORATION_SCALE / 2.0) as f32,
                seed: 1,
            }),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        RouterError::ExplorationScaleBelowFloor { floor, .. } if floor == MIN_EXPLORATION_SCALE
    ));
    // The floor itself is admissible: refusing at the boundary would make
    // the documented minimum a value nothing can use.
    assert!(
        router
            .route_explored(
                &[1.0, 0.5],
                &pool,
                1,
                Some(Exploration {
                    scale: MIN_EXPLORATION_SCALE as f32,
                    seed: 1,
                }),
            )
            .is_ok()
    );
}

/// The gate's own scores survive the draw, the sampled logits are the
/// perturbed ones, and the dropped set is reported as `route` reports it.
///
/// Would fail if `selected_scores` reported the perturbed values, or if the
/// draw recorded `z` instead of `z'`.
#[test]
fn the_gate_scores_survive_the_draw() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.3, 0.2, 0.1]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    // The deterministic arm takes every column so each adapter's gate score
    // is available to compare against, whichever two the draw picks; the
    // drawn arm takes two, because a uniform draw over the whole set cannot
    // reach the action and is refused.
    let deterministic = router.route_detailed(&context, &pool, 3).unwrap();
    let drawn = router
        .route_explored(
            &context,
            &pool,
            2,
            Some(Exploration {
                scale: 1.0,
                seed: 7,
            }),
        )
        .unwrap();
    let recorded = drawn.exploration.as_ref().unwrap();
    for (id, score) in &drawn.selected_scores {
        let expected = deterministic
            .selected_scores
            .iter()
            .find(|(other, _)| other == id)
            .map(|(_, score)| *score)
            .unwrap();
        assert_eq!(
            *score, expected,
            "selected_scores must stay the gate's own scores for {id}"
        );
    }
    // The full sampled logits are the perturbed values, not the gate's
    // own: compared against the deterministic arm's three scores.
    let gate_vector: Vec<f32> = deterministic
        .selected_scores
        .iter()
        .map(|(_, score)| *score)
        .collect();
    assert_eq!(gate_vector.len(), 3);
    assert_ne!(
        recorded.sampled_logits, gate_vector,
        "the recorded sampled logits must be the perturbed ones"
    );
    assert_eq!(recorded.scale, 1.0);
    assert_eq!(recorded.seed, 7);
    assert_eq!(recorded.sampled_logits.len(), 3, "all gate outputs");
    // `sampled_scores` is `z'` for the selected adapters, a slice of the
    // full `z'`.
    for (id, logit) in &drawn.sampled_scores {
        let index = pool.iter().position(|p| p == id).unwrap();
        assert_eq!(*logit, recorded.sampled_logits[index]);
    }
}

/// A draw is not a policy change: the epoch and the router's own state are
/// untouched by drawing.
///
/// Would fail if `route_explored` mutated the router's policy fields.
#[test]
fn a_draw_is_not_a_policy_change() {
    let mut router = AdapterRouter::new(scored_gate(2, &[0.3, 0.2, 0.1]));
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let start = router.policy_epoch();
    let before = router.route_detailed(&[1.0, 0.5], &pool, 2).unwrap();
    router
        .route_explored(
            &[1.0, 0.5],
            &pool,
            2,
            Some(Exploration {
                scale: 1.0,
                seed: 3,
            }),
        )
        .unwrap();
    let after = router.route_detailed(&[1.0, 0.5], &pool, 2).unwrap();
    assert_eq!(
        before, after,
        "drawing must not change the deterministic decision"
    );
    assert_eq!(
        router.policy_epoch(),
        start,
        "sampling an action is not a policy change, so a draw must not advance the epoch"
    );
}

#[test]
fn a_fresh_router_serves_the_initial_epoch() {
    assert_eq!(make_router(2, 2).policy_epoch(), INITIAL_POLICY_EPOCH);
}

/// A successful reload replaces the gate a request runs, so it ends the
/// epoch; a refused one changes nothing and must leave the epoch where it
/// was, or an off-epoch episode would be dated to a gate that never served
/// it.
#[test]
fn a_successful_reload_advances_the_epoch_and_a_failed_one_does_not() {
    let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
    let before = router.policy_epoch();
    router.reload(&fixed_gate(2, 2, 1).to_bytes()).unwrap();
    let after = router.policy_epoch();
    assert_eq!(after, before + 1);
    router.reload(&fixed_gate(3, 2, 1).to_bytes()).unwrap_err();
    assert_eq!(
        router.policy_epoch(),
        after,
        "a refused gate changes nothing a request runs"
    );
}

/// Each setter that changes what a request runs advances the epoch once,
/// and re-applying the same configuration does not: a server that
/// re-applies its config per request must still accumulate on-policy
/// samples.
#[test]
fn a_policy_change_advances_the_epoch_and_re_applying_it_does_not() {
    let mut router = make_router(2, 2);
    let start = router.policy_epoch();
    router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    assert_eq!(router.policy_epoch(), start + 1);
    router.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    assert_eq!(
        router.policy_epoch(),
        start + 1,
        "the same policy again is not a policy change"
    );
    router.set_epsilon(0.1);
    assert_eq!(router.policy_epoch(), start + 2);
    router.set_epsilon(0.1);
    assert_eq!(router.policy_epoch(), start + 2);
    router.set_tau_floor(0.5);
    assert_eq!(router.policy_epoch(), start + 3);
    router.set_tau_floor(0.5);
    assert_eq!(router.policy_epoch(), start + 3);
}

/// The refit guards decide which future gate is accepted and change no
/// served action, so they must not end the epoch of the samples they are
/// about to judge.
#[test]
fn the_refit_guards_do_not_advance_the_epoch() {
    let mut router = make_router(2, 2);
    let start = router.policy_epoch();
    router.set_entropy_floor(0.9);
    router.set_max_consecutive_collapsed(1);
    assert_eq!(
        router.policy_epoch(),
        start,
        "a guard on which future gate is accepted changes no served action"
    );
}

/// An accepted refit loads a gate and advances the epoch through `reload`;
/// a rejected one loads nothing and leaves the epoch alone.
#[test]
fn a_refit_advances_the_epoch_only_when_it_is_accepted() {
    let mut router = AdapterRouter::new(fixed_gate(2, 2, 0));
    let candidate = fixed_gate(2, 2, 1).to_bytes();
    let start = router.policy_epoch();
    assert!(matches!(
        router.submit_refit(&candidate, &[vec![0.5, 0.5]]).unwrap(),
        RefitOutcome::Accepted { .. }
    ));
    assert_eq!(router.policy_epoch(), start + 1);

    router.set_entropy_floor(1.0);
    router.set_max_consecutive_collapsed(1);
    let after_config = router.policy_epoch();
    assert!(matches!(
        router.submit_refit(&candidate, &[vec![1.0, 0.0]]).unwrap(),
        RefitOutcome::Rejected { .. }
    ));
    assert_eq!(
        router.policy_epoch(),
        after_config,
        "a rejected refit loads no gate, so it is still the same policy"
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

/// The seed is a reproducibility contract, so these values are pinned.
///
/// They were produced by an independent reimplementation of FNV-1a plus
/// `splitmix64` rather than by running this function and pasting what it
/// said, which would pin the code to itself and pass through any change to
/// either constant.
#[test]
fn the_completion_seed_is_pinned_and_id_specific() {
    assert_eq!(
        seed_from_completion_id("chatcmpl-1789991322-0"),
        0x03f4_6001_3651_6fde
    );
    assert_eq!(
        seed_from_completion_id("chatcmpl-1789991322-1"),
        0x39d7_2029_2048_2ee4
    );
    // The empty id is not special-cased anywhere, so it has a value like
    // any other; pinned so that a future guard against it is a deliberate
    // change rather than a silent one.
    assert_eq!(seed_from_completion_id(""), 0xc381_7c01_6ba4_ff30);

    // Ids differing in one trailing character must not land near each
    // other: FNV alone leaves those correlated in the low bits, which is
    // what the splitmix round is for. Sequential ids are the ONLY shape
    // this server mints, so this is the case that matters.
    let seeds: Vec<u64> = (0..64)
        .map(|i| seed_from_completion_id(&format!("chatcmpl-1789991322-{i}")))
        .collect();
    let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
    assert_eq!(unique.len(), seeds.len(), "sequential ids collided");
    let close = seeds
        .windows(2)
        .filter(|w| w[0].abs_diff(w[1]) < (1u64 << 48))
        .count();
    assert!(
        close <= 1,
        "{close} adjacent id pairs produced seeds within 2^48 of each other"
    );
}

/// The `router init --tau` claim, measured rather than asserted in prose:
/// over a ZERO gate a softmax bootstrap serves exactly the same mixture a
/// uniform one does, and a draw over it still moves the weights.
///
/// Both halves are load-bearing and they pull in opposite directions. If
/// the first failed, minting a softmax bootstrap would silently change what
/// a baseline serves. If the second failed, the flag would mint an artifact
/// whose exploration record described variation the action never had -- the
/// exact defect `a_draw_that_cannot_change_the_action_is_refused` exists to
/// prevent, arriving through a different door.
#[test]
fn a_softmax_over_a_zero_gate_serves_uniform_and_is_still_explorable() {
    let pool: Vec<AdapterId> = vec!["a".into(), "b".into(), "c".into()];
    let context = [1.0, 0.5];
    let k = pool.len();

    let mut uniform = AdapterRouter::new(scored_gate(2, &[0.0, 0.0, 0.0]));
    let baseline = uniform.route(&context, &pool, k).unwrap();

    let mut softmax = AdapterRouter::new(scored_gate(2, &[0.0, 0.0, 0.0]));
    softmax.set_weight_policy(WeightPolicy::Softmax { tau: 1.0 });
    let served = softmax.route(&context, &pool, k).unwrap();

    assert_eq!(baseline.len(), served.len());
    for ((left_id, left), (right_id, right)) in baseline.iter().zip(served.iter()) {
        assert_eq!(left_id, right_id);
        assert!(
            (left - right).abs() < 1e-6,
            "a softmax over equal logits must serve 1/k: {left} vs {right}"
        );
    }

    // And the draw reaches it. Distinct weight vectors across seeds, on the
    // same gate, at the same k that the uniform policy refuses.
    let mut distinct = std::collections::HashSet::new();
    for seed in 0..32u64 {
        let outcome = softmax
            .route_explored(&context, &pool, k, Some(Exploration { scale: 1.0, seed }))
            .expect("a softmax draw at k == n is reachable");
        let shape: Vec<String> = outcome
            .weights
            .iter()
            .map(|(id, w)| format!("{id}:{w:.6}"))
            .collect();
        distinct.insert(shape.join(","));
    }
    assert!(
        distinct.len() > 1,
        "the draw must move the weights, got one shape across 32 seeds"
    );

    // The control the claim rests on: the same k on the uniform policy is
    // refused, so the reachability above is the policy's doing and not the
    // seed's.
    assert!(
        uniform
            .route_explored(
                &context,
                &pool,
                k,
                Some(Exploration {
                    scale: 1.0,
                    seed: 0
                })
            )
            .is_err()
    );
}

/// Re-applying a NaN floor is not a policy change, and neither is
/// re-applying any other identical floor. `!=` would charge a bump to every
/// repeat of a NaN because NaN compares unequal to itself, which ends the
/// epoch of the samples a server that re-applies its configuration per
/// request is trying to accumulate.
#[test]
fn re_applying_a_nan_floor_does_not_advance_the_epoch() {
    for setter in [
        AdapterRouter::set_epsilon as fn(&mut AdapterRouter, f32),
        AdapterRouter::set_tau_floor as fn(&mut AdapterRouter, f32),
    ] {
        let mut router = make_router(2, 2);
        let start = router.policy_epoch();
        setter(&mut router, f32::NAN);
        assert_eq!(
            router.policy_epoch(),
            start + 1,
            "the first NaN is a change from whatever was there before"
        );
        setter(&mut router, f32::NAN);
        setter(&mut router, f32::NAN);
        assert_eq!(
            router.policy_epoch(),
            start + 1,
            "re-applying the same NaN must not advance the epoch again"
        );
        // A different value is still a change, so the comparison did not
        // simply stop noticing.
        setter(&mut router, 0.25);
        assert_eq!(router.policy_epoch(), start + 2);
        setter(&mut router, 0.25);
        assert_eq!(router.policy_epoch(), start + 2);
    }
}
