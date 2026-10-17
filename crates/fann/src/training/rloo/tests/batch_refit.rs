//! Batch refit over recorded explored actions (ADR-096 decision 3).

use super::*;

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
            .step_with_explored_action(&mut gate, &CTX, &batch_action(&served, &w_full), advantage)
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
/// an outcome, and a NaN reward would reach the policy term. The bad reward
/// is the SECOND sample, so an apply-as-you-go validation would already
/// have moved the gate by the time it noticed.
#[test]
fn an_empty_round_and_a_non_finite_reward_are_refused() {
    let w_full = [0.6_f32, 0.0, 0.4, 0.0];
    let base = test_gate();
    let mut trainer = RlooTrainer::with_seed(RlooConfig::default(), 1);
    let mut gate = test_gate();

    assert!(
        trainer
            .step_batch_with_explored_actions(&mut gate, &[])
            .is_err()
    );

    for reward in [f32::NAN, f32::INFINITY] {
        // A drifted trainer, so the EMA assertion below has content to
        // compare and not just two empty vectors.
        let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
        let z = gate.forward(&CTX).unwrap().to_vec();
        let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let other: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let before = gate.to_bytes();
        let freqs_before = trainer.route_freqs.clone();
        assert!(
            !freqs_before.is_empty(),
            "fixture: the EMA must be sized for the comparison to mean anything"
        );
        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&served, &w_full),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&other, &w_full),
                reward,
            },
        ];
        let err = trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap_err();
        assert!(
            err.to_string().contains("reward"),
            "reward {reward} must be refused and named, got {err}"
        );
        assert_eq!(gate.to_bytes(), before, "reward {reward} moved the gate");
        assert_eq!(
            trainer.route_freqs, freqs_before,
            "reward {reward} folded traffic"
        );
    }
}

/// The batch refuses every malformed action, not only a bad sampled logit:
/// a non-finite `w_full` entry, a non-finite reward, a non-finite sampled
/// logit, a `tau` whose reciprocal overflows, and a `w_full` of the wrong
/// width. Each bad sample sits at index 1 — the non-zero index — and the
/// gate's weights and the traffic EMA are asserted byte-identical across the
/// refusal, so an apply-as-you-go validation would redden even though the
/// first sample is valid.
///
/// Fails if any of the shared checks is dropped, or if the batch validates
/// inside the apply loop rather than before it.
#[test]
fn the_batch_refuses_a_malformed_action_before_applying_any_of_it() {
    let config = RlooConfig {
        learning_rate: 0.5,
        aux_loss_coeff: 1.0,
        z_loss_coeff: 0.1,
    };
    let base = test_gate();
    let good_w = [0.6_f32, 0.0, 0.4, 0.0];
    let nan_w = [0.6_f32, 0.0, f32::NAN, 0.0];
    let inf_w = [0.6_f32, f32::INFINITY, 0.4, 0.0];
    let short_w = [0.5_f32, 0.5];
    let bad_tau = f32::from_bits(1);

    // (label, sampled-logit index to poison (or None), w_full, tau, reward)
    type Case<'a> = (&'a str, Option<usize>, &'a [f32], f32, f32);
    let cases: [Case<'_>; 6] = [
        ("nan w_full", None, &nan_w, 0.5, -1.0),
        ("infinite w_full", None, &inf_w, 0.5, -1.0),
        ("w_full wrong width", None, &short_w, 0.5, -1.0),
        ("nan sampled logit", Some(2), &good_w, 0.5, -1.0),
        ("tau with infinite reciprocal", None, &good_w, bad_tau, -1.0),
        ("nan reward", None, &good_w, 0.5, f32::NAN),
    ];

    for (label, poison_index, bad_w, tau, reward) in cases {
        let (mut gate, mut trainer) = drifted_off_uniform(&base, config.clone());
        let z = gate.forward(&CTX).unwrap().to_vec();
        let first_served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let mut second_served: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        if let Some(j) = poison_index {
            second_served[j] = f32::NAN;
        }
        let before = gate.to_bytes();
        let freqs_before = trainer.route_freqs.clone();

        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&first_served, &good_w),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: ExploredAction {
                    sampled_logits: &second_served,
                    exploration_scale: 1.0,
                    w_full: bad_w,
                    tau,
                },
                reward,
            },
        ];
        let err = trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap_err();
        assert_eq!(
            gate.to_bytes(),
            before,
            "{label}: the refused round moved the gate ({err})"
        );
        assert_eq!(
            trainer.route_freqs, freqs_before,
            "{label}: the refused round folded traffic"
        );
    }
}

/// A round of finite rewards whose leave-one-out advantage overflows is
/// refused before anything is applied.
///
/// `[3e38, -3e38, 3e38]` is all finite as `f32`. The leave-one-out mean of
/// the other two rewards is computed in `f64` and is finite
/// (`3.0000000055e38` for the middle sample), but the advantage
/// `-3e38 - 3.0000000055e38` overflows `f32` to `-inf`, and that infinite
/// advantage reaches the policy term and poisons the gate. Fails if the
/// advantage finiteness check is dropped.
#[test]
fn a_round_with_an_overflowing_advantage_is_refused() {
    let w_full = [0.6_f32, 0.0, 0.4, 0.0];
    let base = test_gate();
    let rewards = [3e38_f32, -3e38, 3e38];
    assert!(
        rewards.iter().all(|r| r.is_finite()),
        "fixture: the rewards must all be finite"
    );
    assert!(
        leave_one_out_advantages(&rewards)
            .iter()
            .any(|a| !a.is_finite()),
        "fixture: an advantage must overflow"
    );

    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
    let before = gate.to_bytes();
    let freqs_before = trainer.route_freqs.clone();
    let samples: Vec<MixtureSample<'_>> = rewards
        .iter()
        .map(|r| MixtureSample {
            context: &CTX,
            action: batch_action(&served, &w_full),
            reward: *r,
        })
        .collect();
    let err = trainer
        .step_batch_with_explored_actions(&mut gate, &samples)
        .unwrap_err();
    assert!(
        err.to_string().contains("advantage"),
        "the overflow must be named as an advantage, got {err}"
    );
    assert_eq!(gate.to_bytes(), before, "the refused round moved the gate");
    assert_eq!(
        trainer.route_freqs, freqs_before,
        "the refused round folded traffic"
    );
}

/// A finite reward that overflows its policy term refuses the WHOLE round,
/// at a non-zero index, leaving the gate and the traffic EMA unchanged.
///
/// Four samples carry rewards `[0, 0, 1e38, 0]`. The third sample's
/// leave-one-out advantage is `1e38` (the other three rewards are zero, so
/// their mean is zero), while the others have advantages of `-3.333e37`.
/// At `s = 1` with `z' = z + 4` the third sample's policy term is
/// `1e38 * -4`, past the `f32` ceiling; the same term for an ordinary
/// sample is `-3.333e37 * -0.5 = 1.667e37`, finite. The fixture asserts
/// both, so the refusal cannot be an accident of the whole round being
/// non-finite.
///
/// Fails if the batch applies as it accumulates (the first two samples
/// would already have moved the gate) or if a non-finite candidate is
/// written.
#[test]
fn a_finite_reward_that_overflows_at_a_non_zero_index_refuses_the_whole_round() {
    let w_full = [0.5_f32, 0.0, 0.5, 0.0];
    let base = test_gate();
    let rewards = [0.0_f32, 0.0, 1e38, 0.0];
    let advantages = leave_one_out_advantages(&rewards);
    assert!(
        advantages.iter().all(|a| a.is_finite()),
        "fixture: every advantage must be finite, got {advantages:?}"
    );
    assert_eq!(
        advantages[2], rewards[2],
        "fixture: the third sample's advantage must be the big reward"
    );
    assert!(
        (advantages[0] * 0.5).is_finite(),
        "fixture: an ordinary sample's policy term must be finite, got {}",
        advantages[0] * 0.5
    );

    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    // Ordinary offsets for the three zero-reward samples; the big reward
    // gets an offset large enough that its advantage times it overflows.
    let served: Vec<Vec<f32>> = vec![
        z.iter().map(|zj| zj + 0.5).collect(),
        z.iter().map(|zj| zj - 0.5).collect(),
        z.iter().map(|zj| zj + 4.0).collect(),
        z.iter().map(|zj| zj - 0.5).collect(),
    ];

    let samples: Vec<MixtureSample<'_>> = (0..4)
        .map(|i| MixtureSample {
            context: &CTX,
            action: batch_action(&served[i], &w_full),
            reward: rewards[i],
        })
        .collect();

    let before = (gate.to_bytes(), trainer.route_freqs.clone());
    let err = trainer
        .step_batch_with_explored_actions(&mut gate, &samples)
        .unwrap_err();
    assert!(
        err.to_string().contains("layer"),
        "the refusal must name the layer: {err}"
    );
    assert_eq!(
        (gate.to_bytes(), trainer.route_freqs.clone()),
        before,
        "the refused round wrote something"
    );
}

/// A zero-advantage episode in an informative round still contributes its
/// regularisers: with both guards on, changing only that episode's `w_full`
/// changes the applied update; with both guards off it does not, because a
/// zero advantage leaves nothing else of that episode to apply.
///
/// The round is `(+1, -1, 0)`, so the third advantage is exactly zero and
/// its policy term is zero whatever its action. The only channel from that
/// episode to the gate is then its load-balance and z-loss terms. Fails if
/// the batch zeroes the output deltas of zero-advantage episodes: the two
/// guards-on arms would then apply identical updates.
#[test]
fn a_zero_advantage_episode_still_contributes_its_regularisers() {
    // `w_full` is the load-balance term's probability input; the two
    // vectors below are chosen to give it visibly different gradients at
    // the trainer's drifted traffic, so the arms cannot agree by accident.
    let w_full_a = [0.6_f32, 0.0, 0.4, 0.0];
    let w_full_b = [0.25_f32, 0.75, 0.0, 0.0];
    let base = test_gate();
    let rewards = [1.0_f32, -1.0, 0.0];
    assert_eq!(
        leave_one_out_advantages(&rewards),
        vec![1.5, -1.5, 0.0],
        "fixture: the third episode must have a zero advantage"
    );

    let run = |config: RlooConfig, w_full_idle: &[f32]| -> Vec<u8> {
        let (mut gate, mut trainer) = drifted_off_uniform(&base, config);
        let z = gate.forward(&CTX).unwrap().to_vec();
        let first: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
        let second: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
        let third: Vec<f32> = z.iter().map(|zj| zj + 0.25).collect();
        let samples = [
            MixtureSample {
                context: &CTX,
                action: batch_action(&first, &w_full_a),
                reward: 1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&second, &w_full_a),
                reward: -1.0,
            },
            MixtureSample {
                context: &CTX,
                action: batch_action(&third, w_full_idle),
                reward: 0.0,
            },
        ];
        let outcome = trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        assert_eq!(
            outcome.informative, 2,
            "fixture: the round must be informative, so the idle episode is \
                 averaged into a real step"
        );
        gate.to_bytes()
    };

    // Both guards on: the idle episode's regularisers are what differ.
    let guards_on = RlooConfig::default();
    assert!(
        guards_on.aux_loss_coeff != 0.0 && guards_on.z_loss_coeff != 0.0,
        "fixture: the default guards must be non-zero"
    );
    assert_ne!(
        run(guards_on.clone(), &w_full_a),
        run(guards_on, &w_full_b),
        "the zero-advantage episode's regularisers did not reach the gate"
    );

    // Both guards off: the idle episode has nothing left to contribute, so
    // the two arms agree and the difference above is attributable to the
    // regularisers alone.
    let guards_off = RlooConfig {
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
        ..RlooConfig::default()
    };
    assert_eq!(
        run(guards_off.clone(), &w_full_a),
        run(guards_off, &w_full_b),
        "with both guards off the idle episode must contribute nothing"
    );
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

/// The round's step is the mean over ALL `n` samples, not over the ones
/// that contributed: a three-episode round with one idle episode applies
/// `2/3` of the delta the same two contributing episodes apply alone.
///
/// Fails if the divisor is the contributor count instead of `n`. The two
/// runs below carry the SAME two advantage-weighted gradients — rewards
/// `(+1, -1, 0)` give advantages `(+1.5, -1.5, 0)`, and `(+0.75, -0.75)`
/// gives `(+1.5, -1.5)` on the same two actions — so the ratio of the
/// applied deltas is exactly the ratio of the divisors, `2/3`. Under a
/// contributor-mean divisor the three-episode round would divide by `2` and
/// the ratio would be `1`.
///
/// Both guards are off and the gate is a single Linear layer, so the idle
/// episode's gradient is exactly zero, the two runs' numerator is
/// identical, and the applied output-layer delta is exactly recoverable
/// from the moved forward output. That isolates the divisor. (With the
/// guards on, the idle episode's aux and z-loss gradients would also be
/// summed and divided by `n`, which is the other half of what averaging
/// over the round means.)
#[test]
fn the_step_is_averaged_over_the_whole_round() {
    let w_full = [0.6_f32, 0.0, 0.4, 0.0];
    let base = linear_gate();
    let config = RlooConfig {
        learning_rate: 0.5,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
    };
    let z = base.clone().forward(&CTX).unwrap().to_vec();
    let first: Vec<f32> = z.iter().map(|zj| zj + 0.5).collect();
    let second: Vec<f32> = z.iter().map(|zj| zj - 0.5).collect();
    assert_eq!(
        leave_one_out_advantages(&[1.0, -1.0, 0.0]),
        vec![1.5, -1.5, 0.0],
        "fixture assumption"
    );
    assert_eq!(
        leave_one_out_advantages(&[0.75, -0.75]),
        vec![1.5, -1.5],
        "fixture assumption"
    );

    let run = |rewards: &[f32]| {
        let mut gate = base.clone();
        let mut trainer = RlooTrainer::with_seed(config.clone(), 1);
        let before = gate.forward(&CTX).unwrap().to_vec();
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
        let after = gate.forward(&CTX).unwrap().to_vec();
        (
            outcome,
            applied_delta(&before, &after, config.learning_rate),
        )
    };

    let (three, delta_three) = run(&[1.0, -1.0, 0.0]);
    assert_eq!(three.samples, 3);
    assert_eq!(
        three.informative, 2,
        "the idle episode is not a contributor"
    );
    let (two, delta_two) = run(&[0.75, -0.75]);
    assert_eq!(two.samples, 2);
    assert_eq!(two.informative, 2);

    for (j, (&d3, &d2)) in delta_three.iter().zip(delta_two.iter()).enumerate() {
        assert!(
            d2.abs() > 1e-6,
            "index {j}: the two-episode delta must be non-zero to take a ratio"
        );
        let ratio = d3 / d2;
        assert!(
            (ratio - 2.0 / 3.0).abs() < 1e-4,
            "index {j}: the three-episode round must divide by 3 and the \
                 two-episode round by 2, a ratio of 2/3, got {ratio} \
                 (three {d3}, two {d2})"
        );
    }
}

/// The round's policy term is continuous in the rewards: moving one
/// judgement by a hair moves the policy term by a hair, because the divisor
/// is `n` whatever the advantages do. A contributor-mean divisor would put
/// a step discontinuity at every exact reward tie.
///
/// The update as a whole is not continuous at the all-equal boundary: a
/// round whose rewards are all equal has every advantage zero and applies
/// nothing, while a round a hair off the tie has a non-zero advantage, so
/// the regularisers (aux and z-loss) enter and the step moves by their mean
/// over the round. That discontinuity comes from ADR-096's no-op rule for a
/// round with nothing to compare; it is a step in the applied update, not in
/// the policy term. This test zeroes both regularisers, so what it measures
/// is the policy term alone: the continuity the `1 / n` divisor buys when
/// the contributor count changes.
///
/// Rewards `[1, 2, 3]` give advantages `[-1.5, 0, 1.5]` — two contributors
/// out of three — and rewards `[1, 2, 3.001]` give three non-zero
/// advantages. Under the `1 / n` divisor the applied deltas differ by under
/// a tenth of a percent; under a contributor-mean divisor the same pair
/// divides by 2 and by 3 and the deltas differ by a third. Fails if the
/// divisor is the contributor count.
///
/// The bound: this fixture's per-column delta is `0.125` at `1 / n`, and
/// the two runs differ by less than `1e-3` of that, well inside the `1e-3`
/// asserted. A contributor-mean mutant differs by `0.334` of it, over 300
/// times the bound.
#[test]
fn the_rounds_step_is_continuous_across_a_reward_tie() {
    let w_full = [0.6_f32, 0.0, 0.4, 0.0];
    let base = linear_gate();
    let config = RlooConfig {
        learning_rate: 0.5,
        aux_loss_coeff: 0.0,
        z_loss_coeff: 0.0,
    };
    let z = base.clone().forward(&CTX).unwrap().to_vec();
    let offsets: [&[f32]; 3] = [
        &[0.5, -0.5, 0.5, -0.5],
        &[-0.5, 0.5, -0.5, 0.5],
        &[0.25, -0.25, 0.25, -0.25],
    ];
    let served: Vec<Vec<f32>> = offsets
        .iter()
        .map(|o| z.iter().zip(o.iter()).map(|(zj, d)| zj + d).collect())
        .collect();

    assert_eq!(
        leave_one_out_advantages(&[1.0, 2.0, 3.0]),
        vec![-1.5, 0.0, 1.5],
        "fixture assumption: the exact tie is a zero advantage"
    );
    assert!(
        leave_one_out_advantages(&[1.0, 2.0, 3.001])
            .iter()
            .all(|a| *a != 0.0),
        "fixture assumption: the nudged round has three contributors"
    );

    let run = |rewards: [f32; 3]| -> Vec<f32> {
        let mut gate = base.clone();
        let mut trainer = RlooTrainer::with_seed(config.clone(), 1);
        let before = gate.forward(&CTX).unwrap().to_vec();
        let samples: Vec<MixtureSample<'_>> = rewards
            .iter()
            .enumerate()
            .map(|(i, r)| MixtureSample {
                context: &CTX,
                action: batch_action(&served[i], &w_full),
                reward: *r,
            })
            .collect();
        trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        let after = gate.forward(&CTX).unwrap().to_vec();
        applied_delta(&before, &after, config.learning_rate)
    };

    let exact = run([1.0, 2.0, 3.0]);
    let nudged = run([1.0, 2.0, 3.001]);
    let scale = exact.iter().fold(0.0_f32, |m, d| m.max(d.abs()));
    assert!(scale > 1e-3, "fixture must apply a non-zero delta");
    let change = exact
        .iter()
        .zip(nudged.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        change / scale < 1e-3,
        "a 1e-3 reward change must move the gate by under 1e-3 of the step, \
             got {change} of {scale} (the contributor-mean divisor would give about 1/3)"
    );
}

/// `mean_loss` divides the round's advantage-weighted loss by `n`, not by
/// the contributor count, in a round where an episode has a zero advantage.
///
/// Rewards `[1, -1, 0]` give advantages `[1.5, -1.5, 0]`, so the third
/// episode contributes nothing and `informative` is 2 of 3. The expected
/// value is computed here from the definition — each episode's advantage
/// times its squared distance from the gate's own logits over `2 s^2`,
/// summed and divided by `n` — rather than from the trainer's own path.
/// Fails if the divisor is the contributor count: `sum / 2` is `1.5`
/// times `sum / 3` whenever the sum is non-zero, and this fixture's sum is
/// non-zero (the two contributing episodes have different offsets, so
/// their losses do not cancel).
#[test]
fn the_rounds_mean_loss_divides_the_advantage_weighted_sum_by_n() {
    let w_full = [0.6_f32, 0.0, 0.4, 0.0];
    let base = test_gate();
    let rewards = [1.0_f32, -1.0, 0.0];
    let advantages = leave_one_out_advantages(&rewards);
    assert_eq!(
        advantages,
        vec![1.5, -1.5, 0.0],
        "fixture: the third episode must be idle"
    );

    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<Vec<f32>> = vec![
        z.iter().map(|zj| zj + 0.5).collect(),
        z.iter().map(|zj| zj - 0.25).collect(),
        z.iter().map(|zj| zj + 0.5).collect(),
    ];

    // Independent expected value: `sum_i A_i * ||z'_i - z||^2 / (2 s^2)`,
    // over `n`. `s = 1`, so `1 / (2 s^2) = 0.5`.
    let per_sample_loss = |a: f32, s: &[f32]| -> f32 {
        let sq: f32 = z
            .iter()
            .zip(s.iter())
            .map(|(zj, sj)| {
                let d = sj - zj;
                d * d
            })
            .sum();
        a * sq * 0.5
    };
    let total: f32 = advantages
        .iter()
        .zip(served.iter())
        .map(|(a, s)| per_sample_loss(*a, s))
        .sum();
    let expected = total / rewards.len() as f32;
    let contributor_mean = total / 2.0;
    assert!(
        expected.abs() > 0.0 && (contributor_mean - expected).abs() > 1e-6,
        "fixture: the two divisors must be distinguishable, \
             n-mean {expected}, contributor-mean {contributor_mean}"
    );

    let samples: Vec<MixtureSample<'_>> = rewards
        .iter()
        .enumerate()
        .map(|(i, r)| MixtureSample {
            context: &CTX,
            action: batch_action(&served[i], &w_full),
            reward: *r,
        })
        .collect();
    let outcome = trainer
        .step_batch_with_explored_actions(&mut gate, &samples)
        .unwrap();

    assert_eq!(outcome.samples, 3);
    assert_eq!(outcome.informative, 2);
    assert!(
        (outcome.mean_loss - expected).abs() <= 1e-6 * expected.abs().max(1.0),
        "mean_loss {} must be the round mean {expected}",
        outcome.mean_loss
    );
}

/// `mean_reward` is the raw reward sum over `n`, not over the contributors,
/// in a round where an episode has a zero advantage.
///
/// Rewards `[1, 2, 3]` give advantages `[-1.5, 0, 1.5]`, so `informative`
/// is 2 of 3 and the round mean is `2.0`, while a contributor mean would be
/// `3.0`. Fails if the divisor is the contributor count.
#[test]
fn the_rounds_mean_reward_is_the_reward_sum_over_n() {
    let w_full = [0.6_f32, 0.0, 0.4, 0.0];
    let base = test_gate();
    let rewards = [1.0_f32, 2.0, 3.0];
    let advantages = leave_one_out_advantages(&rewards);
    assert_eq!(
        advantages,
        vec![-1.5, 0.0, 1.5],
        "fixture: the middle episode must be idle"
    );

    let (mut gate, mut trainer) = drifted_off_uniform(&base, RlooConfig::default());
    let z = gate.forward(&CTX).unwrap().to_vec();
    let served: Vec<Vec<f32>> = vec![
        z.iter().map(|zj| zj + 0.5).collect(),
        z.iter().map(|zj| zj - 0.5).collect(),
        z.iter().map(|zj| zj + 0.25).collect(),
    ];
    let samples: Vec<MixtureSample<'_>> = rewards
        .iter()
        .enumerate()
        .map(|(i, r)| MixtureSample {
            context: &CTX,
            action: batch_action(&served[i], &w_full),
            reward: *r,
        })
        .collect();
    let outcome = trainer
        .step_batch_with_explored_actions(&mut gate, &samples)
        .unwrap();

    assert_eq!(outcome.samples, 3);
    assert_eq!(outcome.informative, 2);
    let expected = rewards.iter().sum::<f32>() / rewards.len() as f32;
    assert_eq!(
        outcome.mean_reward, expected,
        "mean_reward must be the reward sum over n, not over the contributors"
    );
    assert!(
        (expected - rewards.iter().sum::<f32>() / 2.0).abs() > 1e-6,
        "fixture: the two divisors must be distinguishable"
    );
}

/// The outcome reports the raw feedback and the size of the population
/// that contributed, separately.
///
/// This fixture has all four samples informative, so it does NOT separate a
/// round mean from a contributor mean: `mean_reward` is `0.0` either way.
/// What it pins is the shape of the two fields — four samples, four
/// contributors, a round mean of zero, and a finite loss — and the arm that
/// actually separates the two divisors is
/// `the_step_is_averaged_over_the_whole_round`, which carries an idle
/// episode.
#[test]
fn the_outcome_reports_the_round_size_the_contributor_count_and_a_round_mean() {
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
