use super::*;

#[test]
fn test_greedy_picks_argmax() {
    let config = SamplingConfig::greedy();
    let mut sampler = Sampler::new(config);
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    assert_eq!(sampler.sample(&logits), 3);
}

#[test]
fn test_temperature_zero_is_greedy() {
    let config = SamplingConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let mut sampler = Sampler::new(config);
    let logits = vec![1.0, 5.0, 2.0];
    assert_eq!(sampler.sample(&logits), 1);
}

#[test]
fn test_nonfinite_temperature_falls_back_to_argmax() {
    // top_k=2 forces the non-greedy fused-top-k path pre-fix, where a NaN/inf
    // temperature produced `inv_temp = NaN`/`0`, scaling every logit to NaN/0
    // and collapsing the result to token 0. The guard must instead return the
    // true argmax (token 1) for any non-finite temperature.
    let logits = vec![0.0, 100.0, 99.0];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let config = SamplingConfig {
            min_p: 0.0,
            temperature: bad,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config).with_seed(7);
        assert_eq!(
            sampler.sample(&logits),
            1,
            "temperature {bad} must fall back to argmax, not collapse to token 0"
        );
    }
}

#[test]
fn test_sampler_nan_fails_closed_to_argmax() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(config).with_seed(0x5eed_f00d_1234_5678);
    let logits = [0.0, f32::NAN, 0.0];

    assert_eq!(
        sampler.sample(&logits),
        0,
        "a NaN anywhere in the distribution must fall back to the finite argmax"
    );
}

#[test]
fn test_tiny_temperature_falls_back_to_argmax() {
    // A finite, positive, but tiny temperature whose reciprocal overflows f32 to
    // +inf (t < ~2.94e-39). Pre-fix `inv_temp = 1.0 / t = inf` scaled every logit
    // to +inf, destroying their ordering, and the fused top-k collapsed to token 0.
    // The t -> 0+ limit of softmax(logits / t) is argmax, so the highest-logit
    // token (1) must win. top_k=2 forces the non-greedy fused-top-k path.
    let logits = vec![10.0, 11.0, 9.0];
    for tiny in [1e-45_f32, 1e-40, 1e-39] {
        let config = SamplingConfig {
            min_p: 0.0,
            temperature: tiny,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config).with_seed(7);
        assert_eq!(
            sampler.sample(&logits),
            1,
            "tiny temperature {tiny} must fall back to argmax, not collapse to token 0"
        );
    }
}

#[test]
fn test_temperature_degenerate_predicate() {
    // Degenerate: non-finite, non-positive, finite-but-tiny where the reciprocal
    // overflows (1e-45..1e-39), and the residual band where the reciprocal is finite
    // but `logit * inv_temp` would overflow for a plausible logit (1e-38..1e-35,
    // including f32::MIN_POSITIVE whose reciprocal 8.5e37 exceeds f32::MAX / 1e4).
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -1.0,
        1e-45,
        1e-40,
        1e-39,
        f32::MIN_POSITIVE,
        1e-37,
        1e-35,
    ] {
        assert!(
            temperature_degenerate(bad),
            "temperature {bad} should be degenerate"
        );
    }
    // Valid: normal temperatures, plus small-but-safe values whose reciprocal stays
    // far enough below f32::MAX that scaling a plausible logit cannot overflow
    // (1e-30 -> inv 1e30, well under the f32::MAX / 1e4 ≈ 3.4e34 threshold).
    for ok in [1.0, 0.5, 2.0, 0.1, 1e-3, 1e-30] {
        assert!(
            !temperature_degenerate(ok),
            "temperature {ok} should be valid (no scaling overflow)"
        );
    }
}

#[test]
fn test_residual_band_temperature_falls_back_to_argmax() {
    // The residual band: a finite, positive temperature whose reciprocal stays finite
    // (so the pre-fix `!(1.0 / t).is_finite()` check missed it) yet is large enough
    // that `logit * inv_temp` overflows for ordinary logits. Pre-fix this saturated
    // every scaled logit to +inf and the softmax non-finite-max guard returned an
    // arbitrary candidate. argmax (token 1) is the correct t -> 0+ limit.
    let logits = vec![10.0, 11.0, 9.0];
    for band in [f32::MIN_POSITIVE, 1e-37_f32, 1e-36] {
        let config = SamplingConfig {
            min_p: 0.0,
            temperature: band,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config).with_seed(7);
        assert_eq!(
            sampler.sample(&logits),
            1,
            "residual-band temperature {band} must fall back to argmax, not collapse"
        );
    }
}

#[test]
fn test_apply_temperature_nonfinite_is_noop() {
    // A non-finite (or non-positive) temperature must leave logits untouched —
    // `1.0 / NaN` would poison the whole candidate set, `1.0 / inf` would zero it.
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0_f32,
        -2.0_f32,
    ] {
        let mut cs = CandidateSet::from_full_logits(&[1.0, 5.0, 3.0]);
        cs.apply_temperature(bad);
        let logits: Vec<f32> = cs.candidates.iter().map(|c| c.logit).collect();
        assert_eq!(
            logits,
            vec![1.0, 5.0, 3.0],
            "temperature {bad} must be a no-op, leaving logits finite and unscaled"
        );
        assert_eq!(cs.argmax(), 1);
    }
}

#[test]
fn apply_temperature_tiny_positive_temp_stays_greedy() {
    // A finite-but-tiny temperature has a reciprocal that overflows f32. The
    // t -> 0+ limit must stay greedy (argmax). Without the degenerate-branch
    // collapse, `1.0 / 1e-45 == +inf` scales both logits to +inf; the sort then
    // tie-breaks on ascending token_id and `sample_top_p` returns token 0 instead
    // of the argmax (token 1).
    let mut cs = CandidateSet::from_full_logits(&[10.0, 11.0]);
    cs.apply_temperature(1e-45);
    // top_p = 1.0 (no nucleus truncation) + r = 0.0 => pure argmax selection.
    assert_eq!(
        cs.sample_top_p(1.0, 0.0),
        1,
        "tiny positive temperature must resolve to the argmax (greedy), not token 0"
    );
}

#[test]
fn apply_temperature_tiny_temp_is_hard_greedy_for_close_logits() {
    // Even an arbitrarily small logit gap must resolve to hard argmax under a
    // degenerate temperature (the t -> 0+ limit), matching the main Sampler's
    // `temperature_degenerate` route. A finite multiplier (clamped or not) would
    // leave these two near 50/50, so `sample_top_p(1.0, 0.75)` could draw token 0;
    // the one-hot collapse makes the argmax (token 1) the only reachable draw.
    let mut cs = CandidateSet::from_full_logits(&[0.0, f32::from_bits(1)]);
    cs.apply_temperature(1e-45);
    assert_eq!(
        cs.sample_top_p(1.0, 0.75),
        1,
        "degenerate temperature must be hard-greedy even for adjacent logits"
    );
}

#[test]
fn test_top_k_limits_candidates() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(config).with_seed(123);
    // With top_k=2, only the two highest logits (idx 1 and 3) should be sampled
    let logits = vec![0.0, 10.0, 0.0, 9.0, 0.0];
    let mut counts = [0u32; 5];
    for _ in 0..100 {
        let tok = sampler.sample(&logits);
        counts[tok as usize] += 1;
    }
    // Only tokens 1 and 3 should appear
    assert_eq!(counts[0], 0);
    assert_eq!(counts[2], 0);
    assert_eq!(counts[4], 0);
    assert!(counts[1] > 0);
    assert!(counts[3] > 0);
}

#[test]
fn test_repetition_penalty_reduces_probability() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.0, // greedy
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 100.0, // very strong penalty
    };
    let mut sampler = Sampler::new(config);
    let logits = vec![0.0, 5.0, 4.9]; // token 1 is best

    // First sample: picks 1 (highest)
    let first = sampler.sample(&logits);
    assert_eq!(first, 1);

    // Second sample: token 1 is penalized, should pick 2
    let second = sampler.sample(&logits);
    assert_eq!(second, 2);
}

/// A repetition penalty in (0, 1) BOOSTS recent tokens. The greedy fast path
/// must not shortcut to the raw argmax when a boosted recent token would
/// overtake it, even though the raw argmax itself is not recent (the
/// fast-path invariant only holds for penalty >= 1.0).
#[test]
fn test_greedy_sub_one_penalty_boosts_recent_token() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.0, // greedy fast path
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 0.5, // < 1.0 boosts recent tokens
    };
    let mut sampler = Sampler::new(config);

    // Step 1: token 1 is the clear argmax; becomes a recent token.
    assert_eq!(sampler.sample(&[0.0, 10.0]), 1);

    // Step 2: raw argmax is token 0 (5.0) and is NOT recent, but recent token
    // 1 is boosted to 4.0 / 0.5 = 8.0 > 5.0, so the penalized argmax is token
    // 1. Before the fix the fast path returned the raw argmax token 0.
    assert_eq!(sampler.sample(&[5.0, 4.0]), 1);
}

/// The documented always-safe case the conservative gate leaves to the re-scan:
/// a (0, 1) penalty where the raw argmax is ITSELF recent. `penalized_logit` is
/// monotone, so boosting raw_best keeps it the argmax — the fall-through must
/// still return it. Regression guard for the comment at the gate.
#[test]
fn test_greedy_sub_one_penalty_recent_argmax_unchanged() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.0, // greedy fast path
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 0.5, // < 1.0 boosts recent tokens
    };
    let mut sampler = Sampler::new(config);

    // Step 1: token 1 is the clear argmax; becomes a recent token.
    assert_eq!(sampler.sample(&[0.0, 10.0]), 1);

    // Step 2: raw argmax is token 1 (6.0) and IS recent. Boosting it to
    // 6.0 / 0.5 = 12.0 keeps it above token 0's 5.0, so it stays the argmax.
    assert_eq!(sampler.sample(&[5.0, 6.0]), 1);
}

#[test]
fn test_top_p_nucleus_sampling() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5,
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(config).with_seed(456);
    // Token 0 has much higher logit, should dominate with top_p=0.5
    let logits = vec![10.0, 1.0, 1.0, 1.0, 1.0];
    let mut counts = [0u32; 5];
    for _ in 0..100 {
        counts[sampler.sample(&logits) as usize] += 1;
    }
    // Token 0 should get almost all samples (its probability >> 0.5)
    assert!(counts[0] > 90);
}

#[test]
fn top_n_sigma_masks_below_max_minus_population_stddev() {
    let mut logits = [4.0_f32, 3.0, 2.0, 1.0, f32::NEG_INFINITY];
    apply_top_n_sigma(&mut logits, 1.7);

    assert_eq!(logits[0], 4.0);
    assert_eq!(logits[1], 3.0);
    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert_eq!(logits[3], f32::NEG_INFINITY);
    assert_eq!(logits[4], f32::NEG_INFINITY);
}

#[test]
fn top_n_sigma_keeps_logits_equal_to_the_threshold() {
    let mut logits = [4.0_f32, 2.0];
    apply_top_n_sigma(&mut logits, 2.0);

    assert_eq!(logits, [4.0, 2.0]);
}

#[test]
fn top_n_sigma_defaults_disabled() {
    assert!(!top_n_sigma_active(0.0));

    let logits = [4.0_f32, 3.0, 2.0, 1.0];
    let mut unset = Sampler::new(SamplingConfig::default()).with_seed(7);
    let mut disabled = Sampler::new(SamplingConfig::default())
        .with_seed(7)
        .with_top_n_sigma(0.0);
    assert_eq!(unset.sample(&logits), disabled.sample(&logits));
}

#[test]
fn top_n_sigma_selection_is_temperature_invariant() {
    let mut raw = [4.0_f32, 3.0, 2.0, 1.0];
    let mut scaled = raw.map(|logit| logit / 2.5);
    apply_top_n_sigma(&mut raw, 1.0);
    apply_top_n_sigma(&mut scaled, 1.0);

    let raw_mask = raw.map(f32::is_finite);
    let scaled_mask = scaled.map(f32::is_finite);
    assert_eq!(raw_mask, scaled_mask);
}

#[test]
fn top_n_sigma_invalid_values_are_disabled_noops() {
    let original = [4.0_f32, 3.0, 2.0, 1.0];
    for disabled in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut logits = original;
        apply_top_n_sigma(&mut logits, disabled);
        assert_eq!(
            logits.map(f32::to_bits),
            original.map(f32::to_bits),
            "top_n_sigma={disabled:?} must preserve every logit bit"
        );
    }
}

#[test]
fn top_n_sigma_preserves_poisoned_distribution_fallbacks() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let mut nan_sampler = Sampler::new(config.clone())
        .with_seed(7)
        .with_top_n_sigma(1.0);
    assert_eq!(nan_sampler.sample(&[4.0, f32::NAN, 1.0]), 0);

    let mut infinity_sampler = Sampler::new(config.clone())
        .with_seed(7)
        .with_top_n_sigma(1.0);
    assert_eq!(infinity_sampler.sample(&[4.0, f32::INFINITY, 1.0]), 1);

    let mut masked_sampler = Sampler::new(config).with_seed(7).with_top_n_sigma(1.0);
    assert_eq!(masked_sampler.sample(&[f32::NEG_INFINITY; 3]), 0);
}

#[test]
fn top_n_sigma_interacts_with_the_existing_sampling_pipeline() {
    let logits = [10.0_f32, 9.5, 9.0, 8.0, 7.0, 2.0, 1.0, 0.0];
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 2.0,
        top_k: 6,
        top_p: 0.67,
        repetition_penalty: 2.0,
    };
    let mut active_seen = [false; 8];
    let mut disabled_seen = [false; 8];

    for index in 0..512_u64 {
        let seed = index
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut active_sampler = Sampler::new(config.clone())
            .with_seed(seed)
            .with_min_p(0.02)
            .with_top_n_sigma(1.5);
        active_sampler.seed_history(&[0]);
        active_seen[active_sampler.sample(&logits) as usize] = true;

        let mut disabled_sampler = Sampler::new(config.clone())
            .with_seed(seed)
            .with_min_p(0.02)
            .with_top_n_sigma(0.0);
        disabled_sampler.seed_history(&[0]);
        disabled_seen[disabled_sampler.sample(&logits) as usize] = true;
    }

    assert_eq!(
        active_seen,
        [false, true, true, false, false, false, false, false],
        "penalty -> top-n-sigma -> temperature -> top-k -> min-p -> top-p \
             must leave exactly tokens 1 and 2 reachable"
    );
    assert!(
        disabled_seen[3],
        "the disabled control must reach token 3 or the top-n-sigma route assertion is vacuous"
    );
}

#[test]
fn top_n_sigma_runs_after_repetition_penalty() {
    let mut sampler = Sampler::new(SamplingConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 2.0,
    })
    .with_seed(7)
    .with_top_n_sigma(1.0);
    sampler.seed_history(&[0]);

    assert_eq!(sampler.sample(&[10.0, 9.0, 0.0, 0.0]), 1);
    assert_eq!(sampler.logit_scratch[0], f32::NEG_INFINITY);
    assert_eq!(sampler.logit_scratch[1], 9.0);
}

#[test]
fn sample_full_logits_top_n_sigma_masks_candidates() {
    let logits = [10.0_f32, 9.5, 9.0, 8.0, 7.0, 2.0, 1.0, 0.0];
    let cfg = GenerateConfig {
        temperature: 2.0,
        top_k: 6,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    let mut active_seen = [false; 8];
    let mut disabled_seen = [false; 8];
    for index in 0..512_u64 {
        let mut active_rng = index
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407)
            | 1;
        let mut disabled_rng = active_rng;
        active_seen[sample_full_logits(&logits, &cfg, &[], &mut active_rng, 0.0, 1.5) as usize] =
            true;
        disabled_seen
            [sample_full_logits(&logits, &cfg, &[], &mut disabled_rng, 0.0, 0.0) as usize] = true;
    }

    // mean = 5.8125, population stddev ~= 3.856, threshold = 10 - 1.5 * 3.856
    // ~= 4.22: tokens 0-4 survive, tokens 5-7 (logits 2, 1, 0) are masked.
    assert!(
        active_seen.iter().skip(5).all(|&seen| !seen),
        "top_n_sigma=1.5 must mask every token below max - 1.5 * stddev; saw {active_seen:?}"
    );
    assert!(
        disabled_seen[5],
        "the disabled control (top_k = 6 keeps tokens 0-5) must reach token 5, \
             which the filter masks, or the assertion above is vacuous; saw {disabled_seen:?}"
    );
}

#[test]
fn sample_full_logits_top_k_one_stays_argmax_with_top_n_sigma() {
    let logits = [1.0_f32, 4.0, 2.0, 3.0];
    let cfg = GenerateConfig {
        temperature: 0.8,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    for top_n_sigma in [0.0, 0.5, 100.0] {
        let mut rng = 7_u64;
        assert_eq!(
            sample_full_logits(&logits, &cfg, &[], &mut rng, 0.0, top_n_sigma),
            1,
            "top_k == 1 must select the argmax regardless of top_n_sigma={top_n_sigma}"
        );
    }
}

#[test]
fn min_p_filters_by_probability_relative_to_maximum() {
    let candidates = || {
        CandidateSet::from_candidates(vec![
            Candidate {
                token_id: 7,
                logit: 0.0,
            },
            Candidate {
                token_id: 8,
                logit: 0.49_f32.ln(),
            },
            Candidate {
                token_id: 9,
                logit: 0.1_f32.ln(),
            },
        ])
    };

    assert_eq!(
        candidates().sample_min_p_top_p(0.0, 1.0, 0.8),
        8,
        "control draw must reach the 0.49-relative-probability candidate"
    );
    assert_eq!(
        candidates().sample_min_p_top_p(0.5, 1.0, 0.8),
        7,
        "min_p=0.5 must remove every candidate below half of max probability"
    );
}

#[test]
fn min_p_nan_is_a_documented_disabled_noop() {
    let candidates = || {
        CandidateSet::from_candidates(vec![
            Candidate {
                token_id: 7,
                logit: 0.0,
            },
            Candidate {
                token_id: 8,
                logit: 0.49_f32.ln(),
            },
        ])
    };

    assert_eq!(
        candidates().sample_min_p_top_p(f32::NAN, 1.0, 0.8),
        candidates().sample_min_p_top_p(0.0, 1.0, 0.8),
        "NaN min-p must preserve the explicitly disabled distribution"
    );
}

/// Reproduces the reviewer's mutation for #1263: changing the shared
/// `if min_p > 0.0` branch condition in `sample_min_p_top_p_with_scratch`
/// to `if false` left the *previous* version of this test green, because
/// both top-k survivors cleared `min_p` and top-p kept both regardless.
/// A test that cannot fail when min-p is disabled proves nothing about
/// min-p, so the assertion here is chosen to require min-p's exclusion:
/// see [`min_p_interacts_with_penalty_temperature_top_k_and_top_p`] for
/// the derivation and the discriminating replacement.
#[test]
fn min_p_interacts_with_penalty_temperature_and_top_k() {
    let logits = [8.0, 3.0, 0.0];

    // penalty=4 on token 0 (raw logit 8.0 > 0, so 8.0 / 4.0 = 2.0) drops it
    // below token 1 (3.0), so top-k=2 keeps {token 1: 3.0, token 0: 2.0}
    // and excludes token 2 (0.0) regardless of min-p.
    for seed in [0x1234_5678_9abc_def0, 0x9e37_79b9_7f4a_7c15] {
        let mut sampler = Sampler::new(SamplingConfig {
            min_p: 0.0,
            temperature: 2.0,
            top_k: 2,
            top_p: 0.9,
            repetition_penalty: 4.0,
        })
        .with_seed(seed)
        .with_min_p(0.7);
        sampler.seed_history(&[0]);
        let token = sampler.sample(&logits);
        assert_ne!(
            token, 2,
            "penalty → temperature → top-k must exclude token 2"
        );
    }
}

/// Discriminating replacement for the #1263 finding: after penalty (4x on
/// token 0), temperature (2.0) and top-k (2), the surviving candidates are
/// token 1 (scaled logit 1.5) and token 0 (scaled logit 1.0). Their
/// relative weight is `exp(1.0 - 1.5) = exp(-0.5) ≈ 0.6065`, which sits
/// BELOW `min_p = 0.7`: min-p prunes token 0, leaving only token 1
/// reachable. With min-p disabled (0.0), both survive top-p=0.9
/// (`token 1` gets ≈0.6225 of the renormalised mass, `token 0` the rest),
/// and a draw with `r >= 0.6225` reaches token 0.
///
/// The two seeds below were chosen (via the crate's own
/// `xorshift64_next`/`uniform_f32_from_u64`) to produce first draws
/// `r ≈ 0.9941` and `r ≈ 0.8598`, both comfortably above the 0.6225 cut —
/// so disabling min-p changes the sampled token for both, which is the
/// discriminating behavior a min-p test must exercise.
#[test]
fn min_p_interacts_with_penalty_temperature_top_k_and_top_p() {
    let logits = [8.0, 3.0, 0.0];
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 2.0,
        top_k: 2,
        top_p: 0.9,
        repetition_penalty: 4.0,
    };

    for seed in [0x1234_5678_9abc_def0, 0x9e37_79b9_7f4a_7c15] {
        let mut sampler_on = Sampler::new(config.clone()).with_seed(seed).with_min_p(0.7);
        sampler_on.seed_history(&[0]);
        let token_on = sampler_on.sample(&logits);

        let mut sampler_off = Sampler::new(config.clone()).with_seed(seed).with_min_p(0.0);
        sampler_off.seed_history(&[0]);
        let token_off = sampler_off.sample(&logits);

        assert_eq!(
            token_on, 1,
            "min_p=0.7 must prune token 0 (weight ≈0.6065 < 0.7), \
                 leaving only token 1 reachable"
        );
        assert_eq!(
            token_off, 0,
            "with min_p disabled, top-p=0.9 keeps both survivors and this \
                 seed's draw (r >= 0.6225) must reach token 0"
        );
        assert_ne!(
            token_on, token_off,
            "min-p must change the sampled token for seed {seed:#x}"
        );
    }
}

#[test]
fn min_p_runs_before_top_p_and_renormalizes_survivors() {
    let mut candidates = CandidateSet::from_candidates(vec![
        Candidate {
            token_id: 0,
            logit: 0.0,
        },
        Candidate {
            token_id: 1,
            logit: 0.6_f32.ln(),
        },
        Candidate {
            token_id: 2,
            logit: 0.5_f32.ln(),
        },
    ]);

    assert_eq!(
        candidates.sample_min_p_top_p(0.55, 0.6, 0.8),
        0,
        "min-p must first remove token 2; after renormalization top-p=0.6 \
             retains only token 0"
    );
}

#[test]
fn min_p_one_keeps_all_max_probability_ties() {
    let mut candidates = CandidateSet::from_candidates(vec![
        Candidate {
            token_id: 3,
            logit: 4.0,
        },
        Candidate {
            token_id: 4,
            logit: 4.0,
        },
        Candidate {
            token_id: 5,
            logit: 3.0,
        },
    ]);

    assert_eq!(
        candidates.sample_min_p_top_p(1.0, 1.0, 0.75),
        4,
        "min_p=1 must retain every token tied at the maximum, not collapse \
             equal maxima to one arbitrary id"
    );
}

#[test]
fn min_p_does_not_hide_nan_beyond_a_finite_cutoff() {
    let mut candidates = CandidateSet::from_candidates(vec![
        Candidate {
            token_id: 0,
            logit: 4.0,
        },
        Candidate {
            token_id: 1,
            logit: 4.0,
        },
        Candidate {
            token_id: 2,
            logit: 1.0,
        },
        Candidate {
            token_id: 3,
            logit: f32::NAN,
        },
    ]);

    assert_eq!(
        candidates.sample_min_p_top_p(0.5, 1.0, 0.75),
        0,
        "a NaN after the min-p cutoff must preserve the argmax fail-closed fallback"
    );
}

/// Reference oracle for #1394 item 2: the min-p algorithm exactly as it
/// stood before the logit-threshold tail rejection in
/// `CandidateSet::sample_min_p_top_p_with_scratch` -- exponentiate every
/// sorted candidate first, then filter the unnormalized weights against
/// `min_p`. Kept byte-for-byte (down to the `.max(1)` cutoffs and the
/// fail-closed NaN scan) purely for the differential tests below; not used
/// on any production path.
fn sample_min_p_top_p_reference(
    cs: &mut CandidateSet,
    min_p: f32,
    top_p: f32,
    r: f32,
    probs: &mut Vec<f32>,
) -> u32 {
    if cs.candidates.is_empty() {
        return 0;
    }
    let min_p = if min_p.is_nan() {
        0.0
    } else {
        min_p.clamp(0.0, 1.0)
    };
    let top_p = if top_p.is_nan() {
        1.0
    } else {
        top_p.clamp(0.0, 1.0)
    };

    cs.candidates.sort_by(candidate_order);
    let max_logit = cs.candidates[0].logit;
    if !max_logit.is_finite() {
        return cs.candidates[0].token_id;
    }
    probs.clear();
    probs.extend(cs.candidates.iter().map(|c| (c.logit - max_logit).exp()));

    if min_p > 0.0 {
        if probs.iter().any(|weight| !weight.is_finite()) {
            return cs.candidates[0].token_id;
        }
        let cutoff = probs
            .iter()
            .position(|&weight| weight < min_p)
            .unwrap_or(probs.len())
            .max(1);
        probs.truncate(cutoff);
        cs.candidates.truncate(cutoff);
    }

    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return cs.candidates[0].token_id;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        cs.candidates.truncate(cutoff);
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    let mut cumsum = 0.0f32;
    for (c, &p) in cs.candidates.iter().zip(probs.iter()) {
        cumsum += p;
        if r < cumsum {
            return c.token_id;
        }
    }
    cs.candidates.last().map(|c| c.token_id).unwrap_or(0)
}

/// #1394 item 2: at logit offsets straddling the exact log-space threshold
/// by as little as 1e-7 (comparable to the ULP of `ln(0.5)` itself, ~4e-8),
/// the optimized path's logit-threshold prefix must still land on the same
/// side as the reference algorithm's exact `exp(logit - max) < min_p` test
/// -- either by classifying the candidate identically outright, or by
/// pulling it into the exact-recheck prefix so the untouched p-space
/// comparison decides it. Both the selected token and the full final
/// probability vector must match exactly (not just to a tolerance): with
/// the same survivor set, both paths run the identical `exp`/normalize
/// sequence in the identical order, so bit-exact equality is the correct
/// expectation, not merely a close one.
#[test]
fn min_p_logit_threshold_boundary_band_matches_reference() {
    let ln_half = 0.5_f32.ln();
    for &eps in &[-1e-3_f32, -1e-5, -1e-7, 0.0, 1e-7, 1e-5, 1e-3] {
        let boundary_logit = ln_half + eps;
        let build = || {
            CandidateSet::from_candidates(vec![
                Candidate {
                    token_id: 0,
                    logit: 0.0,
                },
                Candidate {
                    token_id: 1,
                    logit: boundary_logit,
                },
                Candidate {
                    token_id: 2,
                    logit: -5.0,
                },
            ])
        };
        for &r in &[0.05_f32, 0.3, 0.6, 0.8, 0.95] {
            let mut probs_new = Vec::new();
            let mut cs_new = build();
            let token_new = cs_new.sample_min_p_top_p_with_scratch(0.5, 1.0, r, &mut probs_new);

            let mut probs_ref = Vec::new();
            let mut cs_ref = build();
            let token_ref = sample_min_p_top_p_reference(&mut cs_ref, 0.5, 1.0, r, &mut probs_ref);

            assert_eq!(
                token_new, token_ref,
                "boundary eps={eps:e} r={r}: optimized vs reference token mismatch"
            );
            assert_eq!(
                probs_new, probs_ref,
                "boundary eps={eps:e} r={r}: final probability vectors differ"
            );
        }
    }
}

/// #1394 item 2: differential test between the pre-optimization reference
/// algorithm above and the production logit-threshold path, over many
/// seeded random logit vectors x min_p x temperature x top_k (including
/// top_k=0, the full-vocabulary case the issue calls out). The selected
/// token must match exactly (both draw off the same `r` against the same
/// cumulative distribution), and the final probabilities to within 1 ULP:
/// both paths compute a surviving candidate's weight with the identical
/// `(logit - max_logit).exp()` expression in the identical sorted order,
/// so the only way probabilities could differ is a different survivor set
/// -- exactly what a length/value mismatch here would catch. Every case
/// generated in this run held bitwise (0 ULP); 1 ULP is declared as the
/// tolerance rather than 0 because a future libm rounding change to `exp`
/// could move both call sites (they are literally the same expression) by
/// the same sub-ULP amount without being a regression in this algorithm.
#[test]
fn min_p_logit_threshold_matches_reference_across_seeds() {
    let min_ps = [0.0_f32, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.999_999_9, 1.0];
    let temperatures = [1.0_f32, 0.7, 2.0];
    let top_ks = [0usize, 8, 64];
    let vocab_size = 512usize;

    let mut seed: u64 = 0x0C0F_FEE1_5BAD_5EED_u64;

    for &min_p in &min_ps {
        for &temperature in &temperatures {
            for &top_k in &top_ks {
                for case in 0..20u32 {
                    let mut logits = Vec::with_capacity(vocab_size);
                    for i in 0..vocab_size {
                        let u = uniform_f32_from_u64(xorshift64_next(&mut seed));
                        let mut logit = (u - 0.5) * 16.0;
                        if i % 251 == 0 && i > 0 {
                            // Force an exact tie with the previous logit
                            // occasionally, exercising the tie-break path.
                            logit = logits[i - 1];
                        }
                        logits.push(logit);
                    }
                    if case % 7 == 0 {
                        // `select_top_k`'s Phase 1 seed pass sanitizes every
                        // NaN scaled logit to NEG_INFINITY before a
                        // `Candidate` is ever built (see its own doc
                        // comment), for every `top_k` including 0 (whose
                        // `take(k)` covers the whole vocabulary) -- so a
                        // NaN injected upstream of `select_top_k` can never
                        // reach `CandidateSet` as an actual NaN here. This
                        // still exercises a real, distinct edge case
                        // (a finite-weight NEG_INFINITY tail candidate,
                        // `exp(-inf - max) == 0.0`), just not the
                        // fail-closed NaN contract -- that is covered
                        // separately by
                        // `min_p_logit_threshold_matches_reference_with_nan_candidates`
                        // below, which builds a `CandidateSet` directly.
                        let idx = (seed as usize) % vocab_size;
                        logits[idx] = f32::NAN;
                    }

                    let inv_temp = 1.0 / temperature;
                    let mut candidate_scratch = Vec::new();
                    select_top_k(&logits, top_k, inv_temp, 0.0, &mut candidate_scratch);

                    let r = uniform_f32_from_u64(xorshift64_next(&mut seed));

                    let mut probs_new = Vec::new();
                    let mut cs_new = CandidateSet {
                        candidates: candidate_scratch.clone(),
                    };
                    let token_new =
                        cs_new.sample_min_p_top_p_with_scratch(min_p, 0.95, r, &mut probs_new);

                    let mut probs_ref = Vec::new();
                    let mut cs_ref = CandidateSet {
                        candidates: candidate_scratch,
                    };
                    let token_ref =
                        sample_min_p_top_p_reference(&mut cs_ref, min_p, 0.95, r, &mut probs_ref);

                    assert_eq!(
                        token_new, token_ref,
                        "min_p={min_p} temperature={temperature} top_k={top_k} \
                             case={case} r={r}: optimized vs reference token mismatch"
                    );
                    assert_eq!(
                        probs_new.len(),
                        probs_ref.len(),
                        "min_p={min_p} temperature={temperature} top_k={top_k} \
                             case={case} r={r}: final probability vector lengths differ"
                    );
                    for (i, (&a, &b)) in probs_new.iter().zip(probs_ref.iter()).enumerate() {
                        let ulp_distance = a.to_bits().abs_diff(b.to_bits());
                        assert!(
                            ulp_distance <= 1,
                            "min_p={min_p} temperature={temperature} top_k={top_k} \
                                 case={case} r={r} index={i}: probability {a} vs \
                                 reference {b} differ by {ulp_distance} ULP"
                        );
                    }
                }
            }
        }
    }
}

/// #1394 item 2: differential test for the NaN fail-closed contract itself
/// (the fuzz test above cannot reach it -- see the comment at its NaN
/// injection site). Builds `CandidateSet`s directly so an actual NaN
/// logit reaches `sample_min_p_top_p_with_scratch`, at several positions
/// relative to where min-p would otherwise have truncated, and asserts
/// the optimized path's pre-sort linear scan (`min_p_prefilter_and_sort`,
/// which detects any NaN before sorting or truncating anything) returns
/// the same argmax fallback token as the reference's full-array scan in
/// every case. Only the returned token is compared, not the
/// `probs` scratch buffer: on this fail-closed path the optimized
/// function returns before ever touching `probs` (the check now runs
/// before the `exp()` pass, which is the whole point of moving it), while
/// the reference populates `probs` with the NaN-poisoned weights first --
/// a real difference in leftover scratch-buffer content, but not an
/// output difference, since no caller reads `probs`/`prob_scratch` after
/// the call; only the returned token is part of the observable contract.
#[test]
fn min_p_logit_threshold_matches_reference_with_nan_candidates() {
    // logits, chosen so a NaN can land strictly before, at, or after
    // where min_p=0.5 (threshold weight 0.5 relative to max) truncates:
    // max=4.0; token weight 0.5*max survivor at logit ln(0.5)+4.0≈3.307;
    // token 2 (1.0) and token 4 (0.5) sit below the cutoff.
    let base = [
        (0u32, 4.0_f32),
        (1u32, 3.5_f32), // weight exp(-0.5) ≈ 0.6065, survives min_p=0.5
        (2u32, 1.0_f32), // weight exp(-3.0) ≈ 0.0498, pruned by min_p=0.5
        (3u32, 0.5_f32), // weight exp(-3.5) ≈ 0.0302, pruned by min_p=0.5
    ];

    for nan_token in [0u32, 1, 2, 3] {
        let build = || {
            CandidateSet::from_candidates(
                base.iter()
                    .map(|&(token_id, logit)| Candidate {
                        token_id,
                        logit: if token_id == nan_token {
                            f32::NAN
                        } else {
                            logit
                        },
                    })
                    .collect(),
            )
        };
        for &r in &[0.1_f32, 0.5, 0.9] {
            let mut probs_new = Vec::new();
            let mut cs_new = build();
            let token_new = cs_new.sample_min_p_top_p_with_scratch(0.5, 1.0, r, &mut probs_new);

            let mut probs_ref = Vec::new();
            let mut cs_ref = build();
            let token_ref = sample_min_p_top_p_reference(&mut cs_ref, 0.5, 1.0, r, &mut probs_ref);

            assert_eq!(
                token_new, token_ref,
                "nan_token={nan_token} r={r}: optimized vs reference token mismatch"
            );
            // `probs_new`/`probs_ref` are deliberately not compared here — see the
            // doc comment above.
        }
    }
}

/// #1394 item 2: pins the order-equivalence claim
/// `min_p_prefilter_and_sort` relies on directly, at the level of the
/// candidate list itself rather than the final sampled output -- for
/// randomized logits including exact ties and NaNs, either (a) the
/// pre-sort-filter-then-sort survivors equal the prefix of the same
/// length that a full `sort_by(candidate_order)` produces, or (b) the
/// prefilter's fail-closed return equals the token id a full sort would
/// have put first. Builds `CandidateSet`s directly (not via
/// `select_top_k`) so injected NaN logits reach the function under test
/// unsanitized -- see the comment at the NaN injection site in
/// `min_p_logit_threshold_matches_reference_across_seeds` for why that
/// distinction matters.
#[test]
fn min_p_prefilter_sort_matches_full_sort_prefix() {
    let mut seed: u64 = 0xF17E_5EED_1394_0002_u64;
    let min_ps = [0.02_f32, 0.05, 0.1, 0.3, 0.5, 0.9, 0.999_999_9];
    let vocab_size = 256usize;

    for &min_p in &min_ps {
        for case in 0..40u32 {
            let mut logits = Vec::with_capacity(vocab_size);
            for i in 0..vocab_size {
                let u = uniform_f32_from_u64(xorshift64_next(&mut seed));
                let mut logit = (u - 0.5) * 16.0;
                if i % 37 == 0 && i > 0 {
                    // Force an exact tie with the previous logit,
                    // exercising the ascending-token_id tie-break path.
                    logit = logits[i - 1];
                }
                logits.push(logit);
            }
            if case % 5 == 0 {
                let idx = (seed as usize) % vocab_size;
                logits[idx] = f32::NAN;
            }
            let candidates: Vec<Candidate> = logits
                .iter()
                .enumerate()
                .map(|(i, &logit)| Candidate {
                    token_id: i as u32,
                    logit,
                })
                .collect();

            let mut prefilter_set = CandidateSet {
                candidates: candidates.clone(),
            };
            let prefilter_result = prefilter_set.min_p_prefilter_and_sort(min_p);

            let mut full_sorted = candidates.clone();
            full_sorted.sort_by(candidate_order);

            match prefilter_result {
                Err(token_id) => {
                    assert_eq!(
                        token_id, full_sorted[0].token_id,
                        "min_p={min_p} case={case}: fail-closed token mismatch"
                    );
                }
                Ok(max_logit) => {
                    assert_eq!(
                        max_logit, full_sorted[0].logit,
                        "min_p={min_p} case={case}: max_logit mismatch"
                    );
                    let k = prefilter_set.candidates.len();
                    assert_eq!(
                        prefilter_set.candidates.as_slice(),
                        &full_sorted[..k],
                        "min_p={min_p} case={case}: prefilter+sort survivors \
                             (len={k}) do not equal the full-sort prefix of the \
                             same length"
                    );
                }
            }
        }
    }
}

#[test]
fn test_sampler_reset() {
    let config = SamplingConfig::greedy();
    let mut sampler = Sampler::new(config);
    sampler.sample(&[1.0, 2.0, 3.0]);
    assert!(!sampler.recent_tokens.is_empty());
    sampler.reset();
    assert!(sampler.recent_tokens.is_empty());
}

// ── argmax_f32 NEON parity tests ──────────────────────────────────────────

fn check_argmax_parity(logits: &[f32]) {
    let scalar = argmax_f32_scalar(logits);
    let dispatch = argmax_f32(logits);
    assert_eq!(
        scalar,
        dispatch,
        "argmax_f32 dispatch differs from scalar for len={}",
        logits.len()
    );
}

#[test]
fn test_argmax_full_vocab() {
    // Deterministic logits covering full Qwen vocab length.
    let n = 248_320usize;
    let logits: Vec<f32> = (0..n)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            (h as f32 / u64::MAX as f32) * 20.0 - 10.0
        })
        .collect();
    check_argmax_parity(&logits);
}

#[test]
fn test_argmax_lower_id_tie_wins() {
    // Both token 2 and 5 have the same max value; lower id (2) must win.
    let mut logits = vec![0.0f32; 10];
    logits[2] = 5.0;
    logits[5] = 5.0;
    assert_eq!(argmax_f32_scalar(&logits), 2, "scalar tie");
    assert_eq!(argmax_f32(&logits), 2, "dispatch tie");
}

#[test]
fn test_argmax_nan_loses_to_real() {
    // NaN at index 0 must not win; index 3 holds the real max.
    let logits = vec![f32::NAN, 1.0, 2.0, 9.0, 0.5];
    assert_eq!(argmax_f32_scalar(&logits), 3, "scalar nan");
    assert_eq!(argmax_f32(&logits), 3, "dispatch nan");
}

#[test]
fn test_argmax_all_nan_returns_0() {
    let logits = vec![f32::NAN, f32::NAN, f32::NAN];
    assert_eq!(argmax_f32_scalar(&logits), 0, "scalar all-nan");
    assert_eq!(argmax_f32(&logits), 0, "dispatch all-nan");
}

#[test]
fn test_argmax_all_neg_inf_returns_0() {
    let logits = vec![f32::NEG_INFINITY; 8];
    assert_eq!(argmax_f32_scalar(&logits), 0, "scalar neg-inf");
    assert_eq!(argmax_f32(&logits), 0, "dispatch neg-inf");
}

// ── CandidateSet::argmax correctness ─────────────────────────────────────

#[test]
fn test_candidateset_argmax_all_masked_returns_first_in_set_token() {
    // After grammar or top-k masking every logit in a compacted CandidateSet
    // can be NEG_INFINITY (the full vocabulary was masked except for these
    // candidates, whose own scores are also suppressed).  When that happens
    // the comparison loop never fires, so argmax must return the *first
    // candidate's token_id* — an in-set token — not the bare sentinel 0.
    // Token ids 7 and 9 are deliberately chosen to be non-zero; if the
    // initializer regresses to `0u32` this test returns 0 instead of 7 and
    // fails, catching the regression.
    let cs = CandidateSet::from_candidates(vec![
        Candidate {
            token_id: 7,
            logit: f32::NEG_INFINITY,
        },
        Candidate {
            token_id: 9,
            logit: f32::NEG_INFINITY,
        },
    ]);
    assert_eq!(
        cs.argmax(),
        7,
        "all-masked compact set must return first in-set token id"
    );
}

#[test]
fn test_argmax_partial_chunk_lengths() {
    // Test tail lengths 1-7 (0 mod 4 through 3 mod 4, and longer).
    for tail in 1usize..=7 {
        let mut logits = vec![0.0f32; 8 + tail];
        logits[8 + tail - 1] = 99.0; // max is always in the tail
        check_argmax_parity(&logits);
        assert_eq!(
            argmax_f32(&logits),
            (8 + tail - 1) as u32,
            "tail len={tail}"
        );
    }
}

// ── H3: candidate_order determinism tests ────────────────────────────────

#[test]
fn test_candidate_order_higher_logit_wins() {
    use std::cmp::Ordering;
    let a = Candidate {
        token_id: 10,
        logit: 5.0,
    };
    let b = Candidate {
        token_id: 20,
        logit: 3.0,
    };
    assert_eq!(
        candidate_order(&a, &b),
        Ordering::Less,
        "higher logit must sort first"
    );
}

#[test]
fn test_candidate_order_tie_lower_token_id_wins() {
    use std::cmp::Ordering;
    let a = Candidate {
        token_id: 5,
        logit: 2.0,
    };
    let b = Candidate {
        token_id: 9,
        logit: 2.0,
    };
    assert_eq!(
        candidate_order(&a, &b),
        Ordering::Less,
        "equal logit: lower token_id is first"
    );
    assert_eq!(candidate_order(&b, &a), Ordering::Greater);
}

#[test]
fn test_candidate_order_nan_loses() {
    use std::cmp::Ordering;
    let nan = Candidate {
        token_id: 0,
        logit: f32::NAN,
    };
    let real = Candidate {
        token_id: 99,
        logit: -1000.0,
    };
    assert_eq!(
        candidate_order(&nan, &real),
        Ordering::Greater,
        "NaN must sort last"
    );
    assert_eq!(candidate_order(&real, &nan), Ordering::Less);
}

#[test]
fn test_retain_top_k_tie_breaking() {
    // Three candidates with the same logit; top-2 must keep the two lowest token_ids.
    let mut cs = CandidateSet {
        candidates: vec![
            Candidate {
                token_id: 7,
                logit: 1.0,
            },
            Candidate {
                token_id: 2,
                logit: 1.0,
            },
            Candidate {
                token_id: 5,
                logit: 1.0,
            },
        ],
    };
    cs.retain_top_k(2);
    cs.candidates.sort_by(candidate_order);
    assert_eq!(cs.candidates[0].token_id, 2);
    assert_eq!(cs.candidates[1].token_id, 5);
}

#[test]
fn test_candidate_order_nan_nan_antisymmetric() {
    use std::cmp::Ordering;
    // Two NaN-logit candidates must produce a valid total order: the
    // comparator has to be antisymmetric (a<b ⇒ b>a) and break the tie
    // deterministically by token_id. The old `(true, _) => Greater` form
    // returned Greater for both (a,b) and (b,a), violating the strict-weak
    // order contract of sort_by/select_nth_unstable_by.
    let a = Candidate {
        token_id: 3,
        logit: f32::NAN,
    };
    let b = Candidate {
        token_id: 7,
        logit: f32::NAN,
    };
    assert_eq!(candidate_order(&a, &b), Ordering::Less);
    assert_eq!(candidate_order(&b, &a), Ordering::Greater);
    assert_eq!(candidate_order(&a, &a), Ordering::Equal);

    // A multi-NaN set sorts deterministically by token_id.
    let mut nans = [
        Candidate {
            token_id: 9,
            logit: f32::NAN,
        },
        Candidate {
            token_id: 1,
            logit: f32::NAN,
        },
        Candidate {
            token_id: 4,
            logit: f32::NAN,
        },
    ];
    nans.sort_by(candidate_order);
    assert_eq!(
        nans.iter().map(|c| c.token_id).collect::<Vec<_>>(),
        vec![1, 4, 9],
        "all-NaN set must sort by ascending token_id"
    );
}

#[test]
fn test_sample_top_p_tail_nan_returns_argmax() {
    // A NaN logit in a non-max position sorts last (so the finite max guard
    // does not catch it), but its exp() poisons the softmax sum to NaN. The
    // weighted-sample loop then never matches `r < cumsum` and would fall
    // through to candidates.last() — the NaN-sorted-last token (id 1) — a
    // garbage selection. The sum-finite guard must instead return the argmax
    // (token 0, the highest finite logit). select_top_k sanitises NaN→-inf in
    // the Sampler path, so this state is reachable via the public CandidateSet
    // API; constructing one directly exercises the residual hole.
    let mut cs = CandidateSet::from_candidates(vec![
        Candidate {
            token_id: 0,
            logit: 100.0,
        },
        Candidate {
            token_id: 1,
            logit: f32::NAN,
        },
        Candidate {
            token_id: 2,
            logit: 50.0,
        },
    ]);
    // r in the middle of the distribution; pre-fix this returns last() = 1.
    let token = cs.sample_top_p(1.0, 0.5);
    assert_eq!(
        token, 0,
        "tail-NaN poisons the softmax sum; must fall back to argmax (token 0), not last()"
    );
}

#[test]
fn test_sample_top_p_invalid_top_p_normalized() {
    // #328: an out-of-range top_p must resolve to defined behaviour, not the
    // accidental fall-through of the raw `top_p < 1.0` gate. NaN / +Inf / >1
    // map to 1.0 (no nucleus truncation, identical to top_p == 1.0); a
    // negative top_p collapses to greedy (argmax only). A fresh candidate set
    // is built per call because sample_top_p truncates its buffers in place.
    let make = || {
        CandidateSet::from_candidates(vec![
            Candidate {
                token_id: 0,
                logit: 3.0,
            },
            Candidate {
                token_id: 1,
                logit: 2.0,
            },
            Candidate {
                token_id: 2,
                logit: 1.0,
            },
            Candidate {
                token_id: 3,
                logit: 0.0,
            },
        ])
    };

    for &r in &[0.0f32, 0.25, 0.5, 0.75, 0.999] {
        let baseline = make().sample_top_p(1.0, r);
        assert_eq!(
            make().sample_top_p(f32::NAN, r),
            baseline,
            "NaN top_p must behave as top_p == 1.0 at r={r}"
        );
        assert_eq!(
            make().sample_top_p(1.5, r),
            baseline,
            ">1 top_p must behave as top_p == 1.0 at r={r}"
        );
        assert_eq!(
            make().sample_top_p(f32::INFINITY, r),
            baseline,
            "+Inf top_p must behave as top_p == 1.0 at r={r}"
        );
        assert_eq!(
            make().sample_top_p(-0.5, r),
            0,
            "negative top_p must collapse to greedy argmax at r={r}"
        );
    }
}

// ── H1: select_top_k correctness tests ──────────────────────────────────

fn check_top_k_parity(logits: &[f32], k: usize) {
    let mut scalar_out = Vec::new();
    select_top_k_scalar(logits, k, 1.0, 0.0, &mut scalar_out);
    scalar_out.sort_by(candidate_order);

    let mut dispatch_out = Vec::new();
    select_top_k(logits, k, 1.0, 0.0, &mut dispatch_out);
    dispatch_out.sort_by(candidate_order);

    assert_eq!(
        scalar_out.len(),
        dispatch_out.len(),
        "k={k}: length mismatch"
    );
    for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
        assert_eq!(
            s.token_id, d.token_id,
            "k={k}: position {i} token_id mismatch (scalar={} dispatch={})",
            s.token_id, d.token_id
        );
    }
}

#[test]
fn test_select_top_k_basic() {
    let logits = vec![1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0];
    let mut out = Vec::new();
    select_top_k_scalar(&logits, 3, 1.0, 0.0, &mut out);
    out.sort_by(candidate_order);
    let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
    assert_eq!(ids, vec![3, 5, 1], "top-3 from [1,5,3,9,2,7]");
}

#[test]
fn test_select_top_k_tie_breaking() {
    // Two tokens with logit=5.0; top-1 must select the lower token_id.
    let logits = vec![5.0f32, 5.0, 1.0];
    let mut out = Vec::new();
    select_top_k_scalar(&logits, 1, 1.0, 0.0, &mut out);
    assert_eq!(out[0].token_id, 0, "tie: lower id must win");
}

#[test]
fn test_select_top_k_nan_excluded() {
    // NaN at position 0 must not appear in top-k.
    let logits = vec![f32::NAN, 2.0, 9.0, 3.0];
    let mut out = Vec::new();
    select_top_k_scalar(&logits, 2, 1.0, 0.0, &mut out);
    out.sort_by(candidate_order);
    let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
    assert_eq!(ids, vec![2, 3], "NaN must not appear in top-2");
}

#[test]
fn test_select_top_k_full_vocab_parity() {
    let n = 248_320usize;
    let logits: Vec<f32> = (0..n)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            (h as f32 / u64::MAX as f32) * 20.0 - 10.0
        })
        .collect();
    check_top_k_parity(&logits, 50);
}

#[test]
fn test_select_top_k_dispatch_tie_breaking() {
    // When two logits are equal, lower token_id must win.
    let logits = vec![5.0f32, 5.0, 5.0, 1.0, 1.0];
    let mut out = Vec::new();
    select_top_k(&logits, 2, 1.0, 0.0, &mut out);
    out.sort_by(candidate_order);
    let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
    assert_eq!(
        ids,
        vec![0, 1],
        "dispatch: tie must break by lower token_id"
    );
}

#[test]
fn test_select_top_k_dispatch_nan_excluded() {
    // NaN logits must never appear in top-k output from dispatch.
    let logits = vec![f32::NAN, 4.0, 9.0, f32::NAN, 7.0];
    let mut out = Vec::new();
    select_top_k(&logits, 2, 1.0, 0.0, &mut out);
    out.sort_by(candidate_order);
    let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
    assert_eq!(ids, vec![2, 4], "dispatch: NaN must not appear in top-2");
}

#[test]
fn test_select_top_k_dispatch_nan_in_seed() {
    // NaN values in the first k elements (seed phase) must not corrupt output.
    // Build logits: [NaN, 2.0, 3.0, NaN, 5.0, 6.0, 7.0, NaN, 9.0, 10.0, 1..=990 as f32]
    let mut logits = vec![
        f32::NAN,
        2.0,
        3.0,
        f32::NAN,
        5.0,
        6.0,
        7.0,
        f32::NAN,
        9.0,
        10.0,
    ];
    logits.extend((1..=990u32).map(|x| x as f32));
    // k=10: both dispatch and scalar must agree after sorting
    let mut scalar_out = Vec::new();
    select_top_k_scalar(&logits, 10, 1.0, 0.0, &mut scalar_out);
    scalar_out.sort_by(candidate_order);

    let mut dispatch_out = Vec::new();
    select_top_k(&logits, 10, 1.0, 0.0, &mut dispatch_out);
    dispatch_out.sort_by(candidate_order);

    assert_eq!(scalar_out.len(), dispatch_out.len());
    for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
        assert_eq!(
            s.token_id, d.token_id,
            "nan_in_seed: position {i} token_id mismatch (scalar={} dispatch={})",
            s.token_id, d.token_id
        );
    }
}

#[test]
fn test_select_top_k_applies_inv_temp_to_candidate_logits() {
    // inv_temp=0.5 halves all logits; top-2 must be token 3 (2.0) and token 2 (1.5).
    let logits = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut out = Vec::new();
    select_top_k(&logits, 2, 0.5, 0.0, &mut out);
    out.sort_by(candidate_order);
    assert_eq!(out[0].token_id, 3);
    assert_eq!(out[1].token_id, 2);
    assert!(
        (out[0].logit - 2.0).abs() < 1e-6,
        "logit[3] must be 4.0*0.5=2.0"
    );
    assert!(
        (out[1].logit - 1.5).abs() < 1e-6,
        "logit[2] must be 3.0*0.5=1.5"
    );
}

// Finite logits above `temperature * f32::MAX`: scaling them before centering
// overflows both to +inf, and the tie then breaks on the lower token id (0).
const HUGE_FINITE_LOGITS: [f32; 2] = [2e38, 3e38];

#[test]
fn apply_temperature_centers_huge_finite_logits_before_scaling() {
    let mut cs = CandidateSet::from_full_logits(&HUGE_FINITE_LOGITS);
    cs.apply_temperature(0.5);
    assert_eq!(
        cs.argmax(),
        1,
        "the larger finite logit must win after temperature scaling"
    );
}

#[test]
fn select_top_k_centers_huge_finite_logits_before_scaling() {
    // The padded row puts both huge logits in the NEON 4-wide prefilter (after the
    // k = 1 seed), which the two-element row never reaches.
    let padded = [0.0_f32, 2e38, 0.0, 0.0, 0.0, 3e38, 0.0, 0.0, 0.0];
    for (logits, expected) in [(&HUGE_FINITE_LOGITS[..], 1u32), (&padded[..], 5)] {
        let mut scalar_out = Vec::new();
        select_top_k_scalar(logits, 1, 2.0, 3e38, &mut scalar_out);
        let mut dispatch_out = Vec::new();
        select_top_k(logits, 1, 2.0, 3e38, &mut dispatch_out);
        for out in [&scalar_out, &dispatch_out] {
            assert_eq!(out.len(), 1);
            assert_eq!(
                out[0].token_id, expected,
                "top-1 must be the larger finite logit"
            );
        }
    }
}

#[test]
fn sampler_centers_huge_finite_logits_before_scaling() {
    // top_k = 2 keeps the fused top-k route; top_k = 1 takes the greedy fast path,
    // which never scales. After centering the smaller logit's weight is exp(-2e38),
    // exactly 0.0, so every seed must draw token 1.
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.5,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(config).with_seed(7);
    assert_eq!(sampler.sample(&HUGE_FINITE_LOGITS), 1);
}

#[test]
fn sample_full_logits_centers_huge_finite_logits_before_scaling() {
    let cfg = GenerateConfig {
        temperature: 0.5,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let mut rng_state = 7;
    assert_eq!(
        sample_full_logits(&HUGE_FINITE_LOGITS, &cfg, &[], &mut rng_state, 0.0, 0.0),
        1
    );
}

// Opposite-direction overflow: `f32::MAX` passes `temperature_degenerate`
// (its reciprocal is tiny but finite), and `inv_temp <= 1.0` so centering
// must be skipped (see `select_top_k`'s doc comment). Centering
// unconditionally sends the negative logit to `-inf` even though plain
// scaling keeps both finite at about `[1.0, -1.0]`.
const HUGE_LOGITS_BOTH_SIGNS: [f32; 2] = [f32::MAX, -f32::MAX];

#[test]
fn apply_temperature_scales_without_centering_when_inv_le_one() {
    let mut cs = CandidateSet::from_full_logits(&HUGE_LOGITS_BOTH_SIGNS);
    cs.apply_temperature(f32::MAX);
    assert!(
        cs.candidates[0].logit.is_finite(),
        "positive logit must stay finite"
    );
    assert!(
        cs.candidates[1].logit.is_finite(),
        "negative logit must stay finite, not center-and-overflow to -inf"
    );
    let diff = (cs.candidates[0].logit - cs.candidates[1].logit).abs();
    assert!(
        (diff - 2.0).abs() < 1e-3,
        "logit difference must be ~2.0, got {diff}"
    );
}

#[test]
fn select_top_k_scales_without_centering_when_inv_le_one() {
    // Mirrors `select_top_k_centers_huge_finite_logits_before_scaling`: the
    // padded row places both extreme logits inside the NEON 4-wide phase.
    // The filler is `NEG_INFINITY`, not `0.0`: with `k == 2` the function
    // keeps only the two largest values, and `0.0` filler would itself
    // outrank `-f32::MAX`, evicting it from the returned set and defeating
    // the point of the test (`-f32::MAX` is the most negative *finite*
    // value, so only `-inf` filler ranks below it).
    let padded = [
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        -f32::MAX,
    ];
    let inv_temp = 1.0 / f32::MAX;
    for logits in [&HUGE_LOGITS_BOTH_SIGNS[..], &padded[..]] {
        let mut scalar_out = Vec::new();
        select_top_k_scalar(logits, 2, inv_temp, 0.0, &mut scalar_out);
        let mut dispatch_out = Vec::new();
        select_top_k(logits, 2, inv_temp, 0.0, &mut dispatch_out);
        for out in [&scalar_out, &dispatch_out] {
            assert_eq!(out.len(), 2);
            for c in out.iter() {
                assert!(
                    c.logit.is_finite(),
                    "logit for token {} must be finite",
                    c.token_id
                );
            }
            let diff = (out[0].logit - out[1].logit).abs();
            assert!(
                (diff - 2.0).abs() < 1e-3,
                "logit difference must be ~2.0, got {diff}"
            );
        }
    }
}

#[test]
fn sampler_sample_scales_without_centering_when_inv_le_one() {
    // `Sampler::sample` exposes only a sampled token id through its public
    // signature; verified through the private `candidate_scratch` it
    // restores after sampling (this test module is nested inside the same
    // module, so the field is visible here).
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: f32::MAX,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(config).with_seed(7);
    sampler.sample(&HUGE_LOGITS_BOTH_SIGNS);
    assert_eq!(sampler.candidate_scratch.len(), 2);
    for c in &sampler.candidate_scratch {
        assert!(
            c.logit.is_finite(),
            "logit for token {} must be finite",
            c.token_id
        );
    }
    let diff = (sampler.candidate_scratch[0].logit - sampler.candidate_scratch[1].logit).abs();
    assert!(
        (diff - 2.0).abs() < 1e-3,
        "logit difference must be ~2.0, got {diff}"
    );
}

#[test]
fn sample_full_logits_scales_without_centering_when_inv_le_one() {
    // `sample_full_logits` also exposes only a sampled token id; verified
    // through the thread-local candidate scratch it restores before
    // returning.
    let cfg = GenerateConfig {
        temperature: f32::MAX,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let mut rng_state = 7;
    sample_full_logits(&HUGE_LOGITS_BOTH_SIGNS, &cfg, &[], &mut rng_state, 0.0, 0.0);
    FULL_LOGIT_SCRATCH.with(|cell| {
        let scratch = cell.borrow();
        assert_eq!(scratch.candidate_scratch.len(), 2);
        for c in &scratch.candidate_scratch {
            assert!(
                c.logit.is_finite(),
                "logit for token {} must be finite",
                c.token_id
            );
        }
        let diff = (scratch.candidate_scratch[0].logit - scratch.candidate_scratch[1].logit).abs();
        assert!(
            (diff - 2.0).abs() < 1e-3,
            "logit difference must be ~2.0, got {diff}"
        );
    });
}

#[test]
fn test_select_top_k_zero_keeps_all() {
    // k == 0 ("disabled") must keep ALL candidates, never an empty set.
    let logits = vec![1.0f32, 5.0, 3.0, 9.0, 2.0];
    let mut scalar_out = Vec::new();
    select_top_k_scalar(&logits, 0, 1.0, 0.0, &mut scalar_out);
    assert_eq!(scalar_out.len(), 5, "scalar: k=0 must keep all candidates");
    let mut dispatch_out = Vec::new();
    select_top_k(&logits, 0, 1.0, 0.0, &mut dispatch_out);
    assert_eq!(
        dispatch_out.len(),
        5,
        "dispatch: k=0 must keep all candidates"
    );
}

#[test]
fn test_top_k_zero_is_no_filtering_not_token_zero() {
    // Regression: top_k=0 used to make select_top_k return empty, so the sampler
    // could only ever emit token 0. With it disabled it must sample across the vocab.
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 1.0,
        top_k: 0,   // disabled
        top_p: 1.0, // disabled
        repetition_penalty: 1.0,
    };
    let mut sampler = Sampler::new(config).with_seed(42);
    let logits = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut counts = [0u32; 8];
    for _ in 0..400 {
        counts[sampler.sample(&logits) as usize] += 1;
    }
    let nonzero = counts.iter().filter(|&&c| c > 0).count();
    assert!(
        nonzero >= 5,
        "top_k=0 must sample across the vocab, got counts={counts:?}"
    );
}

#[test]
fn test_sample_top_p_handles_pos_inf_logit() {
    // A +INF logit must select that token, not poison the softmax and fall
    // through to the worst candidate.
    let mut cs = CandidateSet {
        candidates: vec![
            Candidate {
                token_id: 0,
                logit: 1.0,
            },
            Candidate {
                token_id: 1,
                logit: f32::INFINITY,
            },
            Candidate {
                token_id: 2,
                logit: 2.0,
            },
        ],
    };
    let mut scratch = Vec::new();
    let tok = cs.sample_top_p_with_scratch(1.0, 0.999, &mut scratch);
    assert_eq!(tok, 1, "+INF logit token must be selected");
}

/// #611 defense-in-depth: even if a grammar's `has_finite_logit` guard
/// were ever bypassed upstream, the sampler itself must not let an
/// all-masked (all-`NEG_INFINITY`) `CandidateSet` poison softmax into NaN
/// or panic. The temperature=0 argmax path already has
/// `test_argmax_all_neg_inf_returns_0` and
/// `test_candidateset_argmax_all_masked_returns_first_in_set_token`; this
/// covers the same all-masked input through the temperature>0 / top-p
/// nucleus path (`sample_top_p_with_scratch`), a distinct code path with
/// its own pair of non-finite guards.
///
/// Mutation sensitivity (verified empirically, not just asserted): this
/// input is defended by *two* independent guards — `!max_logit.is_finite()`
/// and `!sum.is_finite() || sum <= 0.0` a few lines below it. Disabling
/// either ONE alone still passes (the other one catches it — genuine
/// defense-in-depth, confirmed by temporarily disabling each in turn).
/// Disabling BOTH simultaneously computes `(NEG_INFINITY -
/// NEG_INFINITY).exp()` = `NaN.exp()` = `NaN` for every candidate, and the
/// `r < cumsum` weighted-sample loop is always false against a NaN cumsum,
/// so it falls through to the *last* sorted candidate (token_id 3, the
/// highest id, per the code's own comment) instead of this test's expected
/// token_id 1 — the assertion below then fails with `left: 3, right: 1`.
#[test]
fn test_sample_top_p_all_neg_inf_falls_back_to_first_sorted_candidate() {
    let mut cs = CandidateSet {
        candidates: vec![
            Candidate {
                token_id: 3,
                logit: f32::NEG_INFINITY,
            },
            Candidate {
                token_id: 1,
                logit: f32::NEG_INFINITY,
            },
            Candidate {
                token_id: 2,
                logit: f32::NEG_INFINITY,
            },
        ],
    };
    let mut scratch = Vec::new();
    // top_p < 1.0 so the nucleus-truncation branch is in play too, not just
    // the full-distribution case.
    let tok = cs.sample_top_p_with_scratch(0.9, 0.5, &mut scratch);
    // All logits tie at NEG_INFINITY, so `candidate_order`'s tie-break
    // (ascending token_id) sorts token_id 1 to `candidates[0]` — the
    // non-finite-max guard's defined fallback answer.
    assert_eq!(
        tok, 1,
        "an all-NEG_INFINITY CandidateSet must deterministically fall back \
             to the first sorted candidate's token id, not NaN-derived garbage"
    );
}

#[test]
fn test_penalized_logit_invalid_penalty_is_noop() {
    // Non-positive or non-finite penalties must not flip a logit's sign.
    for &bad in &[0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            penalized_logit(5.0, bad),
            5.0,
            "positive logit unchanged for penalty={bad}"
        );
        assert_eq!(
            penalized_logit(-5.0, bad),
            -5.0,
            "negative logit unchanged for penalty={bad}"
        );
    }
    // Valid penalty still applies as before.
    assert!(
        (penalized_logit(2.0, 2.0) - 1.0).abs() < 1e-6,
        "2.0/2.0 == 1.0"
    );
    assert!(
        (penalized_logit(-2.0, 2.0) - -4.0).abs() < 1e-6,
        "-2.0*2.0 == -4.0"
    );
}

#[test]
fn test_repetition_penalty_applied_once_per_token() {
    // `recent_tokens` accumulates duplicates (`push_token` never dedups), so a
    // token repeated within the window must be penalized exactly ONCE — matching
    // HF's gather-once semantics and the Metal `CandidateSet` path. Applying the
    // penalty per occurrence compounds it (penalty^N) and silently over-suppresses
    // common tokens as a sequence grows.
    let mut s = Sampler::new(SamplingConfig {
        min_p: 0.0,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 2.0,
    })
    .with_seed(1);
    // Step 1: token 1 is the clear argmax; recent_tokens -> [1].
    assert_eq!(s.sample(&[0.0, 10.0, 0.0]), 1);
    // Step 2: token 1 penalized once (10/2 = 5) still wins; recent_tokens -> [1, 1].
    assert_eq!(s.sample(&[0.0, 10.0, 0.0]), 1);
    // Step 3: token 1 penalized ONCE (10/2 = 5) must beat token 2 (logit 3).
    // The per-occurrence bug penalizes twice (10/2/2 = 2.5) and wrongly picks token 2.
    assert_eq!(
        s.sample(&[0.0, 10.0, 3.0]),
        1,
        "duplicate history id must be penalized once, not per occurrence"
    );
}

// ── canonical RNG primitive tests ────────────────────────────────────────

#[test]
fn uniform_f32_from_u64_is_always_in_unit_interval() {
    // The canonical 24-bit conversion must be in [0, 1) for all u64 values —
    // in particular u64::MAX must NOT round up to exactly 1.0 (the latent bug
    // in the 53-bit `(s >> 11) as f32 / 2^53` path).
    for x in [
        0u64,
        1,
        u64::MAX,
        0xFFFF_FFFF_FFFF_FFFF,
        1u64 << 40,
        (1u64 << 40) - 1,
        0x8000_0000_0000_0000,
    ] {
        let f = uniform_f32_from_u64(x);
        assert!(
            (0.0..1.0).contains(&f),
            "uniform_f32_from_u64({x:#018x}) = {f} is not in [0, 1)"
        );
    }
}

// ── #387 repetition-penalty history contract tests ───────────────────────

/// First-token prompt penalty: seeded prompt tokens must be penalized on the
/// very first `sample` call.
///
/// Mutation-sensitive: reverting `seed_history` at a generation entry point
/// leaves `recent_tokens` empty, so token 1 is NOT penalized
/// and wins with logit 10.0, failing this assertion.
#[test]
fn test_seed_history_penalizes_prompt_token_on_first_sample() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.0, // greedy
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 2.0,
    };
    let mut sampler = Sampler::new(config).with_seed(1);
    // Seed token 1 as a prompt token.
    sampler.seed_history(&[1u32]);
    // logits: token 1 = 10.0 wins raw, but 10.0/2.0 = 5.0 < token 0's 6.0 after penalty.
    let token = sampler.sample(&[6.0, 10.0]);
    assert_eq!(
        token, 0,
        "token 1 was seeded in history; penalty 2.0 reduces its adjusted logit \
             below token 0; without seed_history, token 1 wins (mutation: omit seed_history)"
    );
}

/// Beyond-64-token history: a token at position 0 in a 65-entry history must
/// still be penalized after the old 64-entry cap is removed.
///
/// Mutation-sensitive: re-introducing the `max_recent` truncation evicts token 1
/// from the window, leaving it unpenalized, so token 1 wins with logit 10.0
/// instead of token 0 with logit 6.0.
#[test]
fn test_uncapped_history_penalizes_tokens_beyond_64() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.0, // greedy
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 2.0,
    };
    let mut sampler = Sampler::new(config).with_seed(1);
    // Build a 65-token history THROUGH push_token, the path the removed 64-cap
    // actually lived on. token 1 is the oldest entry (pushed first). Filler tokens
    // use ids >= 2 so they do not compete with tokens 0 or 1.
    sampler.push_token(1);
    for t in 2u32..66 {
        sampler.push_token(t); // 64 filler tokens → total history length = 65
    }
    // Without cap: token 1 penalty applies → 10.0/2.0 = 5.0 < 6.0 → token 0 wins.
    // With old 64-cap restored in push_token: token 1 was evicted from the window →
    // raw_best (token 1) is no longer in history so the greedy shortcut returns it
    // un-penalized → token 1 wins.
    let token = sampler.sample(&[6.0, 10.0]);
    assert_eq!(
        token, 0,
        "token 1 at position 0 in a 65-entry history must still be penalized; \
             the old 64-cap silently dropped it (mutation: restore max_recent truncation)"
    );
}

/// Penalize-once / no compounding: a token repeated many times in history is
/// penalized exactly once, not penalty^N times.
///
/// Mutation-sensitive: reverting the HashSet dedup to per-occurrence penalty
/// compounds to penalty^4 = 16.0, reducing the adjusted logit from 5.0 to 0.625,
/// so token 0 wins instead of token 1.
#[test]
fn test_penalty_applied_exactly_once_for_repeated_history_token() {
    let config = SamplingConfig {
        min_p: 0.0,
        temperature: 0.0, // greedy
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 2.0,
    };
    let mut sampler = Sampler::new(config).with_seed(1);
    // Token 1 repeated 4 times in history.
    sampler.seed_history(&[1, 1, 1, 1]);
    // Once penalized: 10.0/2.0 = 5.0 > 4.9 → token 1 wins (correct).
    // Per-occurrence (penalty^4): 10.0/16.0 = 0.625 < 4.9 → token 0 wins (wrong).
    let token = sampler.sample(&[4.9, 10.0]);
    assert_eq!(
        token, 1,
        "token 1 repeated 4× must be penalized once (5.0 > 4.9, token 1 wins); \
             per-occurrence compounding yields 0.625 and wrongly selects token 0 \
             (mutation: replace HashSet dedup with per-occurrence penalty)"
    );
}

// --- logprobs (#585) ----------------------------------------------------

/// Hand-computed softmax reference for `logits = [1.0, 2.0, 3.0]`,
/// temperature 1.0: `p = softmax(logits)`, `ln(p) = logits - ln(sum(exp(logits)))`.
/// `sum(exp) = e^1 + e^2 + e^3 = 2.718282 + 7.389056 + 20.085537 = 30.192875`,
/// `ln(sum) = 3.407606`. So `ln(p[i]) = logits[i] - 3.407606`.
fn reference_ln_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let log_sum = sum.ln();
    logits.iter().map(|&v| (v - max) - log_sum).collect()
}

#[test]
fn test_compute_step_logprobs_matches_hand_computed_softmax() {
    let logits = [1.0f32, 2.0, 3.0];
    let reference = reference_ln_softmax(&logits);

    let (logprob, top) = compute_step_logprobs(&logits, 2, 1.0, 0);
    assert!(
        (logprob - reference[2]).abs() < 1e-4,
        "token 2 logprob {logprob} should match reference {}",
        reference[2]
    );
    assert!(top.is_empty(), "top_n=0 must return no alternatives");

    for (idx, &want) in reference.iter().enumerate() {
        let (lp, _) = compute_step_logprobs(&logits, idx as u32, 1.0, 0);
        assert!(
            (lp - want).abs() < 1e-4,
            "token {idx} logprob {lp} should match reference {want}"
        );
    }
}

#[test]
fn test_compute_step_logprobs_top_n_sorted_descending_by_probability() {
    let logits = [1.0f32, 2.0, 3.0];
    let reference = reference_ln_softmax(&logits);

    let (_, top) = compute_step_logprobs(&logits, 2, 1.0, 2);
    assert_eq!(top.len(), 2, "top_logprobs=2 must return exactly 2 entries");
    // Descending logit == descending probability at fixed temperature.
    assert_eq!(top[0].token_id, 2, "highest-logit token must be first");
    assert_eq!(
        top[1].token_id, 1,
        "second-highest-logit token must be second"
    );
    assert!((top[0].logprob - reference[2]).abs() < 1e-4);
    assert!((top[1].logprob - reference[1]).abs() < 1e-4);
    assert!(
        top[0].logprob > top[1].logprob,
        "entries must be sorted descending by logprob"
    );
}

#[test]
fn test_compute_step_logprobs_top_n_clamped_to_vocab_size() {
    let logits = [1.0f32, 2.0, 3.0];
    // Requesting more alternatives than the vocabulary must not panic or
    // fabricate entries -- it returns at most `vocab_size` of them.
    let (_, top) = compute_step_logprobs(&logits, 0, 1.0, 20);
    assert_eq!(top.len(), 3);
}

#[test]
fn test_compute_step_logprobs_degenerate_temperature_reports_unscaled_softmax() {
    // temperature <= 0 is degenerate (see `temperature_degenerate`): the
    // *sampler* falls back to greedy argmax, but the reporting distribution
    // here must fall back to an UNSCALED (temperature = 1.0) softmax, not a
    // one-hot `t -> 0+` distribution -- argmax is invariant to positive
    // temperature scaling, so reporting log(1.0) = 0.0 / -inf would erase
    // the model's actual confidence rather than describe it.
    let logits = [1.0f32, 2.0, 3.0];
    let reference = reference_ln_softmax(&logits);
    let (logprob, top) = compute_step_logprobs(&logits, 2, 0.0, 1);
    assert!(
        (logprob - reference[2]).abs() < 1e-4,
        "degenerate temperature must report the T=1.0 softmax logprob \
             ({}), not one-hot 0.0; mutation would collapse this to 0.0",
        reference[2]
    );
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].token_id, 2);
}

#[test]
fn test_compute_step_logprobs_all_nonfinite_logits_falls_back_to_sentinel() {
    let logits = [f32::NAN, f32::NAN, f32::NAN];
    let (logprob, top) = compute_step_logprobs(&logits, 1, 1.0, 3);
    assert_eq!(logprob, LOGPROB_NEG_SENTINEL);
    assert_eq!(
        top,
        vec![TopLogprob {
            token_id: 1,
            logprob: LOGPROB_NEG_SENTINEL
        }],
        "the requested token_id must still be reported (as the sentinel), \
             not dropped or replaced by an arbitrary index"
    );
}

#[test]
fn test_compute_step_logprobs_huge_finite_logit_below_one_temperature_is_not_sentinel() {
    let logits = [3.0e38f32, 0.0];
    let (logprob, top) = compute_step_logprobs(&logits, 0, 0.5, 1);
    assert!(
        logprob.abs() < 1e-6,
        "argmax of [3e38, 0] at T=0.5 has logprob 0.0, got {logprob}"
    );
    assert_ne!(logprob, LOGPROB_NEG_SENTINEL);
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].token_id, 0);
    assert!(
        top[0].logprob.abs() < 1e-6,
        "top-1 logprob {}",
        top[0].logprob
    );
}

#[test]
fn test_compute_step_logprobs_at_or_above_one_temperature_matches_uncentered_formula_bitwise() {
    let logits = [1.5f32, -2.25, 0.0, 7.0, 3.125, -0.5];
    for &t in &[1.0f32, 1.5, 2.0, 10.0] {
        let scale = 1.0 / t;
        let max_scaled = logits
            .iter()
            .map(|&v| v * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&v| (v * scale - max_scaled).exp()).sum();
        let log_sum = sum.ln();
        for (idx, &v) in logits.iter().enumerate() {
            let expected = (v * scale - max_scaled) - log_sum;
            let (lp, _) = compute_step_logprobs(&logits, idx as u32, t, 0);
            assert_eq!(lp.to_bits(), expected.to_bits(), "T={t} token {idx}");
        }
    }
}

#[test]
fn test_compute_step_logprobs_neg_inf_entry_below_one_temperature_reports_sentinel() {
    let logits = [2.0f32, f32::NEG_INFINITY, 1.0];
    let (lp_neg, _) = compute_step_logprobs(&logits, 1, 0.5, 0);
    assert_eq!(lp_neg, LOGPROB_NEG_SENTINEL);
    let (lp_best, top) = compute_step_logprobs(&logits, 0, 0.5, 3);
    assert!(lp_best.is_finite() && lp_best < 0.0 && lp_best != LOGPROB_NEG_SENTINEL);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].token_id, 0);
    assert_eq!(top[2].token_id, 1);
    assert_eq!(top[2].logprob, LOGPROB_NEG_SENTINEL);
}

#[test]
fn test_compute_step_logprobs_top_n_orders_by_logit_when_centering_overflows_tail() {
    // The row spans more than f32::MAX, so centering at the maximum sends
    // both finite tail logits to -inf. The top-N order must still follow
    // the raw logits (0.6M, -0.4M, -0.5M, -inf), not token id.
    let m = f32::MAX;
    let logits = [0.6 * m, f32::NEG_INFINITY, -0.5 * m, -0.4 * m];
    let (_, top) = compute_step_logprobs(&logits, 0, 0.9, 4);
    let ids: Vec<u32> = top.iter().map(|t| t.token_id).collect();
    assert_eq!(ids, vec![0, 3, 2, 1]);
}

#[test]
fn test_compute_step_logprobs_top_n_at_or_above_one_temperature_matches_uncentered_formula_bitwise()
{
    let logits = [1.5f32, -2.25, 0.0, 7.0, 3.125, -0.5, 4.75];
    for &t in &[1.0f32, 1.5, 3.0] {
        let scale = 1.0 / t;
        let max_scaled = logits
            .iter()
            .map(|&v| v * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&v| (v * scale - max_scaled).exp()).sum();
        let log_sum = sum.ln();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| {
            (logits[b] * scale)
                .partial_cmp(&(logits[a] * scale))
                .unwrap()
                .then(a.cmp(&b))
        });
        let (_, top) = compute_step_logprobs(&logits, 0, t, logits.len());
        assert_eq!(top.len(), logits.len(), "T={t}");
        for (rank, (entry, &idx)) in top.iter().zip(order.iter()).enumerate() {
            let expected = (logits[idx] * scale - max_scaled) - log_sum;
            assert_eq!(entry.token_id, idx as u32, "T={t} rank {rank}");
            assert_eq!(
                entry.logprob.to_bits(),
                expected.to_bits(),
                "T={t} rank {rank}"
            );
        }
    }
}

#[test]
fn test_compute_step_logprobs_out_of_vocab_token_id_falls_back_to_sentinel() {
    let logits = [1.0f32, 2.0, 3.0];
    let reference = reference_ln_softmax(&logits);
    // token_id beyond the logits slice: the requested token's logprob is
    // the sentinel, but the top-N alternatives are still computed
    // correctly over the real vocabulary (unaffected by the bad id).
    let (logprob, top) = compute_step_logprobs(&logits, 99, 1.0, 1);
    assert_eq!(logprob, LOGPROB_NEG_SENTINEL);
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].token_id, 2);
    assert!((top[0].logprob - reference[2]).abs() < 1e-4);
}

// `record_logprob`'s noop/appends behavior moved to
// `model::qwen35::generation`'s `DecodePolicy::record_logprob` tests
// (PR #787) -- see
// `decode_policy_record_logprob_noop_when_not_requested` and
// `transition_records_one_logprob_per_generated_token` there. The
// freestanding `pub(crate) record_logprob` this module used to export no
// longer exists: only `compute_step_logprobs` (pure computation) remains
// shared, and mutating `token_logprobs` is exclusively `DecodePolicy`'s.

// -------------------------------------------------------------------
// CPU-side microbench (issue #650): before/after `sample_full_logits`.
//
// `#[ignore]`d so normal `cargo test` runs stay fast; run explicitly with:
//   cargo test -p lattice-inference --lib --release -- --ignored --nocapture \
//     sampling::tests::microbench_sample_full_logits_default_issue_config
// -------------------------------------------------------------------

/// Literal copy of the pre-optimization `sample_token` algorithm (full
/// vocab clone, fresh per-call `HashSet`, full `Vec<usize>` index
/// allocation, allocating softmax-probs `Vec`) — the "before" arm of the
/// microbench. Mirrors what both `model/qwen35/sampling.rs::sample_token`
/// and the Metal CPU fallback did before this change; kept local to the
/// microbench so it has no effect on any production or library code path.
fn old_full_vocab_sample(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    previous_ids: &[u32],
    rng_state: &mut u64,
) -> u32 {
    let vocab_size = logits.len();
    let mut adjusted = logits.to_vec();

    if repetition_penalty != 1.0 {
        let mut seen = std::collections::HashSet::with_capacity(previous_ids.len());
        for &id in previous_ids {
            let idx = id as usize;
            if idx < vocab_size && seen.insert(id) {
                adjusted[idx] = penalized_logit(adjusted[idx], repetition_penalty);
            }
        }
    }

    if temperature_degenerate(temperature) {
        return argmax_f32(&adjusted);
    }

    if temperature != 1.0 {
        let inv_temp = 1.0 / temperature;
        for v in &mut adjusted {
            *v *= inv_temp;
        }
    }

    let mut indices: Vec<usize> = (0..vocab_size).collect();
    if top_k > 0 && top_k < vocab_size {
        indices.select_nth_unstable_by(top_k - 1, |&a, &b| {
            adjusted[b]
                .partial_cmp(&adjusted[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        indices.truncate(top_k);
    }
    indices.sort_unstable_by(|&a, &b| {
        adjusted[b]
            .partial_cmp(&adjusted[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let max_logit = indices
        .iter()
        .map(|&i| adjusted[i])
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return argmax_f32(&adjusted);
    }
    let mut probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i, (adjusted[i] - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_f32(&adjusted);
    }
    for (_, p) in &mut probs {
        *p /= sum;
    }

    if top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        let new_sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= new_sum;
        }
    }

    let r = uniform_f32_from_u64(xorshift64_next(rng_state));
    let mut cumsum = 0.0f32;
    for &(idx, p) in &probs {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

#[test]
#[ignore = "perf microbench, not a correctness check; run explicitly with --ignored"]
fn microbench_sample_full_logits_default_issue_config() {
    const VOCAB_SIZE: usize = 248_320; // Qwen3.5 real vocab (issue #650).
    const ITERS: usize = 300;

    let logits: Vec<f32> = (0..VOCAB_SIZE as u64)
        .map(|i| {
            let h = i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (h as f32 / u64::MAX as f32) * 20.0 - 10.0
        })
        .collect();
    // A 64-token recent-history window, as a real decode loop would pass
    // (previous_ids = prompt + generated-so-far), with duplicates.
    let previous_ids: Vec<u32> = (0..64u32).map(|i| (i * 4099) % VOCAB_SIZE as u32).collect();

    let cfg = SamplingConfig {
        min_p: 0.0,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };
    let gen_cfg = GenerateConfig {
        temperature: cfg.temperature,
        top_k: cfg.top_k,
        top_p: cfg.top_p,
        repetition_penalty: cfg.repetition_penalty,
        ..Default::default()
    };

    let mut rng_before = 0xDEAD_BEEFu64;
    let before_start = std::time::Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(old_full_vocab_sample(
            &logits,
            cfg.temperature,
            cfg.top_k,
            cfg.top_p,
            cfg.repetition_penalty,
            &previous_ids,
            &mut rng_before,
        ));
    }
    let before_elapsed = before_start.elapsed();

    let mut rng_after = 0xDEAD_BEEFu64;
    let after_start = std::time::Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(sample_full_logits(
            &logits,
            &gen_cfg,
            &previous_ids,
            &mut rng_after,
            0.0,
            0.0,
        ));
    }
    let after_elapsed = after_start.elapsed();

    let before_us_per_call = before_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    let after_us_per_call = after_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    eprintln!(
        "microbench sample_full_logits @ vocab={VOCAB_SIZE}, iters={ITERS}, cfg=(temp=0.7,top_k=40,top_p=0.9,rep_penalty=1.1):\n\
             before (old_full_vocab_sample): {before_elapsed:?} total, {before_us_per_call:.1} us/call, {:.1} tok/s\n\
             after  (sample_full_logits):    {after_elapsed:?} total, {after_us_per_call:.1} us/call, {:.1} tok/s",
        1e6 / before_us_per_call,
        1e6 / after_us_per_call,
    );

    assert!(
        after_elapsed < before_elapsed,
        "optimized sample_full_logits ({after_elapsed:?}) must be faster than \
             the old full-vocab-allocating algorithm ({before_elapsed:?}) at the \
             issue's default config and vocab size"
    );
}
