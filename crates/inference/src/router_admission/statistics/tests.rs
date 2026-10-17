use super::*;

fn contract() -> AdmissionContract {
    AdmissionContract {
        split_salt: "registered experiment salt".to_owned(),
        heldout_fraction: 0.25,
        parent_logit_tolerance: 0.0,
        bootstrap_replicates: 64,
        bootstrap_seed: 17,
        lower_quantile: 0.25,
        regression_limit: 0.01,
        min_heldout_events: 8,
        min_effective_sample_size: 4.0,
        min_reference_tokens: 32,
    }
}

fn events() -> Vec<WeightedEvent> {
    vec![
        WeightedEvent {
            log_weight: 3.0_f64.ln(),
            reward: true,
        },
        WeightedEvent {
            log_weight: 0.0,
            reward: false,
        },
    ]
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn every_registered_setting_is_required() {
    let value = serde_json::to_value(contract()).unwrap();
    let fields: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
    assert_eq!(fields.len(), 10);
    for field in fields {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(&field);
        assert!(
            serde_json::from_value::<AdmissionContract>(missing).is_err(),
            "missing {field} was accepted"
        );
    }
    assert!(serde_json::from_str::<AdmissionContract>("{}").is_err());
}

#[test]
fn malformed_and_unknown_contract_settings_refuse() {
    assert!(serde_json::from_str::<AdmissionContract>("not json").is_err());
    let mut value = serde_json::to_value(contract()).unwrap();
    value["invented_setting"] = serde_json::json!(1);
    assert!(serde_json::from_value::<AdmissionContract>(value).is_err());
}

#[test]
fn nonfinite_registered_numbers_refuse_evaluation_and_hashing() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for field in 0..5 {
            let mut settings = contract();
            match field {
                0 => settings.heldout_fraction = value,
                1 => settings.parent_logit_tolerance = value,
                2 => settings.lower_quantile = value,
                3 => settings.regression_limit = value,
                _ => settings.min_effective_sample_size = value,
            }
            assert!(settings.validate().is_err());
            assert!(settings.hash().is_err());
            assert!(settings.split("completion").is_err());
            assert!(evaluate(&events(), &settings).is_err());
        }
    }
}

#[test]
fn out_of_domain_settings_refuse_without_numeric_defaults() {
    for value in [-1.0, 0.0, 1.0, 2.0] {
        let mut settings = contract();
        settings.heldout_fraction = value;
        assert!(settings.validate().is_err());
        settings = contract();
        settings.lower_quantile = value;
        assert!(settings.validate().is_err());
    }
    for field in 0..6 {
        let mut settings = contract();
        match field {
            0 => settings.split_salt.clear(),
            1 => settings.parent_logit_tolerance = -0.1,
            2 => settings.bootstrap_replicates = 0,
            3 => settings.min_heldout_events = 0,
            4 => settings.min_effective_sample_size = 0.0,
            _ => settings.min_reference_tokens = 0,
        }
        assert!(settings.validate().is_err());
    }
    let mut settings = contract();
    settings.regression_limit = -0.25;
    settings.bootstrap_seed = 0;
    assert!(settings.validate().is_ok());
}

#[test]
fn contract_hash_covers_every_registered_value() {
    let settings = contract();
    let hash = settings.hash().unwrap();
    assert_eq!(hash.len(), 64);
    assert_eq!(hash, settings.clone().hash().unwrap());
    let mut changed = Vec::new();
    for field in 0..10 {
        let mut candidate = settings.clone();
        match field {
            0 => candidate.split_salt.push('x'),
            1 => candidate.heldout_fraction = 0.5,
            2 => candidate.parent_logit_tolerance = 0.1,
            3 => candidate.bootstrap_replicates += 1,
            4 => candidate.bootstrap_seed += 1,
            5 => candidate.lower_quantile = 0.5,
            6 => candidate.regression_limit = 0.02,
            7 => candidate.min_heldout_events += 1,
            8 => candidate.min_effective_sample_size = 5.0,
            _ => candidate.min_reference_tokens += 1,
        }
        changed.push(candidate.hash().unwrap());
    }
    assert!(changed.iter().all(|candidate| candidate != &hash));
}

#[test]
fn split_is_reproducible_and_uses_the_declared_fixed_point_rule() {
    let settings = contract();
    let id = "completion-17";
    let mut bytes = SPLIT_DOMAIN.to_vec();
    bytes.extend_from_slice(&(settings.split_salt.len() as u64).to_be_bytes());
    bytes.extend_from_slice(settings.split_salt.as_bytes());
    bytes.extend_from_slice(&(id.len() as u64).to_be_bytes());
    bytes.extend_from_slice(id.as_bytes());
    let digest = Sha256::digest(bytes);
    let prefix: [u8; 8] = digest[..8].try_into().unwrap();
    let bucket = u64::from_be_bytes(prefix) >> 11;
    let fraction = bucket as f64 / EXACT_INTEGER_LIMIT as f64;
    let expected = if fraction < settings.heldout_fraction {
        Population::HeldOut
    } else {
        Population::Training
    };
    assert_eq!(settings.split(id).unwrap(), expected);
    assert_eq!(settings.clone().split(id).unwrap(), expected);
    let mut other_seed = settings.clone();
    other_seed.bootstrap_seed += 1;
    assert_eq!(other_seed.split(id).unwrap(), expected);
    assert_eq!(
        settings.split(""),
        Err(StatisticsError::EmptyCompletionIdentity)
    );
}

#[test]
fn split_salt_and_completion_identity_are_length_framed() {
    let mut first = Sha256::new();
    hash_framed(&mut first, b"ab").unwrap();
    hash_framed(&mut first, b"c").unwrap();
    let mut second = Sha256::new();
    hash_framed(&mut second, b"a").unwrap();
    hash_framed(&mut second, b"bc").unwrap();
    assert_ne!(first.finalize(), second.finalize());
    let mut salted = contract();
    let original: Vec<_> = (0..128)
        .map(|id| salted.split(&id.to_string()).unwrap())
        .collect();
    salted.split_salt.push('x');
    let changed: Vec<_> = (0..128)
        .map(|id| salted.split(&id.to_string()).unwrap())
        .collect();
    assert_ne!(original, changed);
    assert!(original.contains(&Population::Training));
    assert!(original.contains(&Population::HeldOut));
}

#[test]
fn weighted_means_and_ess_match_hand_calculation() {
    let result = evaluate(&events(), &contract()).unwrap();
    assert_eq!(result.count, 2);
    close(result.candidate_mean, 0.75);
    close(result.parent_mean, 0.5);
    close(result.delta, 0.25);
    close(result.effective_sample_size, 1.6);
}

#[test]
fn paired_bootstrap_uses_the_same_resampled_parent_population() {
    let data = events();
    let all_helpful = moments(&data, [0, 0].into_iter()).unwrap();
    close(all_helpful.candidate_mean, 1.0);
    close(all_helpful.parent_mean, 1.0);
    close(all_helpful.delta, 0.0);
    let all_unhelpful = moments(&data, [1, 1].into_iter()).unwrap();
    close(all_unhelpful.delta, 0.0);
    let mixed = moments(&data, [1, 0].into_iter()).unwrap();
    close(mixed.delta, 0.25);
    let estimate = evaluate(&data, &contract()).unwrap();
    close(estimate.lower_bound, 0.0);
}

#[test]
fn seeded_bootstrap_bound_matches_a_hand_paired_resample() {
    let data = [
        WeightedEvent {
            log_weight: 3.0_f64.ln(),
            reward: true,
        },
        WeightedEvent {
            log_weight: 0.0,
            reward: false,
        },
        WeightedEvent {
            log_weight: 7.0_f64.ln(),
            reward: true,
        },
    ];
    let mut settings = contract();
    settings.bootstrap_seed = 0;
    settings.bootstrap_replicates = 1;
    // The pinned first three SplitMix64 words modulo three are [1, 0, 1].
    // Their weights are [1, 3, 1], and their rewards are [0, 1, 0].
    let result = evaluate(&data, &settings).unwrap();
    close(result.lower_bound, 3.0 / 5.0 - 1.0 / 3.0);
    assert!((result.lower_bound - result.delta).abs() > 0.01);
}

#[test]
fn constant_weights_have_zero_paired_difference_in_every_resample() {
    let data: Vec<_> = [true, false, true, true, false]
        .into_iter()
        .map(|reward| WeightedEvent {
            log_weight: -1_000.0,
            reward,
        })
        .collect();
    let estimate = evaluate(&data, &contract()).unwrap();
    close(estimate.candidate_mean, 0.6);
    close(estimate.parent_mean, 0.6);
    close(estimate.effective_sample_size, 5.0);
    close(estimate.delta, 0.0);
    close(estimate.lower_bound, 0.0);
}

#[test]
fn common_log_weight_shift_does_not_change_evidence() {
    let data = events();
    let expected = evaluate(&data, &contract()).unwrap();
    for shift in [-10_000.0, 10_000.0] {
        let shifted: Vec<_> = data
            .iter()
            .map(|event| WeightedEvent {
                log_weight: event.log_weight + shift,
                reward: event.reward,
            })
            .collect();
        let result = evaluate(&shifted, &contract()).unwrap();
        close(result.candidate_mean, expected.candidate_mean);
        close(result.delta, expected.delta);
        close(result.effective_sample_size, expected.effective_sample_size);
        close(result.lower_bound, expected.lower_bound);
    }
}

#[test]
fn each_resample_normalizes_even_if_all_its_weights_underflow_globally() {
    let data = [
        WeightedEvent {
            log_weight: 0.0,
            reward: false,
        },
        WeightedEvent {
            log_weight: -1_000.0,
            reward: true,
        },
        WeightedEvent {
            log_weight: -1_001.0,
            reward: false,
        },
    ];
    let only_small = moments(&data, [1, 2, 1].into_iter()).unwrap();
    close(only_small.candidate_mean, 2.0 / (2.0 + (-1.0_f64).exp()));
    close(only_small.parent_mean, 2.0 / 3.0);
    assert!(
        evaluate(&data, &contract())
            .unwrap()
            .lower_bound
            .is_finite()
    );
}

#[test]
fn finite_extreme_log_weights_remain_stable() {
    let data = [
        WeightedEvent {
            log_weight: f64::MAX,
            reward: true,
        },
        WeightedEvent {
            log_weight: -f64::MAX,
            reward: false,
        },
    ];
    let result = evaluate(&data, &contract()).unwrap();
    close(result.candidate_mean, 1.0);
    close(result.parent_mean, 0.5);
    close(result.effective_sample_size, 1.0);
    close(moments(&data, [1, 1].into_iter()).unwrap().delta, 0.0);
}

#[test]
fn degenerate_rewards_and_single_event_are_evidence_not_an_admission() {
    for reward in [false, true] {
        let data = [WeightedEvent {
            log_weight: -10_000.0,
            reward,
        }];
        let result = evaluate(&data, &contract()).unwrap();
        assert_eq!(result.count, 1);
        close(result.effective_sample_size, 1.0);
        close(result.delta, 0.0);
        close(result.lower_bound, 0.0);
    }
}

#[test]
fn insufficient_registered_floors_do_not_discard_computable_evidence() {
    let mut settings = contract();
    settings.min_heldout_events = 1_000;
    settings.min_effective_sample_size = 1_000.0;
    settings.min_reference_tokens = 1_000;
    let result = evaluate(&events(), &settings).unwrap();
    assert_eq!(result.count, 2);
    close(result.effective_sample_size, 1.6);
}

#[test]
fn empty_and_nonfinite_events_refuse() {
    assert_eq!(
        evaluate(&[], &contract()),
        Err(StatisticsError::EmptyPopulation)
    );
    for log_weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut data = events();
        data[1].log_weight = log_weight;
        assert_eq!(
            evaluate(&data, &contract()),
            Err(StatisticsError::NonFiniteLogWeight { event: 1 })
        );
    }
}

#[test]
fn empirical_quantile_is_inverse_cdf_without_interpolation() {
    let mut values = [0.4, -0.2, 0.2, 0.0];
    close(empirical_quantile(&mut values, 0.25).unwrap(), -0.2);
    close(empirical_quantile(&mut values, 0.5).unwrap(), 0.0);
    close(empirical_quantile(&mut values, 0.500_001).unwrap(), 0.2);
    close(empirical_quantile(&mut values, 0.99).unwrap(), 0.4);
    assert!(empirical_quantile(&mut [f64::NAN], 0.5).is_err());
    assert!(empirical_quantile(&mut [], 0.5).is_err());
}

#[test]
fn bootstrap_is_reproducible_from_the_registered_seed() {
    let settings = contract();
    assert_eq!(
        evaluate(&events(), &settings).unwrap(),
        evaluate(&events(), &settings).unwrap()
    );
    let mut first = SplitMix64(settings.bootstrap_seed);
    let mut same = SplitMix64(settings.bootstrap_seed);
    let mut different = SplitMix64(settings.bootstrap_seed + 1);
    let draws: Vec<_> = (0..32).map(|_| first.index(7).unwrap()).collect();
    let repeated: Vec<_> = (0..32).map(|_| same.index(7).unwrap()).collect();
    let changed: Vec<_> = (0..32).map(|_| different.index(7).unwrap()).collect();
    assert_eq!(draws, repeated);
    assert_ne!(draws, changed);
    assert!(draws.iter().all(|&index| index < 7));
}

#[test]
fn splitmix64_sequence_is_pinned() {
    let mut rng = SplitMix64(0);
    assert_eq!(rng.next(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(rng.next(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(rng.next(), 0x06c4_5d18_8009_454f);
    assert_eq!(rng.index(1).unwrap(), 0);
    assert_eq!(rng.index(0), Err(StatisticsError::EmptyPopulation));
}

#[test]
fn impossible_allocation_is_a_checked_error() {
    assert_eq!(
        reserved_vec::<f64>(usize::MAX, "test"),
        Err(StatisticsError::Capacity { quantity: "test" })
    );
    #[cfg(target_pointer_width = "64")]
    {
        let mut settings = contract();
        settings.bootstrap_replicates = usize::MAX;
        assert_eq!(
            settings.validate(),
            Err(StatisticsError::Unrepresentable {
                quantity: "bootstrap_replicates",
            })
        );
    }
}
