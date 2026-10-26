use super::*;
use crate::mixture::Exploration;
use crate::router_refit::test_router::LiveRouter;
use crate::router_refit::{RefitBuilder, RefitConfig, RefitVerdict};
use crate::router_state::{
    RouterArtifact, RouterLineage, RouterPolicy, RouterProvenance, TrainedRepresentation,
};
use lattice_fann::training::RlooConfig;
use lattice_fann::{Activation, Layer, Network};

const REFERENCE_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct Fixture {
    live: LiveRouter,
    source: RefitSource,
    candidate: RefitCandidate,
    builder: RefitBuilder,
    episodes: EpisodeStore,
    contract: AdmissionContract,
}

fn contract() -> AdmissionContract {
    AdmissionContract {
        split_salt: "admission-integration-fixture".into(),
        heldout_fraction: 0.5,
        parent_logit_tolerance: 0.0,
        bootstrap_replicates: 32,
        bootstrap_seed: 7,
        lower_quantile: 0.1,
        regression_limit: 0.25,
        min_heldout_events: 16,
        min_effective_sample_size: 1.0,
        min_reference_tokens: 4,
    }
}

fn artifact(entropy_floor: f64, maximum_collapsed: u32, epsilon: f64) -> RouterArtifact {
    let mut layer = Layer::zeros(2, 3, Activation::Linear).unwrap();
    layer
        .weights_mut()
        .copy_from_slice(&[0.3, 0.1, -0.2, 0.15, 0.05, -0.25]);
    layer.biases_mut().copy_from_slice(&[0.2, -0.1, 0.05]);
    RouterArtifact {
        version: 1,
        adapter_names: vec!["a".into(), "b".into(), "c".into()],
        representation: TrainedRepresentation {
            embedding_model: "fixture-embedder".into(),
            pooling: "mean".into(),
            prompt_source: "last_user_message".into(),
            loader_format: "safetensors".into(),
            input_width: 2,
        },
        gate_bytes: Network::new(vec![layer]).unwrap().to_bytes(),
        provenance: RouterProvenance::recorded(
            RouterPolicy {
                kind: "softmax".into(),
                tau: Some(0.7),
                epsilon,
                tau_floor: 0.1,
                entropy_floor,
                max_consecutive_collapsed: maximum_collapsed,
                training_state: "bootstrap".into(),
                objective_version: None,
                exploration_scale: Some(0.5),
            },
            RouterLineage {
                schema_id: "fixture-schema".into(),
                base_model: "fixture-base".into(),
                parent: None,
                admission_record: None,
            },
        ),
    }
}

fn training(rate: f32) -> RlooConfig {
    RlooConfig {
        learning_rate: rate,
        aux_loss_coeff: 0.2,
        z_loss_coeff: 0.01,
    }
}

fn candidate(verdict: RefitVerdict) -> RefitCandidate {
    match verdict {
        RefitVerdict::Candidate(value) => value,
        other => panic!("fixture did not produce a candidate: {other:?}"),
    }
}

fn record(store: &mut EpisodeStore, episode: RouterEpisode, signal: EpisodeSignal) {
    let id = episode.completion_id.clone();
    let epoch = episode.policy.epoch;
    store.record(episode);
    store.judge(&id, signal, Some(epoch)).unwrap();
}

fn episode(source: &RefitSource, id: &str, context: Vec<f32>) -> RouterEpisode {
    let route = source
        .policy_router(source.gate_bytes())
        .unwrap()
        .route_explored(
            &context,
            source.adapter_names(),
            source.adapter_names().len(),
            Some(Exploration {
                scale: source.policy().exploration_scale.unwrap() as f32,
                seed: seed_from_completion_id(id),
            }),
        )
        .unwrap();
    RouterEpisode {
        completion_id: id.into(),
        policy: source.identity().clone(),
        adapter_names: source.adapter_names().to_vec(),
        context_vector: context,
        selected_scores: route.selected_scores,
        sampled_scores: route.sampled_scores,
        pre_floor_weights: route.pre_floor_weights,
        applied_weights: route.weights,
        dropped: route.dropped,
        tau: source.policy().tau.map(|tau| tau as f32),
        exploration: route.exploration,
    }
}

fn ids(contract: &AdmissionContract, population: Population, count: usize) -> Vec<String> {
    let result: Vec<_> = (0..4096)
        .map(|index| format!("integration-{index}"))
        .filter(|id| contract.split(id).unwrap() == population)
        .take(count)
        .collect();
    assert_eq!(result.len(), count);
    result
}

fn fixture_with(
    contract: AdmissionContract,
    rate: f32,
    floor: f64,
    maximum: u32,
    epsilon: f64,
) -> Fixture {
    let live = LiveRouter::new(artifact(floor, maximum, epsilon));
    let source = live.refit_source().unwrap();
    let mut episodes = EpisodeStore::new(64);
    for (index, id) in ids(&contract, Population::Training, 4).iter().enumerate() {
        let (context, signal) = if index % 2 == 0 {
            (vec![0.4, -0.2], EpisodeSignal::Helpful)
        } else {
            (vec![-0.3, 0.8], EpisodeSignal::Unhelpful)
        };
        record(&mut episodes, episode(&source, id, context), signal);
    }
    let mut builder = RefitBuilder::new_with_admission(
        training(rate),
        RefitConfig {
            minimum_round: 2,
            maximum_round: 2,
        },
        contract.clone(),
    )
    .unwrap();
    let proposed = candidate(builder.build_candidate(&source, &episodes).unwrap());
    assert_ne!(proposed.gate_bytes(), source.gate_bytes());
    let mut parent = source.policy_router(source.gate_bytes()).unwrap();
    let mut challenger = source.policy_router(proposed.gate_bytes()).unwrap();
    let mut ranked: Vec<_> = ids(&contract, Population::HeldOut, 128)
        .into_iter()
        .map(|id| {
            let episode = episode(&source, &id, vec![0.7, -0.4]);
            let (event, difference) = evaluate_event(
                &source,
                &mut parent,
                &mut challenger,
                &episode,
                EpisodeSignal::Helpful,
                0.0,
            )
            .unwrap();
            assert_eq!(difference, 0.0);
            (event.log_weight, episode)
        })
        .collect();
    ranked.sort_by(|left, right| left.0.total_cmp(&right.0));
    assert!(
        ranked[15].0 < ranked[112].0,
        "candidate must change the held-out density ordering"
    );
    // Synthetic fixture labels correlate with actual density ratios. The
    // integration still computes gate outputs, Gaussian ratios and bootstrap.
    for (_, event) in ranked.iter().take(16) {
        record(&mut episodes, event.clone(), EpisodeSignal::Unhelpful);
    }
    for (_, event) in ranked.iter().skip(112) {
        record(&mut episodes, event.clone(), EpisodeSignal::Helpful);
    }
    let mut reordered: Vec<_> = episodes
        .ordered_on_policy_samples(source.identity().epoch)
        .map(|(event, signal)| (event.clone(), signal))
        .collect();
    reordered.sort_by_key(|(event, _)| {
        contract.split(&event.completion_id).unwrap() == Population::Training
    });
    let mut episodes = EpisodeStore::new(64);
    for (event, signal) in reordered {
        record(&mut episodes, event, signal);
    }
    let mut builder = RefitBuilder::new_with_admission(
        training(rate),
        RefitConfig {
            minimum_round: 2,
            maximum_round: 2,
        },
        contract.clone(),
    )
    .unwrap();
    let rebound = candidate(builder.build_candidate(&source, &episodes).unwrap());
    assert_eq!(
        rebound.gate_bytes(),
        proposed.gate_bytes(),
        "held-out events recorded first must not enter the bounded gradient batch"
    );
    Fixture {
        live,
        source,
        candidate: rebound,
        builder,
        episodes,
        contract,
    }
}

fn fixture() -> Fixture {
    fixture_with(contract(), 0.03, 0.0, 1, 0.0)
}

fn reference(fixture: &Fixture, tokens: usize, nlls: [f64; 3]) -> ReferenceEvidence {
    ReferenceEvidence::fixture(
        &fixture.source,
        &fixture.candidate,
        &fixture.contract,
        REFERENCE_HASH,
        tokens,
        nlls,
    )
}

fn bounds() -> AdmissionBounds {
    AdmissionBounds {
        maximum_evaluations: 16,
    }
}

fn evaluate(
    fixture: &Fixture,
    evidence: &ReferenceEvidence,
) -> Result<AdmissionRecord, AdmissionError> {
    AdmissionEvaluator::new(&fixture.source, &fixture.contract, bounds())?.evaluate_candidate(
        &fixture.source,
        &fixture.candidate,
        &fixture.episodes,
        evidence,
        REFERENCE_HASH,
    )
}

fn mutate_heldout(
    fixture: &mut Fixture,
    change: impl FnOnce(&mut RouterEpisode),
) -> (RouterEpisode, EpisodeSignal) {
    let (original, signal) = fixture
        .episodes
        .ordered_on_policy_samples(fixture.source.identity().epoch)
        .into_iter()
        .find(|(episode, _)| {
            fixture.contract.split(&episode.completion_id).unwrap() == Population::HeldOut
        })
        .map(|(episode, signal)| (episode.clone(), signal))
        .unwrap();
    let mut changed = original.clone();
    change(&mut changed);
    record(&mut fixture.episodes, changed, signal);
    (original, signal)
}

fn assert_reason(record: &AdmissionRecord, expected: &str) {
    assert_eq!(record.verdict(), AdmissionVerdict::Rejected);
    assert!(
        record
            .reasons()
            .iter()
            .any(|reason| reason.contains(expected)),
        "{:?}",
        record.reasons()
    );
}

#[test]
fn real_candidate_positive_paired_bound_admits_without_changing_the_live_parent() {
    let fixture = fixture();
    let before = fixture.live.artifact().clone();
    let epoch = fixture.live.policy_epoch();
    let evidence = reference(&fixture, 8, [2.0, 2.0, 1000.0]);
    let record = evaluate(&fixture, &evidence).unwrap();
    assert_eq!(
        record.verdict(),
        AdmissionVerdict::Admitted,
        "{:?}",
        record.reasons()
    );
    assert!(record.importance.as_ref().unwrap().lower_bound > 0.0);
    assert_eq!(record.heldout_count, 32);
    assert_eq!(record.reference_token_count, 8);
    assert_eq!(fixture.candidate.round().population.held_out, 32);
    assert_eq!(
        record.base_mean_nll, 1000.0,
        "base NLL is recorded without gating"
    );
    assert!(record.reasons().is_empty());
    assert_eq!(record.content_hash().unwrap().len(), 64);
    assert_eq!(fixture.live.artifact(), &before);
    assert_eq!(fixture.live.policy_epoch(), epoch);
    assert_eq!(fixture.source.gate_bytes(), before.gate_bytes);
    assert!(
        fixture
            .candidate
            .round()
            .completion_ids
            .iter()
            .all(|id| fixture.contract.split(id).unwrap() == Population::Training)
    );
}

#[test]
fn full_gaussian_ratio_matches_an_independent_all_column_formula() {
    let fixture = fixture();
    let (event, signal) = fixture
        .episodes
        .ordered_on_policy_samples(fixture.source.identity().epoch)
        .into_iter()
        .find(|(event, _)| {
            fixture.contract.split(&event.completion_id).unwrap() == Population::HeldOut
        })
        .unwrap();
    let mut parent_gate = Network::from_bytes(fixture.source.gate_bytes()).unwrap();
    let mut candidate_gate = Network::from_bytes(fixture.candidate.gate_bytes()).unwrap();
    let parent_logits = parent_gate.forward(&event.context_vector).unwrap();
    let candidate_logits = candidate_gate.forward(&event.context_vector).unwrap();
    let draw = event.exploration.as_ref().unwrap();
    let terms: Vec<f64> = parent_logits
        .iter()
        .zip(candidate_logits)
        .zip(&draw.sampled_logits)
        .map(|((&parent, &candidate), &sample)| {
            let parent = f64::from(parent);
            let candidate = f64::from(candidate);
            (candidate - parent) * (f64::from(sample) - (parent + candidate) / 2.0)
                / f64::from(draw.scale).powi(2)
        })
        .collect();
    assert_eq!(terms.len(), 3);
    assert!(
        terms[2].abs() > 1e-10,
        "last-column contribution must be observable"
    );
    let mut parent = fixture
        .source
        .policy_router(fixture.source.gate_bytes())
        .unwrap();
    let mut proposed = fixture
        .source
        .policy_router(fixture.candidate.gate_bytes())
        .unwrap();
    let (weighted, difference) = evaluate_event(
        &fixture.source,
        &mut parent,
        &mut proposed,
        event,
        signal,
        0.0,
    )
    .unwrap();
    assert_eq!(difference, 0.0);
    assert!((weighted.log_weight - terms.iter().sum::<f64>()).abs() < 1e-10);
    let mut identical = fixture
        .source
        .policy_router(fixture.source.gate_bytes())
        .unwrap();
    let (unchanged, _) = evaluate_event(
        &fixture.source,
        &mut parent,
        &mut identical,
        event,
        signal,
        0.0,
    )
    .unwrap();
    assert_eq!(unchanged.log_weight, 0.0);
}

#[test]
fn each_registered_population_floor_rejects_independently() {
    for (index, reason) in [
        (0, "event count"),
        (1, "effective sample size"),
        (2, "reference token count"),
    ] {
        let mut settings = contract();
        match index {
            0 => settings.min_heldout_events = 33,
            1 => settings.min_effective_sample_size = 33.0,
            _ => settings.min_reference_tokens = 9,
        }
        let fixture = fixture_with(settings, 0.03, 0.0, 1, 0.0);
        let record = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])).unwrap();
        assert_reason(&record, reason);
        assert_eq!(
            record.reasons().len(),
            1,
            "each floor must be an independent refusal"
        );
    }
}

#[test]
fn regression_limit_is_inclusive_and_base_nll_does_not_replace_parent_nll() {
    let fixture = fixture();
    let equal = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.25, 0.0])).unwrap();
    assert_eq!(equal.verdict(), AdmissionVerdict::Admitted);
    assert_eq!(equal.regression, 0.25);
    let beyond = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.5, 1000.0])).unwrap();
    assert_reason(&beyond, "NLL regression");
    assert_eq!(beyond.regression, 0.5);
}

#[test]
fn entropy_guard_rejects_collapsed_and_empty_applied_mixtures() {
    for epsilon in [0.0, 1.0] {
        let fixture = fixture_with(contract(), 0.03, 10.0, 1, epsilon);
        if epsilon == 1.0 {
            assert!(
                fixture
                    .candidate
                    .outputs()
                    .iter()
                    .all(|output| output.route.weights.is_empty())
            );
        }
        let record = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])).unwrap();
        assert_reason(&record, "entropy guard");
        assert_eq!(record.consecutive_collapsed_rounds, 1);
    }
}

#[test]
fn missing_unreadable_or_incomplete_contract_never_defaults() {
    assert!(matches!(
        parse_contract(None),
        Err(AdmissionError::MissingContract)
    ));
    assert!(parse_contract(Some(b"not-json")).is_err());
    assert!(parse_contract(Some(b"{}")).is_err());
    let encoded = serde_json::to_vec(&contract()).unwrap();
    assert_eq!(parse_contract(Some(&encoded)).unwrap(), contract());
}

#[test]
fn legacy_candidate_and_mismatched_registered_contract_are_rejected() {
    let mut fixture = fixture();
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    let mut legacy = RefitBuilder::new(
        training(0.03),
        RefitConfig {
            minimum_round: 2,
            maximum_round: 2,
        },
    )
    .unwrap();
    let mut training_only = EpisodeStore::new(4);
    for (event, signal) in fixture
        .episodes
        .ordered_on_policy_samples(fixture.source.identity().epoch)
    {
        if fixture.contract.split(&event.completion_id).unwrap() == Population::Training {
            record(&mut training_only, event.clone(), signal);
        }
    }
    fixture.candidate = candidate(
        legacy
            .build_candidate(&fixture.source, &training_only)
            .unwrap(),
    );
    assert!(matches!(
        evaluate(&fixture, &evidence),
        Err(AdmissionError::CandidateBinding(_))
    ));

    let mut fixture = self::fixture();
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    fixture.contract.bootstrap_seed += 1;
    assert!(matches!(
        evaluate(&fixture, &evidence),
        Err(AdmissionError::CandidateBinding(_))
    ));
}

#[test]
fn mismatched_source_and_reference_bindings_are_rejected() {
    let fixture = fixture();
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    let mut different = artifact(0.0, 1, 0.0);
    different.version += 1;
    let other_source = LiveRouter::new(different).refit_source().unwrap();
    assert!(
        AdmissionEvaluator::new(&other_source, &fixture.contract, bounds())
            .unwrap()
            .evaluate_candidate(
                &other_source,
                &fixture.candidate,
                &fixture.episodes,
                &evidence,
                REFERENCE_HASH,
            )
            .is_err()
    );

    let mut wrong_reference = evidence.clone();
    wrong_reference.candidate_gate_hash.push('x');
    assert!(matches!(
        evaluate(&fixture, &wrong_reference),
        Err(AdmissionError::ReferenceBinding(_))
    ));
    assert!(matches!(
        AdmissionEvaluator::new(&fixture.source, &fixture.contract, bounds())
            .unwrap()
            .evaluate_candidate(
                &fixture.source,
                &fixture.candidate,
                &fixture.episodes,
                &evidence,
                "different-reference",
            ),
        Err(AdmissionError::ReferenceBinding(_))
    ));
}

fn shifted_record(source: &RefitSource, event: &mut RouterEpisode) -> f64 {
    let mut layer = Layer::zeros(
        source.representation().input_width as usize,
        source.adapter_names().len(),
        Activation::Linear,
    )
    .unwrap();
    for (index, name) in source.adapter_names().iter().enumerate() {
        layer.biases_mut()[index] = event
            .selected_scores
            .iter()
            .find(|(column, _)| column == name)
            .unwrap()
            .1;
    }
    let original = layer.biases()[0];
    let shifted = f32::from_bits(original.to_bits() + 1);
    layer.biases_mut()[0] = shifted;
    let difference = (f64::from(shifted) - f64::from(original)).abs();
    let gate = Network::new(vec![layer]).unwrap();
    let draw = event.exploration.as_ref().unwrap();
    let route = source
        .policy_router(&gate.to_bytes())
        .unwrap()
        .route_explored(
            &event.context_vector,
            source.adapter_names(),
            source.adapter_names().len(),
            Some(Exploration {
                scale: draw.scale,
                seed: draw.seed,
            }),
        )
        .unwrap();
    event.selected_scores = route.selected_scores;
    event.sampled_scores = route.sampled_scores;
    event.pre_floor_weights = route.pre_floor_weights;
    event.applied_weights = route.weights;
    event.dropped = route.dropped;
    event.exploration = route.exploration;
    difference
}

#[test]
fn parent_logit_tolerance_is_registered_and_inclusive_without_relaxing_draw_replay() {
    for multiplier in [0.0, 1.0, 2.0] {
        let mut fixture = fixture();
        let (mut original, signal) = mutate_heldout(&mut fixture, |_| {});
        let difference = shifted_record(&fixture.source, &mut original);
        assert!(difference > 0.0);
        record(&mut fixture.episodes, original, signal);
        fixture.contract.parent_logit_tolerance = difference * multiplier;
        let mut builder = RefitBuilder::new_with_admission(
            training(0.03),
            RefitConfig {
                minimum_round: 2,
                maximum_round: 2,
            },
            fixture.contract.clone(),
        )
        .unwrap();
        fixture.candidate = candidate(
            builder
                .build_candidate(&fixture.source, &fixture.episodes)
                .unwrap(),
        );
        let result = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0]));
        if multiplier == 0.0 {
            assert!(matches!(result, Err(AdmissionError::Provenance { .. })));
        } else {
            let record = result.unwrap();
            assert_eq!(record.maximum_parent_logit_difference, difference);
            assert_eq!(record.verdict(), AdmissionVerdict::Admitted);
        }
    }
}

#[test]
fn named_logits_may_reorder_but_the_trained_schema_must_match() {
    let mut fixture = fixture();
    mutate_heldout(&mut fixture, |event| event.selected_scores.reverse());
    assert_eq!(
        evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0]))
            .unwrap()
            .verdict(),
        AdmissionVerdict::Admitted
    );
    mutate_heldout(&mut fixture, |event| event.adapter_names.swap(0, 1));
    assert!(matches!(
        evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])),
        Err(AdmissionError::Provenance { .. })
    ));
}

#[test]
fn malformed_named_logits_context_and_draw_are_rejected() {
    for fault in 0..8 {
        let mut fixture = fixture();
        mutate_heldout(&mut fixture, |event| match fault {
            0 => event.selected_scores[1].0 = event.selected_scores[0].0.clone(),
            1 => {
                event.selected_scores.pop();
            }
            2 => event.selected_scores[0].1 = f32::NAN,
            3 => event.context_vector[0] = f32::INFINITY,
            4 => event.exploration.as_mut().unwrap().sampled_logits[0] = f32::NAN,
            5 => event.exploration.as_mut().unwrap().log_density = f64::INFINITY,
            6 => event.exploration.as_mut().unwrap().scale *= 2.0,
            _ => event.exploration.as_mut().unwrap().seed ^= 1,
        });
        assert!(
            matches!(
                evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])),
                Err(AdmissionError::Provenance { .. })
            ),
            "fault {fault}"
        );
    }
}

#[test]
fn finite_draw_density_or_applied_weight_tampering_fails_replay() {
    for fault in 0..3 {
        let mut fixture = fixture();
        mutate_heldout(&mut fixture, |event| match fault {
            0 => event.exploration.as_mut().unwrap().sampled_logits[0] += 0.1,
            1 => event.exploration.as_mut().unwrap().log_density += 0.1,
            _ => event.applied_weights[0].1 += 0.1,
        });
        assert!(
            matches!(
                evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])),
                Err(AdmissionError::Provenance { .. })
            ),
            "fault {fault}"
        );
    }
}

#[test]
fn a_far_actual_candidate_loses_effective_sample_size_and_is_rejected() {
    let mut settings = contract();
    settings.min_effective_sample_size = 4.0;
    let fixture = fixture_with(settings, 10_000.0, 0.0, 1, 0.0);
    let record = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])).unwrap();
    assert!(record.importance.as_ref().unwrap().effective_sample_size < 4.0);
    assert_reason(&record, "effective sample size");
}

#[test]
fn absent_heldout_evidence_and_constant_rewards_cannot_pass() {
    let mut fixture = fixture();
    let records: Vec<_> = fixture
        .episodes
        .ordered_on_policy_samples(fixture.source.identity().epoch)
        .map(|(event, _)| event.clone())
        .collect();
    for event in &records {
        if fixture.contract.split(&event.completion_id).unwrap() == Population::HeldOut {
            record(&mut fixture.episodes, event.clone(), EpisodeSignal::Helpful);
        }
    }
    let constant = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])).unwrap();
    assert_eq!(constant.importance.as_ref().unwrap().lower_bound, 0.0);
    assert_reason(&constant, "strictly positive");

    fixture.episodes = EpisodeStore::new(4);
    for event in records {
        if fixture.contract.split(&event.completion_id).unwrap() == Population::Training {
            record(&mut fixture.episodes, event, EpisodeSignal::Helpful);
        }
    }
    let absent = evaluate(&fixture, &reference(&fixture, 8, [2.0, 2.0, 2.0])).unwrap();
    assert_eq!(absent.heldout_count, 0);
    assert!(absent.importance.is_none());
    assert_reason(&absent, "event count");
    assert_reason(&absent, "effective sample size");
    assert_reason(&absent, "strictly positive");
}

#[test]
fn entropy_history_advances_once_per_completed_round_and_not_on_errors() {
    let mut fixture = fixture_with(contract(), 0.03, 10.0, 2, 0.0);
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    let mut evaluator =
        AdmissionEvaluator::new(&fixture.source, &fixture.contract, bounds()).unwrap();
    let first = evaluator
        .evaluate_candidate(
            &fixture.source,
            &fixture.candidate,
            &fixture.episodes,
            &evidence,
            REFERENCE_HASH,
        )
        .unwrap();
    assert_eq!(first.consecutive_collapsed_rounds, 1);
    assert_eq!(first.verdict(), AdmissionVerdict::Admitted);
    assert!(matches!(
        evaluator.evaluate_candidate(
            &fixture.source,
            &fixture.candidate,
            &fixture.episodes,
            &evidence,
            REFERENCE_HASH
        ),
        Err(AdmissionError::AlreadyEvaluated)
    ));
    fixture.candidate = candidate(
        fixture
            .builder
            .build_candidate(&fixture.source, &fixture.episodes)
            .unwrap(),
    );
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    let (original, signal) =
        mutate_heldout(&mut fixture, |event| event.context_vector[0] = f32::NAN);
    assert!(matches!(
        evaluator.evaluate_candidate(
            &fixture.source,
            &fixture.candidate,
            &fixture.episodes,
            &evidence,
            REFERENCE_HASH
        ),
        Err(AdmissionError::Provenance { .. })
    ));
    record(&mut fixture.episodes, original, signal);
    let second = evaluator
        .evaluate_candidate(
            &fixture.source,
            &fixture.candidate,
            &fixture.episodes,
            &evidence,
            REFERENCE_HASH,
        )
        .unwrap();
    assert_eq!(second.consecutive_collapsed_rounds, 2);
    assert_reason(&second, "entropy guard");
}

#[test]
fn explicit_evaluation_budget_refuses_without_advancing_history() {
    let mut fixture = fixture_with(contract(), 0.03, 10.0, 2, 0.0);
    assert!(
        AdmissionEvaluator::new(
            &fixture.source,
            &fixture.contract,
            AdmissionBounds {
                maximum_evaluations: 0
            },
        )
        .is_err()
    );
    let mut evaluator = AdmissionEvaluator::new(
        &fixture.source,
        &fixture.contract,
        AdmissionBounds {
            maximum_evaluations: 1,
        },
    )
    .unwrap();
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    let first = evaluator
        .evaluate_candidate(
            &fixture.source,
            &fixture.candidate,
            &fixture.episodes,
            &evidence,
            REFERENCE_HASH,
        )
        .unwrap();
    assert_eq!(first.maximum_evaluations, 1);
    assert_eq!(evaluator.consecutive_collapsed_rounds(), 1);
    fixture.candidate = candidate(
        fixture
            .builder
            .build_candidate(&fixture.source, &fixture.episodes)
            .unwrap(),
    );
    let evidence = reference(&fixture, 8, [2.0, 2.0, 2.0]);
    assert!(matches!(
        evaluator.evaluate_candidate(
            &fixture.source,
            &fixture.candidate,
            &fixture.episodes,
            &evidence,
            REFERENCE_HASH
        ),
        Err(AdmissionError::EvaluationBudgetExhausted)
    ));
    assert_eq!(evaluator.consecutive_collapsed_rounds(), 1);
}

#[cfg(all(
    feature = "serving-support",
    feature = "f16",
    target_os = "macos",
    feature = "metal-gpu"
))]
mod reference_embedder_contract {
    use super::*;
    use crate::forward::metal_qwen35::{
        ChatMessage, MetalQwen35State, format_chat_template, reference_scoring_fixture,
    };
    use crate::measurement::gpu_test_lock;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::cell::Cell;

    const REPRESENTATION_REFUSAL: &str =
        "embedder representation differs from registered serving representation";

    struct RecordingEmbedder {
        identity: String,
        loader_format: String,
        prompt_source: String,
        dimensions: usize,
        pooling_name: String,
        pooling_calls: Cell<usize>,
        embed_calls: Cell<usize>,
    }

    impl RecordingEmbedder {
        fn new(source: &RefitSource) -> Self {
            let representation = source.representation();
            Self {
                identity: representation.embedding_model.clone(),
                loader_format: representation.loader_format.clone(),
                prompt_source: representation.prompt_source.clone(),
                dimensions: usize::try_from(representation.input_width).unwrap(),
                pooling_name: representation.pooling.clone(),
                pooling_calls: Cell::new(0),
                embed_calls: Cell::new(0),
            }
        }
    }

    impl ReferenceEmbedder for RecordingEmbedder {
        type Pooling = ();

        fn identity(&self) -> &str {
            &self.identity
        }

        fn loader_format(&self) -> &str {
            &self.loader_format
        }

        fn prompt_source(&self) -> &str {
            &self.prompt_source
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn max_context(&self) -> usize {
            128
        }

        fn parse_pooling(&self, name: &str) -> Result<Self::Pooling, String> {
            self.pooling_calls.set(self.pooling_calls.get() + 1);
            if name != self.pooling_name {
                Err(format!("recording embedder refuses pooling name: {name}"))
            } else {
                Ok(())
            }
        }

        fn embed(&self, _text: &str, _pooling: &Self::Pooling) -> Result<Vec<f32>, String> {
            self.embed_calls.set(self.embed_calls.get() + 1);
            Err("unexpected embedding before refusal".into())
        }
    }

    fn score_contract(
        fixture: &Fixture,
        state: &mut MetalQwen35State,
        embedder: &RecordingEmbedder,
    ) -> Result<ReferenceEvidence, ReferenceError> {
        assert!(!state.has_lora_adapter());
        assert!(fixture.candidate.matches_source(&fixture.source));
        let contract_hash = fixture.contract.hash().unwrap();
        assert_eq!(
            fixture.candidate.admission_contract_hash(),
            Some(contract_hash.as_str())
        );
        let tokenizer_json = serde_json::json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
            "added_tokens": [{"id": 2, "content": "x".repeat(128), "special": true}]
        });
        let tokenizer = BpeTokenizer::from_tokenizer_json_str(&tokenizer_json.to_string()).unwrap();
        let examples = [ReferenceExample {
            user_prompt: "a".into(),
            completion: "b".into(),
        }];
        let bounds = ReferenceBounds {
            max_examples: 1,
            max_tokens: 16,
        };
        assert!(examples[0].user_prompt.len() + 2 <= embedder.max_context());
        let rendered = format_chat_template(&[ChatMessage::user(&examples[0].user_prompt)]);
        let span = lattice_tokenizer::__internal::max_token_byte_span(&tokenizer);
        assert_eq!(span, 128);
        let token_bound = (rendered.len() + examples[0].completion.len()).div_ceil(span);
        assert!(token_bound <= state.max_context());
        assert!(token_bound <= bounds.max_tokens);
        score_metal_reference(
            &fixture.source,
            &fixture.candidate,
            &fixture.contract,
            &examples,
            bounds,
            state,
            &tokenizer,
            embedder,
            &[],
            fixture.source.base_model(),
        )
    }

    fn assert_refusal(
        result: Result<ReferenceEvidence, ReferenceError>,
        embedder: &RecordingEmbedder,
        expected: &str,
        pooling_calls: usize,
    ) {
        assert_eq!(
            embedder.embed_calls.get(),
            0,
            "refusal must precede embedding"
        );
        match result {
            Err(ReferenceError::Invalid(message)) => assert_eq!(message, expected),
            _ => panic!("expected the specific Invalid refusal"),
        }
        assert_eq!(embedder.pooling_calls.get(), pooling_calls);
    }

    #[test]
    fn reference_embedder_identity_mismatch_before_embedding() {
        let _gpu = gpu_test_lock();
        let (cfg, weights) = reference_scoring_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 8)
            .expect("representation witness requires a real Metal device");
        let fixture = fixture();
        let mut embedder = RecordingEmbedder::new(&fixture.source);
        embedder.identity.push_str("-different");
        let result = score_contract(&fixture, &mut state, &embedder);
        assert_refusal(result, &embedder, REPRESENTATION_REFUSAL, 0);
    }

    #[test]
    fn reference_embedder_loader_mismatch_before_embedding() {
        let _gpu = gpu_test_lock();
        let (cfg, weights) = reference_scoring_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 8)
            .expect("representation witness requires a real Metal device");
        let fixture = fixture();
        let mut embedder = RecordingEmbedder::new(&fixture.source);
        embedder.loader_format.push_str("-different");
        let result = score_contract(&fixture, &mut state, &embedder);
        assert_refusal(result, &embedder, REPRESENTATION_REFUSAL, 0);
    }

    #[test]
    fn reference_embedder_prompt_source_mismatch_before_embedding() {
        let _gpu = gpu_test_lock();
        let (cfg, weights) = reference_scoring_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 8)
            .expect("representation witness requires a real Metal device");
        let fixture = fixture();
        let mut embedder = RecordingEmbedder::new(&fixture.source);
        embedder.prompt_source.push_str("-different");
        let result = score_contract(&fixture, &mut state, &embedder);
        assert_refusal(result, &embedder, REPRESENTATION_REFUSAL, 0);
    }

    #[test]
    fn reference_embedder_width_mismatch_before_embedding() {
        let _gpu = gpu_test_lock();
        let (cfg, weights) = reference_scoring_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 8)
            .expect("representation witness requires a real Metal device");
        let fixture = fixture();
        let mut embedder = RecordingEmbedder::new(&fixture.source);
        embedder.dimensions += 1;
        let result = score_contract(&fixture, &mut state, &embedder);
        assert_refusal(result, &embedder, REPRESENTATION_REFUSAL, 0);
    }

    #[test]
    fn reference_embedder_invalid_pooling_before_embedding() {
        let _gpu = gpu_test_lock();
        let (cfg, weights) = reference_scoring_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 8)
            .expect("representation witness requires a real Metal device");
        let fixture = fixture();
        let mut embedder = RecordingEmbedder::new(&fixture.source);
        assert_eq!(fixture.source.representation().pooling, "mean");
        embedder.pooling_name = "mean_visual".into();
        let result = score_contract(&fixture, &mut state, &embedder);
        assert_refusal(
            result,
            &embedder,
            "recording embedder refuses pooling name: mean",
            1,
        );
    }
}
