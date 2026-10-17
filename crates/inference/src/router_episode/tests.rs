use super::*;

fn policy(epoch: u64) -> PolicyIdentity {
    PolicyIdentity {
        artifact_version: "1:8363b3f04c702da3".into(),
        schema_id: "schema-tech-casual-v1".into(),
        epoch,
    }
}

fn episode(id: &str, epoch: u64) -> RouterEpisode {
    RouterEpisode {
        completion_id: id.into(),
        policy: policy(epoch),
        adapter_names: vec!["technical".into(), "casual".into()],
        context_vector: vec![0.25, -0.5, 1.0],
        selected_scores: vec![("technical".into(), 0.8), ("casual".into(), 0.2)],
        sampled_scores: vec![("technical".into(), 0.8), ("casual".into(), 0.2)],
        pre_floor_weights: vec![("technical".into(), 0.6), ("casual".into(), 0.4)],
        applied_weights: vec![("technical".into(), 0.6), ("casual".into(), 0.4)],
        dropped: Vec::new(),
        tau: Some(0.7),
        #[cfg(feature = "mixture")]
        exploration: None,
    }
}

/// Nothing serving is not epoch zero. A server with no gate loaded has no
/// policy for an episode to be on, so a judgement of one is stale -- and
/// the refusals still answer as themselves, because an absent epoch must
/// not turn an unknown id into a stale judgement of nothing.
#[test]
fn a_judgement_with_nothing_serving_is_stale() {
    let mut store = EpisodeStore::new(4);
    store.record(episode("c-1", 0));
    // Control: the same episode, judged against the epoch that produced
    // it, records. Without this the staleness below would also be what a
    // store that refuses everything looks like.
    let mut control = EpisodeStore::new(4);
    control.record(episode("c-1", 0));
    assert_eq!(
        control.judge("c-1", EpisodeSignal::Helpful, Some(0)),
        Ok(JudgementOutcome::Recorded)
    );

    assert_eq!(
        store.judge("c-1", EpisodeSignal::Helpful, None),
        Ok(JudgementOutcome::Stale)
    );
    assert_eq!(store.census().stale, 1);
    match store.judge("never-routed", EpisodeSignal::Helpful, None) {
        Err(JudgementError::UnknownCompletion { .. }) => {}
        other => panic!("an unknown id must refuse, got {other:?}"),
    }
}

/// The ordinary path, and the control for every refusal below: a first
/// judgement of an on-policy episode records.
#[test]
fn a_first_judgement_of_an_on_policy_episode_records() {
    let mut store = EpisodeStore::new(8);
    store.record(episode("c-1", 3));
    assert_eq!(
        store.judge("c-1", EpisodeSignal::Helpful, Some(3)),
        Ok(JudgementOutcome::Recorded)
    );
    let census = store.census();
    assert_eq!(census.recorded, 1);
    assert_eq!(census.judged, 1);
    assert_eq!(census.unjudged, 0);
    assert_eq!(census.stale, 0);
}

/// A repeat carrying the same signal is the ordinary retry. A repeat
/// carrying a different one is refused: one of the two is wrong and the
/// store cannot tell which, so overwriting would discard a real judgement
/// on the strength of arrival order.
#[test]
fn an_identical_repeat_is_idempotent_and_a_conflicting_one_refuses() {
    let mut store = EpisodeStore::new(8);
    store.record(episode("c-1", 1));
    assert_eq!(
        store.judge("c-1", EpisodeSignal::Helpful, Some(1)),
        Ok(JudgementOutcome::Recorded)
    );
    assert_eq!(
        store.judge("c-1", EpisodeSignal::Helpful, Some(1)),
        Ok(JudgementOutcome::Idempotent),
        "an identical repeat must not refuse"
    );
    match store.judge("c-1", EpisodeSignal::Unhelpful, Some(1)) {
        Err(JudgementError::ConflictingRepeat {
            existing, incoming, ..
        }) => {
            assert_eq!(existing, "helpful");
            assert_eq!(incoming, "unhelpful");
        }
        other => panic!("expected a conflict, got {other:?}"),
    }
    // The first judgement survived the conflict rather than being replaced.
    let (_, signal) = store.get("c-1").expect("the episode is still there");
    assert_eq!(signal, Some(EpisodeSignal::Helpful));
    assert_eq!(store.census().judged, 1);
}

/// A judgement arriving after its episode's epoch ended is real, counted,
/// and not an on-policy sample for the gate that is serving now. The
/// control is the same judgement at the matching epoch.
#[test]
fn a_judgement_after_its_epoch_is_stale_and_never_an_on_policy_sample() {
    let mut store = EpisodeStore::new(8);
    store.record(episode("old", 1));
    store.record(episode("new", 2));

    assert_eq!(
        store.judge("new", EpisodeSignal::Helpful, Some(2)),
        Ok(JudgementOutcome::Recorded),
        "control: a matching epoch records"
    );
    assert_eq!(
        store.judge("old", EpisodeSignal::Helpful, Some(2)),
        Ok(JudgementOutcome::Stale)
    );

    let census = store.census();
    assert_eq!(census.judged, 2, "a stale judgement is still a judgement");
    assert_eq!(census.stale, 1);

    let samples = store.on_policy_samples(2);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0.completion_id, "new");
}

/// An unjudged episode is missing, not zero. Nothing in the store turns
/// silence into a signal, and the census reports the third number rather
/// than a ratio that would hide it.
#[test]
fn an_unjudged_episode_is_missing_and_never_a_zero() {
    let mut store = EpisodeStore::new(8);
    for (i, id) in ["a", "b", "c"].iter().enumerate() {
        store.record(episode(id, 1));
        if i == 0 {
            store
                .judge(id, EpisodeSignal::Unhelpful, Some(1))
                .expect("judge");
        }
    }
    let census = store.census();
    assert_eq!(census.recorded, 3);
    assert_eq!(census.judged, 1);
    assert_eq!(census.unjudged, 2);
    assert_eq!(
        store.on_policy_samples(1).len(),
        1,
        "the two unjudged episodes must not enter the sample at any weight"
    );
}

/// An id that was never routed, or has aged out, is not a silent no-op:
/// a caller judging it would otherwise believe the feedback landed.
#[test]
fn judging_an_unknown_completion_refuses() {
    let mut store = EpisodeStore::new(8);
    store.record(episode("known", 1));
    assert!(
        store
            .judge("known", EpisodeSignal::Helpful, Some(1))
            .is_ok()
    );
    match store.judge("never-routed", EpisodeSignal::Helpful, Some(1)) {
        Err(JudgementError::UnknownCompletion { completion_id }) => {
            assert_eq!(completion_id, "never-routed");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// The store is a window, and says so. An eviction count above zero is how
/// a reader knows the census is not a history.
#[test]
fn eviction_is_counted_so_the_census_reads_as_a_window() {
    let mut store = EpisodeStore::new(2);
    for id in ["a", "b", "c"] {
        store.record(episode(id, 1));
    }
    let census = store.census();
    assert_eq!(census.recorded, 2);
    assert_eq!(census.evicted, 1);
    assert!(
        store.get("a").is_none(),
        "the oldest episode should have been evicted"
    );
    assert!(store.get("c").is_some());
    // And a judgement of the evicted one refuses rather than reporting a
    // success against nothing.
    assert!(matches!(
        store.judge("a", EpisodeSignal::Helpful, Some(1)),
        Err(JudgementError::UnknownCompletion { .. })
    ));
}

/// A capacity of zero would accept every episode and hold none, so every
/// judgement would refuse and the path would read as broken rather than
/// switched off.
#[test]
fn a_zero_capacity_store_still_holds_one_episode() {
    let mut store = EpisodeStore::new(0);
    store.record(episode("c-1", 1));
    assert_eq!(store.census().recorded, 1);
    assert!(store.judge("c-1", EpisodeSignal::Helpful, Some(1)).is_ok());
}

/// The same id producing two different mixtures means the first record was
/// about a different decision, so its judgement cannot carry over.
#[test]
fn re_recording_an_id_replaces_the_episode_and_drops_its_judgement() {
    let mut store = EpisodeStore::new(8);
    store.record(episode("c-1", 1));
    store
        .judge("c-1", EpisodeSignal::Helpful, Some(1))
        .expect("judge");
    assert_eq!(store.census().judged, 1);

    let mut second = episode("c-1", 1);
    second.applied_weights = vec![("technical".into(), 1.0)];
    store.record(second.clone());

    let (held, signal) = store.get("c-1").expect("still present");
    assert_eq!(held.applied_weights, second.applied_weights);
    assert_eq!(signal, None, "the old judgement was about the old mixture");
    let census = store.census();
    assert_eq!(census.recorded, 1, "re-recording must not double-count");
    assert_eq!(census.judged, 0);
}

/// The wire vocabulary is closed in both directions, with a must-match
/// control so a parser that refused everything would not read as strict.
#[test]
fn the_signal_vocabulary_is_closed_and_round_trips() {
    for signal in [EpisodeSignal::Helpful, EpisodeSignal::Unhelpful] {
        assert_eq!(EpisodeSignal::parse(signal.as_str()), Some(signal));
    }
    for bad in ["", "good", "Helpful", "1", "true", "neutral"] {
        assert_eq!(
            EpisodeSignal::parse(bad),
            None,
            "{bad:?} is not a signal this store can act on"
        );
    }
}

/// The response bodies and the refusal-code mapping are library functions
/// so a Linux host exercises them; this pins the shape a caller reads.
#[test]
fn the_wire_bodies_and_refusal_codes_are_the_documented_ones() {
    let body = feedback_response("c-1", EpisodeSignal::Helpful, JudgementOutcome::Stale);
    assert_eq!(body["object"], "router.feedback");
    assert_eq!(body["completion_id"], "c-1");
    assert_eq!(body["signal"], "helpful");
    assert_eq!(body["outcome"], "stale");

    let mut store = EpisodeStore::new(4);
    store.record(episode("a", 1));
    store
        .judge("a", EpisodeSignal::Helpful, Some(1))
        .expect("judge");
    let body = census_response(&store.census());
    assert_eq!(body["object"], "router.episodes");
    assert_eq!(body["recorded"], 1);
    assert_eq!(body["judged"], 1);
    assert_eq!(body["unjudged"], 0);
    assert_eq!(body["stale"], 0);
    assert_eq!(body["evicted"], 0);

    assert_eq!(
        judgement_error_code(&JudgementError::UnknownCompletion {
            completion_id: "x".into(),
        }),
        "unknown_completion"
    );
    assert_eq!(
        judgement_error_code(&JudgementError::ConflictingRepeat {
            completion_id: "x".into(),
            existing: "helpful".into(),
            incoming: "unhelpful".into(),
        }),
        "conflicting_judgement"
    );
}

/// The census reports the epoch's on-policy population by the same filter
/// the learner uses, so a judged episode from a different epoch is neither
/// a sample nor an unjudged one: it is judged and off-epoch, and the two
/// counts say so rather than folding it into either.
#[test]
fn an_off_epoch_judgement_is_neither_a_sample_nor_unjudged() {
    let mut store = EpisodeStore::new(8);
    store.record(episode("current", 4));
    store.record(episode("older", 3));
    store
        .judge("current", EpisodeSignal::Helpful, Some(4))
        .expect("judge");
    store
        .judge("older", EpisodeSignal::Helpful, Some(4))
        .expect("judge");

    let census = store.census();
    assert_eq!(census.judged, 2, "both carry a judgement");
    assert_eq!(census.unjudged, 0, "neither is missing feedback");
    assert_eq!(census.stale, 1);
    let samples = store.on_policy_samples(4);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0.completion_id, "current");
}
