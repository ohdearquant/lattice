//! What a routed completion recorded about itself, and the judgements against
//! it (ADR-096 decision 3).
//!
//! # Why the server holds the provenance
//!
//! A learner needs to know which mixture produced a completion before it can
//! credit anything. The obvious surface -- let the caller post a context
//! vector and an adapter index alongside a rating -- cannot work, and not
//! because callers are untrusted: a completion produced by a weighted mixture
//! *does not identify an index*. Choosing the largest-weight adapter, or
//! rewarding every adapter equally, manufactures credit the feedback never
//! supplied. So the provenance is captured here, at the moment the routing
//! decision is made, and a judgement carries nothing but the completion id and
//! a signal.
//!
//! # Missing feedback is missing, not zero
//!
//! An unjudged episode is not a negative one. The estimand this store supports
//! is utility among the population that gave explicit feedback, and that is a
//! statement about a biased sample -- said out loud here because the
//! alternative is silent: a store that filled unjudged episodes with zero
//! would produce a number with no missing data and no way to see the
//! substitution. [`EpisodeStore::census`] therefore reports recorded, judged
//! and unjudged separately and never a ratio that hides the third.
//!
//! # Idempotency and staleness are different refusals
//!
//! One judgement per completed request. A repeat carrying the same signal is
//! the ordinary retry and is accepted without changing anything; a repeat
//! carrying a *different* signal is refused, because one of the two is wrong
//! and the store cannot tell which.
//!
//! An episode produced under one policy epoch and judged after that epoch
//! ended is **stale**: it is counted, and it is not handed to a learner as an
//! on-policy sample for a different gate. Stale is a third outcome rather than
//! a failure, because the judgement is real and the operator wants to know how
//! much feedback is arriving too late.

use std::collections::HashMap;
use std::collections::VecDeque;

/// Which way a judgement points, and how strongly.
///
/// Deliberately not a bare `f32`: a caller-supplied magnitude with no declared
/// scale is a number every consumer normalises differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeSignal {
    /// The completion was good.
    Helpful,
    /// The completion was bad.
    Unhelpful,
}

impl EpisodeSignal {
    /// How the wire spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Helpful => "helpful",
            Self::Unhelpful => "unhelpful",
        }
    }

    /// Parse the wire spelling, refusing anything else.
    ///
    /// A closed vocabulary rather than a free string: an unrecognised value
    /// accepted here would be stored, counted as feedback, and mean nothing.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "helpful" => Some(Self::Helpful),
            "unhelpful" => Some(Self::Unhelpful),
            _ => None,
        }
    }
}

/// The identity of the gate and policy a mixture was formed under.
///
/// Carried per episode rather than read from the live server at judgement
/// time. The gate can be replaced between the completion and its judgement,
/// and the question a learner asks is which gate produced this, not which one
/// is serving now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyIdentity {
    /// The artifact's version label: counter and hash together.
    pub artifact_version: String,
    /// The schema this lineage is named by.
    pub schema_id: String,
    /// The policy epoch this episode was produced in. Advances whenever the
    /// serving gate or policy changes.
    pub epoch: u64,
}

/// One routed completion, as the server saw it.
///
/// Everything here was captured at the routing decision and is not
/// reconstructible afterwards: the gate's next forward pass has no memory of
/// what it scored for this request, and a caller cannot supply the context
/// vector because a caller's vector would describe a mixture the server never
/// formed.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterEpisode {
    /// The completion this is about. The id the response carried.
    pub completion_id: String,
    /// The gate and policy that produced the mixture.
    pub policy: PolicyIdentity,
    /// The ordered adapter list the gate was built with: column `i` of the
    /// gate's output is `adapter_names[i]`.
    ///
    /// Recorded per episode so the mapping from an adapter name to its gate
    /// column is a property of the decision rather than of whatever artifact
    /// happens to be served when a learner reads the episode. A schema change
    /// ends the lineage and the epoch, so an on-policy episode's list always
    /// agrees with the served one -- but a record that cannot be read without
    /// the live server is not the record decision 3 asks for.
    pub adapter_names: Vec<String>,
    /// The context vector the server computed and routed on. Server-captured:
    /// a caller cannot supply one, because a caller's vector would describe a
    /// mixture the server never formed.
    pub context_vector: Vec<f32>,
    /// The gate scores `z` of the selected columns, in selection order.
    pub selected_scores: Vec<(String, f32)>,
    /// The logits the selection and the weight policy consumed, for the
    /// selected columns, in selection order: `z` on the deterministic path
    /// and `z'` under a draw.
    ///
    /// Distinct from `selected_scores` only under a draw, and distinct from
    /// `exploration`'s `sampled_logits` in ordering: that one is every gate
    /// output in gate column order, this one is the selected columns in
    /// selection order. The learner takes the former; this records what the
    /// weight policy was actually handed.
    pub sampled_scores: Vec<(String, f32)>,
    /// The weights the policy produced, before the floor.
    pub pre_floor_weights: Vec<(String, f32)>,
    /// The weights the request actually ran with.
    pub applied_weights: Vec<(String, f32)>,
    /// Adapters the `epsilon` floor removed between the pre-floor and applied
    /// weights. Recoverable from those two vectors by set difference, and
    /// recorded rather than left to be re-derived so a reader does not have to
    /// know the floor's rule to see that an adapter was dropped.
    pub dropped: Vec<String>,
    /// The temperature the mixture was formed at, or `None` for a policy that
    /// reads no temperature (uniform).
    ///
    /// Read off the routing call rather than re-derived from the served
    /// artifact, so the episode records the temperature that produced its
    /// weights and not one installed since.
    pub tau: Option<f32>,
    /// The draw, when one happened.
    ///
    /// `None` means the decision was deterministic, not that a draw went
    /// unrecorded. Those are different claims: a learner can attribute an
    /// episode carrying a record and must not attribute one without it. The
    /// record holds the sampled logits `z'` over every gate output, the
    /// exploration scale `s` and seed the draw ran at, and the log density of
    /// `z'` under the draw. There is no action probability: the draw is
    /// Gaussian on the logits, so what the learner scores is the density of
    /// `z'` (see [`crate::mixture::ExplorationOutcome::log_density`]).
    #[cfg(feature = "mixture")]
    pub exploration: Option<crate::mixture::ExplorationOutcome>,
}

/// What happened to a judgement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgementOutcome {
    /// First judgement for this episode, and it is on-policy.
    Recorded,
    /// A repeat carrying the same signal. Nothing changed.
    Idempotent,
    /// Real, counted, and not an on-policy sample: the episode's epoch has
    /// ended.
    Stale,
}

impl JudgementOutcome {
    /// How the wire spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Recorded => "recorded",
            Self::Idempotent => "idempotent",
            Self::Stale => "stale",
        }
    }
}

/// Why a judgement was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum JudgementError {
    /// No episode with that completion id. Either it was never routed, or it
    /// aged out of the bounded store.
    #[error(
        "no routed completion {completion_id:?} is on record; it was either \
         never routed through a gate or has aged out of the episode store"
    )]
    UnknownCompletion {
        /// The id asked about.
        completion_id: String,
    },

    /// A second judgement disagreeing with the first. One of them is wrong and
    /// the store cannot tell which, so it keeps the first and says so.
    #[error(
        "completion {completion_id:?} was already judged {existing:?} and this \
         judgement says {incoming:?}; one judgement per completed request, and \
         a conflicting repeat is refused rather than silently overwriting"
    )]
    ConflictingRepeat {
        /// The id asked about.
        completion_id: String,
        /// What is already on record.
        existing: String,
        /// What arrived.
        incoming: String,
    },
}

/// What the store holds, as counts.
///
/// Three numbers rather than a rate. A rate needs a denominator and the honest
/// denominator here is contested: judged-over-recorded describes response
/// behaviour, not utility, and reporting only that invites reading it as
/// coverage of the population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EpisodeCensus {
    /// Episodes currently held.
    pub recorded: usize,
    /// Of those, how many carry a judgement.
    pub judged: usize,
    /// Of those, how many do not. NOT zeros: unjudged is missing.
    pub unjudged: usize,
    /// Judgements that arrived after their episode's epoch ended. Counted
    /// here so late feedback is visible rather than merely discarded.
    pub stale: usize,
    /// Episodes evicted by the capacity bound since the store was created.
    /// A learner reading a census with this above zero is reading a window,
    /// not a history.
    pub evicted: usize,
}

/// A bounded, in-memory record of routed completions and their judgements.
///
/// In memory and bounded on purpose. A durable store is a schema, a migration
/// and a retention policy, and none of those are decided by ADR-096; a
/// process-lifetime window is honest about what it is as long as the census
/// reports its evictions, which it does.
#[derive(Debug)]
pub struct EpisodeStore {
    capacity: usize,
    order: VecDeque<String>,
    episodes: HashMap<String, (RouterEpisode, Option<EpisodeSignal>)>,
    stale: usize,
    evicted: usize,
}

impl EpisodeStore {
    /// A store holding at most `capacity` episodes.
    ///
    /// A capacity of zero would accept every episode and hold none, so every
    /// judgement would refuse with `UnknownCompletion` and the feedback path
    /// would look broken rather than switched off. It is clamped to one.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            episodes: HashMap::new(),
            stale: 0,
            evicted: 0,
        }
    }

    /// Record a routed completion, evicting the oldest if at capacity.
    ///
    /// Re-recording the same completion id replaces the episode in place and
    /// drops any judgement against it: the same id producing two different
    /// mixtures means the first record was about a different decision, and a
    /// judgement of that decision cannot be carried over to this one.
    pub fn record(&mut self, episode: RouterEpisode) {
        let id = episode.completion_id.clone();
        if self.episodes.insert(id.clone(), (episode, None)).is_some() {
            return;
        }
        self.order.push_back(id);
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.episodes.remove(&oldest);
                self.evicted += 1;
            }
        }
    }

    /// Judge a completion, against the epoch that is serving now.
    ///
    /// `current_epoch` is passed in rather than held, because the store is not
    /// what decides when a policy epoch ends -- the thing that swaps the gate
    /// is. A store that tracked its own epoch would be a second opinion about
    /// the serving state, and the two would diverge silently.
    ///
    /// `None` says nothing is serving: no gate is loaded, so whatever policy
    /// produced a recorded episode has certainly ended and the judgement is
    /// stale. An `Option` rather than a caller-chosen number because a
    /// stand-in epoch is indistinguishable from a real one -- a caller with no
    /// router that passed zero would silently mark every episode on-policy for
    /// a gate that does not exist. The refusals are ordered ahead of this, so
    /// an unknown id and a conflicting repeat still answer as themselves.
    pub fn judge(
        &mut self,
        completion_id: &str,
        signal: EpisodeSignal,
        current_epoch: Option<u64>,
    ) -> Result<JudgementOutcome, JudgementError> {
        let Some((episode, existing)) = self.episodes.get_mut(completion_id) else {
            return Err(JudgementError::UnknownCompletion {
                completion_id: completion_id.to_string(),
            });
        };
        if let Some(existing) = *existing {
            if existing == signal {
                return Ok(JudgementOutcome::Idempotent);
            }
            return Err(JudgementError::ConflictingRepeat {
                completion_id: completion_id.to_string(),
                existing: existing.as_str().to_string(),
                incoming: signal.as_str().to_string(),
            });
        }
        let on_policy = current_epoch.is_some_and(|epoch| episode.policy.epoch == epoch);
        *existing = Some(signal);
        if on_policy {
            Ok(JudgementOutcome::Recorded)
        } else {
            self.stale += 1;
            Ok(JudgementOutcome::Stale)
        }
    }

    /// The counts, including the ones a ratio would hide.
    pub fn census(&self) -> EpisodeCensus {
        let judged = self
            .episodes
            .values()
            .filter(|(_, signal)| signal.is_some())
            .count();
        EpisodeCensus {
            recorded: self.episodes.len(),
            judged,
            unjudged: self.episodes.len() - judged,
            stale: self.stale,
            evicted: self.evicted,
        }
    }

    /// The judged, on-policy episodes for `epoch`, which is the only
    /// population a learner may treat as on-policy samples.
    ///
    /// Unjudged episodes are absent rather than present with a zero, and
    /// episodes from other epochs are absent rather than reweighted. Both
    /// exclusions are the decision, not an implementation convenience.
    pub fn on_policy_samples(&self, epoch: u64) -> Vec<(&RouterEpisode, EpisodeSignal)> {
        self.episodes
            .values()
            .filter_map(|(episode, signal)| match signal {
                Some(signal) if episode.policy.epoch == epoch => Some((episode, *signal)),
                _ => None,
            })
            .collect()
    }

    /// The episode for `completion_id`, with its judgement if it has one.
    pub fn get(&self, completion_id: &str) -> Option<(&RouterEpisode, Option<EpisodeSignal>)> {
        self.episodes
            .get(completion_id)
            .map(|(episode, signal)| (episode, *signal))
    }
}

/// Parse a `POST /v1/router/feedback` body.
///
/// Two fields and no third. A body carrying a context vector, an adapter index
/// or a weight is refused rather than ignored: accepting and discarding it
/// would let a caller believe it had supplied the provenance, and ADR-096
/// decision 3 puts that on the server precisely because a caller's version of
/// it describes a mixture the server never formed. `deny_unknown_fields` is
/// what makes the refusal happen at the boundary rather than in a reviewer's
/// head.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackRequest {
    /// The completion being judged.
    pub completion_id: String,
    /// `helpful` or `unhelpful`. Parsed against the closed vocabulary.
    pub signal: String,
}

impl FeedbackRequest {
    /// The signal, or `None` when the string is not one this store can act on.
    pub fn parsed_signal(&self) -> Option<EpisodeSignal> {
        EpisodeSignal::parse(&self.signal)
    }
}

/// The stable wire code for a refused judgement.
///
/// Library rather than handler, so the mapping from a refusal to its code is
/// one function with its own test instead of a `match` inside a handler no
/// unit test on a non-Metal host can reach.
pub fn judgement_error_code(error: &JudgementError) -> &'static str {
    match error {
        JudgementError::UnknownCompletion { .. } => "unknown_completion",
        JudgementError::ConflictingRepeat { .. } => "conflicting_judgement",
    }
}

/// The `POST /v1/router/feedback` success body.
pub fn feedback_response(
    completion_id: &str,
    signal: EpisodeSignal,
    outcome: JudgementOutcome,
) -> serde_json::Value {
    serde_json::json!({
        "object": "router.feedback",
        "completion_id": completion_id,
        "signal": signal.as_str(),
        "outcome": outcome.as_str(),
    })
}

/// The `GET /v1/router/episodes` body.
///
/// Counts, not a rate: the honest denominator for a feedback rate is
/// contested, and judged-over-recorded describes response behaviour rather
/// than utility. `unjudged` is published because missing feedback is missing
/// and not zero, and `evicted` because a census over a bounded window is a
/// window, which a reader cannot otherwise tell.
pub fn census_response(census: &EpisodeCensus) -> serde_json::Value {
    serde_json::json!({
        "object": "router.episodes",
        "recorded": census.recorded,
        "judged": census.judged,
        "unjudged": census.unjudged,
        "stale": census.stale,
        "evicted": census.evicted,
    })
}

#[cfg(test)]
mod tests {
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
}
