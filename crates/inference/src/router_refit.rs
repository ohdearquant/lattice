//! Bounded construction of router candidates from explicit feedback.
//!
//! A round uses a fresh snapshot of the producing gate. Its outputs are
//! fitting-population diagnostics, not held-out evidence or admission. Nothing
//! in this module writes an artifact, advances a head, or changes a live router.
//!
//! The episode store is trusted server state. Replay checks consistency with
//! the producing policy; it cannot authenticate who constructed a public episode.

use std::collections::{HashMap, HashSet};

use lattice_fann::Network;
use lattice_fann::training::{
    BatchRefitOutcome, ExploredAction, MixtureSample, RlooConfig, RlooTrainer,
};

use crate::mixture::{
    AdapterRouter, Exploration, RouteOutcome, seed_from_completion_id, weight_entropy,
};
use crate::router_admission::{AdmissionContract, Population};
use crate::router_episode::{EpisodeSignal, EpisodeStore, PolicyIdentity, RouterEpisode};
use crate::router_state::{RouterPolicy, TrainedRepresentation};

/// Required population bounds for one round. Qualification supplies the values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefitConfig {
    /// Fewest new, judged, explored episodes that permit a round.
    pub minimum_round: usize,
    /// Most episodes a call replays, trains on, and evaluates.
    pub maximum_round: usize,
}

/// A refusal leaves the builder's trainer and consumption history unchanged.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RefitError {
    /// The supplied bounds or training coefficients are unusable.
    #[error("invalid refit configuration: {0}")]
    InvalidConfig(String),
    /// The snapshot does not describe an eligible producing policy.
    #[error("ineligible refit source: {0}")]
    IneligibleSource(String),
    /// A source predates the observed epoch or changes the lineage.
    #[error("refit source changed incompatibly: {0}")]
    SourceChanged(String),
    /// Recorded provenance does not replay under the producing policy.
    #[error("episode {completion_id:?} is not usable: {problem}")]
    InvalidEpisode {
        /// The completion carrying the inconsistent record.
        completion_id: String,
        /// The inconsistency that was refused.
        problem: String,
    },
    /// The objective refused the complete batch.
    #[error("refit training failed: {0}")]
    Training(String),
    /// The candidate could not produce finite serving-policy outputs.
    #[error("candidate output failed for {completion_id:?}: {problem}")]
    CandidateOutput {
        /// The fitting completion being evaluated.
        completion_id: String,
        /// The routing or numeric refusal.
        problem: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceKey {
    identity: PolicyIdentity,
    artifact_hash: String,
    base_model: String,
    adapter_names: Vec<String>,
    representation: TrainedRepresentation,
}

/// An immutable snapshot captured by `ServingRouter::refit_source`.
///
/// Obtain a fresh snapshot for each round. A builder rejects epochs older than
/// it has observed, but cannot discover a live change made after this capture.
/// The snapshot does not hold a serving lock while a candidate is constructed.
pub struct RefitSource {
    key: SourceKey,
    gate_bytes: Vec<u8>,
    policy: RouterPolicy,
    tau: f32,
    scale: f32,
    router: AdapterRouter,
}

impl RefitSource {
    #[cfg(feature = "serve")]
    pub(crate) fn from_serving(
        artifact: crate::router_state::RouterArtifact,
        router: AdapterRouter,
        learning_enabled: bool,
        exploration_scale: Option<f64>,
    ) -> Result<Self, RefitError> {
        let refuse = |problem: &str| RefitError::IneligibleSource(problem.to_string());
        if !learning_enabled {
            return Err(refuse("learning is not enabled"));
        }
        if !artifact.provenance.is_learner_eligible() {
            return Err(refuse("the artifact is not learner eligible"));
        }
        let policy = artifact
            .provenance
            .policy()
            .ok_or_else(|| refuse("no recorded policy"))?;
        let lineage = artifact
            .provenance
            .lineage()
            .ok_or_else(|| refuse("no recorded lineage"))?;
        policy
            .validate()
            .map_err(|e| RefitError::IneligibleSource(e.to_string()))?;
        if policy.kind != "softmax" {
            return Err(refuse("the producing policy is not softmax"));
        }
        if lineage.schema_id.is_empty() || lineage.base_model.is_empty() {
            return Err(refuse("schema and base-model identities must be nonempty"));
        }
        let tau = policy
            .tau
            .ok_or_else(|| refuse("the policy records no temperature"))? as f32;
        let scale = policy
            .exploration_scale
            .ok_or_else(|| refuse("the policy records no exploration scale"))?
            as f32;
        if !tau.is_finite() || tau <= 0.0 || !(1.0 / tau).is_finite() {
            return Err(refuse(
                "the served temperature and its reciprocal must be finite and positive",
            ));
        }
        if !scale.is_finite()
            || f64::from(scale) < crate::router_state::MIN_EXPLORATION_SCALE
            || !(1.0 / (scale * scale)).is_finite()
            || 1.0 / (scale * scale) <= 0.0
        {
            return Err(refuse("the served exploration scale is unusable"));
        }
        for value in [policy.epsilon, policy.tau_floor, policy.entropy_floor] {
            if value < 0.0 || !(value as f32).is_finite() {
                return Err(refuse(
                    "policy floors must narrow to finite nonnegative values",
                ));
            }
        }
        if router.weight_policy() != (crate::mixture::WeightPolicy::Softmax { tau })
            || router.epsilon().to_bits() != (policy.epsilon as f32).to_bits()
            || exploration_scale.map(f64::to_bits) != policy.exploration_scale.map(f64::to_bits)
        {
            return Err(refuse("the applied policy differs from the artifact"));
        }
        let gate_bytes = router.gate_bytes();
        if gate_bytes != artifact.gate_bytes {
            return Err(refuse("the live gate differs from the named artifact"));
        }
        let gate = Network::from_bytes(&gate_bytes)
            .map_err(|e| RefitError::IneligibleSource(e.to_string()))?;
        if gate.num_inputs() as u64 != artifact.representation.input_width
            || gate.num_outputs() != artifact.adapter_names.len()
            || gate.num_outputs() < 2
        {
            return Err(refuse(
                "the gate dimensions do not describe an explorable schema",
            ));
        }
        if !matches!(
            gate.layers().last().map(lattice_fann::Layer::activation),
            Some(lattice_fann::Activation::Linear)
        ) {
            return Err(refuse("the gate output must contain raw linear logits"));
        }
        if gate.layers().iter().any(|layer| {
            layer
                .weights()
                .iter()
                .chain(layer.biases())
                .any(|v| !v.is_finite())
        }) {
            return Err(refuse("the producing gate contains a nonfinite parameter"));
        }
        let mut names = HashSet::new();
        if artifact
            .adapter_names
            .iter()
            .any(|name| name.is_empty() || !names.insert(name))
        {
            return Err(refuse("adapter names must be nonempty and unique"));
        }
        Ok(Self {
            key: SourceKey {
                identity: PolicyIdentity {
                    artifact_version: artifact.version_label(),
                    schema_id: lineage.schema_id.clone(),
                    epoch: router.policy_epoch(),
                },
                artifact_hash: artifact.content_hash(),
                base_model: lineage.base_model.clone(),
                adapter_names: artifact.adapter_names.clone(),
                representation: artifact.representation.clone(),
            },
            gate_bytes,
            policy: policy.clone(),
            tau,
            scale,
            router,
        })
    }

    /// The artifact, schema, and epoch that produced the eligible episodes.
    pub fn identity(&self) -> &PolicyIdentity {
        &self.key.identity
    }

    /// The exact producing network, before any candidate update.
    pub fn gate_bytes(&self) -> &[u8] {
        &self.gate_bytes
    }

    /// The recorded policy whose serving transformation candidates reuse.
    pub fn policy(&self) -> &RouterPolicy {
        &self.policy
    }

    pub(crate) fn adapter_names(&self) -> &[String] {
        &self.key.adapter_names
    }

    pub(crate) fn base_model(&self) -> &str {
        &self.key.base_model
    }

    pub(crate) fn artifact_hash(&self) -> &str {
        &self.key.artifact_hash
    }

    pub(crate) fn representation(&self) -> &TrainedRepresentation {
        &self.key.representation
    }

    pub(crate) fn policy_router(&self, gate_bytes: &[u8]) -> Result<AdapterRouter, RefitError> {
        let mut router = self.router.refit_snapshot();
        router
            .reload(gate_bytes)
            .map_err(|e| RefitError::Training(e.to_string()))?;
        Ok(router)
    }
}

/// The population considered when too few new explored judgements are available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefitPopulation {
    /// The producing identity examined.
    pub source: PolicyIdentity,
    /// Judged episodes in this epoch, including previously consumed episodes.
    pub judged: usize,
    /// Judged episodes carrying no exploration draw.
    pub deterministic: usize,
    /// Explored judgements reserved by the registered held-out split.
    pub held_out: usize,
    /// Explored episodes already consumed by this builder.
    pub already_consumed: usize,
    /// New explored training judgements available, before applying the maximum.
    pub available: usize,
    /// The caller's required minimum.
    pub minimum: usize,
}

/// Observations about one completed fitting round, without a quality verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct RefitRound {
    /// The population from which the oldest eligible episodes were taken.
    pub population: RefitPopulation,
    /// Consumed completion IDs, in recording order.
    pub completion_ids: Vec<String>,
    /// The objective's batch statistics.
    pub update: BatchRefitOutcome,
}

/// Candidate-derived coefficients for one fitting context and recorded draw seed.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateOutput {
    /// The fitting completion supplying the context, selection count, and seed.
    pub completion_id: String,
    /// The candidate's result through the existing serving policy transformation.
    pub route: RouteOutcome,
}

/// Candidate bytes and diagnostics. This value carries no admission authority.
#[derive(Debug, Clone, PartialEq)]
pub struct RefitCandidate {
    source: Box<SourceKey>,
    admission_contract_hash: Option<String>,
    gate_bytes: Vec<u8>,
    round: RefitRound,
    outputs: Vec<CandidateOutput>,
    mean_pre_floor_entropy: f32,
}

impl RefitCandidate {
    pub(crate) fn admission_contract_hash(&self) -> Option<&str> {
        self.admission_contract_hash.as_deref()
    }

    pub(crate) fn matches_source(&self, source: &RefitSource) -> bool {
        *self.source == source.key
    }

    /// The candidate network, which has not been published or activated.
    pub fn gate_bytes(&self) -> &[u8] {
        &self.gate_bytes
    }

    /// Fitting-population statistics and producing identity.
    pub fn round(&self) -> &RefitRound {
        &self.round
    }

    /// Outputs recomputed from candidate bytes, never copied from parent episodes.
    pub fn outputs(&self) -> &[CandidateOutput] {
        &self.outputs
    }

    /// Descriptive entropy of candidate pre-floor weights on the fitting contexts.
    pub fn mean_pre_floor_entropy(&self) -> f32 {
        self.mean_pre_floor_entropy
    }
}

/// Construction outcomes; none admits, publishes, or activates a policy.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RefitVerdict {
    /// The minimum population was not met and no episodes were consumed.
    BelowMinimum(RefitPopulation),
    /// Traffic was observed, but the gate remained byte-identical.
    Unchanged(RefitRound),
    /// Candidate bytes and fitting diagnostics were constructed successfully.
    Candidate(RefitCandidate),
}

/// Stateful, explicitly invoked candidate construction outside the request path.
///
/// The trainer retains its traffic EMA within an epoch. Consumption IDs are
/// retained only while their episodes remain in the bounded store. Completion
/// IDs must not be reused after eviction: the store cannot distinguish such a
/// reuse from a new completion. A higher epoch in the same lineage starts a new
/// trainer; a different schema or representation requires a new builder.
pub struct RefitBuilder {
    config: RefitConfig,
    training: RlooConfig,
    trainer: RlooTrainer,
    source: Option<SourceKey>,
    consumed: HashSet<String>,
    admission: Option<AdmissionBinding>,
}

struct AdmissionBinding {
    contract: AdmissionContract,
    hash: String,
}

impl RefitBuilder {
    /// Construct a diagnostic-only builder with no registered admission split.
    ///
    /// Validates explicit bounds and finite, nonnegative training coefficients.
    /// Its candidates are unbound and cannot be used for admission.
    pub fn new(training: RlooConfig, config: RefitConfig) -> Result<Self, RefitError> {
        if config.minimum_round == 0 || config.maximum_round < config.minimum_round {
            return Err(RefitError::InvalidConfig(
                "require 0 < minimum_round <= maximum_round".into(),
            ));
        }
        for (name, value) in [
            ("learning_rate", training.learning_rate),
            ("aux_loss_coeff", training.aux_loss_coeff),
            ("z_loss_coeff", training.z_loss_coeff),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RefitError::InvalidConfig(format!(
                    "{name} must be finite and nonnegative"
                )));
            }
        }
        Ok(Self {
            config,
            trainer: RlooTrainer::new(training.clone()),
            training,
            source: None,
            consumed: HashSet::new(),
            admission: None,
        })
    }

    /// Construct a builder with an immutable, validated admission contract.
    ///
    /// The same registered split applies to every round and epoch. Held-out
    /// events never enter training, and candidates retain the contract's hash
    /// for later evaluation. Construction itself grants no admission authority.
    pub fn new_with_admission(
        training: RlooConfig,
        config: RefitConfig,
        contract: AdmissionContract,
    ) -> Result<Self, RefitError> {
        let hash = contract
            .hash()
            .map_err(|error| RefitError::InvalidConfig(error.to_string()))?;
        let mut builder = Self::new(training, config)?;
        builder.admission = Some(AdmissionBinding { contract, hash });
        Ok(builder)
    }

    /// Build one candidate from the oldest eligible, unconsumed judgements.
    ///
    /// Other epochs and absent feedback are excluded. Same-epoch identity
    /// disagreements refuse. A registered split excludes held-out events before
    /// preparing gradients. At most `maximum_round` training episodes are replayed
    /// and trained; population counting and ID pruning scan the bounded store.
    /// Every fallible operation completes before trainer history or consumed
    /// IDs change. Equal rewards still reach the trainer and observe traffic.
    pub fn build_candidate(
        &mut self,
        source: &RefitSource,
        store: &EpisodeStore,
    ) -> Result<RefitVerdict, RefitError> {
        let new_epoch = self.check_source(source)?;
        let mut consumed = if new_epoch {
            HashSet::new()
        } else {
            self.consumed.clone()
        };
        consumed.retain(|id| store.get(id).is_some());
        let mut trainer = if new_epoch {
            RlooTrainer::new(self.training.clone())
        } else {
            self.trainer.clone()
        };
        let mut population = RefitPopulation {
            source: source.identity().clone(),
            judged: 0,
            deterministic: 0,
            held_out: 0,
            already_consumed: 0,
            available: 0,
            minimum: self.config.minimum_round,
        };
        let mut selected = Vec::new();
        for (episode, signal) in store.ordered_on_policy_samples(source.identity().epoch) {
            population.judged += 1;
            validate_identity(source, episode)?;
            if episode.exploration.is_none() {
                population.deterministic += 1;
            } else if self
                .admission
                .as_ref()
                .map(|binding| binding.contract.split(&episode.completion_id))
                .transpose()
                .map_err(|error| invalid(episode, error.to_string()))?
                == Some(Population::HeldOut)
            {
                population.held_out += 1;
            } else if consumed.contains(&episode.completion_id) {
                population.already_consumed += 1;
            } else {
                population.available += 1;
                if selected.len() < self.config.maximum_round {
                    selected.push((episode, signal));
                }
            }
        }
        if selected.len() < self.config.minimum_round {
            self.trainer = trainer;
            self.source = Some(source.key.clone());
            self.consumed = consumed;
            return Ok(RefitVerdict::BelowMinimum(population));
        }

        let mut producing = source.policy_router(source.gate_bytes())?;
        let columns: HashMap<&str, usize> = source
            .key
            .adapter_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let mut prepared = Vec::with_capacity(selected.len());
        for (episode, signal) in selected {
            prepared.push(prepare(source, &mut producing, episode, signal, &columns)?);
        }
        let samples: Vec<_> = prepared
            .iter()
            .map(|p| MixtureSample {
                context: &p.episode.context_vector,
                action: ExploredAction {
                    sampled_logits: &p.draw.sampled_logits,
                    exploration_scale: p.draw.scale,
                    w_full: &p.w_full,
                    tau: source.tau,
                },
                reward: p.reward,
            })
            .collect();
        let mut candidate = Network::from_bytes(source.gate_bytes())
            .map_err(|e| RefitError::Training(e.to_string()))?;
        let update = trainer
            .step_batch_with_explored_actions(&mut candidate, &samples)
            .map_err(|e| RefitError::Training(e.to_string()))?;
        if !update.mean_reward.is_finite() || !update.mean_loss.is_finite() {
            return Err(RefitError::Training(
                "batch statistics are not finite".into(),
            ));
        }
        let gate_bytes = candidate.to_bytes();
        let round = RefitRound {
            population,
            completion_ids: prepared
                .iter()
                .map(|p| p.episode.completion_id.clone())
                .collect(),
            update,
        };
        let verdict = if gate_bytes == source.gate_bytes {
            RefitVerdict::Unchanged(round)
        } else {
            let mut policy_router = source.policy_router(&gate_bytes)?;
            let mut outputs = Vec::with_capacity(prepared.len());
            let mut entropy_total = 0.0_f64;
            for p in &prepared {
                let refusal = |problem: String| RefitError::CandidateOutput {
                    completion_id: p.episode.completion_id.clone(),
                    problem,
                };
                let route = policy_router
                    .route_explored(
                        &p.episode.context_vector,
                        &source.key.adapter_names,
                        p.episode.selected_scores.len(),
                        Some(Exploration {
                            scale: source.scale,
                            seed: p.draw.seed,
                        }),
                    )
                    .map_err(|e| refusal(e.to_string()))?;
                validate_route(&route).map_err(refusal)?;
                let weights: Vec<f32> = route.pre_floor_weights.iter().map(|(_, w)| *w).collect();
                entropy_total += f64::from(weight_entropy(&weights));
                outputs.push(CandidateOutput {
                    completion_id: p.episode.completion_id.clone(),
                    route,
                });
            }
            let mean_pre_floor_entropy = (entropy_total / outputs.len() as f64) as f32;
            if !mean_pre_floor_entropy.is_finite() {
                return Err(RefitError::Training(
                    "candidate entropy is not finite".into(),
                ));
            }
            RefitVerdict::Candidate(RefitCandidate {
                source: Box::new(source.key.clone()),
                admission_contract_hash: self
                    .admission
                    .as_ref()
                    .map(|binding| binding.hash.clone()),
                gate_bytes,
                round,
                outputs,
                mean_pre_floor_entropy,
            })
        };
        consumed.extend(prepared.iter().map(|p| p.episode.completion_id.clone()));
        self.trainer = trainer;
        self.source = Some(source.key.clone());
        self.consumed = consumed;
        Ok(verdict)
    }

    fn check_source(&self, source: &RefitSource) -> Result<bool, RefitError> {
        let Some(previous) = &self.source else {
            return Ok(false);
        };
        if previous.identity.schema_id != source.key.identity.schema_id
            || previous.base_model != source.key.base_model
            || previous.adapter_names != source.key.adapter_names
            || previous.representation != source.key.representation
        {
            return Err(RefitError::SourceChanged(
                "a schema, base-model, or representation change requires a new builder".into(),
            ));
        }
        if source.identity().epoch < previous.identity.epoch {
            return Err(RefitError::SourceChanged(
                "the snapshot predates the observed epoch".into(),
            ));
        }
        if source.identity().epoch == previous.identity.epoch && previous != &source.key {
            return Err(RefitError::SourceChanged(
                "one epoch names different producing artifacts or policies".into(),
            ));
        }
        Ok(source.identity().epoch > previous.identity.epoch)
    }
}

struct Prepared<'a> {
    episode: &'a RouterEpisode,
    draw: &'a crate::mixture::ExplorationOutcome,
    w_full: Vec<f32>,
    reward: f32,
}

fn invalid(episode: &RouterEpisode, problem: impl Into<String>) -> RefitError {
    RefitError::InvalidEpisode {
        completion_id: episode.completion_id.clone(),
        problem: problem.into(),
    }
}

fn validate_identity(source: &RefitSource, episode: &RouterEpisode) -> Result<(), RefitError> {
    if episode.completion_id.is_empty()
        || episode.policy != source.key.identity
        || episode.adapter_names != source.key.adapter_names
    {
        return Err(invalid(
            episode,
            "completion, epoch, artifact, schema, or ordered columns disagree",
        ));
    }
    Ok(())
}

fn prepare<'a>(
    source: &RefitSource,
    producing: &mut AdapterRouter,
    episode: &'a RouterEpisode,
    signal: EpisodeSignal,
    columns: &HashMap<&str, usize>,
) -> Result<Prepared<'a>, RefitError> {
    let draw = episode
        .exploration
        .as_ref()
        .ok_or_else(|| invalid(episode, "no recorded exploration"))?;
    if episode.context_vector.len() != producing.input_size()
        || episode.context_vector.iter().any(|v| !v.is_finite())
        || draw.sampled_logits.len() != columns.len()
        || draw.sampled_logits.iter().any(|v| !v.is_finite())
        || !draw.log_density.is_finite()
    {
        return Err(invalid(
            episode,
            "context or sampled logits have invalid dimensions or nonfinite values",
        ));
    }
    if episode.tau.map(f32::to_bits) != Some(source.tau.to_bits())
        || draw.scale.to_bits() != source.scale.to_bits()
        || draw.seed != seed_from_completion_id(&episode.completion_id)
    {
        return Err(invalid(
            episode,
            "temperature, exploration scale, or completion seed disagrees with the producing policy",
        ));
    }
    if episode.selected_scores.is_empty()
        || episode.sampled_scores.is_empty()
        || episode.pre_floor_weights.is_empty()
    {
        return Err(invalid(
            episode,
            "a selected score or pre-floor weight vector is empty",
        ));
    }
    for values in [
        &episode.selected_scores,
        &episode.sampled_scores,
        &episode.pre_floor_weights,
        &episode.applied_weights,
    ] {
        let mut names = HashSet::new();
        if values.iter().any(|(name, value)| {
            !value.is_finite() || !columns.contains_key(name.as_str()) || !names.insert(name)
        }) {
            return Err(invalid(
                episode,
                "a score or weight vector has empty, duplicate, unknown, or nonfinite entries",
            ));
        }
    }
    let expected = producing
        .route_explored(
            &episode.context_vector,
            &source.key.adapter_names,
            episode.selected_scores.len(),
            Some(Exploration {
                scale: source.scale,
                seed: draw.seed,
            }),
        )
        .map_err(|e| invalid(episode, e.to_string()))?;
    validate_route(&expected).map_err(|e| invalid(episode, e))?;
    let expected_draw = expected
        .exploration
        .as_ref()
        .ok_or_else(|| invalid(episode, "replay produced no exploration"))?;
    if !named_equal(&expected.selected_scores, &episode.selected_scores)
        || !named_equal(&expected.sampled_scores, &episode.sampled_scores)
        || !named_equal(&expected.pre_floor_weights, &episode.pre_floor_weights)
        || !named_equal(&expected.weights, &episode.applied_weights)
        || expected.dropped != episode.dropped
        || !bits_equal(&expected_draw.sampled_logits, &draw.sampled_logits)
        || expected_draw.log_density.to_bits() != draw.log_density.to_bits()
    {
        return Err(invalid(
            episode,
            "recorded decision does not replay exactly under its producing gate",
        ));
    }
    let mut w_full = vec![0.0; columns.len()];
    for (name, weight) in &episode.pre_floor_weights {
        let column = columns
            .get(name.as_str())
            .ok_or_else(|| invalid(episode, "unknown pre-floor column"))?;
        w_full[*column] = *weight;
    }
    Ok(Prepared {
        episode,
        draw,
        w_full,
        reward: match signal {
            EpisodeSignal::Helpful => 1.0,
            EpisodeSignal::Unhelpful => -1.0,
        },
    })
}

fn named_equal(left: &[(String, f32)], right: &[(String, f32)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((ln, lv), (rn, rv))| ln == rn && lv.to_bits() == rv.to_bits())
}

fn bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(l, r)| l.to_bits() == r.to_bits())
}

fn validate_route(route: &RouteOutcome) -> Result<(), String> {
    if route
        .selected_scores
        .iter()
        .chain(&route.sampled_scores)
        .any(|(_, value)| !value.is_finite())
        || route.pre_floor_weights.is_empty()
        || route
            .pre_floor_weights
            .iter()
            .chain(&route.weights)
            .any(|(_, value)| !value.is_finite() || *value < 0.0 || *value > 1.0)
        || route.exploration.as_ref().is_none_or(|draw| {
            !draw.log_density.is_finite() || draw.sampled_logits.iter().any(|v| !v.is_finite())
        })
    {
        return Err(
            "the serving policy produced empty or nonfinite scores or invalid weights".into(),
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "serve"))]
mod tests;
