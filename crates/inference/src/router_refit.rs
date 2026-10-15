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
    source: SourceKey,
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
        self.source == source.key
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
                source: source.key.clone(),
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
mod tests {
    use super::*;
    use crate::router_state::{RouterArtifact, RouterLineage, RouterProvenance};
    use crate::serve::routing::ServingRouter;
    use lattice_fann::{Activation, Layer};

    fn artifact() -> RouterArtifact {
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
                    epsilon: 0.0,
                    tau_floor: 0.1,
                    entropy_floor: 0.35,
                    max_consecutive_collapsed: 3,
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

    fn source() -> RefitSource {
        ServingRouter::new(artifact(), true)
            .unwrap()
            .refit_source()
            .unwrap()
    }

    fn training() -> RlooConfig {
        RlooConfig {
            learning_rate: 0.03,
            aux_loss_coeff: 0.2,
            z_loss_coeff: 0.01,
        }
    }

    fn builder(minimum: usize, maximum: usize) -> RefitBuilder {
        RefitBuilder::new(
            training(),
            RefitConfig {
                minimum_round: minimum,
                maximum_round: maximum,
            },
        )
        .unwrap()
    }

    fn episode(source: &RefitSource, id: &str, context: Vec<f32>, k: usize) -> RouterEpisode {
        let route = source
            .router
            .refit_snapshot()
            .route_explored(
                &context,
                &source.key.adapter_names,
                k,
                Some(Exploration {
                    scale: source.scale,
                    seed: seed_from_completion_id(id),
                }),
            )
            .unwrap();
        RouterEpisode {
            completion_id: id.into(),
            policy: source.identity().clone(),
            adapter_names: source.key.adapter_names.clone(),
            context_vector: context,
            selected_scores: route.selected_scores,
            sampled_scores: route.sampled_scores,
            pre_floor_weights: route.pre_floor_weights,
            applied_weights: route.weights,
            dropped: route.dropped,
            tau: Some(source.tau),
            exploration: route.exploration,
        }
    }

    fn record(store: &mut EpisodeStore, episode: RouterEpisode, signal: EpisodeSignal) {
        let id = episode.completion_id.clone();
        let epoch = episode.policy.epoch;
        store.record(episode);
        store.judge(&id, signal, Some(epoch)).unwrap();
    }

    fn pair(source: &RefitSource, capacity: usize) -> EpisodeStore {
        let mut store = EpisodeStore::new(capacity);
        record(
            &mut store,
            episode(source, "first", vec![0.4, -0.2], 3),
            EpisodeSignal::Helpful,
        );
        record(
            &mut store,
            episode(source, "second", vec![-0.3, 0.8], 3),
            EpisodeSignal::Unhelpful,
        );
        store
    }

    fn candidate(verdict: RefitVerdict) -> RefitCandidate {
        match verdict {
            RefitVerdict::Candidate(candidate) => candidate,
            other => panic!("expected candidate, got {other:?}"),
        }
    }

    fn admission_contract() -> AdmissionContract {
        AdmissionContract {
            split_salt: "refit-partition-fixture".into(),
            heldout_fraction: 0.5,
            parent_logit_tolerance: 0.0,
            bootstrap_replicates: 16,
            bootstrap_seed: 7,
            lower_quantile: 0.1,
            regression_limit: 0.2,
            min_heldout_events: 2,
            min_effective_sample_size: 1.5,
            min_reference_tokens: 2,
        }
    }

    fn partition_ids(
        contract: &AdmissionContract,
        population: Population,
        count: usize,
    ) -> Vec<String> {
        let ids: Vec<_> = (0..1024)
            .map(|index| format!("partition-{index}"))
            .filter(|id| contract.split(id).unwrap() == population)
            .take(count)
            .collect();
        assert_eq!(ids.len(), count, "fixture must populate both partitions");
        ids
    }

    fn registered_builder(contract: AdmissionContract) -> RefitBuilder {
        RefitBuilder::new_with_admission(
            training(),
            RefitConfig {
                minimum_round: 2,
                maximum_round: 2,
            },
            contract,
        )
        .unwrap()
    }

    fn record_training_ids(source: &RefitSource, store: &mut EpisodeStore, ids: &[String]) {
        for (index, id) in ids.iter().enumerate() {
            let (context, signal) = if index % 2 == 0 {
                (vec![0.4, -0.2], EpisodeSignal::Helpful)
            } else {
                (vec![-0.3, 0.8], EpisodeSignal::Unhelpful)
            };
            record(store, episode(source, id, context, 3), signal);
        }
    }

    #[test]
    fn admission_constructor_requires_a_valid_explicit_contract() {
        let mut contract = admission_contract();
        contract.heldout_fraction = f64::NAN;
        assert!(matches!(
            RefitBuilder::new_with_admission(
                training(),
                RefitConfig {
                    minimum_round: 2,
                    maximum_round: 2,
                },
                contract,
            ),
            Err(RefitError::InvalidConfig(_))
        ));
    }

    #[test]
    fn admission_heldout_events_never_enter_gradients_or_traffic_history() {
        let source = source();
        let contract = admission_contract();
        let training_ids = partition_ids(&contract, Population::Training, 4);
        let heldout_ids = partition_ids(&contract, Population::HeldOut, 2);
        let mut mixed = EpisodeStore::new(6);
        for id in &heldout_ids {
            let mut heldout = episode(&source, id, vec![0.7, -0.8], 3);
            // Only the later admission evaluator validates held-out numerics.
            // Replaying this record during gradient preparation must fail.
            heldout.context_vector[0] = f32::NAN;
            record(&mut mixed, heldout, EpisodeSignal::Helpful);
        }
        record_training_ids(&source, &mut mixed, &training_ids);
        let mut training_only = EpisodeStore::new(4);
        record_training_ids(&source, &mut training_only, &training_ids);
        let mut registered = registered_builder(contract.clone());
        let mut diagnostic = builder(2, 2);
        for round in 0..2 {
            let actual = candidate(registered.build_candidate(&source, &mixed).unwrap());
            let expected = candidate(diagnostic.build_candidate(&source, &training_only).unwrap());
            assert_eq!(actual.gate_bytes(), expected.gate_bytes());
            assert_eq!(actual.round().update, expected.round().update);
            assert_eq!(actual.outputs(), expected.outputs());
            assert_eq!(actual.round().population.held_out, 2);
            assert_eq!(actual.round().population.available, 4 - round * 2);
            assert_eq!(
                actual.round().completion_ids,
                training_ids[round * 2..round * 2 + 2]
            );
            assert!(
                actual
                    .round()
                    .completion_ids
                    .iter()
                    .all(|id| { contract.split(id).unwrap() == Population::Training })
            );
        }
        assert_eq!(registered.consumed, diagnostic.consumed);
        assert!(
            heldout_ids
                .iter()
                .all(|id| !registered.consumed.contains(id))
        );
    }

    #[test]
    fn admission_heldout_population_cannot_satisfy_the_training_minimum() {
        let source = source();
        let contract = admission_contract();
        let heldout_ids = partition_ids(&contract, Population::HeldOut, 4);
        let mut store = EpisodeStore::new(4);
        record_training_ids(&source, &mut store, &heldout_ids);
        let mut registered = registered_builder(contract);
        match registered.build_candidate(&source, &store).unwrap() {
            RefitVerdict::BelowMinimum(population) => {
                assert_eq!(population.judged, 4);
                assert_eq!(population.held_out, 4);
                assert_eq!(population.available, 0);
            }
            other => panic!("held-out-only store produced a training round: {other:?}"),
        }
        assert!(registered.consumed.is_empty());
    }

    #[test]
    fn admission_contract_binding_is_immutable_across_rounds_and_epochs() {
        let source = source();
        let mut contract = admission_contract();
        let expected_hash = contract.hash().unwrap();
        let training_ids = partition_ids(&contract, Population::Training, 4);
        let mut registered = registered_builder(contract.clone());
        contract.split_salt.push_str("-changed-outside-builder");
        assert_ne!(contract.hash().unwrap(), expected_hash);
        let mut store = EpisodeStore::new(4);
        record_training_ids(&source, &mut store, &training_ids);
        for _ in 0..2 {
            let result = candidate(registered.build_candidate(&source, &store).unwrap());
            assert_eq!(
                result.admission_contract_hash(),
                Some(expected_hash.as_str())
            );
            assert!(result.matches_source(&source));
        }

        let mut next = super::tests::source();
        next.key.identity.epoch += 1;
        let mut next_store = EpisodeStore::new(4);
        record_training_ids(&next, &mut next_store, &training_ids);
        let result = candidate(registered.build_candidate(&next, &next_store).unwrap());
        assert_eq!(
            result.admission_contract_hash(),
            Some(expected_hash.as_str())
        );
        assert_eq!(result.round().completion_ids, training_ids[..2]);
        assert_eq!(result.round().population.already_consumed, 0);
        assert!(result.matches_source(&next));
        assert!(!result.matches_source(&source));
    }

    #[test]
    fn admission_legacy_candidates_remain_unbound_and_match_the_full_source() {
        let source = source();
        let result = candidate(
            builder(2, 2)
                .build_candidate(&source, &pair(&source, 2))
                .unwrap(),
        );
        assert_eq!(result.admission_contract_hash(), None);
        assert!(result.matches_source(&source));
        let mut changed = super::tests::source();
        changed.key.artifact_hash.push('x');
        assert_eq!(changed.identity(), source.identity());
        assert!(!result.matches_source(&changed));
    }

    #[test]
    fn configuration_requires_explicit_valid_bounds_and_coefficients() {
        for (minimum_round, maximum_round) in [(0, 0), (0, 2), (3, 2)] {
            assert!(
                RefitBuilder::new(
                    training(),
                    RefitConfig {
                        minimum_round,
                        maximum_round
                    }
                )
                .is_err()
            );
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
            for index in 0..3 {
                let mut config = training();
                match index {
                    0 => config.learning_rate = bad,
                    1 => config.aux_loss_coeff = bad,
                    _ => config.z_loss_coeff = bad,
                }
                assert!(
                    RefitBuilder::new(
                        config,
                        RefitConfig {
                            minimum_round: 1,
                            maximum_round: 2
                        }
                    )
                    .is_err()
                );
            }
        }
        assert!(
            RefitBuilder::new(
                RlooConfig {
                    learning_rate: 0.0,
                    aux_loss_coeff: 0.0,
                    z_loss_coeff: 0.0
                },
                RefitConfig {
                    minimum_round: 1,
                    maximum_round: 1
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn source_requires_enabled_recorded_exploration_and_exact_live_gate() {
        assert!(RefitSource::from_serving(artifact(), source().router, true, Some(0.5)).is_ok());
        assert!(matches!(
            RefitSource::from_serving(artifact(), source().router, false, Some(0.5)),
            Err(RefitError::IneligibleSource(problem)) if problem == "learning is not enabled"
        ));
        let mut art = artifact();
        art.provenance = RouterProvenance::Legacy;
        assert!(matches!(
            RefitSource::from_serving(art, source().router, true, Some(0.5)),
            Err(RefitError::IneligibleSource(problem)) if problem == "the artifact is not learner eligible"
        ));
        let mut art = artifact();
        if let RouterProvenance::Recorded(recorded) = &mut art.provenance {
            recorded.policy.exploration_scale = None;
        }
        assert!(
            ServingRouter::new(art, true)
                .unwrap()
                .refit_source()
                .is_err()
        );

        let art = artifact();
        let mut changed = source().router;
        let zero = Network::new(vec![Layer::zeros(2, 3, Activation::Linear).unwrap()]).unwrap();
        changed.reload(&zero.to_bytes()).unwrap();
        assert!(matches!(
            RefitSource::from_serving(art, changed, true, Some(0.5)),
            Err(RefitError::IneligibleSource(_))
        ));
    }

    #[test]
    fn source_refuses_duplicate_columns_nonfinite_parameters_and_unusable_narrowing() {
        let mut art = artifact();
        art.adapter_names[1] = art.adapter_names[0].clone();
        assert!(
            ServingRouter::new(art, true)
                .unwrap()
                .refit_source()
                .is_err()
        );
        let mut art = artifact();
        let mut gate = Network::from_bytes(&art.gate_bytes).unwrap();
        gate.layer_mut(0).unwrap().weights_mut()[0] = f32::NAN;
        art.gate_bytes = gate.to_bytes();
        assert!(
            ServingRouter::new(art, true)
                .unwrap()
                .refit_source()
                .is_err()
        );
        for scale in [f64::MAX, 1e30] {
            let mut art = artifact();
            if let RouterProvenance::Recorded(recorded) = &mut art.provenance {
                recorded.policy.exploration_scale = Some(scale);
            }
            assert!(
                ServingRouter::new(art, true)
                    .unwrap()
                    .refit_source()
                    .is_err()
            );
        }
    }

    #[test]
    fn minimum_counts_only_new_explicit_explored_judgements_in_the_current_epoch() {
        let source = source();
        let mut store = EpisodeStore::new(8);
        record(
            &mut store,
            episode(&source, "judged", vec![0.4, -0.2], 3),
            EpisodeSignal::Helpful,
        );
        store.record(episode(&source, "missing", vec![0.4, -0.2], 3));
        let mut deterministic = episode(&source, "deterministic", vec![0.4, -0.2], 3);
        deterministic.exploration = None;
        record(&mut store, deterministic, EpisodeSignal::Unhelpful);
        let mut stale = episode(&source, "stale", vec![0.4, -0.2], 3);
        stale.policy.epoch += 1;
        record(&mut store, stale, EpisodeSignal::Unhelpful);
        let mut builder = builder(2, 3);
        match builder.build_candidate(&source, &store).unwrap() {
            RefitVerdict::BelowMinimum(population) => {
                assert_eq!(population.judged, 2);
                assert_eq!(population.deterministic, 1);
                assert_eq!(population.available, 1);
            }
            other => panic!("unexpected round: {other:?}"),
        }
        assert!(builder.consumed.is_empty());
    }

    #[test]
    fn maximum_uses_recording_order_and_successful_rounds_do_not_reuse_episodes() {
        let source = source();
        let mut store = pair(&source, 8);
        record(
            &mut store,
            episode(&source, "third", vec![0.7, 0.1], 3),
            EpisodeSignal::Helpful,
        );
        record(
            &mut store,
            episode(&source, "fourth", vec![-0.2, 0.6], 3),
            EpisodeSignal::Unhelpful,
        );
        let mut builder = builder(2, 2);
        let first = candidate(builder.build_candidate(&source, &store).unwrap());
        assert_eq!(first.round().completion_ids, ["first", "second"]);
        assert_eq!(first.round().population.available, 4);
        let second = candidate(builder.build_candidate(&source, &store).unwrap());
        assert_eq!(second.round().completion_ids, ["third", "fourth"]);
        assert_eq!(second.round().population.already_consumed, 2);
        assert!(matches!(
            builder.build_candidate(&source, &store).unwrap(),
            RefitVerdict::BelowMinimum(RefitPopulation { available: 0, .. })
        ));
    }

    #[test]
    fn constant_rewards_preserve_gate_bytes_and_still_update_traffic_history() {
        let source = source();
        let mut store = EpisodeStore::new(2);
        for id in ["constant-one", "constant-two"] {
            record(
                &mut store,
                episode(&source, id, vec![4.0, -0.2], 3),
                EpisodeSignal::Helpful,
            );
        }
        let original = source.gate_bytes().to_vec();
        let mut builder = builder(2, 2);
        let unchanged = builder.build_candidate(&source, &store).unwrap();
        assert!(matches!(
            unchanged,
            RefitVerdict::Unchanged(RefitRound {
                update: BatchRefitOutcome { informative: 0, .. },
                ..
            })
        ));
        assert_eq!(source.gate_bytes(), original);
        assert_eq!(builder.consumed.len(), 2);
        record(
            &mut store,
            episode(&source, "new-one", vec![0.4, -0.2], 3),
            EpisodeSignal::Helpful,
        );
        record(
            &mut store,
            episode(&source, "new-two", vec![-0.3, 0.8], 3),
            EpisodeSignal::Unhelpful,
        );
        let with_history = candidate(builder.build_candidate(&source, &store).unwrap());
        let without_history = candidate(
            super::tests::builder(2, 2)
                .build_candidate(&source, &store)
                .unwrap(),
        );
        assert_ne!(with_history.gate_bytes(), without_history.gate_bytes());
        assert_eq!(
            builder.consumed.len(),
            2,
            "evicted consumption records must be pruned"
        );
    }

    #[test]
    fn one_episode_is_a_byte_noop_and_is_consumed() {
        let source = source();
        let mut store = EpisodeStore::new(1);
        record(
            &mut store,
            episode(&source, "single", vec![0.4, -0.2], 2),
            EpisodeSignal::Helpful,
        );
        let mut builder = builder(1, 1);
        assert!(matches!(
            builder.build_candidate(&source, &store).unwrap(),
            RefitVerdict::Unchanged(_)
        ));
        assert!(matches!(
            builder.build_candidate(&source, &store).unwrap(),
            RefitVerdict::BelowMinimum(_)
        ));
    }

    #[test]
    fn recorded_provenance_must_replay_exactly_including_order_and_numeric_fields() {
        let mutations: &[fn(&mut RouterEpisode)] = &[
            |e| e.policy.artifact_version.push('x'),
            |e| e.policy.schema_id.push('x'),
            |e| e.adapter_names.swap(0, 1),
            |e| e.context_vector[0] += 1.0,
            |e| e.context_vector[0] = f32::NAN,
            |e| {
                e.context_vector.pop();
            },
            |e| e.tau = Some(0.8),
            |e| e.tau = None,
            |e| e.exploration.as_mut().unwrap().scale = 0.6,
            |e| e.exploration.as_mut().unwrap().seed ^= 1,
            |e| e.exploration.as_mut().unwrap().sampled_logits[0] += 1.0,
            |e| e.exploration.as_mut().unwrap().sampled_logits[0] = f32::INFINITY,
            |e| {
                e.exploration.as_mut().unwrap().sampled_logits.pop();
            },
            |e| e.exploration.as_mut().unwrap().log_density += 1.0,
            |e| e.exploration.as_mut().unwrap().log_density = f64::NAN,
            |e| e.selected_scores.swap(0, 1),
            |e| e.sampled_scores.swap(0, 1),
            |e| e.pre_floor_weights.swap(0, 1),
            |e| e.applied_weights.swap(0, 1),
            |e| e.selected_scores[0].0 = "unknown".into(),
            |e| e.selected_scores[0].0 = e.selected_scores[1].0.clone(),
            |e| e.sampled_scores[0].1 = f32::NAN,
            |e| e.pre_floor_weights[0].1 = -0.1,
            |e| e.applied_weights[0].1 = 0.0,
            |e| e.dropped.push("unknown".into()),
        ];
        let source = source();
        for (index, mutate) in mutations.iter().enumerate() {
            let mut store = pair(&source, 2);
            let mut bad = store.get("first").unwrap().0.clone();
            mutate(&mut bad);
            record(&mut store, bad, EpisodeSignal::Helpful);
            let mut builder = builder(2, 2);
            assert!(
                matches!(
                    builder.build_candidate(&source, &store),
                    Err(RefitError::InvalidEpisode { .. })
                ),
                "mutation {index} must refuse"
            );
            assert!(builder.source.is_none());
            assert!(builder.consumed.is_empty());
        }
    }

    #[test]
    fn constant_rewards_do_not_bypass_provenance_validation() {
        let source = source();
        let mut store = EpisodeStore::new(2);
        for id in ["one", "two"] {
            let mut event = episode(&source, id, vec![0.4, -0.2], 3);
            event.context_vector[0] += 1.0;
            record(&mut store, event, EpisodeSignal::Helpful);
        }
        assert!(matches!(
            builder(2, 2).build_candidate(&source, &store),
            Err(RefitError::InvalidEpisode { .. })
        ));
    }

    #[test]
    fn candidate_outputs_come_from_candidate_bytes_and_the_existing_floor_policy() {
        let mut art = artifact();
        if let RouterProvenance::Recorded(recorded) = &mut art.provenance {
            recorded.policy.epsilon = 0.6;
        }
        let live = ServingRouter::new(art, true).unwrap();
        let source = live.refit_source().unwrap();
        let store = pair(&source, 2);
        let before_epoch = live.policy_epoch();
        let before_artifact = live.artifact().clone();
        let result = candidate(builder(2, 2).build_candidate(&source, &store).unwrap());
        assert_ne!(result.gate_bytes(), source.gate_bytes());
        let mut evaluated = source.policy_router(result.gate_bytes()).unwrap();
        let mut coefficient_changed = false;
        let mut floor_observed = false;
        let mut entropy = 0.0_f64;
        for output in result.outputs() {
            let event = store.get(&output.completion_id).unwrap().0;
            let expected = evaluated
                .route_explored(
                    &event.context_vector,
                    &event.adapter_names,
                    event.selected_scores.len(),
                    Some(Exploration {
                        scale: source.scale,
                        seed: seed_from_completion_id(&event.completion_id),
                    }),
                )
                .unwrap();
            assert_eq!(output.route, expected);
            coefficient_changed |= output.route.pre_floor_weights != event.pre_floor_weights;
            floor_observed |= output.route.pre_floor_weights != output.route.weights;
            entropy += f64::from(weight_entropy(
                &output
                    .route
                    .pre_floor_weights
                    .iter()
                    .map(|(_, w)| *w)
                    .collect::<Vec<_>>(),
            ));
        }
        assert!(coefficient_changed);
        assert!(floor_observed);
        assert_eq!(
            result.mean_pre_floor_entropy(),
            (entropy / result.outputs().len() as f64) as f32
        );
        assert_eq!(live.policy_epoch(), before_epoch);
        assert_eq!(live.artifact(), &before_artifact);
    }

    #[test]
    fn a_floor_that_drops_every_adapter_remains_a_valid_policy_output() {
        let mut art = artifact();
        if let RouterProvenance::Recorded(recorded) = &mut art.provenance {
            recorded.policy.epsilon = 1.0;
        }
        let source = ServingRouter::new(art, true)
            .unwrap()
            .refit_source()
            .unwrap();
        let store = pair(&source, 2);
        let result = candidate(builder(2, 2).build_candidate(&source, &store).unwrap());
        assert!(
            result
                .outputs()
                .iter()
                .all(|output| output.route.weights.is_empty() && output.route.dropped.len() == 3)
        );
    }

    #[test]
    fn a_failed_candidate_evaluation_does_not_consume_ids_or_commit_trainer_history() {
        let mut art = artifact();
        art.gate_bytes = Network::new(vec![Layer::zeros(2, 3, Activation::Linear).unwrap()])
            .unwrap()
            .to_bytes();
        let source = ServingRouter::new(art, true)
            .unwrap()
            .refit_source()
            .unwrap();
        let mut store = EpisodeStore::new(2);
        record(
            &mut store,
            episode(&source, "overflow-one", vec![1e20, 0.0], 3),
            EpisodeSignal::Helpful,
        );
        record(
            &mut store,
            episode(&source, "overflow-two", vec![1e20, 0.0], 3),
            EpisodeSignal::Unhelpful,
        );
        let mut builder = builder(2, 2);
        assert!(matches!(
            builder.build_candidate(&source, &store),
            Err(RefitError::CandidateOutput { .. })
        ));
        assert!(builder.source.is_none());
        assert!(builder.consumed.is_empty());
        let good = pair(&source, 2);
        let recovered = candidate(builder.build_candidate(&source, &good).unwrap());
        let pristine = candidate(
            super::tests::builder(2, 2)
                .build_candidate(&source, &good)
                .unwrap(),
        );
        assert_eq!(recovered, pristine);
    }

    #[test]
    fn a_failed_training_update_does_not_consume_ids_or_commit_trainer_history() {
        let mut art = artifact();
        if let RouterProvenance::Recorded(recorded) = &mut art.provenance {
            recorded.policy.exploration_scale = Some(f64::from(5e18_f32));
        }
        let bad_source = ServingRouter::new(art, true)
            .unwrap()
            .refit_source()
            .unwrap();
        let mut store = EpisodeStore::new(2);
        record(
            &mut store,
            episode(&bad_source, "late-overflow-430", vec![0.4, -0.2], 3),
            EpisodeSignal::Helpful,
        );
        record(
            &mut store,
            episode(&bad_source, "large-two", vec![-0.3, 0.8], 3),
            EpisodeSignal::Unhelpful,
        );
        let mut builder = builder(2, 2);
        assert!(matches!(
            builder.build_candidate(&bad_source, &store),
            Err(RefitError::Training(problem)) if problem == "batch statistics are not finite"
        ));
        assert!(builder.source.is_none());
        assert!(builder.consumed.is_empty());

        let good_source = source();
        let good = pair(&good_source, 2);
        let recovered = candidate(builder.build_candidate(&good_source, &good).unwrap());
        let pristine = candidate(
            super::tests::builder(2, 2)
                .build_candidate(&good_source, &good)
                .unwrap(),
        );
        assert_eq!(recovered, pristine);

        // The scale keeps gradients finite but overflows the scalar loss only
        // after the trainer has observed both traffic masses. Retaining that
        // trainer must change the follow-up, or this fixture proves nothing.
        let mut producing = bad_source.policy_router(bad_source.gate_bytes()).unwrap();
        let columns = bad_source
            .key
            .adapter_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let prepared: Vec<_> = store
            .ordered_on_policy_samples(bad_source.identity().epoch)
            .map(|(event, signal)| {
                prepare(&bad_source, &mut producing, event, signal, &columns).unwrap()
            })
            .collect();
        let samples: Vec<_> = prepared
            .iter()
            .map(|p| MixtureSample {
                context: &p.episode.context_vector,
                action: ExploredAction {
                    sampled_logits: &p.draw.sampled_logits,
                    exploration_scale: p.draw.scale,
                    w_full: &p.w_full,
                    tau: bad_source.tau,
                },
                reward: p.reward,
            })
            .collect();
        let mut contaminated = super::tests::builder(2, 2);
        let mut gate = Network::from_bytes(bad_source.gate_bytes()).unwrap();
        let update = contaminated
            .trainer
            .step_batch_with_explored_actions(&mut gate, &samples)
            .unwrap();
        assert!(!update.mean_loss.is_finite());
        let contaminated = candidate(contaminated.build_candidate(&good_source, &good).unwrap());
        assert_ne!(contaminated.gate_bytes(), pristine.gate_bytes());
    }

    #[test]
    fn backwards_epochs_and_same_epoch_artifact_replacement_refuse() {
        let source = source();
        let mut builder = builder(2, 2);
        builder
            .build_candidate(&source, &EpisodeStore::new(1))
            .unwrap();
        let mut older = super::tests::source();
        older.key.identity.epoch -= 1;
        assert!(matches!(
            builder.build_candidate(&older, &EpisodeStore::new(1)),
            Err(RefitError::SourceChanged(_))
        ));
        let mut changed = super::tests::source();
        changed.key.artifact_hash.push('x');
        assert!(matches!(
            builder.build_candidate(&changed, &EpisodeStore::new(1)),
            Err(RefitError::SourceChanged(_))
        ));
        assert_eq!(builder.source.as_ref().unwrap(), &source.key);
    }

    #[test]
    fn a_schema_or_representation_change_requires_an_explicit_new_builder() {
        let source = source();
        let mut builder = builder(2, 2);
        builder
            .build_candidate(&source, &EpisodeStore::new(1))
            .unwrap();
        for index in 0..4 {
            let mut changed = super::tests::source();
            changed.key.identity.epoch += 1;
            match index {
                0 => changed.key.identity.schema_id.push('x'),
                1 => changed.key.base_model.push('x'),
                2 => changed.key.adapter_names.swap(0, 1),
                _ => changed.key.representation.pooling.push('x'),
            }
            assert!(matches!(
                builder.build_candidate(&changed, &EpisodeStore::new(1)),
                Err(RefitError::SourceChanged(_))
            ));
        }
    }

    #[test]
    fn a_new_same_lineage_epoch_resets_consumption_and_traffic_history() {
        let source = source();
        let mut builder = builder(2, 2);
        builder.build_candidate(&source, &pair(&source, 2)).unwrap();
        let mut next = super::tests::source();
        next.key.identity.epoch += 1;
        let store = pair(&next, 2);
        let continued = candidate(builder.build_candidate(&next, &store).unwrap());
        let fresh = candidate(
            super::tests::builder(2, 2)
                .build_candidate(&next, &store)
                .unwrap(),
        );
        assert_eq!(continued, fresh);
        assert_eq!(continued.round().population.already_consumed, 0);
    }
}
