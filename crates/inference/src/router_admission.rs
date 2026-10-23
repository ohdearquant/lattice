//! Registered, held-out router admission evidence (ADR-096 decision 5).
//!
//! Evaluation consumes a partition-bound candidate and reference measurements
//! produced by the teacher-forced scorer. It does not publish an artifact or
//! change a serving gate. An admission record is evidence, not a storage grant.
//! Episodes and model handles belong to the server's bounded refit task; this
//! in-process API does not authenticate caller-created episode stores.

mod reference;
mod statistics;

#[cfg(all(
    feature = "serve",
    feature = "f16",
    target_os = "macos",
    feature = "metal-gpu"
))]
pub use reference::{ReferenceAdapter, score_metal_reference};
pub use reference::{
    ReferenceBounds, ReferenceError, ReferenceEvidence, ReferenceExample, ReferencePayloadSummary,
    reference_content_hash,
};
pub use statistics::{AdmissionContract, Population, StatisticsError};

use std::collections::HashSet;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::mixture::{
    AdapterRouter, Exploration, RefitOutcome, RouteOutcome, seed_from_completion_id,
};
use crate::router_episode::{EpisodeSignal, EpisodeStore, RouterEpisode};
use crate::router_refit::{RefitCandidate, RefitSource};
use statistics::{ImportanceEstimate, WeightedEvent};

/// A failed evaluation never supplies a positive admission record.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdmissionError {
    /// A qualification contract is mandatory, including for a previously trained gate.
    #[error("router admission requires a registered contract")]
    MissingContract,
    /// One candidate round contributes at most once to the entropy history.
    #[error("this candidate round has already been evaluated")]
    AlreadyEvaluated,
    /// The registered task budget has been exhausted.
    #[error("the registered candidate-evaluation budget is exhausted")]
    EvaluationBudgetExhausted,
    /// A registered setting or numerical operation is invalid.
    #[error(transparent)]
    Statistics(#[from] StatisticsError),
    /// The candidate was not fitted under this source and partition.
    #[error("candidate admission binding disagrees: {0}")]
    CandidateBinding(String),
    /// Reference measurements were not made for this evaluation's exact inputs.
    #[error("reference evidence disagrees: {0}")]
    ReferenceBinding(String),
    /// A judged event cannot be attributed to the producing policy.
    #[error("held-out episode {completion_id:?} failed provenance: {problem}")]
    Provenance {
        /// The event whose evidence was rejected.
        completion_id: String,
        /// The failed invariant, including a tolerance mismatch when applicable.
        problem: String,
    },
    /// A gate or serving transformation could not be evaluated.
    #[error("router admission evaluation failed: {0}")]
    Evaluation(String),
}

/// All registered requirements must pass; there is no inconclusive positive result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionVerdict {
    /// The evaluated candidate met every registered requirement.
    Admitted,
    /// At least one registered requirement failed.
    Rejected,
}

/// Immutable evidence produced by evaluation, never deserialized as authority.
#[derive(Debug, Clone, Serialize)]
pub struct AdmissionRecord {
    contract_hash: String,
    contract: AdmissionContract,
    maximum_evaluations: usize,
    policy: crate::router_state::RouterPolicy,
    parent_artifact_hash: String,
    parent_epoch: u64,
    candidate_gate_hash: String,
    reference_hash: String,
    reference_binding_hash: String,
    heldout_ids_hash: String,
    heldout_count: usize,
    importance: Option<ImportanceEstimate>,
    maximum_parent_logit_difference: f64,
    reference_token_count: usize,
    candidate_mean_nll: f64,
    parent_mean_nll: f64,
    base_mean_nll: f64,
    regression: f64,
    mean_entropy: f32,
    consecutive_collapsed_rounds: usize,
    verdict: AdmissionVerdict,
    reasons: Vec<String>,
}

impl AdmissionRecord {
    /// Whether every registered requirement passed for these bound inputs.
    pub fn verdict(&self) -> AdmissionVerdict {
        self.verdict
    }

    /// Every failed requirement, in stable evaluation order.
    pub fn reasons(&self) -> &[String] {
        &self.reasons
    }

    /// SHA-256 of the complete versioned record, suitable for artifact provenance.
    pub fn content_hash(&self) -> Result<String, AdmissionError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| AdmissionError::Evaluation(error.to_string()))?;
        let mut hash = Sha256::new();
        hash.update(b"lattice-router-admission-record-v1\0");
        hash.update(bytes);
        Ok(format!("{:x}", hash.finalize()))
    }
}

/// Bounded-task admission state for one producing epoch and registered contract.
///
/// The server retains one evaluator across that epoch's rounds. The initial
/// collapse count is captured from the producing router; subsequent evaluated
/// rounds advance it once, including quality rejections. Failed evidence checks
/// leave it unchanged. A new producing epoch requires a new evaluator.
pub struct AdmissionEvaluator {
    contract: AdmissionContract,
    parent_artifact_hash: String,
    epoch: u64,
    guard: AdapterRouter,
    evaluated_rounds: HashSet<String>,
    bounds: AdmissionBounds,
}

/// Explicit task resource limits, supplied by the qualification run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionBounds {
    /// Maximum distinct candidate rounds retained and evaluated in this task.
    pub maximum_evaluations: usize,
}

impl AdmissionEvaluator {
    /// Bind the evaluator to explicit qualification settings and producing state.
    pub fn new(
        source: &RefitSource,
        contract: &AdmissionContract,
        bounds: AdmissionBounds,
    ) -> Result<Self, AdmissionError> {
        contract.validate()?;
        if bounds.maximum_evaluations == 0 {
            return Err(AdmissionError::Evaluation(
                "maximum_evaluations must be positive".into(),
            ));
        }
        Ok(Self {
            contract: contract.clone(),
            parent_artifact_hash: source.artifact_hash().to_string(),
            epoch: source.identity().epoch,
            guard: source
                .policy_router(source.gate_bytes())
                .map_err(|error| AdmissionError::Evaluation(error.to_string()))?,
            evaluated_rounds: HashSet::new(),
            bounds,
        })
    }

    /// Number of consecutive collapsed candidate rounds observed in this epoch.
    pub fn consecutive_collapsed_rounds(&self) -> usize {
        self.guard.consecutive_collapsed_rounds()
    }

    /// Evaluate once without publishing, activating, or mutating the serving gate.
    pub fn evaluate_candidate(
        &mut self,
        source: &RefitSource,
        candidate: &RefitCandidate,
        episodes: &EpisodeStore,
        reference: &ReferenceEvidence,
        expected_reference_hash: &str,
    ) -> Result<AdmissionRecord, AdmissionError> {
        if source.artifact_hash() != self.parent_artifact_hash
            || source.identity().epoch != self.epoch
        {
            return Err(AdmissionError::CandidateBinding(
                "evaluator belongs to a different producing epoch".into(),
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"lattice-router-admission-round-v1\0");
        hash.update(Sha256::digest(candidate.gate_bytes()));
        for id in &candidate.round().completion_ids {
            hash.update((id.len() as u64).to_be_bytes());
            hash.update(id.as_bytes());
        }
        let round = format!("{:x}", hash.finalize());
        if self.evaluated_rounds.contains(&round) {
            return Err(AdmissionError::AlreadyEvaluated);
        }
        if self.evaluated_rounds.len() >= self.bounds.maximum_evaluations {
            return Err(AdmissionError::EvaluationBudgetExhausted);
        }
        let mut guard = self.guard.refit_snapshot();
        let record = evaluate_candidate(
            source,
            candidate,
            episodes,
            &self.contract,
            reference,
            expected_reference_hash,
            &mut guard,
            self.bounds,
        )?;
        self.guard = guard;
        self.evaluated_rounds.insert(round);
        Ok(record)
    }
}

/// Parse a required qualification contract without filling missing fields.
pub fn parse_contract(bytes: Option<&[u8]>) -> Result<AdmissionContract, AdmissionError> {
    let bytes = bytes.ok_or(AdmissionError::MissingContract)?;
    let contract: AdmissionContract = serde_json::from_slice(bytes)
        .map_err(|error| AdmissionError::Evaluation(format!("unreadable contract: {error}")))?;
    contract.validate()?;
    Ok(contract)
}

fn evaluate_candidate(
    source: &RefitSource,
    candidate: &RefitCandidate,
    episodes: &EpisodeStore,
    contract: &AdmissionContract,
    reference: &ReferenceEvidence,
    expected_reference_hash: &str,
    guard: &mut AdapterRouter,
    bounds: AdmissionBounds,
) -> Result<AdmissionRecord, AdmissionError> {
    let contract_hash = contract.hash()?;
    if !candidate.matches_source(source)
        || candidate.admission_contract_hash() != Some(contract_hash.as_str())
    {
        return Err(AdmissionError::CandidateBinding(
            "source identity or registered training partition differs".into(),
        ));
    }
    if candidate.round().completion_ids.is_empty()
        || candidate.round().completion_ids.len() != candidate.outputs().len()
    {
        return Err(AdmissionError::CandidateBinding(
            "candidate has no complete fitting population".into(),
        ));
    }
    let mut training_ids = HashSet::new();
    for id in &candidate.round().completion_ids {
        if !training_ids.insert(id.as_str()) || contract.split(id)? != Population::Training {
            return Err(AdmissionError::CandidateBinding(
                "a duplicate or held-out completion entered candidate fitting".into(),
            ));
        }
    }
    reference
        .matches(source, candidate, contract)
        .map_err(|error| AdmissionError::ReferenceBinding(error.to_string()))?;
    if [
        reference.parent_mean_nll,
        reference.candidate_mean_nll,
        reference.base_mean_nll,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
        || !reference.regression.is_finite()
        || reference.regression != reference.candidate_mean_nll - reference.parent_mean_nll
    {
        return Err(AdmissionError::ReferenceBinding(
            "reference measurements are nonfinite or internally inconsistent".into(),
        ));
    }
    if expected_reference_hash.is_empty() || reference.reference_hash != expected_reference_hash {
        return Err(AdmissionError::ReferenceBinding(
            "reference content differs from the predeclared set".into(),
        ));
    }

    let mut parent = source
        .policy_router(source.gate_bytes())
        .map_err(|error| AdmissionError::Evaluation(error.to_string()))?;
    let mut proposed = source
        .policy_router(candidate.gate_bytes())
        .map_err(|error| AdmissionError::Evaluation(error.to_string()))?;
    let mut evidence = Vec::new();
    let mut heldout_ids = Vec::new();
    let mut maximum_parent_logit_difference = 0.0_f64;
    for (episode, signal) in episodes.ordered_on_policy_samples(source.identity().epoch) {
        if episode.policy != *source.identity() || episode.adapter_names != source.adapter_names() {
            return Err(provenance(
                episode,
                "producing artifact or ordered schema differs",
            ));
        }
        if contract.split(&episode.completion_id)? != Population::HeldOut {
            continue;
        }
        // Deterministic traffic has no density under the recorded exploration policy.
        if episode.exploration.is_none() {
            continue;
        }
        let (event, difference) = evaluate_event(
            source,
            &mut parent,
            &mut proposed,
            episode,
            signal,
            contract.parent_logit_tolerance,
        )?;
        maximum_parent_logit_difference = maximum_parent_logit_difference.max(difference);
        evidence.push(event);
        heldout_ids.push(episode.completion_id.as_str());
    }
    let importance = if evidence.is_empty() {
        None
    } else {
        Some(statistics::evaluate(&evidence, contract)?)
    };

    let mut round_weights = Vec::with_capacity(candidate.outputs().len());
    for (id, output) in candidate
        .round()
        .completion_ids
        .iter()
        .zip(candidate.outputs())
    {
        if id != &output.completion_id {
            return Err(AdmissionError::CandidateBinding(
                "candidate output population differs from fitting identities".into(),
            ));
        }
        let weights: Vec<f32> = output
            .route
            .weights
            .iter()
            .map(|(_, weight)| *weight)
            .collect();
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0 || *weight > 1.0)
        {
            return Err(AdmissionError::Evaluation(
                "candidate entropy inputs are invalid".into(),
            ));
        }
        round_weights.push(weights);
    }
    let outcome = guard
        .submit_refit(candidate.gate_bytes(), &round_weights)
        .map_err(|error| AdmissionError::Evaluation(error.to_string()))?;
    let (entropy_passed, mean_entropy) = match outcome {
        RefitOutcome::Accepted { mean_entropy } => (true, mean_entropy),
        RefitOutcome::Rejected { mean_entropy, .. } => (false, mean_entropy),
    };
    if !mean_entropy.is_finite() {
        return Err(AdmissionError::Evaluation(
            "candidate entropy is not finite".into(),
        ));
    }

    let reasons = failed_requirements(
        contract,
        evidence.len(),
        importance.as_ref(),
        reference,
        entropy_passed,
    );
    let verdict = if reasons.is_empty() {
        AdmissionVerdict::Admitted
    } else {
        AdmissionVerdict::Rejected
    };
    let mut ids_hash = Sha256::new();
    ids_hash.update(b"lattice-router-heldout-ids-v1\0");
    for id in heldout_ids {
        ids_hash.update((id.len() as u64).to_be_bytes());
        ids_hash.update(id.as_bytes());
    }
    Ok(AdmissionRecord {
        contract_hash,
        contract: contract.clone(),
        maximum_evaluations: bounds.maximum_evaluations,
        policy: source.policy().clone(),
        parent_artifact_hash: source.artifact_hash().to_string(),
        parent_epoch: source.identity().epoch,
        candidate_gate_hash: format!("{:x}", Sha256::digest(candidate.gate_bytes())),
        reference_hash: reference.reference_hash.clone(),
        reference_binding_hash: reference.binding_hash.clone(),
        heldout_ids_hash: format!("{:x}", ids_hash.finalize()),
        heldout_count: evidence.len(),
        importance,
        maximum_parent_logit_difference,
        reference_token_count: reference.token_count,
        candidate_mean_nll: reference.candidate_mean_nll,
        parent_mean_nll: reference.parent_mean_nll,
        base_mean_nll: reference.base_mean_nll,
        regression: reference.regression,
        mean_entropy,
        consecutive_collapsed_rounds: guard.consecutive_collapsed_rounds(),
        verdict,
        reasons,
    })
}

fn failed_requirements(
    contract: &AdmissionContract,
    count: usize,
    importance: Option<&ImportanceEstimate>,
    reference: &ReferenceEvidence,
    entropy_passed: bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if count < contract.min_heldout_events {
        reasons.push("insufficient held-out event count".into());
    }
    if importance
        .is_none_or(|estimate| estimate.effective_sample_size < contract.min_effective_sample_size)
    {
        reasons.push("insufficient held-out effective sample size".into());
    }
    if reference.token_count < contract.min_reference_tokens {
        reasons.push("insufficient reference token count".into());
    }
    if importance.is_none_or(|estimate| estimate.lower_bound <= 0.0) {
        reasons.push("paired improvement lower bound is not strictly positive".into());
    }
    if reference.regression > contract.regression_limit {
        reasons.push("reference NLL regression exceeds the registered limit".into());
    }
    if !entropy_passed {
        reasons.push("candidate failed the recorded entropy guard".into());
    }
    reasons
}

fn provenance(episode: &RouterEpisode, problem: impl Into<String>) -> AdmissionError {
    AdmissionError::Provenance {
        completion_id: episode.completion_id.clone(),
        problem: problem.into(),
    }
}

fn ordered_scores(
    episode: &RouterEpisode,
    values: &[(String, f32)],
    names: &[String],
) -> Result<Vec<f64>, AdmissionError> {
    if values.len() != names.len() {
        return Err(provenance(
            episode,
            "admission requires every trained column's recorded raw logit; top-k episodes are ineligible",
        ));
    }
    let mut ordered = Vec::with_capacity(names.len());
    for name in names {
        let mut matches = values.iter().filter(|(column, _)| column == name);
        let Some((_, value)) = matches.next() else {
            return Err(provenance(episode, "missing logit column"));
        };
        if matches.next().is_some() || !value.is_finite() {
            return Err(provenance(episode, "duplicate or nonfinite logit column"));
        }
        ordered.push(f64::from(*value));
    }
    Ok(ordered)
}

fn route_logits(
    router: &mut AdapterRouter,
    episode: &RouterEpisode,
    names: &[String],
) -> Result<Vec<f64>, AdmissionError> {
    let RouteOutcome {
        selected_scores, ..
    } = router
        .route_detailed(&episode.context_vector, names, names.len())
        .map_err(|error| provenance(episode, error.to_string()))?;
    ordered_scores(episode, &selected_scores, names)
}

fn evaluate_event(
    source: &RefitSource,
    parent: &mut AdapterRouter,
    candidate: &mut AdapterRouter,
    episode: &RouterEpisode,
    signal: EpisodeSignal,
    tolerance: f64,
) -> Result<(WeightedEvent, f64), AdmissionError> {
    let draw = episode
        .exploration
        .as_ref()
        .ok_or_else(|| provenance(episode, "no exploration draw"))?;
    let names = source.adapter_names();
    let expected_scale = source.policy().exploration_scale.map(|scale| scale as f32);
    if episode.context_vector.len() != parent.input_size()
        || episode
            .context_vector
            .iter()
            .any(|value| !value.is_finite())
        || draw.sampled_logits.len() != names.len()
        || draw.sampled_logits.iter().any(|value| !value.is_finite())
        || !draw.log_density.is_finite()
        || !draw.scale.is_finite()
        || draw.scale <= 0.0
        || expected_scale.map(f32::to_bits) != Some(draw.scale.to_bits())
        || episode.tau.map(f32::to_bits) != source.policy().tau.map(|tau| (tau as f32).to_bits())
        || draw.seed != seed_from_completion_id(&episode.completion_id)
    {
        return Err(provenance(
            episode,
            "context, draw, scale, temperature or seed is inconsistent",
        ));
    }
    let recorded = ordered_scores(episode, &episode.selected_scores, names)?;
    let recorded_f32: Vec<f32> = recorded.iter().map(|value| *value as f32).collect();
    let replay = parent
        .replay_recorded_logits(
            &recorded_f32,
            names,
            Exploration {
                scale: draw.scale,
                seed: draw.seed,
            },
        )
        .map_err(|error| provenance(episode, error.to_string()))?;
    let replayed_draw = replay
        .exploration
        .as_ref()
        .ok_or_else(|| provenance(episode, "recorded draw did not replay"))?;
    if !same_named_values(&replay.selected_scores, &episode.selected_scores)
        || !same_named_values(&replay.sampled_scores, &episode.sampled_scores)
        || !same_named_values(&replay.pre_floor_weights, &episode.pre_floor_weights)
        || !same_named_values(&replay.weights, &episode.applied_weights)
        || replay.dropped != episode.dropped
        || replayed_draw.log_density.to_bits() != draw.log_density.to_bits()
        || replayed_draw
            .sampled_logits
            .iter()
            .zip(&draw.sampled_logits)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(provenance(
            episode,
            "recorded draw or applied action does not replay from recorded parent logits",
        ));
    }
    let parent_logits = route_logits(parent, episode, names)?;
    let candidate_logits = route_logits(candidate, episode, names)?;
    let mut difference = 0.0_f64;
    let mut log_weight = 0.0_f64;
    let scale = f64::from(draw.scale);
    for (((recorded, parent), candidate), drawn) in recorded
        .iter()
        .zip(&parent_logits)
        .zip(&candidate_logits)
        .zip(&draw.sampled_logits)
    {
        difference = difference.max((parent - recorded).abs());
        let parent_distance = (f64::from(*drawn) - parent) / scale;
        let candidate_distance = (f64::from(*drawn) - candidate) / scale;
        log_weight +=
            0.5 * (parent_distance * parent_distance - candidate_distance * candidate_distance);
    }
    if difference > tolerance {
        return Err(provenance(
            episode,
            format!(
                "recomputed parent logits differ by {difference}, beyond registered tolerance {tolerance}"
            ),
        ));
    }
    if !log_weight.is_finite() {
        return Err(provenance(
            episode,
            "Gaussian density ratio is not representable",
        ));
    }
    Ok((
        WeightedEvent {
            log_weight,
            reward: signal == EpisodeSignal::Helpful,
        },
        difference,
    ))
}

fn same_named_values(left: &[(String, f32)], right: &[(String, f32)]) -> bool {
    left.len() == right.len()
        && left.iter().all(|(name, value)| {
            let mut matches = right.iter().filter(|(other, _)| other == name);
            matches
                .next()
                .is_some_and(|(_, other)| value.to_bits() == other.to_bits())
                && matches.next().is_none()
        })
}

#[cfg(all(test, feature = "serve"))]
mod tests;
