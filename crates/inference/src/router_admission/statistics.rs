//! Deterministic numerical evidence for ADR-096 D5; this module grants no authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CONTRACT_DOMAIN: &[u8] = b"lattice-router-admission-contract-v1\0";
const SPLIT_DOMAIN: &[u8] = b"lattice-router-admission-split-v1\0";
const EXACT_INTEGER_LIMIT: u64 = 1_u64 << 53;

/// Every value is registered explicitly; deserialization never fills missing settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionContract {
    /// Registered nonempty salt for deterministic completion-identity splitting.
    pub split_salt: String,
    /// Held-out fraction, strictly between zero and one.
    pub heldout_fraction: f64,
    /// Nonnegative absolute tolerance for recomputed parent logits; zero is valid.
    pub parent_logit_tolerance: f64,
    /// Positive number of paired bootstrap resamples.
    pub bootstrap_replicates: usize,
    /// Seed for the versioned deterministic bootstrap sampler; zero is valid.
    pub bootstrap_seed: u64,
    /// Empirical lower-bound quantile, strictly between zero and one.
    pub lower_quantile: f64,
    /// Maximum candidate-minus-parent reference NLL; a negative limit is valid.
    pub regression_limit: f64,
    /// Positive minimum number of held-out events.
    pub min_heldout_events: usize,
    /// Positive, finite minimum effective sample size.
    pub min_effective_sample_size: f64,
    /// Positive minimum number of scored reference-completion tokens.
    pub min_reference_tokens: usize,
}

/// Deterministic membership assigned before fitting or evaluating a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Population {
    /// The event may enter candidate fitting after other provenance checks.
    Training,
    /// The event is reserved for held-out evaluation and must not enter fitting.
    HeldOut,
}

/// The caller derives the density ratio from verified provenance and gate outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WeightedEvent {
    pub(crate) log_weight: f64,
    pub(crate) reward: bool,
}

/// Numerical evidence only: registered floors and the admission rule are checked elsewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ImportanceEstimate {
    pub(crate) count: usize,
    pub(crate) candidate_mean: f64,
    pub(crate) parent_mean: f64,
    pub(crate) delta: f64,
    pub(crate) effective_sample_size: f64,
    pub(crate) lower_bound: f64,
}

/// Invalid settings or unavailable numerical evidence; never an admission verdict.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StatisticsError {
    /// A required setting lies outside its declared mathematical domain.
    #[error("invalid admission setting {field}: {reason}")]
    InvalidSetting {
        /// Name of the invalid registered setting.
        field: &'static str,
        /// Required property that the setting failed.
        reason: &'static str,
    },
    /// An empty string cannot identify a completion for population assignment.
    #[error("completion identity must be nonempty")]
    EmptyCompletionIdentity,
    /// No numerical evidence can be calculated without held-out events.
    #[error("held-out population is empty")]
    EmptyPopulation,
    /// A verified event's numerical weight must have a finite logarithm.
    #[error("event {event} has a nonfinite log importance weight")]
    NonFiniteLogWeight {
        /// Index in the supplied event population.
        event: usize,
    },
    /// A count or index does not fit the calculation's exact representation.
    #[error("{quantity} cannot be represented exactly for this calculation")]
    Unrepresentable {
        /// Name of the unrepresentable quantity.
        quantity: &'static str,
    },
    /// The requested numerical workspace cannot be reserved.
    #[error("unable to reserve storage for {quantity}")]
    Capacity {
        /// Name of the workspace allocation.
        quantity: &'static str,
    },
    /// A calculation produced an invalid result instead of usable evidence.
    #[error("nonfinite or invalid numerical result: {quantity}")]
    Numerical {
        /// Name of the failed intermediate or result.
        quantity: &'static str,
    },
    /// The validated registered contract could not be serialized for hashing.
    #[error("admission contract serialization failed: {0}")]
    Serialization(String),
}

impl AdmissionContract {
    /// Check all registered domains without filling settings or admitting a candidate.
    pub fn validate(&self) -> Result<(), StatisticsError> {
        if self.split_salt.is_empty() {
            return Err(invalid("split_salt", "must be nonempty"));
        }
        unit_interval("heldout_fraction", self.heldout_fraction)?;
        unit_interval("lower_quantile", self.lower_quantile)?;
        if !self.parent_logit_tolerance.is_finite() || self.parent_logit_tolerance < 0.0 {
            return Err(invalid(
                "parent_logit_tolerance",
                "must be finite and nonnegative",
            ));
        }
        if !self.regression_limit.is_finite() {
            return Err(invalid("regression_limit", "must be finite"));
        }
        if self.bootstrap_replicates == 0 {
            return Err(invalid("bootstrap_replicates", "must be positive"));
        }
        exact_count(self.bootstrap_replicates, "bootstrap_replicates")?;
        if self.min_heldout_events == 0 {
            return Err(invalid("min_heldout_events", "must be positive"));
        }
        if !self.min_effective_sample_size.is_finite() || self.min_effective_sample_size <= 0.0 {
            return Err(invalid(
                "min_effective_sample_size",
                "must be finite and positive",
            ));
        }
        if self.min_reference_tokens == 0 {
            return Err(invalid("min_reference_tokens", "must be positive"));
        }
        Ok(())
    }

    /// Versioned SHA-256 of the validated struct's field-ordered JSON encoding.
    pub fn hash(&self) -> Result<String, StatisticsError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| StatisticsError::Serialization(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(CONTRACT_DOMAIN);
        hasher.update(encoded);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// SHA-256 over a domain and u64-big-endian-length-framed salt/identity.
    ///
    /// The first 53 digest bits form an exact integer `h`; held-out means
    /// `h / 2^53 < heldout_fraction`. This fixed-point split has resolution 2^-53.
    /// It is independent of event order, rewards, and bootstrap seed.
    pub fn split(&self, completion_id: &str) -> Result<Population, StatisticsError> {
        self.validate()?;
        if completion_id.is_empty() {
            return Err(StatisticsError::EmptyCompletionIdentity);
        }
        let mut hasher = Sha256::new();
        hasher.update(SPLIT_DOMAIN);
        hash_framed(&mut hasher, self.split_salt.as_bytes())?;
        hash_framed(&mut hasher, completion_id.as_bytes())?;
        let digest = hasher.finalize();
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&digest[..8]);
        let bucket = u64::from_be_bytes(prefix) >> 11;
        let fraction = bucket as f64 / EXACT_INTEGER_LIMIT as f64;
        Ok(if fraction < self.heldout_fraction {
            Population::HeldOut
        } else {
            Population::Training
        })
    }
}

/// Paired bootstrap of self-normalized importance reward minus plain reward.
///
/// Each replicate draws `events.len()` indices with replacement, using SplitMix64
/// from the registered seed and rejection sampling to remove modulo bias. Both
/// means use that same index sequence. Each replicate subtracts its own maximum
/// log weight before exponentiation; a resample containing only globally tiny
/// weights therefore still has a positive normalizer.
///
/// The bound is the empirical inverse CDF: sort the replicate differences and
/// select the one-based rank `ceil(lower_quantile * bootstrap_replicates)`, with
/// no interpolation. IEEE-754 f64 arithmetic defines that rank calculation.
/// Input order is significant for the seeded resampling and must be stable at
/// the caller. No evidence floor or pass rule is applied by this function.
pub(crate) fn evaluate(
    events: &[WeightedEvent],
    contract: &AdmissionContract,
) -> Result<ImportanceEstimate, StatisticsError> {
    contract.validate()?;
    let full = moments(events, 0..events.len())?;
    let bound = u64::try_from(events.len()).map_err(|_| StatisticsError::Unrepresentable {
        quantity: "held-out count",
    })?;
    let mut sampled = reserved_vec(events.len(), "bootstrap indices")?;
    sampled.resize(events.len(), 0);
    let mut differences = reserved_vec(contract.bootstrap_replicates, "bootstrap differences")?;
    let mut rng = SplitMix64(contract.bootstrap_seed);
    for _ in 0..contract.bootstrap_replicates {
        for index in &mut sampled {
            *index = rng.index(bound)?;
        }
        let replicate = moments(events, sampled.iter().copied())?;
        differences.push(replicate.delta);
    }
    let lower_bound = empirical_quantile(&mut differences, contract.lower_quantile)?;
    Ok(ImportanceEstimate {
        count: events.len(),
        candidate_mean: full.candidate_mean,
        parent_mean: full.parent_mean,
        delta: full.delta,
        effective_sample_size: full.effective_sample_size,
        lower_bound,
    })
}

fn invalid(field: &'static str, reason: &'static str) -> StatisticsError {
    StatisticsError::InvalidSetting { field, reason }
}

fn unit_interval(field: &'static str, value: f64) -> Result<(), StatisticsError> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(invalid(
            field,
            "must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn hash_framed(hasher: &mut Sha256, value: &[u8]) -> Result<(), StatisticsError> {
    let length = u64::try_from(value.len()).map_err(|_| StatisticsError::Unrepresentable {
        quantity: "split input length",
    })?;
    hasher.update(length.to_be_bytes());
    hasher.update(value);
    Ok(())
}

fn exact_count(count: usize, quantity: &'static str) -> Result<f64, StatisticsError> {
    let count = u64::try_from(count).map_err(|_| StatisticsError::Unrepresentable { quantity })?;
    if count > EXACT_INTEGER_LIMIT {
        return Err(StatisticsError::Unrepresentable { quantity });
    }
    Ok(count as f64)
}

fn reserved_vec<T>(length: usize, quantity: &'static str) -> Result<Vec<T>, StatisticsError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| StatisticsError::Capacity { quantity })?;
    Ok(values)
}

#[derive(Debug)]
struct Moments {
    candidate_mean: f64,
    parent_mean: f64,
    delta: f64,
    effective_sample_size: f64,
}

fn moments(
    events: &[WeightedEvent],
    indices: impl Iterator<Item = usize> + Clone,
) -> Result<Moments, StatisticsError> {
    let mut max_log_weight = f64::NEG_INFINITY;
    let mut count = 0_usize;
    let mut helpful = 0_usize;
    for index in indices.clone() {
        let event = event_at(events, index)?;
        if !event.log_weight.is_finite() {
            return Err(StatisticsError::NonFiniteLogWeight { event: index });
        }
        max_log_weight = max_log_weight.max(event.log_weight);
        count = count
            .checked_add(1)
            .ok_or(StatisticsError::Unrepresentable {
                quantity: "resample count",
            })?;
        if event.reward {
            helpful += 1;
        }
    }
    if count == 0 {
        return Err(StatisticsError::EmptyPopulation);
    }
    let count = exact_count(count, "resample count")?;
    let mut weight_sum = Sum::new();
    let mut square_sum = Sum::new();
    let mut reward_sum = Sum::new();
    for index in indices {
        let event = event_at(events, index)?;
        // Finite extremes may subtract to -infinity, whose exponential is the
        // correctly underflowed zero. A maximum element always contributes one.
        let weight = (event.log_weight - max_log_weight).exp();
        weight_sum.add(weight);
        square_sum.add(weight * weight);
        if event.reward {
            reward_sum.add(weight);
        }
    }
    let sum = positive_finite(weight_sum.value, "importance normalizer")?;
    let squared = positive_finite(square_sum.value, "squared importance normalizer")?;
    let candidate_mean = finite(reward_sum.value / sum, "candidate mean")?;
    let parent_mean = finite(
        exact_count(helpful, "helpful count")? / count,
        "parent mean",
    )?;
    let delta = finite(candidate_mean - parent_mean, "paired difference")?;
    let effective_sample_size = positive_finite((sum / squared) * sum, "effective sample size")?;
    Ok(Moments {
        candidate_mean,
        parent_mean,
        delta,
        effective_sample_size,
    })
}

fn event_at(events: &[WeightedEvent], index: usize) -> Result<&WeightedEvent, StatisticsError> {
    events.get(index).ok_or(StatisticsError::Numerical {
        quantity: "bootstrap index",
    })
}

fn finite(value: f64, quantity: &'static str) -> Result<f64, StatisticsError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StatisticsError::Numerical { quantity })
    }
}

fn positive_finite(value: f64, quantity: &'static str) -> Result<f64, StatisticsError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(StatisticsError::Numerical { quantity })
    }
}

fn empirical_quantile(values: &mut [f64], quantile: f64) -> Result<f64, StatisticsError> {
    unit_interval("lower_quantile", quantile)?;
    if values.is_empty() {
        return Err(StatisticsError::EmptyPopulation);
    }
    for &value in values.iter() {
        finite(value, "bootstrap difference")?;
    }
    let count = exact_count(values.len(), "bootstrap_replicates")?;
    let rank = (quantile * count).ceil();
    if !rank.is_finite() || rank < 1.0 || rank > count {
        return Err(StatisticsError::Numerical {
            quantity: "bootstrap quantile rank",
        });
    }
    let rank = usize::try_from(rank as u64).map_err(|_| StatisticsError::Unrepresentable {
        quantity: "bootstrap quantile rank",
    })?;
    let index = rank.checked_sub(1).ok_or(StatisticsError::Numerical {
        quantity: "bootstrap quantile rank",
    })?;
    values.sort_unstable_by(f64::total_cmp);
    values
        .get(index)
        .copied()
        .ok_or(StatisticsError::Numerical {
            quantity: "bootstrap quantile rank",
        })
}

struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    fn new() -> Self {
        Self {
            value: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, term: f64) {
        let corrected = term - self.correction;
        let next = self.value + corrected;
        self.correction = (next - self.value) - corrected;
        self.value = next;
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, bound: u64) -> Result<usize, StatisticsError> {
        if bound == 0 {
            return Err(StatisticsError::EmptyPopulation);
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return usize::try_from(value % bound).map_err(|_| {
                    StatisticsError::Unrepresentable {
                        quantity: "bootstrap index",
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
}
