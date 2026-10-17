//! Candidate-versus-production comparison for deployment decisions.
//!
//! Callers select traffic, run both models, and record observations. A session
//! transitions only after its sample minimum; it never promotes a candidate.
//!
//! See `docs/registry.md` for shadow configuration, state transitions, and metrics.

use chrono::{DateTime, Utc};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of comparing shadow model outputs to production.
///
/// Contains aggregate statistics from the shadow evaluation period.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ShadowComparison {
    /// Agreement rate between shadow and production outputs [0.0, 1.0]
    pub agreement_rate: f64,
    /// Number of samples compared
    pub sample_count: usize,
    /// Average latency difference (shadow - production) in milliseconds.
    /// Positive means shadow is slower.
    pub latency_diff_ms: f64,
}

/// Configuration for shadow evaluation.
///
/// Defines sampling and acceptance criteria for evaluating a candidate.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ShadowConfig {
    /// Fraction of traffic to sample for shadow evaluation [0.0, 1.0]
    pub sample_rate: f64,
    /// Minimum samples required before evaluation can complete
    pub min_samples: usize,
    /// Minimum agreement rate to pass [0.0, 1.0]
    pub min_agreement: f64,
    /// Maximum acceptable latency increase in milliseconds
    pub max_latency_increase_ms: f64,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            sample_rate: 0.1,              // 10% of traffic
            min_samples: 1000,             // At least 1000 samples
            min_agreement: 0.95,           // 95% agreement
            max_latency_increase_ms: 50.0, // Max 50ms slower
        }
    }
}

impl ShadowConfig {
    /// Create a config for quick validation (smaller sample size).
    pub fn quick() -> Self {
        Self {
            sample_rate: 0.2,
            min_samples: 100,
            min_agreement: 0.90,
            max_latency_increase_ms: 100.0,
        }
    }

    /// Create a strict config for production-critical models.
    pub fn strict() -> Self {
        Self {
            sample_rate: 0.1,
            min_samples: 10000,
            min_agreement: 0.99,
            max_latency_increase_ms: 20.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.sample_rate) {
            return Err(format!(
                "sample_rate must be in [0.0, 1.0], got {}",
                self.sample_rate
            ));
        }
        if self.min_samples == 0 {
            return Err("min_samples must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.min_agreement) {
            return Err(format!(
                "min_agreement must be in [0.0, 1.0], got {}",
                self.min_agreement
            ));
        }
        if self.max_latency_increase_ms < 0.0 {
            return Err(format!(
                "max_latency_increase_ms must be >= 0, got {}",
                self.max_latency_increase_ms
            ));
        }
        Ok(())
    }
}

/// State of a shadow evaluation session.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum ShadowState {
    /// Shadow evaluation is in progress, collecting samples.
    Running {
        /// When the session started
        started_at: DateTime<Utc>,
        /// Number of samples collected so far
        samples_collected: usize,
    },
    /// Shadow evaluation passed all criteria.
    Passed {
        /// Final comparison statistics
        comparison: ShadowComparison,
        /// When evaluation completed
        completed_at: DateTime<Utc>,
    },
    /// Shadow evaluation failed one or more criteria.
    Failed {
        /// Final comparison statistics
        comparison: ShadowComparison,
        /// Human-readable failure reason
        reason: String,
        /// When evaluation completed
        completed_at: DateTime<Utc>,
    },
    /// Shadow evaluation was manually cancelled.
    Cancelled {
        /// Reason for cancellation
        reason: String,
        /// When cancellation occurred
        cancelled_at: DateTime<Utc>,
    },
}

/// A shadow evaluation session for a candidate model.
///
/// Tracks comparison samples between production and candidate models,
/// then evaluates whether the candidate meets promotion criteria.
#[derive(Debug)]
pub struct ShadowSession {
    /// Unique session ID
    pub id: Uuid,
    /// The production model being compared against
    pub production_model_id: Uuid,
    /// The candidate model being evaluated
    pub candidate_model_id: Uuid,
    /// Configuration for this session
    pub config: ShadowConfig,
    /// Current state of the session
    state: ShadowState,
    /// Collected samples: (agreed: bool, latency_diff_ms: f64)
    samples: Vec<(bool, f64)>,
}

impl ShadowSession {
    /// Creates a running session for a production/candidate pair and its configuration.
    /// See [`docs/registry.md`](../../docs/registry.md#shadowsession-public-methods) for sampling and transition rules.
    pub fn new(production_model_id: Uuid, candidate_model_id: Uuid, config: ShadowConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            production_model_id,
            candidate_model_id,
            config,
            state: ShadowState::Running {
                started_at: Utc::now(),
                samples_collected: 0,
            },
            samples: Vec::new(),
        }
    }

    /// Create a session with a specific ID (for testing or persistence).
    pub fn with_id(
        id: Uuid,
        production_model_id: Uuid,
        candidate_model_id: Uuid,
        config: ShadowConfig,
    ) -> Self {
        let mut session = Self::new(production_model_id, candidate_model_id, config);
        session.id = id;
        session
    }

    /// Records a comparison while the session is running.
    /// `latency_diff_ms` is candidate minus production latency; positive means slower.
    /// See [`docs/registry.md`](../../docs/registry.md#shadowsession-public-methods) for sampling rules.
    pub fn record_sample(&mut self, agreed: bool, latency_diff_ms: f64) {
        if let ShadowState::Running {
            samples_collected, ..
        } = &mut self.state
        {
            self.samples.push((agreed, latency_diff_ms));
            *samples_collected = self.samples.len();
        }
    }

    /// Evaluates a running session after its minimum sample count and returns its state.
    /// Both agreement and latency thresholds must pass for [`ShadowState::Passed`].
    /// See [`docs/registry.md`](../../docs/registry.md#shadowsession-public-methods) for transition rules.
    pub fn evaluate(&mut self) -> &ShadowState {
        if let ShadowState::Running { .. } = &self.state
            && self.samples.len() >= self.config.min_samples
        {
            let comparison = self.compute_comparison();

            if comparison.agreement_rate >= self.config.min_agreement
                && comparison.latency_diff_ms <= self.config.max_latency_increase_ms
            {
                self.state = ShadowState::Passed {
                    comparison,
                    completed_at: Utc::now(),
                };
            } else {
                let reason = self.failure_reason(&comparison);
                self.state = ShadowState::Failed {
                    comparison,
                    reason,
                    completed_at: Utc::now(),
                };
            }
        }
        &self.state
    }

    /// Cancel the shadow evaluation session.
    ///
    /// # Arguments
    ///
    /// * `reason` - Human-readable reason for cancellation
    pub fn cancel(&mut self, reason: impl Into<String>) {
        self.state = ShadowState::Cancelled {
            reason: reason.into(),
            cancelled_at: Utc::now(),
        };
    }

    /// Get the current session state.
    pub fn state(&self) -> &ShadowState {
        &self.state
    }

    /// Get the number of samples collected.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Check if the session passed evaluation.
    pub fn passed(&self) -> bool {
        matches!(self.state, ShadowState::Passed { .. })
    }

    /// Check if the session failed evaluation.
    pub fn failed(&self) -> bool {
        matches!(self.state, ShadowState::Failed { .. })
    }

    /// Check if the session is still running.
    pub fn is_running(&self) -> bool {
        matches!(self.state, ShadowState::Running { .. })
    }

    /// Check if the session is complete (passed, failed, or cancelled).
    pub fn is_complete(&self) -> bool {
        !self.is_running()
    }

    /// Get the current comparison statistics.
    ///
    /// Can be called at any time to see progress, even before evaluation.
    pub fn current_comparison(&self) -> ShadowComparison {
        self.compute_comparison()
    }

    /// Get progress towards minimum samples as a fraction [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        (self.samples.len() as f64 / self.config.min_samples as f64).min(1.0)
    }

    fn compute_comparison(&self) -> ShadowComparison {
        if self.samples.is_empty() {
            return ShadowComparison {
                agreement_rate: 0.0,
                sample_count: 0,
                latency_diff_ms: 0.0,
            };
        }

        let agreed_count = self.samples.iter().filter(|(a, _)| *a).count();
        let total_latency: f64 = self.samples.iter().map(|(_, l)| *l).sum();

        ShadowComparison {
            agreement_rate: agreed_count as f64 / self.samples.len() as f64,
            sample_count: self.samples.len(),
            latency_diff_ms: total_latency / self.samples.len() as f64,
        }
    }

    fn failure_reason(&self, comparison: &ShadowComparison) -> String {
        let mut reasons = Vec::new();

        if comparison.agreement_rate < self.config.min_agreement {
            reasons.push(format!(
                "Agreement rate {:.2}% below threshold {:.2}%",
                comparison.agreement_rate * 100.0,
                self.config.min_agreement * 100.0
            ));
        }

        if comparison.latency_diff_ms > self.config.max_latency_increase_ms {
            reasons.push(format!(
                "Latency increase {:.1}ms exceeds threshold {:.1}ms",
                comparison.latency_diff_ms, self.config.max_latency_increase_ms
            ));
        }

        if reasons.is_empty() {
            "Unknown failure".to_string()
        } else {
            reasons.join("; ")
        }
    }
}

#[cfg(test)]
mod tests;
