//! Shadow evaluation for safe model deployment.
//!
//! Runs candidate models in parallel with production to compare outputs
//! before promotion. This enables data-driven deployment decisions based
//! on real traffic rather than synthetic benchmarks.
//!
//! # Workflow
//!
//! 1. Create a [`ShadowSession`] with production and candidate model IDs
//! 2. For each sampled request, run both models and call [`record_sample`]
//! 3. Call [`evaluate`] periodically to check if criteria are met
//! 4. If [`passed`] returns true, promote the candidate to production
//!
//! # Example
//!
//! ```
//! use lattice_tune::registry::{ShadowSession, ShadowConfig, ShadowState};
//! use uuid::Uuid;
//!
//! let production_id = Uuid::new_v4();
//! let candidate_id = Uuid::new_v4();
//!
//! let config = ShadowConfig {
//!     sample_rate: 0.1,      // Sample 10% of traffic
//!     min_samples: 100,      // Need at least 100 samples
//!     min_agreement: 0.95,   // 95% agreement threshold
//!     max_latency_increase_ms: 50.0,
//! };
//!
//! let mut session = ShadowSession::new(production_id, candidate_id, config);
//!
//! // Record samples (in real usage, from actual inference)
//! for _ in 0..100 {
//!     session.record_sample(true, 5.0); // agreed, 5ms slower
//! }
//!
//! // Evaluate
//! let state = session.evaluate();
//! assert!(session.passed());
//! ```

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
/// Defines the sampling strategy and acceptance criteria for
/// promoting a candidate model.
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
    /// Create a new shadow evaluation session.
    ///
    /// # Arguments
    ///
    /// * `production_model_id` - ID of the current production model
    /// * `candidate_model_id` - ID of the candidate model to evaluate
    /// * `config` - Evaluation configuration
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

    /// Record a sample comparison.
    ///
    /// This should be called for each sampled request where both models
    /// were invoked. Only records if the session is still running.
    ///
    /// # Arguments
    ///
    /// * `agreed` - Whether the production and shadow outputs agreed
    /// * `latency_diff_ms` - Latency difference (shadow - production) in ms.
    ///   Positive means shadow was slower.
    pub fn record_sample(&mut self, agreed: bool, latency_diff_ms: f64) {
        if let ShadowState::Running {
            samples_collected, ..
        } = &mut self.state
        {
            self.samples.push((agreed, latency_diff_ms));
            *samples_collected = self.samples.len();
        }
    }

    /// Evaluate the session and update state if criteria are met.
    ///
    /// If the minimum sample count is reached, computes the comparison
    /// statistics and transitions to [`ShadowState::Passed`] or
    /// [`ShadowState::Failed`].
    ///
    /// If not enough samples yet, returns the current running state.
    ///
    /// # Returns
    ///
    /// The current (possibly updated) session state.
    pub fn evaluate(&mut self) -> &ShadowState {
        if let ShadowState::Running { .. } = &self.state {
            if self.samples.len() >= self.config.min_samples {
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
mod tests {
    use super::*;

    #[test]
    fn test_shadow_config_default() {
        let config = ShadowConfig::default();
        assert_eq!(config.sample_rate, 0.1);
        assert_eq!(config.min_samples, 1000);
        assert_eq!(config.min_agreement, 0.95);
        assert_eq!(config.max_latency_increase_ms, 50.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_shadow_config_quick() {
        let config = ShadowConfig::quick();
        assert_eq!(config.min_samples, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_shadow_config_strict() {
        let config = ShadowConfig::strict();
        assert_eq!(config.min_samples, 10000);
        assert_eq!(config.min_agreement, 0.99);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_shadow_config_validation() {
        let mut config = ShadowConfig::default();

        config.sample_rate = 1.5;
        assert!(config.validate().is_err());

        config.sample_rate = 0.1;
        config.min_samples = 0;
        assert!(config.validate().is_err());

        config.min_samples = 100;
        config.min_agreement = 1.5;
        assert!(config.validate().is_err());

        config.min_agreement = 0.95;
        config.max_latency_increase_ms = -10.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_shadow_session_creation() {
        let prod_id = Uuid::new_v4();
        let cand_id = Uuid::new_v4();
        let config = ShadowConfig::quick();

        let session = ShadowSession::new(prod_id, cand_id, config);

        assert_eq!(session.production_model_id, prod_id);
        assert_eq!(session.candidate_model_id, cand_id);
        assert!(session.is_running());
        assert!(!session.passed());
        assert!(!session.failed());
        assert_eq!(session.sample_count(), 0);
    }

    #[test]
    fn test_shadow_session_record_sample() {
        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

        session.record_sample(true, 5.0);
        session.record_sample(false, 10.0);
        session.record_sample(true, -2.0);

        assert_eq!(session.sample_count(), 3);

        let comparison = session.current_comparison();
        assert_eq!(comparison.sample_count, 3);
        // 2 out of 3 agreed
        assert!((comparison.agreement_rate - 2.0 / 3.0).abs() < 0.001);
        // Average latency: (5 + 10 + (-2)) / 3 = 13/3 = 4.333...
        assert!((comparison.latency_diff_ms - 13.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_shadow_session_evaluate_pass() {
        let config = ShadowConfig {
            sample_rate: 0.1,
            min_samples: 10,
            min_agreement: 0.90,
            max_latency_increase_ms: 50.0,
        };

        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

        // Add 10 samples, all agree, low latency
        for _ in 0..10 {
            session.record_sample(true, 5.0);
        }

        let state = session.evaluate();
        assert!(matches!(state, ShadowState::Passed { .. }));
        assert!(session.passed());
    }

    #[test]
    fn test_shadow_session_evaluate_fail_agreement() {
        let config = ShadowConfig {
            sample_rate: 0.1,
            min_samples: 10,
            min_agreement: 0.90,
            max_latency_increase_ms: 50.0,
        };

        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

        // Add 10 samples, only 5 agree (50% < 90%)
        for i in 0..10 {
            session.record_sample(i < 5, 5.0);
        }

        session.evaluate();
        assert!(session.failed());

        if let ShadowState::Failed { reason, .. } = session.state() {
            assert!(reason.contains("Agreement rate"));
        } else {
            panic!("Expected Failed state");
        }
    }

    #[test]
    fn test_shadow_session_evaluate_fail_latency() {
        let config = ShadowConfig {
            sample_rate: 0.1,
            min_samples: 10,
            min_agreement: 0.90,
            max_latency_increase_ms: 10.0,
        };

        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

        // Add 10 samples, all agree but high latency
        for _ in 0..10 {
            session.record_sample(true, 50.0); // 50ms > 10ms threshold
        }

        session.evaluate();
        assert!(session.failed());

        if let ShadowState::Failed { reason, .. } = session.state() {
            assert!(reason.contains("Latency increase"));
        } else {
            panic!("Expected Failed state");
        }
    }

    #[test]
    fn test_shadow_session_evaluate_not_enough_samples() {
        let config = ShadowConfig {
            sample_rate: 0.1,
            min_samples: 100,
            min_agreement: 0.90,
            max_latency_increase_ms: 50.0,
        };

        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

        // Add only 50 samples
        for _ in 0..50 {
            session.record_sample(true, 5.0);
        }

        session.evaluate();
        // Should still be running
        assert!(session.is_running());
        assert!(!session.passed());
        assert!(!session.failed());
    }

    #[test]
    fn test_shadow_session_cancel() {
        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

        session.record_sample(true, 5.0);
        session.cancel("Manual cancellation for testing");

        assert!(!session.is_running());
        assert!(session.is_complete());
        assert!(!session.passed());

        if let ShadowState::Cancelled { reason, .. } = session.state() {
            assert_eq!(reason, "Manual cancellation for testing");
        } else {
            panic!("Expected Cancelled state");
        }
    }

    #[test]
    fn test_shadow_session_no_samples_after_cancel() {
        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

        session.record_sample(true, 5.0);
        assert_eq!(session.sample_count(), 1);

        session.cancel("done");

        // Should not accept more samples
        session.record_sample(true, 5.0);
        assert_eq!(session.sample_count(), 1);
    }

    #[test]
    fn test_shadow_session_progress() {
        let config = ShadowConfig {
            sample_rate: 0.1,
            min_samples: 100,
            min_agreement: 0.90,
            max_latency_increase_ms: 50.0,
        };

        let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

        assert_eq!(session.progress(), 0.0);

        for _ in 0..50 {
            session.record_sample(true, 5.0);
        }
        assert!((session.progress() - 0.5).abs() < 0.001);

        for _ in 0..50 {
            session.record_sample(true, 5.0);
        }
        assert!((session.progress() - 1.0).abs() < 0.001);

        // Progress caps at 1.0
        for _ in 0..50 {
            session.record_sample(true, 5.0);
        }
        assert!((session.progress() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_shadow_comparison_empty() {
        let session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

        let comparison = session.current_comparison();
        assert_eq!(comparison.sample_count, 0);
        assert_eq!(comparison.agreement_rate, 0.0);
        assert_eq!(comparison.latency_diff_ms, 0.0);
    }

    #[test]
    fn test_shadow_session_with_id() {
        let session_id = Uuid::new_v4();
        let session = ShadowSession::with_id(
            session_id,
            Uuid::new_v4(),
            Uuid::new_v4(),
            ShadowConfig::quick(),
        );

        assert_eq!(session.id, session_id);
    }
}
