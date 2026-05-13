//! Migration controller: state machine executor.

use std::time::Instant;

use super::types::{MigrationError, MigrationPlan, MigrationProgress, MigrationState, SkipReason};

/// Manages the state machine for a single migration.
///
/// # Example
///
/// ```rust
/// use lattice_embed::migration::{MigrationController, MigrationPlan};
/// use lattice_embed::EmbeddingModel;
///
/// let plan = MigrationPlan {
///     id: "mig-001".to_string(),
///     source_model: EmbeddingModel::BgeSmallEnV15,
///     target_model: EmbeddingModel::BgeBaseEnV15,
///     total_embeddings: 100,
///     batch_size: 50,
///     created_at: "2026-01-27T00:00:00Z".to_string(),
/// };
///
/// let mut ctrl = MigrationController::new(plan);
/// ctrl.start().unwrap();
/// ctrl.record_progress(50).unwrap();
///
/// let report = ctrl.progress();
/// assert!(report.state.is_active());
/// assert_eq!(report.state.processed(), 50);
/// ```
#[derive(Debug)]
pub struct MigrationController {
    pub(super) plan: MigrationPlan,
    pub(super) state: MigrationState,
    started_at: Option<Instant>,
    error_count: usize,
    skip_reasons: Vec<SkipReason>,
}

impl MigrationController {
    /// Create a new migration controller from a plan.
    pub fn new(plan: MigrationPlan) -> Self {
        Self {
            plan,
            state: MigrationState::Planned,
            started_at: None,
            error_count: 0,
            skip_reasons: Vec::new(),
        }
    }

    /// Start the migration (`Planned` -> `InProgress`).
    pub fn start(&mut self) -> Result<(), MigrationError> {
        match &self.state {
            MigrationState::Planned => {
                self.state = MigrationState::InProgress {
                    processed: 0,
                    total: self.plan.total_embeddings,
                    skipped: 0,
                };
                self.started_at = Some(Instant::now());
                Ok(())
            }
            other => Err(MigrationError::InvalidTransition {
                from: format!("{other:?}"),
                to: "InProgress".to_string(),
            }),
        }
    }

    /// Record that `newly_processed` embeddings were completed.
    pub fn record_progress(&mut self, newly_processed: usize) -> Result<(), MigrationError> {
        match &self.state {
            MigrationState::InProgress {
                processed,
                total,
                skipped,
            } => {
                let new_processed = processed + newly_processed;
                let effective_total = total.saturating_sub(*skipped);
                if new_processed >= effective_total {
                    let duration = self
                        .started_at
                        .map(|s| s.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    self.state = MigrationState::Completed {
                        processed: new_processed,
                        skipped: *skipped,
                        duration_secs: duration,
                    };
                } else {
                    self.state = MigrationState::InProgress {
                        processed: new_processed,
                        total: *total,
                        skipped: *skipped,
                    };
                }
                Ok(())
            }
            other => Err(MigrationError::InvalidTransition {
                from: format!("{other:?}"),
                to: "InProgress (progress)".to_string(),
            }),
        }
    }

    /// Record a non-fatal error during processing.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Record an item that will be permanently skipped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattice_embed::migration::{MigrationController, MigrationPlan, SkipReason};
    /// use lattice_embed::EmbeddingModel;
    ///
    /// let plan = MigrationPlan {
    ///     id: "mig-001".to_string(),
    ///     source_model: EmbeddingModel::BgeSmallEnV15,
    ///     target_model: EmbeddingModel::BgeBaseEnV15,
    ///     total_embeddings: 100,
    ///     batch_size: 50,
    ///     created_at: "2026-01-27T00:00:00Z".to_string(),
    /// };
    ///
    /// let mut ctrl = MigrationController::new(plan);
    /// ctrl.start().unwrap();
    /// ctrl.record_skip(SkipReason::ContentTooLarge { size: 50000, max: 8192 }).unwrap();
    /// assert_eq!(ctrl.state().skipped(), 1);
    /// ```
    pub fn record_skip(&mut self, reason: SkipReason) -> Result<(), MigrationError> {
        match &self.state {
            MigrationState::InProgress {
                processed,
                total,
                skipped,
            } => {
                self.skip_reasons.push(reason);
                self.state = MigrationState::InProgress {
                    processed: *processed,
                    total: *total,
                    skipped: skipped + 1,
                };
                Ok(())
            }
            other => Err(MigrationError::InvalidTransition {
                from: format!("{other:?}"),
                to: "InProgress (skip)".to_string(),
            }),
        }
    }

    /// Returns the list of reasons why entries were skipped during migration.
    #[inline]
    pub fn skip_reasons(&self) -> &[SkipReason] {
        &self.skip_reasons
    }

    /// Returns the effective coverage fraction (0.0–1.0) of the migration.
    pub fn effective_coverage(&self) -> f64 {
        self.state.effective_coverage()
    }

    /// Pause the migration (`InProgress` -> `Paused`).
    pub fn pause(&mut self, reason: impl Into<String>) -> Result<(), MigrationError> {
        match &self.state {
            MigrationState::InProgress {
                processed,
                total,
                skipped,
            } => {
                self.state = MigrationState::Paused {
                    processed: *processed,
                    total: *total,
                    skipped: *skipped,
                    reason: reason.into(),
                };
                Ok(())
            }
            other => Err(MigrationError::InvalidTransition {
                from: format!("{other:?}"),
                to: "Paused".to_string(),
            }),
        }
    }

    /// Resume the migration (`Paused`/`Failed` -> `InProgress`).
    pub fn resume(&mut self) -> Result<(), MigrationError> {
        match &self.state {
            MigrationState::Paused {
                processed,
                total,
                skipped,
                ..
            }
            | MigrationState::Failed {
                processed,
                total,
                skipped,
                ..
            } => {
                self.state = MigrationState::InProgress {
                    processed: *processed,
                    total: *total,
                    skipped: *skipped,
                };
                if self.started_at.is_none() {
                    self.started_at = Some(Instant::now());
                }
                Ok(())
            }
            other => Err(MigrationError::InvalidTransition {
                from: format!("{other:?}"),
                to: "InProgress (resume)".to_string(),
            }),
        }
    }

    /// Fail the migration (`InProgress` -> `Failed`).
    pub fn fail(&mut self, error: impl Into<String>) -> Result<(), MigrationError> {
        match &self.state {
            MigrationState::InProgress {
                processed,
                total,
                skipped,
            } => {
                self.state = MigrationState::Failed {
                    processed: *processed,
                    total: *total,
                    skipped: *skipped,
                    error: error.into(),
                };
                Ok(())
            }
            other => Err(MigrationError::InvalidTransition {
                from: format!("{other:?}"),
                to: "Failed".to_string(),
            }),
        }
    }

    /// Cancel the migration (any non-terminal state -> `Cancelled`).
    pub fn cancel(&mut self) -> Result<(), MigrationError> {
        if self.state.is_terminal() {
            return Err(MigrationError::InvalidTransition {
                from: format!("{:?}", self.state),
                to: "Cancelled".to_string(),
            });
        }
        let (processed, total, skipped) = match &self.state {
            MigrationState::Planned => (0, self.plan.total_embeddings, 0),
            MigrationState::InProgress {
                processed,
                total,
                skipped,
            } => (*processed, *total, *skipped),
            MigrationState::Paused {
                processed,
                total,
                skipped,
                ..
            } => (*processed, *total, *skipped),
            MigrationState::Failed {
                processed,
                total,
                skipped,
                ..
            } => (*processed, *total, *skipped),
            _ => unreachable!(),
        };
        self.state = MigrationState::Cancelled {
            processed,
            total,
            skipped,
        };
        Ok(())
    }

    /// Get a snapshot of current progress.
    pub fn progress(&self) -> MigrationProgress {
        let throughput = match (&self.state, self.started_at) {
            (MigrationState::InProgress { processed, .. }, Some(start)) => {
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    *processed as f64 / elapsed
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let eta_secs = match &self.state {
            MigrationState::InProgress {
                processed,
                total,
                skipped,
            } if throughput > 0.0 => {
                let effective_total = total.saturating_sub(*skipped);
                let remaining = effective_total.saturating_sub(*processed);
                Some(remaining as f64 / throughput)
            }
            _ => None,
        };

        MigrationProgress {
            migration_id: self.plan.id.clone(),
            state: self.state.clone(),
            skipped: self.state.skipped(),
            effective_total: self.state.effective_total(),
            effective_coverage: self.state.effective_coverage(),
            throughput,
            eta_secs,
            error_count: self.error_count,
        }
    }

    /// Returns the current migration state.
    #[inline]
    pub fn state(&self) -> &MigrationState {
        &self.state
    }

    /// Returns the migration plan.
    #[inline]
    pub fn plan(&self) -> &MigrationPlan {
        &self.plan
    }
}
