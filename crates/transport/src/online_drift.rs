//! Streaming online drift detection via Sinkhorn divergence.
//!
//! Implements `OnlineDriftDetector` as a sample-count sliding-window drift sensor
//! that calls `point_set_sinkhorn_divergence` every `check_interval` observations.
//! The first `window_size` samples populate a frozen reference distribution; all
//! subsequent samples slide through `current_window`.
//!
//! See ADR-055 for design rationale and crate-boundary rules.

use std::collections::VecDeque;

use crate::{
    SinkhornConfig, SinkhornError, SinkhornSolver, SinkhornWorkspace, SquaredEuclidean,
    point_set_sinkhorn_divergence, uniform_weights,
};

/// Minimum `window_size` enforced by `OnlineDriftConfig::normalized`.
pub const MIN_ONLINE_DRIFT_WINDOW_SIZE: usize = 128;

/// Maximum `window_size` enforced by `OnlineDriftConfig::normalized`.
///
/// Each drift check builds three `W×W` Sinkhorn workspaces, so memory scales as
/// O(W²). This cap bounds one detector's workspace footprint to roughly
/// 3 × 4096² × 4 bytes ≈ 192 MB, preventing a deserialized config with an
/// oversized window from exhausting memory.
pub const MAX_ONLINE_DRIFT_WINDOW_SIZE: usize = 4096;

/// Configuration for streaming Sinkhorn drift detection.
///
/// **Stable** (provisional): aggregate config for the online drift API; new fields would be additive with `Default`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnlineDriftConfig {
    /// Sliding window size W. Clamped to `[128, 4096]` by `normalized` (O(W²) cost per check).
    pub window_size: usize,
    /// Compute divergence every N observed samples once both windows are full.
    pub check_interval: usize,
    /// Emit `Drift` when S(reference, current) > threshold.
    pub threshold: f32,
    /// Sinkhorn entropy regularization strength.
    pub epsilon: f32,
}

impl Default for OnlineDriftConfig {
    fn default() -> Self {
        let sinkhorn = SinkhornConfig::default();
        Self {
            window_size: MIN_ONLINE_DRIFT_WINDOW_SIZE,
            check_interval: 16,
            threshold: 0.05,
            epsilon: sinkhorn.epsilon,
        }
    }
}

impl OnlineDriftConfig {
    fn normalized(mut self) -> Self {
        self.window_size = self
            .window_size
            .clamp(MIN_ONLINE_DRIFT_WINDOW_SIZE, MAX_ONLINE_DRIFT_WINDOW_SIZE);
        self.check_interval = self.check_interval.max(1);
        self
    }
}

/// Per-sample online drift status returned by [`OnlineDriftDetector::observe`].
///
/// **Stable** (provisional): four-variant status enum; new variants would be additive.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OnlineDriftSignal {
    /// Not enough samples have been observed to fill both windows.
    Warming {
        /// Total samples observed so far.
        samples_seen: usize,
        /// Samples required before divergence can be computed.
        window_size: usize,
    },
    /// Both windows are full but this sample did not land on `check_interval`.
    Skipped {
        /// Total samples observed so far.
        samples_seen: usize,
        /// Remaining samples until the next divergence check.
        next_check_in: usize,
    },
    /// Divergence was computed and did not exceed the threshold.
    Stable {
        /// S(reference, current) at this check.
        divergence: f32,
        /// Total samples observed when this check ran.
        window_pos: usize,
    },
    /// Divergence was computed and exceeded the threshold.
    Drift {
        /// S(reference, current) at detection.
        divergence: f32,
        /// Total samples observed when this signal fired.
        window_pos: usize,
    },
}

/// Streaming Sinkhorn drift detector.
///
/// The first `window_size` samples auto-fit a frozen reference distribution.
/// Subsequent samples slide through `current_window`; every `check_interval`
/// observations `observe` computes S(reference, current) and emits a signal.
///
/// **Stable** (provisional): `new` + `observe` + `reset_reference` are the stable surface.
#[derive(Debug, Clone)]
pub struct OnlineDriftDetector {
    config: OnlineDriftConfig,
    reference_window: VecDeque<Vec<f32>>,
    current_window: VecDeque<Vec<f32>>,
    samples_seen: usize,
    solver: SinkhornSolver,
    weights: Vec<f32>,
    workspace_xy: SinkhornWorkspace,
    workspace_xx: SinkhornWorkspace,
    workspace_yy: SinkhornWorkspace,
    last_divergence: Option<f32>,
}

impl OnlineDriftDetector {
    /// Create a detector whose first `window_size` observed samples become the frozen reference.
    pub fn new(config: OnlineDriftConfig) -> Self {
        let config = config.normalized();
        let sinkhorn = SinkhornConfig {
            epsilon: config.epsilon,
            ..SinkhornConfig::default()
        };
        let w = config.window_size;
        Self {
            reference_window: VecDeque::with_capacity(w),
            current_window: VecDeque::with_capacity(w),
            samples_seen: 0,
            solver: SinkhornSolver::new(sinkhorn),
            weights: uniform_weights(w),
            workspace_xy: SinkhornWorkspace::new(w, w),
            workspace_xx: SinkhornWorkspace::new(w, w),
            workspace_yy: SinkhornWorkspace::new(w, w),
            last_divergence: None,
            config,
        }
    }

    /// Observe one embedding sample and return the current drift status.
    ///
    /// Returns `Err(SinkhornError)` only on solver failure; status signals are
    /// encoded in the `Ok` variants.
    pub fn observe(&mut self, embedding: Vec<f32>) -> Result<OnlineDriftSignal, SinkhornError> {
        self.samples_seen += 1;

        if self.reference_window.len() < self.config.window_size {
            self.reference_window.push_back(embedding.clone());
        }

        if self.current_window.len() == self.config.window_size {
            self.current_window.pop_front();
        }
        self.current_window.push_back(embedding);

        if self.reference_window.len() < self.config.window_size
            || self.current_window.len() < self.config.window_size
        {
            return Ok(OnlineDriftSignal::Warming {
                samples_seen: self.samples_seen,
                window_size: self.config.window_size,
            });
        }

        if !self.samples_seen.is_multiple_of(self.config.check_interval) {
            return Ok(OnlineDriftSignal::Skipped {
                samples_seen: self.samples_seen,
                next_check_in: self.next_check_in(),
            });
        }

        let reference_points: Vec<&[f32]> =
            self.reference_window.iter().map(Vec::as_slice).collect();
        let current_points: Vec<&[f32]> = self.current_window.iter().map(Vec::as_slice).collect();

        let divergence = point_set_sinkhorn_divergence(
            &self.solver,
            reference_points.as_slice(),
            current_points.as_slice(),
            &self.weights,
            &self.weights,
            SquaredEuclidean,
            &mut self.workspace_xy,
            &mut self.workspace_xx,
            &mut self.workspace_yy,
        )?;

        // Clamp tiny negative floating-point noise; debiased divergence is non-negative by definition.
        let value = divergence.value.max(0.0);
        self.last_divergence = Some(value);

        if value > self.config.threshold {
            Ok(OnlineDriftSignal::Drift {
                divergence: value,
                window_pos: self.samples_seen,
            })
        } else {
            Ok(OnlineDriftSignal::Stable {
                divergence: value,
                window_pos: self.samples_seen,
            })
        }
    }

    /// Promote the current window to the frozen reference distribution.
    ///
    /// Call after an adapter or router refresh to reset the baseline. `samples_seen`
    /// is preserved so `window_pos` in future signals remains monotonically increasing.
    pub fn reset_reference(&mut self) {
        self.reference_window = self.current_window.clone();
        self.workspace_xy.reset();
        self.workspace_xx.reset();
        self.workspace_yy.reset();
        self.last_divergence = None;
    }

    /// Borrow the normalized detector config.
    pub fn config(&self) -> &OnlineDriftConfig {
        &self.config
    }

    /// Total samples observed since construction.
    pub fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    /// Last computed Sinkhorn divergence, if any check has run.
    pub fn last_divergence(&self) -> Option<f32> {
        self.last_divergence
    }

    fn next_check_in(&self) -> usize {
        let remainder = self.samples_seen % self.config.check_interval;
        if remainder == 0 {
            0
        } else {
            self.config.check_interval - remainder
        }
    }
}

#[cfg(test)]
mod tests;
