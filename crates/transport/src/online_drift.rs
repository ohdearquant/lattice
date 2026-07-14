//! Streaming, sliding-window drift detection via Sinkhorn divergence.
//!
//! Extended background: <https://github.com/ohdearquant/lattice/blob/main/crates/transport/docs/algorithms.md>.

use std::collections::VecDeque;

use crate::{
    SinkhornConfig, SinkhornError, SinkhornSolver, SinkhornWorkspace, SquaredEuclidean,
    point_set_sinkhorn_divergence, uniform_weights,
};

/// Minimum `window_size` enforced by `OnlineDriftConfig::normalized`.
pub const MIN_ONLINE_DRIFT_WINDOW_SIZE: usize = 128;

/// Maximum window size, bounding three O(W²) workspaces to about 192 MiB.
pub const MAX_ONLINE_DRIFT_WINDOW_SIZE: usize = 4096;

/// Configuration for streaming Sinkhorn drift detection.
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// Streaming Sinkhorn drift detector with frozen-reference and sliding windows.
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

    /// Observe one sample; solver failures are `Err` and statuses are `Ok`.
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

        // Clamp roundoff: the debiased divergence is non-negative.
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

    /// Promote the current window to the reference while preserving `samples_seen`.
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
mod tests {
    use super::*;

    #[test]
    fn window_size_above_max_is_clamped() {
        let config = OnlineDriftConfig {
            window_size: usize::MAX,
            ..Default::default()
        };
        assert_eq!(
            config.normalized().window_size,
            MAX_ONLINE_DRIFT_WINDOW_SIZE
        );
    }

    #[test]
    fn window_size_below_min_is_raised() {
        let config = OnlineDriftConfig {
            window_size: 1,
            ..Default::default()
        };
        assert_eq!(
            config.normalized().window_size,
            MIN_ONLINE_DRIFT_WINDOW_SIZE
        );
    }

    // Deterministic linear congruential generator so tests don't need `rand`.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 32) as f32 / u32::MAX as f32
        }

        // Box-Muller sample from approximately N(mean, 1).
        fn next_normal(&mut self, mean: f32) -> f32 {
            let u1 = self.next_f32().max(1e-10);
            let u2 = self.next_f32();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + z
        }
    }

    fn make_vector(rng: &mut Lcg, dim: usize, mean: f32) -> Vec<f32> {
        (0..dim).map(|_| rng.next_normal(mean)).collect()
    }

    #[test]
    fn online_drift_warms_until_window_full() {
        let config = OnlineDriftConfig {
            window_size: 128,
            check_interval: 1,
            threshold: 100.0, // never fires drift
            ..Default::default()
        };
        let mut detector = OnlineDriftDetector::new(config);
        let mut rng = Lcg::new(42);

        for i in 1..128 {
            let signal = detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap();
            assert!(
                matches!(
                    signal,
                    OnlineDriftSignal::Warming {
                        samples_seen,
                        window_size: 128
                    } if samples_seen == i
                ),
                "expected Warming at sample {i}, got {signal:?}"
            );
        }
        assert_eq!(detector.samples_seen(), 127);
    }

    #[test]
    fn online_drift_computes_when_window_fills() {
        let config = OnlineDriftConfig {
            window_size: 128,
            check_interval: 1,
            threshold: 100.0,
            ..Default::default()
        };
        let mut detector = OnlineDriftDetector::new(config);
        let mut rng = Lcg::new(99);

        let mut last_signal = None;
        for _ in 0..128 {
            last_signal = Some(detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap());
        }

        let signal = last_signal.unwrap();
        match signal {
            OnlineDriftSignal::Stable {
                divergence,
                window_pos: 128,
            } => {
                assert!(
                    divergence <= 1e-2,
                    "expected near-zero divergence for identical windows, got {divergence}"
                );
            }
            other => panic!("expected Stable at sample 128, got {other:?}"),
        }
        assert!(detector.last_divergence().is_some());
    }

    #[test]
    fn online_drift_skips_between_check_intervals() {
        let config = OnlineDriftConfig {
            window_size: 128,
            check_interval: 8,
            threshold: 100.0,
            ..Default::default()
        };
        let mut detector = OnlineDriftDetector::new(config);
        let mut rng = Lcg::new(7);

        for _ in 0..128 {
            detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap();
        }
        let at_128 = detector.observe(make_vector(&mut rng, 4, 0.0)).unwrap();
        let at_129 = at_128; // we already consumed it; redo:
        let mut detector2 = OnlineDriftDetector::new(OnlineDriftConfig {
            window_size: 128,
            check_interval: 8,
            threshold: 100.0,
            ..Default::default()
        });
        let mut rng2 = Lcg::new(7);
        for _ in 0..128 {
            detector2.observe(make_vector(&mut rng2, 4, 0.0)).unwrap();
        }
        let sig_129 = detector2.observe(make_vector(&mut rng2, 4, 0.0)).unwrap();
        assert!(
            matches!(
                sig_129,
                OnlineDriftSignal::Skipped {
                    samples_seen: 129,
                    next_check_in: 7
                }
            ),
            "expected Skipped at 129 with next_check_in=7, got {sig_129:?}"
        );
        let mut detector3 = OnlineDriftDetector::new(OnlineDriftConfig {
            window_size: 128,
            check_interval: 8,
            threshold: 100.0,
            ..Default::default()
        });
        let mut rng3 = Lcg::new(7);
        for _ in 0..127 {
            detector3.observe(make_vector(&mut rng3, 4, 0.0)).unwrap();
        }
        let sig_128 = detector3.observe(make_vector(&mut rng3, 4, 0.0)).unwrap();
        assert!(
            matches!(
                sig_128,
                OnlineDriftSignal::Stable { .. } | OnlineDriftSignal::Drift { .. }
            ),
            "expected compute at sample 128, got {sig_128:?}"
        );
        let _ = at_129; // silence unused warning
    }

    #[test]
    fn online_drift_detects_gaussian_shift() {
        let config = OnlineDriftConfig {
            window_size: 128,
            check_interval: 1,
            threshold: 0.1,
            ..Default::default()
        };
        let mut detector = OnlineDriftDetector::new(config);
        let mut rng = Lcg::new(2025);

        for _ in 0..128 {
            let sig = detector.observe(make_vector(&mut rng, 8, 0.0)).unwrap();
            let _ = sig;
        }

        let mut found_drift = false;
        for _ in 0..128 {
            let sig = detector.observe(make_vector(&mut rng, 8, 4.0)).unwrap();
            if let OnlineDriftSignal::Drift {
                divergence,
                window_pos,
            } = sig
            {
                assert!(divergence > 0.1, "drift divergence should exceed threshold");
                assert!(
                    window_pos > 128,
                    "window_pos must be after warm-up: got {window_pos}"
                );
                found_drift = true;
                break;
            }
        }
        assert!(
            found_drift,
            "expected at least one Drift signal after distribution shift"
        );
    }

    #[test]
    fn online_drift_zero_divergence_identical_distributions() {
        let config = OnlineDriftConfig {
            window_size: 128,
            check_interval: 1,
            threshold: 100.0,
            ..Default::default()
        };
        let mut detector = OnlineDriftDetector::new(config);
        let mut rng = Lcg::new(55);

        let vecs: Vec<Vec<f32>> = (0..128).map(|_| make_vector(&mut rng, 4, 0.0)).collect();

        for v in &vecs {
            detector.observe(v.clone()).unwrap();
        }

        detector.reset_reference();

        let mut last_signal = None;
        for v in &vecs {
            last_signal = Some(detector.observe(v.clone()).unwrap());
        }

        match last_signal.unwrap() {
            OnlineDriftSignal::Stable { divergence, .. } => {
                assert!(
                    divergence <= 1e-4,
                    "expected near-zero divergence for identical windows after reset, got {divergence}"
                );
            }
            other => panic!("expected Stable for identical distributions, got {other:?}"),
        }
    }

    #[test]
    fn online_drift_reproducible_with_fixed_sequence() {
        let config = OnlineDriftConfig {
            window_size: 128,
            check_interval: 8,
            threshold: 0.05,
            ..Default::default()
        };

        let mut rng = Lcg::new(1337);
        let sequence: Vec<Vec<f32>> = (0..256).map(|_| make_vector(&mut rng, 8, 0.0)).collect();

        let mut det_a = OnlineDriftDetector::new(config.clone());
        let mut det_b = OnlineDriftDetector::new(config);

        let signals_a: Vec<OnlineDriftSignal> = sequence
            .iter()
            .map(|v| det_a.observe(v.clone()).unwrap())
            .collect();
        let signals_b: Vec<OnlineDriftSignal> = sequence
            .iter()
            .map(|v| det_b.observe(v.clone()).unwrap())
            .collect();

        assert_eq!(signals_a.len(), signals_b.len());
        for (i, (a, b)) in signals_a.iter().zip(&signals_b).enumerate() {
            match (a, b) {
                (
                    OnlineDriftSignal::Stable {
                        divergence: da,
                        window_pos: pa,
                    },
                    OnlineDriftSignal::Stable {
                        divergence: db,
                        window_pos: pb,
                    },
                ) => {
                    assert_eq!(pa, pb, "window_pos mismatch at index {i}");
                    assert!(
                        (da - db).abs() <= 1e-6,
                        "divergence mismatch at index {i}: {da} vs {db}"
                    );
                }
                (
                    OnlineDriftSignal::Drift {
                        divergence: da,
                        window_pos: pa,
                    },
                    OnlineDriftSignal::Drift {
                        divergence: db,
                        window_pos: pb,
                    },
                ) => {
                    assert_eq!(pa, pb, "window_pos mismatch at index {i}");
                    assert!(
                        (da - db).abs() <= 1e-6,
                        "divergence mismatch at index {i}: {da} vs {db}"
                    );
                }
                _ => assert_eq!(a, b, "signal variant mismatch at index {i}: {a:?} vs {b:?}"),
            }
        }
    }
}
