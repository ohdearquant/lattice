//! Learning rate schedule configuration

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Result, TuneError};

/// Learning rate schedule
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LRSchedule {
    /// Constant learning rate
    Constant,

    /// Linear warmup then constant
    LinearWarmup {
        /// Number of warmup steps
        warmup_steps: usize,
    },

    /// Step decay (reduce by factor every step_size epochs)
    StepDecay {
        /// Epochs between reductions
        step_size: usize,
        /// Decay factor (e.g., 0.1 = reduce to 10%)
        gamma: f32,
    },

    /// Exponential decay
    ExponentialDecay {
        /// Decay rate per epoch
        gamma: f32,
    },

    /// Cosine annealing
    CosineAnnealing {
        /// Minimum learning rate
        min_lr: f32,
        /// Period in epochs
        t_max: usize,
    },

    /// Cosine annealing with warmup
    CosineAnnealingWarmup {
        /// Number of warmup steps
        warmup_steps: usize,
        /// Minimum learning rate
        min_lr: f32,
        /// Period in epochs (after warmup)
        t_max: usize,
    },

    /// One cycle policy
    OneCycle {
        /// Maximum learning rate
        max_lr: f32,
        /// Percentage of cycle spent increasing LR
        pct_start: f32,
        /// Total steps in cycle
        total_steps: usize,
    },
}

impl Default for LRSchedule {
    fn default() -> Self {
        Self::CosineAnnealingWarmup {
            warmup_steps: 100,
            min_lr: 1e-6,
            t_max: 100,
        }
    }
}

impl LRSchedule {
    /// Validate the schedule parameters to catch division-by-zero and other
    /// invalid configurations before training begins.
    pub fn validate(&self) -> Result<()> {
        match self {
            LRSchedule::Constant => {}

            LRSchedule::LinearWarmup { warmup_steps } => {
                if *warmup_steps == 0 {
                    return Err(TuneError::InvalidConfig(
                        "LinearWarmup: warmup_steps must be > 0".into(),
                    ));
                }
            }

            LRSchedule::StepDecay { step_size, gamma } => {
                if *step_size == 0 {
                    return Err(TuneError::InvalidConfig(
                        "StepDecay: step_size must be > 0".into(),
                    ));
                }
                if !gamma.is_finite() || *gamma <= 0.0 {
                    return Err(TuneError::InvalidConfig(
                        "StepDecay: gamma must be finite and > 0".into(),
                    ));
                }
            }

            LRSchedule::ExponentialDecay { gamma } => {
                if !gamma.is_finite() || *gamma <= 0.0 {
                    return Err(TuneError::InvalidConfig(
                        "ExponentialDecay: gamma must be finite and > 0".into(),
                    ));
                }
            }

            LRSchedule::CosineAnnealing { min_lr, t_max } => {
                if *t_max == 0 {
                    return Err(TuneError::InvalidConfig(
                        "CosineAnnealing: t_max must be > 0".into(),
                    ));
                }
                if !min_lr.is_finite() || *min_lr < 0.0 {
                    return Err(TuneError::InvalidConfig(
                        "CosineAnnealing: min_lr must be finite and >= 0".into(),
                    ));
                }
            }

            LRSchedule::CosineAnnealingWarmup {
                warmup_steps,
                min_lr,
                t_max,
            } => {
                if *t_max == 0 {
                    return Err(TuneError::InvalidConfig(
                        "CosineAnnealingWarmup: t_max must be > 0".into(),
                    ));
                }
                if *warmup_steps == 0 {
                    return Err(TuneError::InvalidConfig(
                        "CosineAnnealingWarmup: warmup_steps must be > 0".into(),
                    ));
                }
                if !min_lr.is_finite() || *min_lr < 0.0 {
                    return Err(TuneError::InvalidConfig(
                        "CosineAnnealingWarmup: min_lr must be finite and >= 0".into(),
                    ));
                }
            }

            LRSchedule::OneCycle {
                max_lr,
                pct_start,
                total_steps,
            } => {
                if *total_steps == 0 {
                    return Err(TuneError::InvalidConfig(
                        "OneCycle: total_steps must be > 0".into(),
                    ));
                }
                if !max_lr.is_finite() || *max_lr <= 0.0 {
                    return Err(TuneError::InvalidConfig(
                        "OneCycle: max_lr must be finite and > 0".into(),
                    ));
                }
                if !pct_start.is_finite() || *pct_start <= 0.0 || *pct_start >= 1.0 {
                    return Err(TuneError::InvalidConfig(
                        "OneCycle: pct_start must be finite and in (0, 1)".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Calculate learning rate at given step
    pub fn get_lr(&self, base_lr: f32, step: usize, epoch: usize) -> f32 {
        match self {
            LRSchedule::Constant => base_lr,

            LRSchedule::LinearWarmup { warmup_steps } => {
                if *warmup_steps == 0 || step >= *warmup_steps {
                    base_lr
                } else {
                    base_lr * (step as f32 / *warmup_steps as f32)
                }
            }

            LRSchedule::StepDecay { step_size, gamma } => {
                if *step_size == 0 {
                    base_lr
                } else {
                    let num_decays = epoch / step_size;
                    base_lr * gamma.powi(num_decays as i32)
                }
            }

            LRSchedule::ExponentialDecay { gamma } => base_lr * gamma.powi(epoch as i32),

            LRSchedule::CosineAnnealing { min_lr, t_max } => {
                if *t_max == 0 {
                    return base_lr;
                }
                let progress = (epoch % t_max) as f32 / *t_max as f32;
                *min_lr
                    + (base_lr - *min_lr) * (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0
            }

            LRSchedule::CosineAnnealingWarmup {
                warmup_steps,
                min_lr,
                t_max,
            } => {
                if *warmup_steps > 0 && step < *warmup_steps {
                    base_lr * (step as f32 / *warmup_steps as f32)
                } else if *t_max == 0 {
                    base_lr
                } else {
                    let progress = (epoch % t_max) as f32 / *t_max as f32;
                    *min_lr
                        + (base_lr - *min_lr) * (1.0 + (progress * std::f32::consts::PI).cos())
                            / 2.0
                }
            }

            LRSchedule::OneCycle {
                max_lr,
                pct_start,
                total_steps,
            } => {
                let pct = step as f32 / *total_steps as f32;
                if pct < *pct_start {
                    // Increasing phase
                    let phase_pct = pct / *pct_start;
                    base_lr + (max_lr - base_lr) * phase_pct
                } else {
                    // Decreasing phase
                    let phase_pct = (pct - *pct_start) / (1.0 - *pct_start);
                    *max_lr * (1.0 - phase_pct) + base_lr * 0.01 * phase_pct
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_schedule() {
        // Constant
        let constant = LRSchedule::Constant;
        assert_eq!(constant.get_lr(0.01, 100, 10), 0.01);

        // Linear warmup
        let warmup = LRSchedule::LinearWarmup { warmup_steps: 100 };
        assert!(warmup.get_lr(0.01, 0, 0) < 0.001); // Start low
        assert!(warmup.get_lr(0.01, 50, 0) < 0.01); // Mid warmup
        assert_eq!(warmup.get_lr(0.01, 100, 1), 0.01); // After warmup

        // Step decay
        let step = LRSchedule::StepDecay {
            step_size: 10,
            gamma: 0.1,
        };
        assert_eq!(step.get_lr(0.01, 0, 0), 0.01);
        assert!((step.get_lr(0.01, 0, 10) - 0.001).abs() < 1e-6);
    }
}
