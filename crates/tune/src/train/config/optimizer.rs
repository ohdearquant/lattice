//! Optimizer configuration

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Result, TuneError};

/// Optimizer type
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    SGDMomentum,
    /// Adam optimizer (default choice)
    Adam,
    /// AdamW (Adam with decoupled weight decay)
    #[default]
    AdamW,
    /// RMSprop
    RMSprop,
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Optimizer::SGD => write!(f, "sgd"),
            Optimizer::SGDMomentum => write!(f, "sgd_momentum"),
            Optimizer::Adam => write!(f, "adam"),
            Optimizer::AdamW => write!(f, "adamw"),
            Optimizer::RMSprop => write!(f, "rmsprop"),
        }
    }
}

/// Optimizer configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer: Optimizer,

    /// Learning rate
    pub learning_rate: f32,

    /// Momentum (for SGD, RMSprop)
    pub momentum: f32,

    /// Beta1 (for Adam variants)
    pub beta1: f32,

    /// Beta2 (for Adam variants)
    pub beta2: f32,

    /// Epsilon for numerical stability
    pub epsilon: f32,

    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer: Optimizer::AdamW,
            learning_rate: 0.001,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl OptimizerConfig {
    /// Create SGD config
    pub fn sgd(learning_rate: f32) -> Self {
        Self {
            optimizer: Optimizer::SGD,
            learning_rate,
            ..Default::default()
        }
    }

    /// Create SGD with momentum config
    pub fn sgd_momentum(learning_rate: f32, momentum: f32) -> Self {
        Self {
            optimizer: Optimizer::SGDMomentum,
            learning_rate,
            momentum,
            ..Default::default()
        }
    }

    /// Create Adam config
    pub fn adam(learning_rate: f32) -> Self {
        Self {
            optimizer: Optimizer::Adam,
            learning_rate,
            ..Default::default()
        }
    }

    /// Create AdamW config
    pub fn adamw(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            optimizer: Optimizer::AdamW,
            learning_rate,
            weight_decay,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(TuneError::InvalidConfig(
                "learning_rate must be finite and > 0".to_string(),
            ));
        }
        if !self.momentum.is_finite() || !(0.0..=1.0).contains(&self.momentum) {
            return Err(TuneError::InvalidConfig(
                "momentum must be finite and between 0 and 1".to_string(),
            ));
        }
        if !self.beta1.is_finite() || self.beta1 < 0.0 || self.beta1 >= 1.0 {
            return Err(TuneError::InvalidConfig(
                "beta1 must be finite and in [0, 1)".to_string(),
            ));
        }
        if !self.beta2.is_finite() || self.beta2 < 0.0 || self.beta2 >= 1.0 {
            return Err(TuneError::InvalidConfig(
                "beta2 must be finite and in [0, 1)".to_string(),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(TuneError::InvalidConfig(
                "epsilon must be finite and > 0".to_string(),
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(TuneError::InvalidConfig(
                "weight_decay must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config() {
        let sgd = OptimizerConfig::sgd(0.01);
        assert_eq!(sgd.optimizer, Optimizer::SGD);
        assert!(sgd.validate().is_ok());

        let adamw = OptimizerConfig::adamw(0.001, 0.01);
        assert_eq!(adamw.optimizer, Optimizer::AdamW);
        assert!(adamw.validate().is_ok());
    }
}
