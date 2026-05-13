//! Regularization configuration

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Result, TuneError};

/// Regularization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegularizationConfig {
    /// Dropout rate (0.0 = no dropout)
    pub dropout: f32,

    /// Label smoothing (0.0 = no smoothing)
    pub label_smoothing: f32,

    /// Gradient clipping (None = no clipping)
    pub gradient_clip: Option<f32>,

    /// Use mixup augmentation
    pub mixup_alpha: Option<f32>,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            dropout: 0.1,
            label_smoothing: 0.1,
            gradient_clip: Some(1.0),
            mixup_alpha: None,
        }
    }
}

impl RegularizationConfig {
    /// No regularization
    pub fn none() -> Self {
        Self {
            dropout: 0.0,
            label_smoothing: 0.0,
            gradient_clip: None,
            mixup_alpha: None,
        }
    }

    /// Clip gradients by their L2 norm
    ///
    /// If the L2 norm of gradients exceeds `max_norm`, scale them down
    /// proportionally so that the norm equals `max_norm`.
    ///
    /// # Arguments
    /// * `gradients` - Mutable slice of gradient values to clip
    /// * `max_norm` - Maximum allowed L2 norm
    ///
    /// # Returns
    /// The original gradient norm (before clipping)
    ///
    /// # Example
    /// ```
    /// use lattice_tune::train::RegularizationConfig;
    ///
    /// let mut grads = vec![3.0, 4.0]; // Norm = 5.0
    /// let original_norm = RegularizationConfig::clip_grad_norm(&mut grads, 1.0);
    /// assert!((original_norm - 5.0).abs() < 1e-6);
    /// // After clipping, norm should be 1.0
    /// let new_norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
    /// assert!((new_norm - 1.0).abs() < 1e-6);
    /// ```
    pub fn clip_grad_norm(gradients: &mut [f32], max_norm: f32) -> f32 {
        // Compute L2 norm
        let norm_sq: f32 = gradients.iter().map(|g| g * g).sum();
        let norm = norm_sq.sqrt();

        // Clip if necessary
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for g in gradients.iter_mut() {
                *g *= scale;
            }
        }

        norm
    }

    /// Apply gradient clipping if configured
    ///
    /// Convenience method that checks if gradient clipping is enabled
    /// and applies it if so.
    ///
    /// # Returns
    /// The original gradient norm if clipping was applied, None otherwise
    pub fn apply_gradient_clip(&self, gradients: &mut [f32]) -> Option<f32> {
        self.gradient_clip
            .map(|max_norm| Self::clip_grad_norm(gradients, max_norm))
    }

    /// Light regularization
    pub fn light() -> Self {
        Self {
            dropout: 0.05,
            label_smoothing: 0.05,
            gradient_clip: Some(1.0),
            mixup_alpha: None,
        }
    }

    /// Strong regularization
    pub fn strong() -> Self {
        Self {
            dropout: 0.3,
            label_smoothing: 0.2,
            gradient_clip: Some(0.5),
            mixup_alpha: Some(0.2),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TuneError::InvalidConfig(format!(
                "dropout must be between 0 and 1, got {}",
                self.dropout
            )));
        }
        if !(0.0..=1.0).contains(&self.label_smoothing) {
            return Err(TuneError::InvalidConfig(format!(
                "label_smoothing must be between 0 and 1, got {}",
                self.label_smoothing
            )));
        }
        if let Some(clip) = self.gradient_clip {
            if clip <= 0.0 {
                return Err(TuneError::InvalidConfig(
                    "gradient_clip must be > 0".to_string(),
                ));
            }
        }
        if let Some(alpha) = self.mixup_alpha {
            if alpha <= 0.0 {
                return Err(TuneError::InvalidConfig(
                    "mixup_alpha must be > 0".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regularization_config() {
        let none = RegularizationConfig::none();
        assert_eq!(none.dropout, 0.0);
        assert!(none.validate().is_ok());

        let strong = RegularizationConfig::strong();
        assert!(strong.dropout > 0.0);
        assert!(strong.validate().is_ok());
    }

    #[test]
    fn test_clip_grad_norm_exceeds_max() {
        // 3-4-5 triangle, norm = 5
        let mut grads = vec![3.0, 4.0];
        let original_norm = RegularizationConfig::clip_grad_norm(&mut grads, 1.0);

        // Original norm should be returned
        assert!((original_norm - 5.0).abs() < 1e-6);

        // After clipping, norm should equal max_norm
        let new_norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 1e-6);

        // Direction should be preserved
        assert!((grads[0] / grads[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_under_max() {
        // Norm = 2, max = 5, should not clip
        let mut grads = vec![1.0, 1.0, 1.0, 1.0]; // norm = 2
        let original_norm = RegularizationConfig::clip_grad_norm(&mut grads, 5.0);

        assert!((original_norm - 2.0).abs() < 1e-6);
        // Values should be unchanged
        assert_eq!(grads, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clip_grad_norm_empty() {
        let mut grads: Vec<f32> = vec![];
        let norm = RegularizationConfig::clip_grad_norm(&mut grads, 1.0);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_norm_zero_grads() {
        let mut grads = vec![0.0, 0.0, 0.0];
        let norm = RegularizationConfig::clip_grad_norm(&mut grads, 1.0);
        assert_eq!(norm, 0.0);
        assert_eq!(grads, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_apply_gradient_clip_enabled() {
        let reg = RegularizationConfig {
            gradient_clip: Some(1.0),
            ..RegularizationConfig::none()
        };

        let mut grads = vec![3.0, 4.0]; // norm = 5
        let result = reg.apply_gradient_clip(&mut grads);

        assert!(result.is_some());
        assert!((result.unwrap() - 5.0).abs() < 1e-6);

        let new_norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_gradient_clip_disabled() {
        let reg = RegularizationConfig::none();
        let mut grads = vec![3.0, 4.0];
        let result = reg.apply_gradient_clip(&mut grads);

        assert!(result.is_none());
        // Grads unchanged
        assert_eq!(grads, vec![3.0, 4.0]);
    }
}
