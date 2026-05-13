//! Gradient utilities for training
//!
//! Provides utilities for gradient checking, sanitization, and guard strategies.

/// Check if a gradient vector contains NaN or Inf values.
///
/// Returns Some(index) of the first bad value, or None if all values are finite.
#[inline]
pub fn find_nan_or_inf(grads: &[f32]) -> Option<usize> {
    grads.iter().position(|&g| !g.is_finite())
}

/// Check multiple gradient vectors for NaN/Inf values.
///
/// Returns a description of where the bad value was found.
pub fn check_gradients_valid(
    weight_grads: &[Vec<f32>],
    bias_grads: &[Vec<f32>],
) -> Result<(), String> {
    for (layer_idx, grads) in weight_grads.iter().enumerate() {
        if let Some(idx) = find_nan_or_inf(grads) {
            return Err(format!(
                "NaN/Inf in weight gradient at layer {layer_idx}, index {idx}, value {}",
                grads[idx]
            ));
        }
    }
    for (layer_idx, grads) in bias_grads.iter().enumerate() {
        if let Some(idx) = find_nan_or_inf(grads) {
            return Err(format!(
                "NaN/Inf in bias gradient at layer {layer_idx}, index {idx}, value {}",
                grads[idx]
            ));
        }
    }
    Ok(())
}

/// Replace NaN/Inf values with zeros (gradient clipping fallback).
///
/// Returns the number of values replaced.
pub fn sanitize_gradients(grads: &mut [f32]) -> usize {
    let mut count = 0;
    for g in grads.iter_mut() {
        if !g.is_finite() {
            *g = 0.0;
            count += 1;
        }
    }
    count
}

/// Strategy for handling NaN/Inf gradients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum GradientGuardStrategy {
    /// Return an error immediately when NaN/Inf detected
    #[default]
    Error,
    /// Replace NaN/Inf with zero and continue (gradient clipping fallback)
    Sanitize,
    /// Skip the batch when NaN/Inf detected
    SkipBatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nan_or_inf() {
        // Normal gradients
        let normal = vec![0.1, 0.2, 0.3, -0.5];
        assert!(find_nan_or_inf(&normal).is_none());

        // NaN value
        let with_nan = vec![0.1, f32::NAN, 0.3];
        assert_eq!(find_nan_or_inf(&with_nan), Some(1));

        // Inf value
        let with_inf = vec![0.1, 0.2, f32::INFINITY];
        assert_eq!(find_nan_or_inf(&with_inf), Some(2));

        // Negative Inf
        let with_neg_inf = vec![f32::NEG_INFINITY, 0.2];
        assert_eq!(find_nan_or_inf(&with_neg_inf), Some(0));
    }

    #[test]
    fn test_check_gradients_valid() {
        let weight_grads = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let bias_grads = vec![vec![0.1], vec![0.2]];
        assert!(check_gradients_valid(&weight_grads, &bias_grads).is_ok());

        // With NaN in weight gradients
        let bad_weight_grads = vec![vec![0.1, f32::NAN], vec![0.3, 0.4]];
        let result = check_gradients_valid(&bad_weight_grads, &bias_grads);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("weight gradient"));

        // With Inf in bias gradients
        let bad_bias_grads = vec![vec![0.1], vec![f32::INFINITY]];
        let result = check_gradients_valid(&weight_grads, &bad_bias_grads);
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_gradients() {
        let mut grads = vec![0.1, f32::NAN, 0.3, f32::INFINITY, -0.5, f32::NEG_INFINITY];
        let count = sanitize_gradients(&mut grads);

        assert_eq!(count, 3); // NaN, Inf, -Inf
        assert_eq!(grads, vec![0.1, 0.0, 0.3, 0.0, -0.5, 0.0]);
    }

    #[test]
    fn test_gradient_guard_strategy_default() {
        // Default should be Error
        let strategy = GradientGuardStrategy::default();
        assert_eq!(strategy, GradientGuardStrategy::Error);
    }
}
