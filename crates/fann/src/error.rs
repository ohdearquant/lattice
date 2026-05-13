//! Error types for lattice-fann

use thiserror::Error;

/// Result type for neural network operations
pub type FannResult<T> = std::result::Result<T, FannError>;

/// Maximum allowed elements in a single tensor allocation.
/// 100 million elements = ~400MB for f32, a reasonable limit to prevent OOM.
pub const MAX_ALLOWED_ELEMENTS: usize = 100_000_000;

/// Validate that an allocation size is within safe limits.
///
/// Returns an error if the requested size exceeds [`MAX_ALLOWED_ELEMENTS`].
/// This prevents denial-of-service attacks via memory exhaustion.
#[inline]
pub fn validate_allocation_size(num_elements: usize) -> FannResult<()> {
    if num_elements > MAX_ALLOWED_ELEMENTS {
        return Err(FannError::ShapeTooLarge {
            requested: num_elements,
            max: MAX_ALLOWED_ELEMENTS,
        });
    }
    Ok(())
}

/// Validate layer dimensions for safe allocation.
///
/// Checks both that dimensions are non-zero and that the total allocation
/// (num_inputs * num_outputs for weights) doesn't exceed safe limits.
#[inline]
pub fn validate_layer_dimensions(num_inputs: usize, num_outputs: usize) -> FannResult<()> {
    if num_inputs == 0 || num_outputs == 0 {
        return Err(FannError::InvalidLayerDimensions {
            inputs: num_inputs,
            outputs: num_outputs,
        });
    }

    // Check for overflow and size limit
    let total = num_inputs
        .checked_mul(num_outputs)
        .ok_or(FannError::ShapeTooLarge {
            requested: usize::MAX,
            max: MAX_ALLOWED_ELEMENTS,
        })?;

    validate_allocation_size(total)
}

/// Errors that can occur in neural network operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FannError {
    /// Input size does not match network's expected input size
    #[error("Input size mismatch: expected {expected}, got {actual}")]
    InputSizeMismatch {
        /// Expected input size
        expected: usize,
        /// Actual input size provided
        actual: usize,
    },

    /// Output size does not match network's expected output size
    #[error("Output size mismatch: expected {expected}, got {actual}")]
    OutputSizeMismatch {
        /// Expected output size
        expected: usize,
        /// Actual output size provided
        actual: usize,
    },

    /// Weight count does not match expected weight count for layer
    #[error("Weight count mismatch: expected {expected}, got {actual}")]
    WeightCountMismatch {
        /// Expected weight count
        expected: usize,
        /// Actual weight count provided
        actual: usize,
    },

    /// Bias count does not match expected bias count for layer
    #[error("Bias count mismatch: expected {expected}, got {actual}")]
    BiasCountMismatch {
        /// Expected bias count
        expected: usize,
        /// Actual bias count provided
        actual: usize,
    },

    /// Layer dimensions are invalid
    #[error("Invalid layer dimensions: inputs={inputs}, outputs={outputs}")]
    InvalidLayerDimensions {
        /// Number of inputs
        inputs: usize,
        /// Number of outputs
        outputs: usize,
    },

    /// Allocation size exceeds maximum allowed elements (DoS prevention)
    #[error("Shape too large: requested {requested} elements, max allowed is {max}")]
    ShapeTooLarge {
        /// Requested number of elements
        requested: usize,
        /// Maximum allowed elements
        max: usize,
    },

    /// Network has no layers
    #[error("Network has no layers")]
    EmptyNetwork,

    /// Builder configuration is invalid
    #[error("Invalid builder configuration: {0}")]
    InvalidBuilder(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Gradient computation failed
    #[error("Gradient computation failed: {0}")]
    GradientError(String),

    /// Numeric instability detected
    #[error("Numeric instability: {0}")]
    NumericInstability(String),

    /// Invalid distribution parameters (e.g., negative std_dev, NaN)
    #[error("Invalid distribution parameters: {0}")]
    InvalidDistributionParams(String),

    /// Serialization error
    #[cfg(feature = "serde")]
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_size_mismatch_display() {
        let err = FannError::InputSizeMismatch {
            expected: 10,
            actual: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_weight_count_mismatch_display() {
        let err = FannError::WeightCountMismatch {
            expected: 100,
            actual: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_invalid_layer_dimensions_display() {
        let err = FannError::InvalidLayerDimensions {
            inputs: 0,
            outputs: 10,
        };
        assert!(err.to_string().contains("inputs=0"));
    }

    #[test]
    fn test_empty_network_display() {
        let err = FannError::EmptyNetwork;
        assert!(err.to_string().contains("no layers"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = FannError::InputSizeMismatch {
            expected: 10,
            actual: 5,
        };
        let err2 = FannError::InputSizeMismatch {
            expected: 10,
            actual: 5,
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_shape_too_large_display() {
        let err = FannError::ShapeTooLarge {
            requested: 200_000_000,
            max: MAX_ALLOWED_ELEMENTS,
        };
        let msg = err.to_string();
        assert!(msg.contains("200000000"));
        assert!(msg.contains(&MAX_ALLOWED_ELEMENTS.to_string()));
    }

    #[test]
    fn test_validate_allocation_size_ok() {
        assert!(validate_allocation_size(0).is_ok());
        assert!(validate_allocation_size(1000).is_ok());
        assert!(validate_allocation_size(MAX_ALLOWED_ELEMENTS).is_ok());
    }

    #[test]
    fn test_validate_allocation_size_too_large() {
        let result = validate_allocation_size(MAX_ALLOWED_ELEMENTS + 1);
        assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
    }

    #[test]
    fn test_validate_layer_dimensions_ok() {
        assert!(validate_layer_dimensions(100, 100).is_ok());
        assert!(validate_layer_dimensions(1000, 1000).is_ok());
    }

    #[test]
    fn test_validate_layer_dimensions_zero() {
        let result = validate_layer_dimensions(0, 100);
        assert!(matches!(
            result,
            Err(FannError::InvalidLayerDimensions { .. })
        ));

        let result = validate_layer_dimensions(100, 0);
        assert!(matches!(
            result,
            Err(FannError::InvalidLayerDimensions { .. })
        ));
    }

    #[test]
    fn test_validate_layer_dimensions_too_large() {
        // 100_001 * 100_001 > MAX_ALLOWED_ELEMENTS (100M)
        let result = validate_layer_dimensions(100_001, 100_001);
        assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
    }

    #[test]
    fn test_validate_layer_dimensions_overflow() {
        // usize::MAX * 2 would overflow
        let result = validate_layer_dimensions(usize::MAX, 2);
        assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
    }
}
