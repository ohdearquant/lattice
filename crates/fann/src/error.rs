//! Error types and allocation-shape validation for `lattice-fann`.
//!
//! Dimension checks reject zero-sized or excessively large tensors before
//! allocation. See `docs/network.md` for the limits and parser invariants.

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
#[non_exhaustive]
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

    /// Input or configuration failed validation.
    ///
    /// Distinct from `InvalidBuilder`, which names a network builder's own
    /// configuration. This one carries validation of caller-supplied values and
    /// input slices — the wording matches what these messages read as before the
    /// router-refit module moved into this crate, so the text a user sees is
    /// unchanged by the move.
    #[error("Validation error: {0}")]
    Validation(String),

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
mod tests;
