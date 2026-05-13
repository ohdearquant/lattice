//! Error types for lattice-tune

use thiserror::Error;
use uuid::Uuid;

/// Result type alias for lattice-tune operations
pub type Result<T> = std::result::Result<T, TuneError>;

/// Error types for lattice-tune training infrastructure
#[derive(Error, Debug)]
pub enum TuneError {
    /// Dataset error
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// Example not found
    #[error("Example not found: {id}")]
    ExampleNotFound {
        /// The example ID that wasn't found
        id: Uuid,
    },

    /// Invalid batch size
    #[error("Invalid batch size: {size} (must be > 0 and <= dataset size {dataset_size})")]
    InvalidBatchSize {
        /// The requested batch size
        size: usize,
        /// The dataset size
        dataset_size: usize,
    },

    /// Dimension mismatch in embeddings
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Teacher API error
    #[error("Teacher API error: {0}")]
    TeacherApi(String),

    /// Teacher timeout
    #[error("Teacher timeout after {timeout_ms}ms")]
    TeacherTimeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },

    /// Training error
    #[error("Training error: {0}")]
    Training(String),

    /// Convergence failure
    #[error("Training did not converge after {epochs} epochs (final loss: {final_loss:.6})")]
    ConvergenceFailure {
        /// Number of epochs trained
        epochs: usize,
        /// Final loss value
        final_loss: f32,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model not found in registry
    #[error("Model not found: {name} v{version}")]
    ModelNotFound {
        /// Model name
        name: String,
        /// Model version
        version: String,
    },

    /// Duplicate model in registry
    #[error("Model already exists: {name} v{version}")]
    DuplicateModel {
        /// Model name
        name: String,
        /// Model version
        version: String,
    },

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Weight integrity error (checksum mismatch)
    #[error("Weight integrity error: expected checksum {expected}, got {actual}")]
    WeightIntegrityError {
        /// Expected checksum
        expected: String,
        /// Actual checksum
        actual: String,
    },

    /// Memory budget exceeded
    #[error("Memory budget exceeded: required {required_mb}MB, budget {budget_mb}MB")]
    MemoryBudgetExceeded {
        /// Required memory in MB
        required_mb: usize,
        /// Budget memory in MB
        budget_mb: usize,
    },
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for TuneError {
    fn from(err: serde_json::Error) -> Self {
        TuneError::Serialization(err.to_string())
    }
}
