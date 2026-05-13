//! Training data types and utilities
//!
//! This module provides the core data structures for training:
//! - [`TrainingExample`]: Individual training examples with embeddings and labels
//! - [`IntentLabels`]: Soft labels from teacher models
//! - [`Dataset`]: Collection of examples with batching support
//!
//! # Example
//!
//! ```
//! use lattice_tune::data::{TrainingExample, IntentLabels, Dataset, DatasetConfig};
//!
//! // Create a dataset from examples
//! let examples = vec![
//!     TrainingExample::new(
//!         vec![vec![0.1, 0.2, 0.3]],  // context embeddings
//!         vec![0.4, 0.5, 0.6],        // message embedding
//!         IntentLabels::continuation(0.8),
//!     ),
//! ];
//!
//! let dataset = Dataset::from_examples(examples);
//! ```

mod dataset;
mod example;

pub use dataset::{Batch, Dataset, DatasetConfig, DatasetStats};
pub use example::{ExampleMetadata, IntentLabels, TrainingExample};
