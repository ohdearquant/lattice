//! Data contracts and in-memory batching for training.
//!
//! `TrainingExample` carries embeddings and soft labels, while `Dataset` turns
//! examples into epoch batches. See `docs/data.md` for the full format,
//! validation rules, and iteration behavior.

mod dataset;
mod example;

pub use dataset::{Batch, Dataset, DatasetConfig, DatasetStats};
pub use example::{ExampleMetadata, IntentLabels, TrainingExample};
