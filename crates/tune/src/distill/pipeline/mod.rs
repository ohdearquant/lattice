//! Distillation pipeline for labeling training data

mod distill;
mod types;

pub use distill::DistillationPipeline;
pub use types::{DistillationStats, LabelingResult, RawExample};

use crate::error::{Result, TuneError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the distillation pipeline
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistillationConfig {
    /// Batch size for labeling requests
    pub batch_size: usize,

    /// Concurrency level (parallel requests)
    pub concurrency: usize,

    /// Whether to apply softmax normalization to labels
    pub normalize_labels: bool,

    /// Minimum confidence threshold (skip low-confidence labels)
    pub min_confidence: Option<f32>,

    /// Whether to save intermediate results
    pub save_intermediate: bool,

    /// Output directory for intermediate results
    pub output_dir: Option<String>,

    /// Progress callback interval (every N examples)
    pub progress_interval: usize,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            concurrency: 5,
            normalize_labels: true,
            min_confidence: None,
            save_intermediate: false,
            output_dir: None,
            progress_interval: 100,
        }
    }
}

impl DistillationConfig {
    /// Create a config optimized for speed
    pub fn fast() -> Self {
        Self {
            batch_size: 20,
            concurrency: 10,
            normalize_labels: true,
            min_confidence: None,
            save_intermediate: false,
            output_dir: None,
            progress_interval: 50,
        }
    }

    /// Create a config optimized for quality
    pub fn quality() -> Self {
        Self {
            batch_size: 5,
            concurrency: 3,
            normalize_labels: true,
            min_confidence: Some(0.5),
            save_intermediate: true,
            output_dir: None,
            progress_interval: 20,
        }
    }

    /// Set the batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set concurrency level
    pub fn concurrency(mut self, level: usize) -> Self {
        self.concurrency = level;
        self
    }

    /// Set output directory
    pub fn output_dir(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = Some(dir.into());
        self.save_intermediate = true;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(TuneError::InvalidConfig(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.concurrency == 0 {
            return Err(TuneError::InvalidConfig(
                "concurrency must be > 0".to_string(),
            ));
        }
        if let Some(conf) = self.min_confidence {
            if !(0.0..=1.0).contains(&conf) {
                return Err(TuneError::InvalidConfig(format!(
                    "min_confidence must be between 0.0 and 1.0, got {conf}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
