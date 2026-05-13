//! Training orchestration module
//!
//! This module provides the training loop and configuration for
//! training neural models on labeled data.
//!
//! # Architecture
//!
//! ```text
//! Dataset → Training Loop → Model Updates → Checkpoints → Registry
//! ```
//!
//! # Example
//!
//! ```ignore
//! use lattice_tune::train::{TrainingConfig, TrainingLoop, TrainingMetrics};
//! use lattice_tune::data::Dataset;
//!
//! // Configure training
//! let config = TrainingConfig::default()
//!     .epochs(100)
//!     .learning_rate(0.001)
//!     .batch_size(32);
//!
//! // Create training loop
//! let mut trainer = TrainingLoop::new(model, config)?;
//!
//! // Train
//! let metrics = trainer.train(&dataset)?;
//! ```
//!
//! # GPU Training
//!
//! With the `gpu` feature enabled, GPU-accelerated training is available:
//!
//! ```ignore
//! use lattice_tune::train::{GpuTrainer, GpuTrainerBuilder, TrainingConfig};
//! use lattice_fann::Activation;
//!
//! // Build GPU trainer
//! let mut trainer = GpuTrainerBuilder::new(768, 6)
//!     .hidden(64, Activation::ReLU)
//!     .hidden(32, Activation::ReLU)
//!     .config(TrainingConfig::default())
//!     .build()?;
//!
//! // Train batches
//! for batch in dataset.batches() {
//!     let loss = trainer.train_batch(&batch)?;
//! }
//! ```

mod config;
mod jit;
mod r#loop;

#[cfg(feature = "gpu")]
mod gpu;

pub use config::{
    EarlyStopping, LRSchedule, MAX_BATCH_SIZE, Optimizer, OptimizerConfig, RegularizationConfig,
    TrainingConfig,
};
pub use jit::{JitAdapter, JitConfig, JitResult, JitStrategy, freeze};
pub use r#loop::{
    Checkpoint, EpochMetrics, LoggingCallback, NoOpCallback, TrainingCallback, TrainingLoop,
    TrainingMetrics, TrainingState,
};

#[cfg(feature = "gpu")]
pub use gpu::{GpuTrainer, GpuTrainerBuilder};
