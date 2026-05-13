#![allow(unused_imports)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::field_reassign_with_default)]

//! lattice-tune - Training infrastructure for Lattice neural models
//!
//! Provides a complete pipeline for training neural networks through knowledge distillation:
//!
//! - **Data**: Training examples, datasets, and batching
//! - **Distill**: Knowledge distillation from teacher models (Claude, GPT, Gemini)
//! - **Train**: Training loop, optimization, and checkpointing
//! - **Registry**: Model versioning, storage, and deployment tracking
//!
//! # Architecture
//!
//! ```text
//! Raw Data → Teacher (LLM) → Soft Labels → Dataset → Training → Model → Registry
//!                                                        ↓
//!                                                   Deployment
//! ```
//!
//! # Quick Start
//!
//! ```rust
//! use lattice_tune::data::{TrainingExample, IntentLabels, Dataset, DatasetConfig};
//!
//! // Create training examples
//! let examples = vec![
//!     TrainingExample::new(
//!         vec![vec![0.1, 0.2, 0.3]],  // context embeddings
//!         vec![0.4, 0.5, 0.6],        // message embedding
//!         IntentLabels::continuation(0.8),
//!     ),
//! ];
//!
//! // Create a dataset
//! let dataset = Dataset::from_examples(examples);
//! let stats = dataset.stats();
//! println!("Dataset has {} examples", stats.num_examples);
//! ```
//!
//! # Distillation Example
//!
//! ```ignore
//! use lattice_tune::distill::{TeacherConfig, DistillationPipeline, RawExample};
//!
//! // Configure teacher model
//! let teacher = TeacherConfig::claude_sonnet();
//!
//! // Create distillation pipeline
//! let mut pipeline = DistillationPipeline::with_teacher(teacher)?;
//!
//! // Create raw examples (text, not embeddings)
//! let raw = RawExample::new(
//!     vec!["Hello".to_string(), "How are you?".to_string()],
//!     "What's the weather like?",
//! );
//!
//! // Label with teacher
//! let result = pipeline.label_single(&raw)?;
//! println!("Labeled with confidence: {}", result.confidence);
//! ```
//!
//! # Training Example
//!
//! ```ignore
//! use lattice_tune::train::{TrainingConfig, TrainingLoop};
//! use lattice_tune::data::Dataset;
//!
//! // Configure training
//! let config = TrainingConfig::default()
//!     .epochs(100)
//!     .batch_size(32)
//!     .learning_rate(0.001);
//!
//! // Train
//! let mut trainer = TrainingLoop::new(config)?;
//! let metrics = trainer.train(&mut dataset)?;
//!
//! println!("Final loss: {:.4}", metrics.final_train_loss);
//! ```
//!
//! # Registry Example
//!
//! ```rust
//! use lattice_tune::registry::{ModelRegistry, RegisteredModel, ModelMetadata};
//!
//! // Create a registry
//! let registry = ModelRegistry::in_memory();
//!
//! // Register a model
//! let metadata = ModelMetadata::classifier(768, 6, 10000);
//! let model = RegisteredModel::new("intent_classifier", "1.0.0")
//!     .with_metadata(metadata)
//!     .with_description("Intent classification model");
//!
//! let weights = vec![0u8; 1000]; // Model weights
//! let id = registry.register(model, &weights).unwrap();
//!
//! // Retrieve the model
//! let loaded = registry.get("intent_classifier", "1.0.0").unwrap();
//! println!("Loaded: {}", loaded.full_name());
//! ```
//!
//! # Design Principles
//!
//! 1. **Data-first**: Well-defined training example format with full traceability
//! 2. **Modular**: Distillation, training, and registry are separate concerns
//! 3. **Extensible**: Support different teacher models (Claude, GPT, Gemini)
//! 4. **Traceable**: All models have version, training config, and metrics
//!
//! # Feature Flags
//!
//! - `std` (default): Standard library support
//! - `serde`: Serialization support for all types

#![warn(missing_docs)]

pub mod data;
pub mod distill;
pub mod error;
pub mod lora;
pub mod registry;
pub mod train;

// Re-exports for convenience
pub use error::{Result, TuneError};

// Data re-exports
pub use data::{
    Batch, Dataset, DatasetConfig, DatasetStats, ExampleMetadata, IntentLabels, TrainingExample,
};

// Distill re-exports
pub use distill::{
    DistillationConfig, DistillationPipeline, DistillationStats, EndpointSecurity, LabelingResult,
    TeacherConfig, TeacherConfigBuilder, TeacherProvider,
};

// Train re-exports
pub use train::{
    Checkpoint, EarlyStopping, EpochMetrics, JitAdapter, JitConfig, JitResult, JitStrategy,
    LRSchedule, LoggingCallback, NoOpCallback, Optimizer, OptimizerConfig, RegularizationConfig,
    TrainingCallback, TrainingConfig, TrainingLoop, TrainingMetrics, TrainingState, freeze,
};

// GPU training re-exports (when feature enabled)
#[cfg(feature = "gpu")]
pub use train::{GpuTrainer, GpuTrainerBuilder};

// LoRA re-exports
pub use lora::{LoraAdapter, LoraConfig, LoraLayer};

// Registry re-exports
pub use registry::{
    LiveModel, ModelMetadata, ModelQuery, ModelRegistry, ModelStatus, RegisteredModel,
    RollbackController, RollbackRecord, ShadowComparison, ShadowConfig, ShadowSession, ShadowState,
    StorageBackend,
};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::data::{
        Batch, Dataset, DatasetConfig, DatasetStats, ExampleMetadata, IntentLabels, TrainingExample,
    };
    pub use crate::distill::{
        DistillationConfig, DistillationPipeline, DistillationStats, EndpointSecurity,
        LabelingResult, TeacherConfig, TeacherProvider,
    };
    pub use crate::error::{Result, TuneError};
    pub use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
    pub use crate::registry::{
        LiveModel, ModelMetadata, ModelRegistry, ModelStatus, RegisteredModel, RollbackController,
        RollbackRecord, ShadowComparison, ShadowConfig, ShadowSession, ShadowState, StorageBackend,
    };
    pub use crate::train::{
        Checkpoint, EarlyStopping, EpochMetrics, JitAdapter, JitConfig, JitResult, JitStrategy,
        LRSchedule, Optimizer, OptimizerConfig, RegularizationConfig, TrainingCallback,
        TrainingConfig, TrainingLoop, TrainingMetrics, TrainingState, freeze,
    };

    #[cfg(feature = "gpu")]
    pub use crate::train::{GpuTrainer, GpuTrainerBuilder};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_to_end_workflow() {
        // 1. Create training examples
        let examples: Vec<TrainingExample> = (0..100)
            .map(|i| {
                let label = match i % 6 {
                    0 => IntentLabels::continuation(0.8),
                    1 => IntentLabels::topic_shift(0.7),
                    2 => IntentLabels::explicit_query(0.9),
                    3 => IntentLabels::person_lookup(0.85),
                    4 => IntentLabels::health_check(0.75),
                    _ => IntentLabels::task_status(0.8),
                };
                TrainingExample::new(
                    vec![vec![0.1, 0.2, 0.3]; 3], // 3 context messages
                    vec![0.4, 0.5, 0.6],          // current message
                    label,
                )
            })
            .collect();

        // 2. Create dataset
        let mut dataset = Dataset::from_examples(examples);
        let config = DatasetConfig::with_batch_size(16).shuffle(true).seed(42);
        dataset.set_config(config).unwrap();

        let stats = dataset.stats();
        assert_eq!(stats.num_examples, 100);
        assert_eq!(stats.embedding_dim, 3);

        // 3. Configure training
        let train_config = TrainingConfig::quick();
        assert!(train_config.validate().is_ok());

        // 4. Create training loop
        let mut trainer = TrainingLoop::new(train_config).unwrap();

        // 5. Train (using placeholder implementation)
        let metrics = trainer.train(&mut dataset).unwrap();
        assert!(metrics.epochs_completed > 0);

        // 6. Create model for registry
        let metadata = ModelMetadata::classifier(3, 6, 1000)
            .dataset("test_dataset", 100)
            .training_metrics(metrics);

        let model = RegisteredModel::new("intent_classifier", "0.1.0")
            .with_metadata(metadata)
            .with_description("Test model from end-to-end workflow");

        // 7. Register in registry
        let registry = ModelRegistry::in_memory();
        let weights = vec![0u8; 1000];
        let id = registry.register(model, &weights).unwrap();

        // 8. Verify registration
        let loaded = registry.get("intent_classifier", "0.1.0").unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.metadata.num_training_examples, 100);
    }

    #[test]
    fn test_distillation_workflow() {
        // 1. Create raw examples
        let raw = distill::RawExample::new(
            vec!["Hello".to_string(), "How are you?".to_string()],
            "What's the weather like?",
        );

        // 2. Verify prompt generation
        let prompt = raw.to_prompt();
        assert!(prompt.contains("Context"));
        assert!(prompt.contains("weather"));

        // 3. Create teacher config
        let teacher = TeacherConfig::claude_sonnet();
        assert!(teacher.validate().is_ok());

        // 4. Create pipeline
        let mut pipeline = DistillationPipeline::with_teacher(teacher).unwrap();

        // 5. Label (placeholder)
        let result = pipeline.label_single(&raw).unwrap();
        assert!(result.is_success());
        assert!(result.confidence > 0.0);

        // 6. Check stats
        let stats = pipeline.stats();
        assert_eq!(stats.successful, 1);
    }
}
