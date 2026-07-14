#![allow(unused_imports)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::field_reassign_with_default)]

//! Training infrastructure for Lattice neural models.
//!
//! The crate connects data preparation, teacher-label distillation, training,
//! LoRA adaptation, and versioned model registration. The public modules
//! retain those boundaries: [`data`], [`distill`], [`train`], [`lora`], and
//! [`registry`].
//!
//! Optional features add serialization, GPU paths, registry storage, LoRA
//! inference integration, and backward-training support.
//!
//! See [`docs/design.md`](https://github.com/ohdearquant/lattice/blob/main/crates/tune/docs/design.md)
//! for architecture, lifecycle boundaries, and links to the subsystem guides.

#![warn(missing_docs)]

pub mod data;
pub mod distill;
pub mod error;
pub mod lora;
pub mod registry;
pub mod train;

pub use error::{Result, TuneError};

pub use data::{
    Batch, Dataset, DatasetConfig, DatasetStats, ExampleMetadata, IntentLabels, TrainingExample,
};

pub use distill::{
    DistillationConfig, DistillationPipeline, DistillationStats, EndpointSecurity, LabelingResult,
    TeacherConfig, TeacherConfigBuilder, TeacherProvider,
};

pub use train::{
    Checkpoint, EarlyStopping, EpochMetrics, JitAdapter, JitConfig, JitResult, JitStrategy,
    LRSchedule, LoggingCallback, NoOpCallback, Optimizer, OptimizerConfig, RegularizationConfig,
    TrainingCallback, TrainingConfig, TrainingLoop, TrainingMetrics, TrainingState, freeze,
};

#[cfg(feature = "gpu")]
pub use train::{GpuTrainer, GpuTrainerBuilder};

#[cfg(all(feature = "safetensors", feature = "serde"))]
pub use lora::LoadedAdapter;
#[cfg(feature = "serde")]
pub use lora::{AdapterId, LoraManifest, ManifestEntry};
pub use lora::{LoraAdapter, LoraConfig, LoraLayer, blend_lora_adapters};

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
    pub use crate::lora::{LoraAdapter, LoraConfig, LoraLayer, blend_lora_adapters};
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

        let mut dataset = Dataset::from_examples(examples);
        let config = DatasetConfig::with_batch_size(16).shuffle(true).seed(42);
        dataset.set_config(config).unwrap();

        let stats = dataset.stats();
        assert_eq!(stats.num_examples, 100);
        assert_eq!(stats.embedding_dim, 3);

        let train_config = TrainingConfig::quick();
        assert!(train_config.validate().is_ok());

        let mut trainer = TrainingLoop::new(train_config).unwrap();

        // The placeholder loop must fail instead of fabricating metrics.
        let error = trainer.train(&mut dataset).unwrap_err();
        assert!(matches!(
            error,
            TuneError::Training(message) if message.contains("not implemented")
        ));
        let metrics = TrainingMetrics::default();

        let metadata = ModelMetadata::classifier(3, 6, 1000)
            .dataset("test_dataset", 100)
            .training_metrics(metrics);

        let model = RegisteredModel::new("intent_classifier", "0.1.0")
            .with_metadata(metadata)
            .with_description("Test model from end-to-end workflow");

        let registry = ModelRegistry::in_memory();
        let weights = vec![0u8; 1000];
        let id = registry.register(model, &weights).unwrap();

        let loaded = registry.get("intent_classifier", "0.1.0").unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.metadata.num_training_examples, 100);
    }

    #[test]
    fn test_distillation_workflow() {
        let raw = distill::RawExample::new(
            vec!["Hello".to_string(), "How are you?".to_string()],
            "What's the weather like?",
        );

        let prompt = raw.to_prompt();
        assert!(prompt.contains("Context"));
        assert!(prompt.contains("weather"));

        let teacher = TeacherConfig::claude_sonnet();
        assert!(teacher.validate().is_ok());

        let mut pipeline = DistillationPipeline::with_teacher(teacher).unwrap();

        let result = pipeline.label_single(&raw).unwrap();
        assert!(result.is_success());
        assert!(result.confidence > 0.0);

        let stats = pipeline.stats();
        assert_eq!(stats.successful, 1);
    }
}
