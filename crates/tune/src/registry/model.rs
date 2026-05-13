//! Registered model types

use crate::train::TrainingMetrics;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Model status in the registry
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum ModelStatus {
    /// Model is registered but not yet validated
    #[default]
    Pending,

    /// Model has passed validation
    Validated,

    /// Model is staged for deployment
    Staged,

    /// Model is in production
    Production,

    /// Model has been archived
    Archived,

    /// Model has been deprecated
    Deprecated,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelStatus::Pending => write!(f, "pending"),
            ModelStatus::Validated => write!(f, "validated"),
            ModelStatus::Staged => write!(f, "staged"),
            ModelStatus::Production => write!(f, "production"),
            ModelStatus::Archived => write!(f, "archived"),
            ModelStatus::Deprecated => write!(f, "deprecated"),
        }
    }
}

/// Metadata about a registered model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelMetadata {
    /// Model architecture description
    pub architecture: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension (number of classes)
    pub output_dim: usize,

    /// Number of parameters
    pub num_parameters: usize,

    /// Training configuration hash (for reproducibility)
    pub config_hash: Option<String>,

    /// Training dataset ID
    pub dataset_id: Option<String>,

    /// Number of training examples
    pub num_training_examples: usize,

    /// Training metrics summary
    pub training_metrics: Option<TrainingMetrics>,

    /// Validation accuracy
    pub validation_accuracy: Option<f32>,

    /// Test accuracy (if evaluated)
    pub test_accuracy: Option<f32>,

    /// Custom tags
    pub tags: Vec<String>,

    /// Additional properties
    #[cfg(feature = "serde")]
    pub extra: Option<serde_json::Value>,

    /// Additional properties (non-serde fallback)
    #[cfg(not(feature = "serde"))]
    pub extra: Option<String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            input_dim: 0,
            output_dim: 0,
            num_parameters: 0,
            config_hash: None,
            dataset_id: None,
            num_training_examples: 0,
            training_metrics: None,
            validation_accuracy: None,
            test_accuracy: None,
            tags: Vec::new(),
            extra: None,
        }
    }
}

impl ModelMetadata {
    /// Create metadata for a classification model
    pub fn classifier(input_dim: usize, output_dim: usize, num_parameters: usize) -> Self {
        Self {
            architecture: format!("classifier_{input_dim}x{output_dim}"),
            input_dim,
            output_dim,
            num_parameters,
            ..Default::default()
        }
    }

    /// Set architecture description
    pub fn architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = arch.into();
        self
    }

    /// Set dataset info
    pub fn dataset(mut self, id: impl Into<String>, num_examples: usize) -> Self {
        self.dataset_id = Some(id.into());
        self.num_training_examples = num_examples;
        self
    }

    /// Set training metrics
    pub fn training_metrics(mut self, metrics: TrainingMetrics) -> Self {
        self.validation_accuracy = metrics.final_val_loss.map(|_| {
            metrics
                .history
                .last()
                .and_then(|m| m.val_accuracy)
                .unwrap_or(0.0)
        });
        self.training_metrics = Some(metrics);
        self
    }

    /// Add a tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// A registered model with versioning
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RegisteredModel {
    /// Unique model ID
    pub id: Uuid,

    /// Model name (e.g., "intent_classifier")
    pub name: String,

    /// Semantic version (e.g., "1.2.3")
    pub version: String,

    /// Model status
    pub status: ModelStatus,

    /// Model metadata
    pub metadata: ModelMetadata,

    /// When the model was registered
    pub registered_at: DateTime<Utc>,

    /// When the model was last updated
    pub updated_at: DateTime<Utc>,

    /// User/system that registered the model
    pub registered_by: Option<String>,

    /// Description
    pub description: Option<String>,

    /// Path to model weights (relative to registry root)
    pub weights_path: Option<String>,

    /// Size of weights file in bytes
    pub weights_size: Option<usize>,

    /// SHA256 hash of weights file
    pub weights_hash: Option<String>,

    /// Parent model (if fine-tuned from another)
    pub parent_id: Option<Uuid>,

    /// Child models (fine-tuned from this)
    pub children: Vec<Uuid>,
}

impl RegisteredModel {
    /// Create a new registered model
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            version: version.into(),
            status: ModelStatus::Pending,
            metadata: ModelMetadata::default(),
            registered_at: now,
            updated_at: now,
            registered_by: None,
            description: None,
            weights_path: None,
            weights_size: None,
            weights_hash: None,
            parent_id: None,
            children: Vec::new(),
        }
    }

    /// Create with specific ID
    pub fn with_id(id: Uuid, name: impl Into<String>, version: impl Into<String>) -> Self {
        let mut model = Self::new(name, version);
        model.id = id;
        model
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self.updated_at = Utc::now();
        self
    }

    /// Set status
    pub fn with_status(mut self, status: ModelStatus) -> Self {
        self.status = status;
        self.updated_at = Utc::now();
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self.updated_at = Utc::now();
        self
    }

    /// Set registered by
    pub fn registered_by(mut self, user: impl Into<String>) -> Self {
        self.registered_by = Some(user.into());
        self
    }

    /// Set parent model
    pub fn with_parent(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self.updated_at = Utc::now();
        self
    }

    /// Set weights information
    pub fn with_weights(
        mut self,
        path: impl Into<String>,
        size: usize,
        hash: impl Into<String>,
    ) -> Self {
        self.weights_path = Some(path.into());
        self.weights_size = Some(size);
        self.weights_hash = Some(hash.into());
        self.updated_at = Utc::now();
        self
    }

    /// Get full model identifier
    pub fn full_name(&self) -> String {
        format!("{}:{}", self.name, self.version)
    }

    /// Parse version components (major, minor, patch)
    pub fn version_tuple(&self) -> Option<(u32, u32, u32)> {
        let parts: Vec<&str> = self.version.split('.').collect();
        if parts.len() == 3 {
            let major = parts[0].parse().ok()?;
            let minor = parts[1].parse().ok()?;
            let patch = parts[2].parse().ok()?;
            Some((major, minor, patch))
        } else {
            None
        }
    }

    /// Check if this version is newer than another
    pub fn is_newer_than(&self, other: &RegisteredModel) -> Option<bool> {
        let self_ver = self.version_tuple()?;
        let other_ver = other.version_tuple()?;
        Some(self_ver > other_ver)
    }

    /// Validate the model
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        if self.version.is_empty() {
            return Err("Model version cannot be empty".to_string());
        }
        Ok(())
    }

    /// Mark as validated
    pub fn mark_validated(&mut self) {
        self.status = ModelStatus::Validated;
        self.updated_at = Utc::now();
    }

    /// Mark as staged
    pub fn mark_staged(&mut self) {
        self.status = ModelStatus::Staged;
        self.updated_at = Utc::now();
    }

    /// Mark as production
    pub fn mark_production(&mut self) {
        self.status = ModelStatus::Production;
        self.updated_at = Utc::now();
    }

    /// Mark as archived
    pub fn mark_archived(&mut self) {
        self.status = ModelStatus::Archived;
        self.updated_at = Utc::now();
    }

    /// Mark as deprecated
    pub fn mark_deprecated(&mut self) {
        self.status = ModelStatus::Deprecated;
        self.updated_at = Utc::now();
    }

    /// Check if model is deployable (validated or staged)
    pub fn is_deployable(&self) -> bool {
        matches!(
            self.status,
            ModelStatus::Validated | ModelStatus::Staged | ModelStatus::Production
        )
    }

    /// Check if model is active (not archived or deprecated)
    pub fn is_active(&self) -> bool {
        !matches!(self.status, ModelStatus::Archived | ModelStatus::Deprecated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = RegisteredModel::new("intent_classifier", "1.0.0");

        assert_eq!(model.name, "intent_classifier");
        assert_eq!(model.version, "1.0.0");
        assert_eq!(model.status, ModelStatus::Pending);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_builder() {
        let metadata = ModelMetadata::classifier(768, 6, 1000);
        let model = RegisteredModel::new("intent_classifier", "1.0.0")
            .with_metadata(metadata)
            .with_description("Test model")
            .registered_by("test_user");

        assert_eq!(model.description, Some("Test model".to_string()));
        assert_eq!(model.registered_by, Some("test_user".to_string()));
        assert_eq!(model.metadata.input_dim, 768);
    }

    #[test]
    fn test_model_status_transitions() {
        let mut model = RegisteredModel::new("test", "1.0.0");

        assert_eq!(model.status, ModelStatus::Pending);
        assert!(!model.is_deployable());

        model.mark_validated();
        assert_eq!(model.status, ModelStatus::Validated);
        assert!(model.is_deployable());

        model.mark_staged();
        assert_eq!(model.status, ModelStatus::Staged);

        model.mark_production();
        assert_eq!(model.status, ModelStatus::Production);

        model.mark_deprecated();
        assert_eq!(model.status, ModelStatus::Deprecated);
        assert!(!model.is_active());
    }

    #[test]
    fn test_version_parsing() {
        let model = RegisteredModel::new("test", "1.2.3");
        assert_eq!(model.version_tuple(), Some((1, 2, 3)));

        let model2 = RegisteredModel::new("test", "invalid");
        assert_eq!(model2.version_tuple(), None);
    }

    #[test]
    fn test_version_comparison() {
        let older = RegisteredModel::new("test", "1.0.0");
        let newer = RegisteredModel::new("test", "1.1.0");

        assert_eq!(newer.is_newer_than(&older), Some(true));
        assert_eq!(older.is_newer_than(&newer), Some(false));
    }

    #[test]
    fn test_full_name() {
        let model = RegisteredModel::new("intent_classifier", "1.0.0");
        assert_eq!(model.full_name(), "intent_classifier:1.0.0");
    }

    #[test]
    fn test_model_metadata() {
        let metadata = ModelMetadata::classifier(768, 6, 10000)
            .architecture("MLP(768, 256, 6)")
            .dataset("train_v1", 50000)
            .tag("production")
            .tag("intent");

        assert_eq!(metadata.input_dim, 768);
        assert_eq!(metadata.output_dim, 6);
        assert_eq!(metadata.num_training_examples, 50000);
        assert_eq!(metadata.tags.len(), 2);
    }

    #[test]
    fn test_model_with_weights() {
        let model = RegisteredModel::new("test", "1.0.0").with_weights(
            "models/test/1.0.0/weights.bin",
            1024 * 1024,
            "abc123",
        );

        assert_eq!(
            model.weights_path,
            Some("models/test/1.0.0/weights.bin".to_string())
        );
        assert_eq!(model.weights_size, Some(1024 * 1024));
        assert_eq!(model.weights_hash, Some("abc123".to_string()));
    }

    #[test]
    fn test_model_with_parent() {
        let parent_id = Uuid::new_v4();
        let model = RegisteredModel::new("test", "1.0.0").with_parent(parent_id);

        assert_eq!(model.parent_id, Some(parent_id));
    }
}
