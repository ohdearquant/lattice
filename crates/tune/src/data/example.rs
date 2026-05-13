//! Training example types

use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single training example for intent classification
///
/// Contains the input embeddings, soft labels from teacher, and metadata
/// for traceability and debugging.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingExample {
    /// Unique identifier for this example
    pub id: Uuid,

    /// Context embeddings from the last N messages
    ///
    /// Each inner Vec represents an embedding vector for one message.
    /// Order: oldest to newest (chronological).
    pub context_embeddings: Vec<Vec<f32>>,

    /// Embedding of the current message to classify
    pub message_embedding: Vec<f32>,

    /// Soft labels from teacher model
    pub labels: IntentLabels,

    /// Metadata about this example
    pub metadata: ExampleMetadata,
}

impl TrainingExample {
    /// Create a new training example with minimal metadata
    pub fn new(
        context_embeddings: Vec<Vec<f32>>,
        message_embedding: Vec<f32>,
        labels: IntentLabels,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            context_embeddings,
            message_embedding,
            labels,
            metadata: ExampleMetadata::default(),
        }
    }

    /// Create a new training example with specific ID
    pub fn with_id(
        id: Uuid,
        context_embeddings: Vec<Vec<f32>>,
        message_embedding: Vec<f32>,
        labels: IntentLabels,
    ) -> Self {
        Self {
            id,
            context_embeddings,
            message_embedding,
            labels,
            metadata: ExampleMetadata::default(),
        }
    }

    /// Set metadata for this example
    pub fn with_metadata(mut self, metadata: ExampleMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the embedding dimension (from message embedding)
    pub fn embedding_dim(&self) -> usize {
        self.message_embedding.len()
    }

    /// Get the context window size (number of context messages)
    pub fn context_size(&self) -> usize {
        self.context_embeddings.len()
    }

    /// Validate the example structure
    pub fn validate(&self) -> Result<(), String> {
        if self.message_embedding.is_empty() {
            return Err("Message embedding cannot be empty".to_string());
        }

        let dim = self.embedding_dim();
        for (i, ctx_emb) in self.context_embeddings.iter().enumerate() {
            if ctx_emb.len() != dim {
                return Err(format!(
                    "Context embedding {} has dimension {} but expected {}",
                    i,
                    ctx_emb.len(),
                    dim
                ));
            }
        }

        self.labels.validate()?;
        Ok(())
    }

    /// Get the dominant intent (highest probability label)
    pub fn dominant_intent(&self) -> (&'static str, f32) {
        self.labels.dominant()
    }
}

/// Soft labels for intent classification
///
/// Each field represents the probability of that intent class.
/// Values should be in [0, 1] and typically sum to ~1.0 (soft labels may not sum exactly).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IntentLabels {
    /// Probability of continuation (continue conversation naturally)
    pub continuation: f32,

    /// Probability of topic shift (user changing subject)
    pub topic_shift: f32,

    /// Probability of explicit query (direct question/request)
    pub explicit_query: f32,

    /// Probability of person lookup (looking up contact/person info)
    pub person_lookup: f32,

    /// Probability of health check (health/wellness related)
    pub health_check: f32,

    /// Probability of task status (checking task/todo status)
    pub task_status: f32,
}

impl IntentLabels {
    /// Create labels with dominant continuation intent
    pub fn continuation(prob: f32) -> Self {
        Self {
            continuation: prob,
            ..Default::default()
        }
    }

    /// Create labels with dominant topic_shift intent
    pub fn topic_shift(prob: f32) -> Self {
        Self {
            topic_shift: prob,
            ..Default::default()
        }
    }

    /// Create labels with dominant explicit_query intent
    pub fn explicit_query(prob: f32) -> Self {
        Self {
            explicit_query: prob,
            ..Default::default()
        }
    }

    /// Create labels with dominant person_lookup intent
    pub fn person_lookup(prob: f32) -> Self {
        Self {
            person_lookup: prob,
            ..Default::default()
        }
    }

    /// Create labels with dominant health_check intent
    pub fn health_check(prob: f32) -> Self {
        Self {
            health_check: prob,
            ..Default::default()
        }
    }

    /// Create labels with dominant task_status intent
    pub fn task_status(prob: f32) -> Self {
        Self {
            task_status: prob,
            ..Default::default()
        }
    }

    /// Create labels from a probability vector
    ///
    /// Order: [continuation, topic_shift, explicit_query, person_lookup, health_check, task_status]
    pub fn from_vec(probs: &[f32]) -> Self {
        Self {
            continuation: probs.first().copied().unwrap_or(0.0),
            topic_shift: probs.get(1).copied().unwrap_or(0.0),
            explicit_query: probs.get(2).copied().unwrap_or(0.0),
            person_lookup: probs.get(3).copied().unwrap_or(0.0),
            health_check: probs.get(4).copied().unwrap_or(0.0),
            task_status: probs.get(5).copied().unwrap_or(0.0),
        }
    }

    /// Convert to probability vector
    ///
    /// Order: [continuation, topic_shift, explicit_query, person_lookup, health_check, task_status]
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.continuation,
            self.topic_shift,
            self.explicit_query,
            self.person_lookup,
            self.health_check,
            self.task_status,
        ]
    }

    /// Number of intent classes
    pub const NUM_CLASSES: usize = 6;

    /// Get all intent names
    pub fn class_names() -> &'static [&'static str] {
        &[
            "continuation",
            "topic_shift",
            "explicit_query",
            "person_lookup",
            "health_check",
            "task_status",
        ]
    }

    /// Get the dominant intent (highest probability)
    pub fn dominant(&self) -> (&'static str, f32) {
        let probs = self.to_vec();
        let names = Self::class_names();
        let (idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));
        (names[idx], prob)
    }

    /// Validate that all probabilities are in [0, 1]
    pub fn validate(&self) -> Result<(), String> {
        let probs = self.to_vec();
        for (i, &p) in probs.iter().enumerate() {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!(
                    "Invalid probability for {}: {} (must be in [0, 1])",
                    Self::class_names()[i],
                    p
                ));
            }
        }
        Ok(())
    }

    /// Apply softmax normalization.
    /// Returns an error if any input is non-finite.
    pub fn softmax_normalize(&mut self) -> Result<(), String> {
        let probs = self.to_vec();
        if let Some(pos) = probs.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "non-finite value {} at index {} in softmax input",
                probs[pos], pos
            ));
        }
        let max_val = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = probs.iter().map(|&p| (p - max_val).exp()).sum();

        self.continuation = ((self.continuation - max_val).exp()) / exp_sum;
        self.topic_shift = ((self.topic_shift - max_val).exp()) / exp_sum;
        self.explicit_query = ((self.explicit_query - max_val).exp()) / exp_sum;
        self.person_lookup = ((self.person_lookup - max_val).exp()) / exp_sum;
        self.health_check = ((self.health_check - max_val).exp()) / exp_sum;
        self.task_status = ((self.task_status - max_val).exp()) / exp_sum;
        Ok(())
    }
}

/// Metadata about a training example
///
/// Provides traceability back to the source data.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExampleMetadata {
    /// Source conversation or session ID
    pub source_id: Option<String>,

    /// Timestamp when the original message was created
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,

    /// Teacher model that generated the labels
    pub teacher_model: Option<String>,

    /// Teacher generation timestamp
    pub labeled_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Confidence of the teacher's labels (0-1)
    pub teacher_confidence: Option<f32>,

    /// Additional metadata as key-value pairs
    #[cfg(feature = "serde")]
    pub extra: Option<serde_json::Value>,

    /// Additional metadata as key-value pairs (non-serde fallback)
    #[cfg(not(feature = "serde"))]
    pub extra: Option<String>,
}

impl ExampleMetadata {
    /// Create metadata with source ID
    pub fn with_source(source_id: impl Into<String>) -> Self {
        Self {
            source_id: Some(source_id.into()),
            ..Default::default()
        }
    }

    /// Set the teacher model
    pub fn teacher(mut self, model: impl Into<String>) -> Self {
        self.teacher_model = Some(model.into());
        self
    }

    /// Set the timestamp
    pub fn timestamp(mut self, ts: chrono::DateTime<chrono::Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Set the labeled_at timestamp
    pub fn labeled_at(mut self, ts: chrono::DateTime<chrono::Utc>) -> Self {
        self.labeled_at = Some(ts);
        self
    }

    /// Set teacher confidence
    pub fn confidence(mut self, conf: f32) -> Self {
        self.teacher_confidence = Some(conf);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_labels_creation() {
        let labels = IntentLabels::continuation(0.8);
        assert_eq!(labels.continuation, 0.8);
        assert_eq!(labels.topic_shift, 0.0);
    }

    #[test]
    fn test_intent_labels_dominant() {
        let labels = IntentLabels {
            continuation: 0.1,
            topic_shift: 0.2,
            explicit_query: 0.5,
            person_lookup: 0.1,
            health_check: 0.05,
            task_status: 0.05,
        };
        let (name, prob) = labels.dominant();
        assert_eq!(name, "explicit_query");
        assert_eq!(prob, 0.5);
    }

    #[test]
    fn test_intent_labels_validation() {
        let valid = IntentLabels::continuation(0.8);
        assert!(valid.validate().is_ok());

        let invalid = IntentLabels {
            continuation: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_training_example_creation() {
        let example = TrainingExample::new(
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            vec![0.7, 0.8, 0.9],
            IntentLabels::explicit_query(0.9),
        );

        assert_eq!(example.embedding_dim(), 3);
        assert_eq!(example.context_size(), 2);
        assert!(example.validate().is_ok());
    }

    #[test]
    fn test_training_example_validation() {
        // Dimension mismatch should fail
        let example = TrainingExample::new(
            vec![vec![0.1, 0.2]], // 2D
            vec![0.7, 0.8, 0.9],  // 3D - mismatch!
            IntentLabels::default(),
        );

        assert!(example.validate().is_err());
    }

    #[test]
    fn test_softmax_normalize() {
        let mut labels = IntentLabels {
            continuation: 2.0,
            topic_shift: 1.0,
            explicit_query: 0.5,
            person_lookup: 0.0,
            health_check: 0.0,
            task_status: 0.0,
        };
        labels.softmax_normalize().expect("test inputs are finite");

        // Sum should be approximately 1.0
        let sum: f32 = labels.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // continuation should have highest probability
        let (name, _) = labels.dominant();
        assert_eq!(name, "continuation");
    }
}
