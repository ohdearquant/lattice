//! Pipeline types: labeling results, statistics, and raw examples.

use crate::data::{ExampleMetadata, IntentLabels};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of labeling a single example
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LabelingResult {
    /// Original example ID
    pub example_id: Uuid,

    /// Generated labels
    pub labels: IntentLabels,

    /// Confidence score from teacher
    pub confidence: f32,

    /// Raw response from teacher (for debugging)
    pub raw_response: Option<String>,

    /// Error message if labeling failed
    pub error: Option<String>,

    /// Latency in milliseconds
    pub latency_ms: u64,
}

impl LabelingResult {
    /// Create a successful result
    pub fn success(
        example_id: Uuid,
        labels: IntentLabels,
        confidence: f32,
        latency_ms: u64,
    ) -> Self {
        Self {
            example_id,
            labels,
            confidence,
            raw_response: None,
            error: None,
            latency_ms,
        }
    }

    /// Create a failed result
    pub fn failure(example_id: Uuid, error: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            example_id,
            labels: IntentLabels::default(),
            confidence: 0.0,
            raw_response: None,
            error: Some(error.into()),
            latency_ms,
        }
    }

    /// Check if labeling succeeded
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    /// Set raw response
    pub fn with_raw_response(mut self, response: impl Into<String>) -> Self {
        self.raw_response = Some(response.into());
        self
    }
}

/// Statistics from distillation
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistillationStats {
    /// Total examples processed
    pub total_processed: usize,

    /// Successfully labeled
    pub successful: usize,

    /// Failed to label
    pub failed: usize,

    /// Skipped (below confidence threshold)
    pub skipped: usize,

    /// Total latency in milliseconds
    pub total_latency_ms: u64,

    /// Average latency per example
    pub avg_latency_ms: f64,

    /// Average confidence score
    pub avg_confidence: f32,

    /// Label distribution
    pub label_distribution: Vec<usize>,
}

impl DistillationStats {
    /// Success rate as a fraction
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            return 0.0;
        }
        self.successful as f64 / self.total_processed as f64
    }

    /// Update stats with a result
    pub fn update(&mut self, result: &LabelingResult) {
        self.total_processed += 1;
        self.total_latency_ms += result.latency_ms;

        if result.is_success() {
            self.successful += 1;
            self.avg_confidence = (self.avg_confidence * (self.successful - 1) as f32
                + result.confidence)
                / self.successful as f32;

            // Update label distribution
            if self.label_distribution.is_empty() {
                self.label_distribution = vec![0; IntentLabels::NUM_CLASSES];
            }
            let (name, _) = result.labels.dominant();
            let idx = IntentLabels::class_names()
                .iter()
                .position(|&n| n == name)
                .unwrap_or(0);
            self.label_distribution[idx] += 1;
        } else {
            self.failed += 1;
        }

        if self.total_processed > 0 {
            self.avg_latency_ms = self.total_latency_ms as f64 / self.total_processed as f64;
        }
    }
}

/// Maximum allowed length for a single message (in characters).
/// Prevents excessive memory usage and potential DoS.
pub const MAX_MESSAGE_LENGTH: usize = 10_000;

/// Maximum total prompt length after formatting.
/// Prevents sending excessively large prompts to teacher models.
pub const MAX_PROMPT_LENGTH: usize = 50_000;

/// Raw input for labeling (before embeddings)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RawExample {
    /// Unique identifier
    pub id: Uuid,

    /// Context messages (oldest to newest)
    pub context: Vec<String>,

    /// Current message to classify
    pub message: String,

    /// Optional metadata
    pub metadata: Option<ExampleMetadata>,
}

impl RawExample {
    /// Create a new raw example
    pub fn new(context: Vec<String>, message: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            context,
            message: message.into(),
            metadata: None,
        }
    }

    /// Create with specific ID
    pub fn with_id(id: Uuid, context: Vec<String>, message: impl Into<String>) -> Self {
        Self {
            id,
            context,
            message: message.into(),
            metadata: None,
        }
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ExampleMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Format for teacher prompt with input sanitization.
    ///
    /// Applies the following sanitization:
    /// - Strips control characters (except newlines and tabs)
    /// - Truncates individual messages to [`MAX_MESSAGE_LENGTH`]
    /// - Truncates total prompt to [`MAX_PROMPT_LENGTH`]
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();

        if !self.context.is_empty() {
            prompt.push_str("Context (previous messages):\n");
            for (i, msg) in self.context.iter().enumerate() {
                let sanitized = Self::sanitize_input(msg);
                prompt.push_str(&format!("{}. {}\n", i + 1, sanitized));
            }
            prompt.push('\n');
        }

        let sanitized_message = Self::sanitize_input(&self.message);
        prompt.push_str(&format!(
            "Current message to classify:\n{sanitized_message}"
        ));

        // Truncate total prompt if needed
        if prompt.len() > MAX_PROMPT_LENGTH {
            prompt.truncate(MAX_PROMPT_LENGTH);
            prompt.push_str("\n[truncated]");
        }

        prompt
    }

    /// Sanitize user input to prevent prompt injection.
    ///
    /// - Strips control characters (except \n, \t, \r)
    /// - Truncates to [`MAX_MESSAGE_LENGTH`]
    fn sanitize_input(input: &str) -> String {
        // Truncate first to avoid processing huge strings
        let truncated = if input.len() > MAX_MESSAGE_LENGTH {
            let boundary = input
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= MAX_MESSAGE_LENGTH)
                .last()
                .unwrap_or(0);
            &input[..boundary]
        } else {
            input
        };

        // Remove control characters except newlines, tabs, carriage returns
        truncated
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t' || *c == '\r')
            .collect()
    }
}
