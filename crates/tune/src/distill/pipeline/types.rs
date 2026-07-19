//! Inputs, outputs, and accounting for the distillation pipeline.
//!
//! `RawExample` sanitizes and formats teacher input; `LabelingResult` carries
//! one outcome, and `DistillationStats` aggregates outcomes. See
//! `docs/distill.md` for prompt limits, result semantics, and statistics rules.

use crate::data::{ExampleMetadata, IntentLabels};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Provenance of a labeling result: did it come from the configured teacher,
/// or from the opt-in deterministic simulation path?
///
/// [`DistillationPipeline::to_training_examples`](super::DistillationPipeline::to_training_examples)
/// uses this to keep simulated labels from being attributed to the real teacher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LabelSource {
    /// Produced by the configured teacher.
    #[default]
    Teacher,
    /// Produced by the deterministic `simulated-teacher` path. Never a real
    /// teacher's output.
    Simulated,
}

/// Capability proving a [`LabelingResult`] was produced by this crate's own
/// teacher-provider path, not fabricated by an external caller.
///
/// `TeacherAuth` has a private field and no public constructor, so code
/// outside `lattice_tune` can never obtain one — and [`LabelingResult::success`]
/// requires one to construct a `Teacher`-sourced result. This closes the gap
/// a private `source` field alone leaves open: without an unforgeable token
/// gating the constructor itself, any caller could still mint a `Teacher`
/// result through the public API with arbitrary labels and confidence.
#[derive(Debug, Clone, Copy)]
pub struct TeacherAuth(());

impl TeacherAuth {
    /// Mint a new capability. Only reachable from within this crate — the
    /// live teacher-provider path is the intended (future) caller. Unused in
    /// production code today because [`DistillationPipeline::label_single`](super::DistillationPipeline::label_single)
    /// fails closed rather than calling a live teacher; tests exercise it in
    /// the meantime.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self(())
    }
}

/// Result of labeling a single example.
///
/// Every field that [`Self::is_success`], [`Self::source`], or
/// [`super::DistillationPipeline::to_training_examples`] reads is private:
/// the only ways to reach a value are [`Self::success`], [`Self::simulated`],
/// and [`Self::failure`]. There is no public setter, so a caller cannot
/// mutate a failed result into an accepted one (clearing `error` and
/// swapping in `labels`) or otherwise disagree with the state a constructor
/// established:
///
/// ```compile_fail
/// # use lattice_tune::LabelingResult;
/// # use uuid::Uuid;
/// let mut result = LabelingResult::failure(Uuid::new_v4(), "boom", 0);
/// result.error = None; // private field — does not compile
/// ```
///
/// `LabelingResult` implements `Serialize` but deliberately never
/// `Deserialize`. A deserialized payload cannot be authenticated as having
/// come from the configured teacher, so there is no path — silent or
/// otherwise — from untrusted bytes back to this type; the only route is to
/// call [`Self::success`], [`Self::simulated`], or [`Self::failure`] again,
/// in-process:
///
/// ```compile_fail
/// # use lattice_tune::LabelingResult;
/// let _: LabelingResult = serde_json::from_str("{}").unwrap();
/// ```
///
/// A caller outside this crate also cannot mint a `Teacher`-sourced result
/// at all: [`Self::success`] requires a [`TeacherAuth`] token, and
/// `TeacherAuth` has no public constructor. There is no `TeacherAuth` value
/// an external caller can pass, so the call site itself does not compile:
///
/// ```compile_fail
/// # use lattice_tune::{IntentLabels, LabelingResult};
/// # use uuid::Uuid;
/// let _ = LabelingResult::success(Uuid::new_v4(), IntentLabels::default(), 0.99, 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct LabelingResult {
    /// Original example ID
    pub example_id: Uuid,

    /// Generated labels
    labels: IntentLabels,

    /// Confidence score, from either the teacher or the simulated path —
    /// see [`Self::source`] for which one produced this result.
    confidence: f32,

    /// Raw response, for debugging — populated for teacher results; absent
    /// for simulated results, which never call the teacher.
    raw_response: Option<String>,

    /// Error message if labeling failed
    error: Option<String>,

    /// Latency in milliseconds
    latency_ms: u64,

    /// Where this result's labels came from.
    ///
    /// Visibility is intentionally narrower than the other fields: provenance
    /// must only ever be set by [`LabelingResult::success`] or
    /// [`LabelingResult::simulated`] at construction time, never mutated
    /// afterward. Read it via [`Self::source`].
    pub(super) source: LabelSource,
}

impl LabelingResult {
    fn new_success(
        example_id: Uuid,
        labels: IntentLabels,
        confidence: f32,
        latency_ms: u64,
        source: LabelSource,
    ) -> Self {
        Self {
            example_id,
            labels,
            confidence,
            raw_response: None,
            error: None,
            latency_ms,
            source,
        }
    }

    /// Create a successful result attributed to the configured teacher.
    ///
    /// Requires a [`TeacherAuth`] token, which only this crate's own
    /// teacher-provider path can construct — an external caller cannot
    /// obtain one, so this constructor cannot be used to fabricate
    /// `Teacher`-sourced labels.
    pub fn success(
        example_id: Uuid,
        labels: IntentLabels,
        confidence: f32,
        latency_ms: u64,
        _auth: TeacherAuth,
    ) -> Self {
        Self::new_success(
            example_id,
            labels,
            confidence,
            latency_ms,
            LabelSource::Teacher,
        )
    }

    /// Create a successful result from the deterministic simulation path.
    /// Never attributable to the configured teacher.
    pub fn simulated(
        example_id: Uuid,
        labels: IntentLabels,
        confidence: f32,
        latency_ms: u64,
    ) -> Self {
        Self::new_success(
            example_id,
            labels,
            confidence,
            latency_ms,
            LabelSource::Simulated,
        )
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
            source: LabelSource::Teacher,
        }
    }

    /// Check if labeling succeeded
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    /// Where this result's labels came from. `source` itself is not settable
    /// from outside this module — provenance is fixed at construction by
    /// [`Self::success`] or [`Self::simulated`].
    pub fn source(&self) -> LabelSource {
        self.source
    }

    /// Generated labels.
    pub fn labels(&self) -> &IntentLabels {
        &self.labels
    }

    /// Confidence score, from either the teacher or the simulated path —
    /// see [`Self::source`] for which one produced this result.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Raw response, for debugging — populated for teacher results; absent
    /// for simulated results, which never call the teacher.
    pub fn raw_response(&self) -> Option<&str> {
        self.raw_response.as_deref()
    }

    /// Error message if labeling failed.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Latency in milliseconds.
    pub fn latency_ms(&self) -> u64 {
        self.latency_ms
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

    /// Format the context and current message as a bounded, sanitized teacher prompt.
    /// See [`docs/distill.md`](../../../docs/distill.md#rawexampleto_prompt) for its layout and limits.
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

    /// Sanitize user input before embedding it in a prompt.
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
