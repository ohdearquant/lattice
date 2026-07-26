//! Orchestration from raw prompts to labeled examples.
//!
//! This implementation validates configuration and records results. Live
//! teacher transport is not yet configured, so normal labeling fails closed.
//! A fixed-label path is available only through the `simulated-teacher` feature.

use super::DistillationConfig;
use super::types::{DistillationStats, LabelSource, LabelingResult, RawExample};
#[cfg(feature = "simulated-teacher")]
use crate::data::IntentLabels;
use crate::data::{ExampleMetadata, TrainingExample};
use crate::distill::teacher::TeacherConfig;
use crate::error::{Result, TuneError};
use chrono::Utc;

/// Distillation pipeline that orchestrates labeling from teacher models.
///
/// Live provider transport is not configured yet. [`Self::label_single`] and
/// [`Self::label_batch`] fail closed instead of producing fabricated labels.
pub struct DistillationPipeline {
    /// Teacher model configuration
    teacher: TeacherConfig,

    /// Pipeline configuration
    config: DistillationConfig,

    /// Statistics
    stats: DistillationStats,
}

impl DistillationPipeline {
    /// Create a new distillation pipeline
    pub fn new(teacher: TeacherConfig, config: DistillationConfig) -> Result<Self> {
        teacher.validate().map_err(TuneError::InvalidConfig)?;
        config.validate()?;

        Ok(Self {
            teacher,
            config,
            stats: DistillationStats::default(),
        })
    }

    /// Create with default configuration
    pub fn with_teacher(teacher: TeacherConfig) -> Result<Self> {
        Self::new(teacher, DistillationConfig::default())
    }

    /// Get the teacher configuration
    pub fn teacher(&self) -> &TeacherConfig {
        &self.teacher
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Get current statistics
    pub fn stats(&self) -> &DistillationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DistillationStats::default();
    }

    /// Label a single raw example using a live teacher.
    ///
    /// Returns [`TuneError::TeacherApi`] until a live provider transport is
    /// configured. It never falls back to simulated labels.
    pub fn label_single(&mut self, _raw: &RawExample) -> Result<LabelingResult> {
        Err(TuneError::TeacherApi(
            "live teacher transport is not configured".to_string(),
        ))
    }

    /// Label a single raw example with deterministic simulated output.
    ///
    /// Produces a fixed label distribution independent of `raw`'s content —
    /// it does not call [`RawExample::to_prompt`] or otherwise inspect the
    /// message/context. This explicit test-only path is available with the
    /// non-default `simulated-teacher` feature and never affects
    /// [`Self::label_single`].
    #[cfg(feature = "simulated-teacher")]
    pub fn label_single_simulated(&mut self, raw: &RawExample) -> Result<LabelingResult> {
        let result = self.simulate_single(raw)?;
        self.stats.update(&result);
        Ok(result)
    }

    /// Core simulated-labeling computation shared by
    /// [`Self::label_single_simulated`] and [`Self::label_batch_simulated`]
    /// (via [`Self::label_batch_with`]). Does not update `self.stats` for a
    /// successful result — the caller owns that accounting, so a result is
    /// counted exactly once whether it comes from the single-item API or a
    /// batch.
    #[cfg(feature = "simulated-teacher")]
    fn simulate_single(&mut self, raw: &RawExample) -> Result<LabelingResult> {
        let start = std::time::Instant::now();

        let mut labels = IntentLabels {
            continuation: 0.4,
            topic_shift: 0.1,
            explicit_query: 0.3,
            person_lookup: 0.1,
            health_check: 0.05,
            task_status: 0.05,
        };

        if self.config.normalize_labels {
            labels
                .softmax_normalize()
                .map_err(TuneError::InvalidConfig)?;
        }

        let confidence = 0.85;
        let latency_ms = start.elapsed().as_millis() as u64;

        // Check confidence threshold
        if let Some(min_conf) = self.config.min_confidence
            && confidence < min_conf
        {
            self.stats.skipped += 1;
            return Err(TuneError::Validation(format!(
                "Confidence {confidence} below threshold {min_conf}"
            )));
        }

        Ok(LabelingResult::simulated(
            raw.id, labels, confidence, latency_ms,
        ))
    }

    /// Label a batch of raw examples
    ///
    /// Returns results for all examples, including failures.
    pub fn label_batch(&mut self, raws: &[RawExample]) -> Vec<LabelingResult> {
        self.label_batch_with(raws, Self::label_single)
    }

    /// Shared batch loop: labels each example with `label_one`, converting
    /// errors into failed [`LabelingResult`]s. Owns statistics accounting for
    /// both branches — every result, success or failure, is counted exactly
    /// once here, so a new `label_one` implementation can never under-count
    /// or double-count by forgetting to touch `self.stats` itself.
    fn label_batch_with(
        &mut self,
        raws: &[RawExample],
        mut label_one: impl FnMut(&mut Self, &RawExample) -> Result<LabelingResult>,
    ) -> Vec<LabelingResult> {
        let mut results = Vec::with_capacity(raws.len());

        for raw in raws {
            let result = match label_one(self, raw) {
                Ok(r) => r,
                Err(e) => LabelingResult::failure(raw.id, e.to_string(), 0),
            };
            self.stats.update(&result);
            results.push(result);
        }

        results
    }

    /// Label a batch with deterministic simulated output.
    ///
    /// This explicit test-only path is available with the non-default
    /// `simulated-teacher` feature and never affects [`Self::label_batch`].
    #[cfg(feature = "simulated-teacher")]
    pub fn label_batch_simulated(&mut self, raws: &[RawExample]) -> Vec<LabelingResult> {
        self.label_batch_with(raws, Self::simulate_single)
    }

    /// Convert labeled results to training examples
    ///
    /// Requires embeddings to be provided separately.
    pub fn to_training_examples(
        &self,
        results: &[LabelingResult],
        context_embeddings: &[Vec<Vec<f32>>],
        message_embeddings: &[Vec<f32>],
    ) -> Result<Vec<TrainingExample>> {
        if results.len() != context_embeddings.len() || results.len() != message_embeddings.len() {
            return Err(TuneError::DimensionMismatch {
                expected: results.len(),
                actual: context_embeddings.len(),
            });
        }

        let eligible = results
            .iter()
            .filter(|result| result.is_success() && result.source() != LabelSource::Simulated)
            .count();
        let mut examples = Vec::with_capacity(eligible);

        for (i, result) in results.iter().enumerate() {
            if !result.is_success() {
                continue;
            }
            // Simulated labels must never be attributed to the configured
            // teacher in a training dataset — reject them here rather than
            // stamp `self.teacher.display_name()` on fabricated output.
            if result.source() == LabelSource::Simulated {
                continue;
            }

            let mut example = TrainingExample::with_id(
                result.example_id,
                context_embeddings[i].clone(),
                message_embeddings[i].clone(),
                result.labels().clone(),
            );

            // Add metadata about the labeling
            let metadata = ExampleMetadata::with_source(result.example_id.to_string())
                .teacher(self.teacher.display_name())
                .labeled_at(Utc::now())
                .confidence(result.confidence());

            example = example.with_metadata(metadata);
            examples.push(example);
        }

        Ok(examples)
    }
}
