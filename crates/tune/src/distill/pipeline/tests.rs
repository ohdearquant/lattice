use super::types::{MAX_MESSAGE_LENGTH, MAX_PROMPT_LENGTH};
use super::*;
use crate::data::IntentLabels;
use crate::distill::teacher::TeacherConfig;
use uuid::Uuid;

#[test]
fn test_distillation_config_defaults() {
    let config = DistillationConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.batch_size, 10);
    assert_eq!(config.concurrency, 5);
}

#[test]
fn test_distillation_config_presets() {
    let fast = DistillationConfig::fast();
    assert!(fast.validate().is_ok());
    assert!(fast.batch_size > DistillationConfig::default().batch_size);

    let quality = DistillationConfig::quality();
    assert!(quality.validate().is_ok());
    assert!(quality.min_confidence.is_some());
}

#[test]
fn test_raw_example_prompt() {
    let raw = RawExample::new(
        vec!["Hello".to_string(), "How are you?".to_string()],
        "What's the weather like?",
    );

    let prompt = raw.to_prompt();
    assert!(prompt.contains("Context"));
    assert!(prompt.contains("Hello"));
    assert!(prompt.contains("weather"));
}

#[test]
fn test_labeling_result() {
    let labels = IntentLabels::explicit_query(0.9);
    let result = LabelingResult::success(Uuid::new_v4(), labels, 0.85, 150, TeacherAuth::new());

    assert!(result.is_success());
    assert_eq!(result.confidence(), 0.85);
    assert_eq!(result.latency_ms(), 150);
}

#[test]
fn test_distillation_stats() {
    let mut stats = DistillationStats::default();

    let result1 = LabelingResult::success(
        Uuid::new_v4(),
        IntentLabels::continuation(0.8),
        0.9,
        100,
        TeacherAuth::new(),
    );
    stats.update(&result1);

    let result2 = LabelingResult::failure(Uuid::new_v4(), "API error", 50);
    stats.update(&result2);

    assert_eq!(stats.total_processed, 2);
    assert_eq!(stats.successful, 1);
    assert_eq!(stats.failed, 1);
    assert_eq!(stats.success_rate(), 0.5);
}

#[test]
fn test_pipeline_creation() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let pipeline = DistillationPipeline::new(teacher, config);

    assert!(pipeline.is_ok());
}

#[test]
fn label_single_fails_closed_without_a_live_teacher() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let raw = RawExample::new(vec!["Hi".to_string()], "What time is it?");
    let error = pipeline.label_single(&raw).unwrap_err();

    assert!(matches!(error, crate::error::TuneError::TeacherApi(_)));
    assert!(error.to_string().contains("not configured"));
    assert_eq!(pipeline.stats().total_processed, 0);
}

#[test]
fn label_batch_reports_missing_live_teacher_as_failures() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let raws: Vec<RawExample> = (0..5)
        .map(|i| RawExample::new(vec!["context".to_string()], format!("message {i}")))
        .collect();

    let results = pipeline.label_batch(&raws);
    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|result| !result.is_success()));
    assert!(results.iter().all(|result| {
        result
            .error()
            .is_some_and(|error| error.contains("not configured"))
    }));
    assert_eq!(pipeline.stats().failed, 5);
}

#[cfg(feature = "simulated-teacher")]
#[test]
fn simulated_labeling_requires_explicit_method() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();
    let raw = RawExample::new(vec!["Hi".to_string()], "What time is it?");

    let result = pipeline.label_single_simulated(&raw).unwrap();

    assert!(result.is_success());
    assert_eq!(result.confidence(), 0.85);
    assert_eq!(pipeline.stats().successful, 1);
    assert_eq!(
        result.source(),
        LabelSource::Simulated,
        "the simulated path must stamp Simulated provenance, not Teacher"
    );

    let results = pipeline.label_batch_simulated(&[raw]);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_success());
    assert_eq!(pipeline.stats().successful, 2);
    assert_eq!(
        results[0].source(),
        LabelSource::Simulated,
        "the simulated batch path must stamp Simulated provenance, not Teacher"
    );
}

#[test]
fn test_sanitize_strips_control_chars() {
    // Create input with control characters
    let input_with_control = "Hello\x00World\x07Test";
    let raw = RawExample::new(vec![], input_with_control);
    let prompt = raw.to_prompt();

    // Control chars should be stripped
    assert!(!prompt.contains('\x00'));
    assert!(!prompt.contains('\x07'));
    assert!(prompt.contains("HelloWorldTest"));
}

#[test]
fn test_sanitize_preserves_newlines() {
    let input = "Line1\nLine2\tTabbed\rCarriage";
    let raw = RawExample::new(vec![], input);
    let prompt = raw.to_prompt();

    // Newlines, tabs, and carriage returns should be preserved
    assert!(prompt.contains('\n'));
    assert!(prompt.contains('\t'));
    assert!(prompt.contains('\r'));
}

#[test]
fn test_sanitize_truncates_long_message() {
    let long_message = "a".repeat(MAX_MESSAGE_LENGTH + 1000);
    let raw = RawExample::new(vec![], long_message.clone());
    let prompt = raw.to_prompt();

    // Should be truncated to MAX_MESSAGE_LENGTH
    assert!(prompt.len() <= MAX_PROMPT_LENGTH + 20); // +20 for formatting
}

#[test]
fn test_sanitize_truncates_long_prompt() {
    // Create enough context to exceed MAX_PROMPT_LENGTH
    let long_context: Vec<String> = (0..1000)
        .map(|i| format!("Context message {i} with some content"))
        .collect();
    let raw = RawExample::new(long_context, "Final message");
    let prompt = raw.to_prompt();

    // Should be truncated with marker
    assert!(prompt.len() <= MAX_PROMPT_LENGTH + 20);
    assert!(prompt.contains("[truncated]") || prompt.len() <= MAX_PROMPT_LENGTH);
}

#[test]
fn label_source_reflects_construction_path() {
    let teacher_result = LabelingResult::success(
        Uuid::new_v4(),
        IntentLabels::continuation(0.8),
        0.9,
        10,
        TeacherAuth::new(),
    );
    assert_eq!(teacher_result.source(), LabelSource::Teacher);

    let simulated_result =
        LabelingResult::simulated(Uuid::new_v4(), IntentLabels::continuation(0.8), 0.85, 5);
    assert_eq!(simulated_result.source(), LabelSource::Simulated);

    // Provenance is compile-enforced, not just constructor-enforced: `source`
    // is `pub(super)` on `LabelingResult`, so it cannot be set in a struct
    // literal or reassigned from outside `pipeline` — there is no path from
    // either result above to the other's provenance short of editing this
    // module.
}

#[cfg(feature = "serde")]
#[test]
fn teacher_result_serializes_but_has_no_deserialize_path_back() {
    // A legitimate, in-process Teacher result.
    let result = LabelingResult::success(
        Uuid::new_v4(),
        IntentLabels::explicit_query(0.9),
        0.85,
        150,
        TeacherAuth::new(),
    );
    assert_eq!(result.source(), LabelSource::Teacher);

    // It serializes fine — `Serialize` is still derived, so the data isn't
    // lost or hidden.
    let json = serde_json::to_string(&result).expect("Teacher result must serialize");
    assert!(json.contains("\"source\":\"Teacher\""));

    // But there is no way back: `LabelingResult` does not implement
    // `Deserialize`, so `serde_json::from_str::<LabelingResult>(&json)` does
    // not type-check (see the `compile_fail` doctests on `LabelingResult`).
    // A result can only ever be reconstructed by calling `success`,
    // `simulated`, or `failure` again, in-process — never by round-tripping
    // through storage. This is a compile-time guarantee, not a runtime
    // discard: there is no silent path from serialized bytes back to a
    // convertible `LabelingResult`, forged or genuine.
    let _ = json;
}

#[test]
fn to_training_examples_converts_a_genuine_teacher_result() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let teacher_result = LabelingResult::success(
        Uuid::new_v4(),
        IntentLabels::explicit_query(0.9),
        0.85,
        150,
        TeacherAuth::new(),
    );

    let context_embeddings = vec![vec![vec![0.0_f32; 4]]];
    let message_embeddings = vec![vec![0.0_f32; 4]];

    let examples = pipeline
        .to_training_examples(&[teacher_result], &context_embeddings, &message_embeddings)
        .unwrap();

    assert_eq!(
        examples.len(),
        1,
        "a genuinely-constructed Teacher result must convert to a training example"
    );
}

#[test]
fn failed_result_cannot_be_promoted_to_success_via_public_api() {
    let result = LabelingResult::failure(Uuid::new_v4(), "boom", 5);

    assert!(!result.is_success());
    assert_eq!(result.error(), Some("boom"));

    // There is no public setter for `error`, `labels`, or `source`: the only
    // way to reach a `LabelingResult` with `is_success() == true` is through
    // `LabelingResult::success` or `LabelingResult::simulated`. See the
    // `compile_fail` doctest on `LabelingResult` for the structural proof
    // that `result.error = None` does not compile.
}

#[test]
fn label_batch_accounts_failures_from_the_shared_helper() {
    // `label_batch` routes every item through `label_batch_with`, which must
    // own failure accounting itself — `label_single` never touches
    // `self.stats`.
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let raws: Vec<RawExample> = (0..3)
        .map(|i| RawExample::new(vec!["context".to_string()], format!("message {i}")))
        .collect();

    let results = pipeline.label_batch(&raws);

    assert_eq!(results.len(), 3);
    assert_eq!(pipeline.stats().total_processed, 3);
    assert_eq!(pipeline.stats().failed, 3);
    assert_eq!(pipeline.stats().successful, 0);
}

#[cfg(feature = "simulated-teacher")]
#[test]
fn mixed_batch_reports_correct_success_and_failure_counts_from_the_helper() {
    // Two batches on one pipeline, one that always fails (no live teacher)
    // and one that always succeeds (simulated), combine into stats neither
    // batch's callback updates on its own — `label_batch_with` is the only
    // place that calls `self.stats.update`, for both branches, so the
    // combined counts prove it owns both, not just failures.
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let failing_raws: Vec<RawExample> = (0..2)
        .map(|i| RawExample::new(vec!["context".to_string()], format!("fail {i}")))
        .collect();
    let succeeding_raws: Vec<RawExample> = (0..3)
        .map(|i| RawExample::new(vec!["context".to_string()], format!("ok {i}")))
        .collect();

    let failed_results = pipeline.label_batch(&failing_raws);
    let succeeded_results = pipeline.label_batch_simulated(&succeeding_raws);

    assert_eq!(failed_results.len(), 2);
    assert_eq!(succeeded_results.len(), 3);
    assert!(failed_results.iter().all(|r| !r.is_success()));
    assert!(succeeded_results.iter().all(LabelingResult::is_success));

    assert_eq!(pipeline.stats().total_processed, 5);
    assert_eq!(pipeline.stats().failed, 2);
    assert_eq!(pipeline.stats().successful, 3);
}

#[test]
fn to_training_examples_rejects_simulated_provenance() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let simulated =
        LabelingResult::simulated(Uuid::new_v4(), IntentLabels::continuation(0.8), 0.85, 5);

    let context_embeddings = vec![vec![vec![0.0_f32; 4]]];
    let message_embeddings = vec![vec![0.0_f32; 4]];

    let examples = pipeline
        .to_training_examples(&[simulated], &context_embeddings, &message_embeddings)
        .unwrap();

    assert!(
        examples.is_empty(),
        "a simulated label must never be converted into a training example \
         attributed to the configured real teacher"
    );
}

#[test]
fn to_training_examples_does_not_over_reserve_for_an_all_skipped_batch() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let failures: Vec<LabelingResult> = (0..64)
        .map(|i| LabelingResult::failure(Uuid::new_v4(), format!("error {i}"), 0))
        .collect();
    let context_embeddings: Vec<Vec<Vec<f32>>> = vec![vec![]; failures.len()];
    let message_embeddings: Vec<Vec<f32>> = vec![vec![]; failures.len()];

    let examples = pipeline
        .to_training_examples(&failures, &context_embeddings, &message_embeddings)
        .unwrap();

    assert!(examples.is_empty());
    assert_eq!(
        examples.capacity(),
        0,
        "the returned Vec must be sized to the eligible (kept) count, not the input count, \
         so a large all-skipped batch does not hold unused reserved memory"
    );
}
