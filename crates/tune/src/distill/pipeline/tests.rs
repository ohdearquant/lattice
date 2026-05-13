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
    let result = LabelingResult::success(Uuid::new_v4(), labels, 0.85, 150);

    assert!(result.is_success());
    assert_eq!(result.confidence, 0.85);
    assert_eq!(result.latency_ms, 150);
}

#[test]
fn test_distillation_stats() {
    let mut stats = DistillationStats::default();

    let result1 =
        LabelingResult::success(Uuid::new_v4(), IntentLabels::continuation(0.8), 0.9, 100);
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
fn test_pipeline_label_single() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let raw = RawExample::new(vec!["Hi".to_string()], "What time is it?");
    let result = pipeline.label_single(&raw);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.is_success());
}

#[test]
fn test_pipeline_label_batch() {
    let teacher = TeacherConfig::claude_sonnet();
    let config = DistillationConfig::default();
    let mut pipeline = DistillationPipeline::new(teacher, config).unwrap();

    let raws: Vec<RawExample> = (0..5)
        .map(|i| RawExample::new(vec!["context".to_string()], format!("message {i}")))
        .collect();

    let results = pipeline.label_batch(&raws);
    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|r| r.is_success()));
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
