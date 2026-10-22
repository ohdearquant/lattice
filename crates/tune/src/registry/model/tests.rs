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
