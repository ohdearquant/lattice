use super::*;
use lattice_inference::BertPooling;
use lattice_inference::bert_pooling_fixture::{DIMENSIONS, INPUTS, write_checkpoint};
use std::path::Path;

fn assert_vector(vector: &[f32], context: &str) {
    assert_eq!(vector.len(), DIMENSIONS, "{context}: dimensions");
    assert!(
        vector.iter().all(|value| value.is_finite()),
        "{context}: nonfinite vector"
    );
    assert!(
        vector.iter().any(|value| *value != 0.0),
        "{context}: zero vector"
    );
}

fn distance(left: &[f32], right: &[f32]) -> f64 {
    assert_vector(left, "distance left");
    assert_vector(right, "distance right");
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_vector(actual, context);
    assert_vector(expected, "explicit pooling reference");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-5 + 1e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: component {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn service_from_checkpoint(
    directory: &Path,
    model: EmbeddingModel,
    expected_name: &str,
) -> NativeEmbeddingService {
    let service = NativeEmbeddingService::with_model(model);
    let mut acquisitions = 0usize;
    let loaded = load_model_sync_with_bert_acquisition(service.model_config, |name| {
        acquisitions += 1;
        assert_eq!(name, expected_name, "production model-name selection");
        BertModel::from_directory(directory).map_err(|error| error.to_string())
    })
    .expect("load the actual checkpoint through native model selection");
    assert_eq!(acquisitions, 1);
    assert!(service.model.set(Ok(loaded)).is_ok());
    service
}

async fn checked_model_vectors(
    directory: &Path,
    model: EmbeddingModel,
    name: &'static str,
    expected_pooling: BertPooling,
    opposite_pooling: BertPooling,
) {
    let service = service_from_checkpoint(directory, model, name);
    let mut reference = BertModel::from_directory(directory).expect("independent BERT load");
    reference.set_pooling(expected_pooling);
    let mut opposite = BertModel::from_directory(directory).expect("opposite BERT load");
    opposite.set_pooling(opposite_pooling);

    let inputs: Vec<String> = INPUTS.iter().map(|text| (*text).to_owned()).collect();
    let references: Vec<Vec<f32>> = INPUTS
        .iter()
        .map(|text| reference.encode(text).expect("explicit pooling reference"))
        .collect();
    let opposites: Vec<Vec<f32>> = INPUTS
        .iter()
        .map(|text| opposite.encode(text).expect("opposite pooling reference"))
        .collect();
    let mut singles = Vec::with_capacity(INPUTS.len());
    for (index, input) in inputs.iter().enumerate() {
        let output = service
            .embed(std::slice::from_ref(input), model)
            .await
            .expect("native single embedding");
        assert_eq!(output.len(), 1);
        assert_close(&output[0], &references[index], "native single");
        assert!(
            distance(&references[index], &opposites[index]) >= 1e-3,
            "{name} input {index}: fixture does not distinguish pooling modes"
        );
        assert!(
            distance(&output[0], &opposites[index]) >= 1e-3,
            "{name} input {index}: single output follows the opposite pooling mode"
        );
        singles.push(output.into_iter().next().expect("one output"));
    }

    let batch = service
        .embed(&inputs, model)
        .await
        .expect("native unequal-length batch embedding");
    assert_eq!(batch.len(), INPUTS.len());
    for (index, actual) in batch.iter().enumerate() {
        assert_close(actual, &references[index], "native batch row");
        assert_close(actual, &singles[index], "native batch versus single");
        assert!(
            distance(actual, &opposites[index]) >= 1e-3,
            "{name} input {index}: batch output follows the opposite pooling mode"
        );
    }
    for (index, vector) in references.iter().enumerate() {
        for other in &references[index + 1..] {
            assert!(
                distance(vector, other) >= 1e-3,
                "{name}: fixture cannot detect batch row reordering"
            );
        }
    }
}

async fn checked_bge(directory: &Path) {
    checked_model_vectors(
        directory,
        EmbeddingModel::BgeSmallEnV15,
        "bge-small-en-v1.5",
        BertPooling::CLS,
        BertPooling::Mean,
    )
    .await
}

async fn checked_minilm(directory: &Path) {
    checked_model_vectors(
        directory,
        EmbeddingModel::AllMiniLmL6V2,
        "all-minilm-l6-v2",
        BertPooling::Mean,
        BertPooling::CLS,
    )
    .await
}

#[tokio::test]
async fn native_pooling_bge_matches_explicit_cls_for_single_and_batch() {
    let temporary = tempfile::tempdir().expect("fixture parent");
    let directory = temporary.path().join("checkpoint");
    write_checkpoint(&directory).expect("write actual BERT checkpoint");
    checked_bge(&directory).await;
}

#[tokio::test]
async fn native_pooling_minilm_matches_explicit_mean_for_single_and_batch() {
    let temporary = tempfile::tempdir().expect("fixture parent");
    let directory = temporary.path().join("checkpoint");
    write_checkpoint(&directory).expect("write actual BERT checkpoint");
    checked_minilm(&directory).await;
}

#[test]
fn native_pooling_acquisition_failure_preserves_model_name_and_error() {
    for (model, expected_name) in [
        (EmbeddingModel::BgeSmallEnV15, "bge-small-en-v1.5"),
        (EmbeddingModel::AllMiniLmL6V2, "all-minilm-l6-v2"),
    ] {
        let mut calls = 0usize;
        let result = load_model_sync_with_bert_acquisition(ModelConfig::new(model), |name| {
            calls += 1;
            assert_eq!(name, expected_name);
            Err(format!("checkpoint acquisition failed for {name}"))
        });
        assert_eq!(calls, 1);
        match result {
            Err(error) => assert_eq!(
                error,
                format!("checkpoint acquisition failed for {expected_name}")
            ),
            Ok(_) => panic!("failed acquisition must not produce a loaded model"),
        }
    }
}
