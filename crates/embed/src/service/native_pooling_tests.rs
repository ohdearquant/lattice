use super::*;
use lattice_inference::BertPooling;
use lattice_inference::bert_pooling_fixture::{
    DIMENSIONS, INPUTS, checkpoint_hashes, write_checkpoint,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
struct ModelVectors {
    name: &'static str,
    pooling: &'static str,
    single: Vec<f32>,
    batch: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct PoolingArtifact {
    schema: u32,
    run_id: String,
    source_id: String,
    checkpoint_hashes: BTreeMap<String, String>,
    dimensions: usize,
    inputs: Vec<String>,
    models: Vec<ModelVectors>,
}

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
    pooling_name: &'static str,
    expected_pooling: BertPooling,
    opposite_pooling: BertPooling,
) -> ModelVectors {
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
    ModelVectors {
        name,
        pooling: pooling_name,
        single: singles.remove(0),
        batch,
    }
}

async fn checked_bge(directory: &Path) -> ModelVectors {
    checked_model_vectors(
        directory,
        EmbeddingModel::BgeSmallEnV15,
        "bge-small-en-v1.5",
        "cls",
        BertPooling::CLS,
        BertPooling::Mean,
    )
    .await
}

async fn checked_minilm(directory: &Path) -> ModelVectors {
    checked_model_vectors(
        directory,
        EmbeddingModel::AllMiniLmL6V2,
        "all-minilm-l6-v2",
        "mean",
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

fn required_hex_identity(name: &str, length: usize) -> String {
    let value = std::env::var(name).unwrap_or_else(|error| panic!("{name} is required: {error}"));
    assert!(
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must contain exactly {length} lowercase hexadecimal characters"
    );
    value
}

fn write_new_artifact(path: &Path, artifact: &PoolingArtifact) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(artifact).map_err(std::io::Error::other)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()
}

#[test]
fn native_pooling_artifact_writer_refuses_to_overwrite() {
    let temporary = tempfile::tempdir().expect("artifact parent");
    let path = temporary.path().join("artifact.json");
    let artifact = PoolingArtifact {
        schema: 1,
        run_id: "0".repeat(32),
        source_id: "1".repeat(64),
        checkpoint_hashes: BTreeMap::new(),
        dimensions: DIMENSIONS,
        inputs: INPUTS.iter().map(|text| (*text).to_owned()).collect(),
        models: Vec::new(),
    };
    write_new_artifact(&path, &artifact).expect("first creation");
    let original = std::fs::read(&path).expect("read first artifact");
    let different = PoolingArtifact {
        run_id: "2".repeat(32),
        ..artifact
    };
    let error = write_new_artifact(&path, &different).expect_err("must refuse existing artifact");
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(
        std::fs::read(path).expect("read preserved artifact"),
        original
    );
}

#[tokio::test]
#[ignore = "explicit cross-runtime export; requires bound artifact, run and source identities"]
async fn native_pooling_exports_bound_cross_runtime_artifact() {
    let run_id = required_hex_identity("LATTICE_BERT_POOLING_RUN_ID", 32);
    let source_id = required_hex_identity("LATTICE_BERT_POOLING_SOURCE_ID", 64);
    let path = std::env::var_os("LATTICE_BERT_POOLING_ARTIFACT")
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .expect("LATTICE_BERT_POOLING_ARTIFACT is required");
    let temporary = tempfile::tempdir().expect("fixture parent");
    let directory = temporary.path().join("checkpoint");
    write_checkpoint(&directory).expect("write actual BERT checkpoint");
    let hashes = checkpoint_hashes(&directory).expect("hash actual checkpoint");
    let bge = checked_bge(&directory).await;
    let minilm = checked_minilm(&directory).await;
    assert_eq!(
        checkpoint_hashes(&directory).expect("rehash checkpoint"),
        hashes,
        "checkpoint changed while producing native vectors"
    );
    let artifact = PoolingArtifact {
        schema: 1,
        run_id,
        source_id,
        checkpoint_hashes: hashes,
        dimensions: DIMENSIONS,
        inputs: INPUTS.iter().map(|text| (*text).to_owned()).collect(),
        models: vec![bge, minilm],
    };
    write_new_artifact(&path, &artifact).expect("create new bound native pooling artifact");
    println!(
        "BERT_POOLING_NATIVE_COMPLETE schema=1 models=2 inputs=3 dimensions={DIMENSIONS} run_id={} source_id={}",
        artifact.run_id, artifact.source_id
    );
}
