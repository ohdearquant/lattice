use super::*;
use lattice_inference::bert_pooling_fixture::{
    DIMENSIONS, INPUTS, MAX_POSITIONS, checkpoint_hashes, write_checkpoint,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tower::ServiceExt as _;

const MODELS: [(&str, &str, BertPooling); 2] = [
    ("bge-small-en-v1.5", "cls", BertPooling::CLS),
    ("all-minilm-l6-v2", "mean", BertPooling::Mean),
];

fn fixture(parent: &Path, name: &str) -> PathBuf {
    let path = parent.join(name);
    write_checkpoint(&path).expect("deterministic BERT checkpoint must be written");
    path
}

fn valid_vector(vector: &[f32]) -> bool {
    vector.len() == DIMENSIONS
        && vector.iter().all(|value| value.is_finite())
        && vector.iter().any(|&value| value != 0.0)
}

fn assert_vector(actual: &[f32], expected: &[f32]) {
    assert!(valid_vector(actual), "invalid actual embedding");
    assert!(valid_vector(expected), "invalid expected embedding");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-5 + 1e-5 * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "embedding lane {index}: {actual} != {expected}"
        );
    }
}

fn distance(left: &[f32], right: &[f32]) -> f64 {
    assert!(valid_vector(left) && valid_vector(right));
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn reference(path: &Path, pooling: BertPooling) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut model = BertModel::from_directory(path).expect("reference checkpoint must load");
    model.set_pooling(pooling);
    let single = model
        .encode(INPUTS[0])
        .expect("reference single must encode");
    let batch = model
        .encode_batch(&INPUTS)
        .expect("reference batch must encode");
    let opposite = match pooling {
        BertPooling::CLS => BertPooling::Mean,
        BertPooling::Mean => BertPooling::CLS,
        _ => panic!("fixture must use one of the two registered pooling modes"),
    };
    model.set_pooling(opposite);
    let wrong_single = model
        .encode(INPUTS[0])
        .expect("opposite single must encode");
    let wrong_batch = model
        .encode_batch(&INPUTS)
        .expect("opposite batch must encode");
    assert!(distance(&single, &wrong_single) >= 1e-3);
    assert_eq!(batch.len(), INPUTS.len());
    assert_eq!(wrong_batch.len(), INPUTS.len());
    for (expected, wrong) in batch.iter().zip(&wrong_batch) {
        assert!(
            distance(expected, wrong) >= 1e-3,
            "pooling fixture is vacuous"
        );
    }
    for first in 0..batch.len() {
        for second in first + 1..batch.len() {
            assert!(distance(&batch[first], &batch[second]) >= 1e-3);
        }
    }
    (single, batch)
}

async fn request(embedding: Arc<EmbeddingState>, input: Value) -> Response {
    let model_id = embedding.model_id.to_string();
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"input": input, "model": model_id}).to_string(),
        ))
        .expect("request must build");
    router(test_app_state_with_embedding(embedding))
        .oneshot(request)
        .await
        .expect("actual router must respond")
}

async fn vectors(response: Response, name: &str, expected_count: usize) -> Vec<Vec<f32>> {
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "list");
    assert_eq!(value["model"], name);
    assert!(value["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        value["usage"]["prompt_tokens"],
        value["usage"]["total_tokens"]
    );
    let rows = value["data"].as_array().unwrap();
    assert_eq!(rows.len(), expected_count);
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            assert_eq!(row["object"], "embedding");
            assert_eq!(row["index"].as_u64().unwrap(), index as u64);
            let vector: Vec<f32> = serde_json::from_value(row["embedding"].clone()).unwrap();
            assert!(valid_vector(&vector));
            vector
        })
        .collect()
}

async fn compare_route(
    embedding: Arc<EmbeddingState>,
    name: &str,
    single: &[f32],
    batch: &[Vec<f32>],
) {
    let actual = vectors(request(embedding.clone(), json!(INPUTS[0])).await, name, 1).await;
    assert_vector(&actual[0], single);
    let actual = vectors(request(embedding, json!(INPUTS)).await, name, INPUTS.len()).await;
    assert_eq!(actual.len(), batch.len());
    for (actual, expected) in actual.iter().zip(batch) {
        assert_vector(actual, expected);
    }
}

#[tokio::test]
async fn embeddings_default_pooling_matches_library_for_single_and_batch() {
    let temp = tempfile::tempdir().unwrap();
    let mut hashes = None;
    for (name, _, pooling) in MODELS {
        let path = fixture(temp.path(), name);
        let current = checkpoint_hashes(&path).unwrap();
        if let Some(expected) = hashes.as_ref() {
            assert_eq!(&current, expected);
        }
        hashes = Some(current);
        let (single, batch) = reference(&path, pooling);
        let embedding = load_embedding_state(&path, None).unwrap();
        assert_eq!(embedding.model.pooling(), pooling);
        compare_route(embedding, name, &single, &batch).await;
    }
}

#[tokio::test]
async fn embeddings_explicit_pooling_override_matches_library() {
    let temp = tempfile::tempdir().unwrap();
    for (name, _, _) in MODELS {
        let path = fixture(temp.path(), name);
        for (argument, pooling) in [("cls", BertPooling::CLS), ("mean", BertPooling::Mean)] {
            let (single, batch) = reference(&path, pooling);
            let embedding = load_embedding_state(&path, Some(argument)).unwrap();
            assert_eq!(embedding.model.pooling(), pooling);
            compare_route(embedding, name, &single, &batch).await;
        }
    }
}

#[test]
fn embedding_startup_preserves_load_and_override_errors() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing");
    assert_eq!(
        load_embedding_state(&missing, Some("invalid"))
            .err()
            .unwrap(),
        format!("embedding model directory not found: {}", missing.display())
    );
    let malformed = temp.path().join("malformed");
    std::fs::create_dir(&malformed).unwrap();
    let load_error = load_embedding_state(&malformed, Some("invalid"))
        .err()
        .unwrap();
    assert!(load_error.starts_with(&format!(
        "embedding model load failed ({}): ",
        malformed.display()
    )));
    let path = fixture(temp.path(), MODELS[0].0);
    assert_eq!(
        load_embedding_state(&path, Some("invalid")).err().unwrap(),
        "invalid --embedding-pooling 'invalid'; expected 'mean' or 'cls'"
    );
}

pub(super) async fn check_context_window_route() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path(), MODELS[0].0);
    let embedding = load_embedding_state(&path, None).unwrap();
    assert_eq!(
        embedding.model.config().max_position_embeddings,
        MAX_POSITIONS
    );
    assert_eq!(embedding.model.tokenizer().max_seq_len(), MAX_POSITIONS);
    let at_limit = "hello ".repeat(MAX_POSITIONS - 2);
    let over_limit = "hello ".repeat(MAX_POSITIONS - 1);
    let at_tokens = embedding.model.tokenizer().tokenize(&at_limit);
    let over_tokens = embedding.model.tokenizer().tokenize(&over_limit);
    assert_eq!(at_tokens.real_length, MAX_POSITIONS);
    assert_eq!(at_tokens.pre_truncation_len, MAX_POSITIONS);
    assert_eq!(over_tokens.real_length, MAX_POSITIONS);
    assert_eq!(over_tokens.pre_truncation_len, MAX_POSITIONS + 1);
    let expected = embedding.model.encode(&at_limit).unwrap();
    let actual = vectors(
        request(embedding.clone(), json!(at_limit)).await,
        MODELS[0].0,
        1,
    )
    .await;
    assert_vector(&actual[0], &expected);
    let (status, code) = error_code_of(request(embedding, json!(over_limit)).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "context_length_exceeded");
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureHashes {
    #[serde(rename = "config.json")]
    config: String,
    #[serde(rename = "model.safetensors")]
    weights: String,
    #[serde(rename = "vocab.txt")]
    vocab: String,
}

impl FixtureHashes {
    fn as_map(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("config.json".to_string(), self.config.clone()),
            ("model.safetensors".to_string(), self.weights.clone()),
            ("vocab.txt".to_string(), self.vocab.clone()),
        ])
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactModel {
    name: String,
    pooling: String,
    single: Vec<f32>,
    batch: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PoolingArtifact {
    schema: u32,
    run_id: String,
    source_id: String,
    checkpoint_hashes: FixtureHashes,
    dimensions: usize,
    inputs: Vec<String>,
    models: Vec<ArtifactModel>,
}

fn valid_id(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_artifact(
    bytes: &[u8],
    run_id: &str,
    source_id: &str,
    hashes: &BTreeMap<String, String>,
) -> Result<PoolingArtifact, String> {
    if !valid_id(run_id, 32) || !valid_id(source_id, 64) {
        return Err("invalid expected run/source identity".into());
    }
    let artifact: PoolingArtifact = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if artifact.schema != 1
        || artifact.run_id != run_id
        || artifact.source_id != source_id
        || artifact.checkpoint_hashes.as_map() != *hashes
        || artifact.dimensions != DIMENSIONS
        || !artifact.inputs.iter().map(String::as_str).eq(INPUTS)
        || artifact.models.len() != MODELS.len()
    {
        return Err("artifact identity or population mismatch".into());
    }
    for (model, (name, pooling, _)) in artifact.models.iter().zip(MODELS) {
        if model.name != name
            || model.pooling != pooling
            || !valid_vector(&model.single)
            || model.batch.len() != INPUTS.len()
            || !model.batch.iter().all(|row| valid_vector(row))
        {
            return Err("artifact model or vector mismatch".into());
        }
    }
    Ok(artifact)
}

#[test]
fn pooling_artifact_rejects_invalid_identity_schema_and_vectors() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path(), MODELS[0].0);
    let hashes = checkpoint_hashes(&path).unwrap();
    let run_id = "a".repeat(32);
    let source_id = "b".repeat(64);
    let models: Vec<Value> = MODELS
        .iter()
        .map(|&(name, pooling, mode)| {
            let (single, batch) = reference(&path, mode);
            json!({"name": name, "pooling": pooling, "single": single, "batch": batch})
        })
        .collect();
    let valid = json!({
        "schema": 1, "run_id": run_id, "source_id": source_id,
        "checkpoint_hashes": hashes, "dimensions": DIMENSIONS,
        "inputs": INPUTS, "models": models,
    });
    let bytes = serde_json::to_vec(&valid).unwrap();
    assert!(parse_artifact(&bytes, &run_id, &source_id, &hashes).is_ok());
    for (pointer, bad) in [
        ("/schema", json!(2)),
        ("/run_id", json!("c".repeat(32))),
        ("/source_id", json!("c".repeat(64))),
        ("/checkpoint_hashes/config.json", json!("c".repeat(64))),
        ("/dimensions", json!(DIMENSIONS - 1)),
        ("/inputs/0", json!("wrong input")),
        ("/models/0/name", json!(MODELS[1].0)),
        ("/models/0/pooling", json!("mean")),
        ("/models/0/single", json!([])),
        ("/models/0/batch", json!([])),
        ("/models/0/batch/0", json!(vec![0.0; DIMENSIONS])),
        ("/models/0/single/0", json!(null)),
    ] {
        let mut changed = valid.clone();
        *changed.pointer_mut(pointer).unwrap() = bad;
        assert!(
            parse_artifact(
                &serde_json::to_vec(&changed).unwrap(),
                &run_id,
                &source_id,
                &hashes
            )
            .is_err(),
            "accepted invalid {pointer}"
        );
    }
    let mut reordered = valid.clone();
    reordered["models"].as_array_mut().unwrap().swap(0, 1);
    assert!(
        parse_artifact(
            &serde_json::to_vec(&reordered).unwrap(),
            &run_id,
            &source_id,
            &hashes
        )
        .is_err()
    );
    let mut unknown = valid.clone();
    unknown["extra"] = json!(true);
    assert!(
        parse_artifact(
            &serde_json::to_vec(&unknown).unwrap(),
            &run_id,
            &source_id,
            &hashes
        )
        .is_err()
    );
    let duplicate = format!(
        "{{\"schema\":1,{}",
        &String::from_utf8(bytes.clone()).unwrap()[1..]
    );
    assert!(parse_artifact(duplicate.as_bytes(), &run_id, &source_id, &hashes).is_err());
    assert!(parse_artifact(&bytes, "", &source_id, &hashes).is_err());
    assert!(parse_artifact(&bytes, &run_id, &source_id.to_ascii_uppercase(), &hashes).is_err());
}

#[test]
fn pooling_comparison_rejects_nonfinite_wrong_shape_and_wrong_values() {
    let expected = vec![0.25; DIMENSIONS];
    assert_vector(&expected, &expected);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5] {
        let mut wrong = expected.clone();
        wrong[DIMENSIONS - 1] = value;
        assert!(std::panic::catch_unwind(|| assert_vector(&wrong, &expected)).is_err());
        assert!(std::panic::catch_unwind(|| assert_vector(&expected, &wrong)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| assert_vector(&expected[..DIMENSIONS - 1], &expected)).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| assert_vector(&expected, &expected[..DIMENSIONS - 1])).is_err()
    );
}

#[tokio::test]
#[ignore = "requires bound LATTICE_BERT_POOLING_ARTIFACT from the native export gate"]
async fn embeddings_pooling_matches_bound_native_artifact() {
    let artifact_path = std::env::var_os("LATTICE_BERT_POOLING_ARTIFACT")
        .expect("LATTICE_BERT_POOLING_ARTIFACT is required");
    let run_id = std::env::var("LATTICE_BERT_POOLING_RUN_ID").expect("run identity is required");
    let source_id =
        std::env::var("LATTICE_BERT_POOLING_SOURCE_ID").expect("source identity is required");
    let bytes = std::fs::read(&artifact_path).expect("native artifact must exist and be readable");
    let temp = tempfile::tempdir().unwrap();
    let paths: Vec<PathBuf> = MODELS
        .iter()
        .map(|&(name, _, _)| fixture(temp.path(), name))
        .collect();
    let hashes = checkpoint_hashes(&paths[0]).unwrap();
    assert_eq!(checkpoint_hashes(&paths[1]).unwrap(), hashes);
    let artifact = parse_artifact(&bytes, &run_id, &source_id, &hashes)
        .expect("native artifact must match exact source/run/checkpoint/input identities");
    for ((model, path), &(name, _, pooling)) in artifact.models.iter().zip(&paths).zip(&MODELS) {
        let (single, batch) = reference(path, pooling);
        assert_vector(&model.single, &single);
        for (native, expected) in model.batch.iter().zip(&batch) {
            assert_vector(native, expected);
        }
        compare_route(
            load_embedding_state(path, None).unwrap(),
            name,
            &model.single,
            &model.batch,
        )
        .await;
    }
    println!(
        "BERT_POOLING_HTTP_COMPLETE schema=1 models=2 inputs=3 dimensions={DIMENSIONS} run_id={run_id} source_id={source_id}"
    );
}
