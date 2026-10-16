use super::*;
use lattice_inference::bert_pooling_fixture::{
    DIMENSIONS, INPUTS, MAX_POSITIONS, checkpoint_hashes, write_checkpoint,
};
use std::path::{Path, PathBuf};
use tower::ServiceExt as _;

const MODELS: [(&str, BertPooling); 2] = [
    ("bge-small-en-v1.5", BertPooling::CLS),
    ("all-minilm-l6-v2", BertPooling::Mean),
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
    for (name, pooling) in MODELS {
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
    for (name, _) in MODELS {
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

// Regression: the context-window guard used to measure each item's
// *post-truncation* token count (`WordPieceTokenizer` truncates to its own
// `max_seq_len` during tokenization), and `BertModel::from_directory` derives
// that cap as `max_position_embeddings.min(2048)`. For every checkpoint whose
// config sets `max_position_embeddings` at or below that 2048 cap -- including
// the synthetic fixture below, whose config sets it to `MAX_POSITIONS` (16)
// with no `tokenizer_config.json` to raise it -- the two limits are equal by
// construction. A post-truncation count can never exceed `max_seq_len`, so the
// old `count > max_position_embeddings` test was unsatisfiable and an
// over-limit input was silently truncated and embedded instead of rejected.
// The `assert_eq!` below pins the fixture in exactly that collision case
// (`tokenizer.max_seq_len() == config.max_position_embeddings`), and the
// `pre_truncation_len` assertions pin the token counts the guard must compare.
// This is a generated checkpoint, not a real `bge-small-en-v1.5` download.
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
