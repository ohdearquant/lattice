use super::bert_pooling_tests::{assert_vector, fixture, request, request_json, success_body};
use super::*;
use axum::http::StatusCode;
use axum::response::Response;
use lattice_inference::bert_pooling_fixture::{INPUTS, MAX_POSITIONS};
use lattice_inference::serve::embeddings::{
    MAX_EMBEDDINGS_BATCH_SIZE, MAX_EMBEDDINGS_TOTAL_TOKENS,
};
use serde_json::{Value, json};
use std::sync::Arc;

async fn response_value(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body must be readable");
    let value = serde_json::from_slice(&bytes).expect("response must be valid JSON");
    (status, value)
}

fn error_body(message: impl Into<String>, kind: &str, code: &str) -> Value {
    json!({
        "error": {
            "message": message.into(),
            "type": kind,
            "code": code,
            "param": null
        }
    })
}

async fn assert_response(response: Response, status: StatusCode, body: Value) {
    let (actual_status, actual_body) = response_value(response).await;
    assert_eq!(actual_status, status);
    assert_eq!(actual_body, body);
}

fn hold_all_embedding_permits(
    embedding: &Arc<EmbeddingState>,
) -> Vec<tokio::sync::OwnedSemaphorePermit> {
    (0..EMBEDDING_MAX_CONCURRENT_JOBS)
        .map(|_| {
            try_acquire_embedding_slot(&embedding.admission)
                .expect("every embedding permit must be acquired")
        })
        .collect()
}

#[tokio::test]
async fn embedding_route_model_identity_uses_checkpoint_directory_name() {
    let temp = tempfile::tempdir().expect("temporary directory must be created");
    let path = fixture(temp.path(), "bge-route-fixture");
    let embedding = load_embedding_state(&path, None).expect("fixture checkpoint must load");
    let served_id = "bge-route-fixture";
    assert_eq!(embedding.model_id.as_ref(), served_id);

    for body in [
        json!({"input": "hello", "model": served_id}),
        json!({"input": "hello"}),
    ] {
        let (status, value) =
            response_value(request_json(Some(embedding.clone()), body).await).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["model"], served_id);
    }

    for requested in ["", "another-model"] {
        assert_response(
            request_json(
                Some(embedding.clone()),
                json!({"input": "hello", "model": requested}),
            )
            .await,
            StatusCode::BAD_REQUEST,
            error_body(
                format!("model '{requested}' is not loaded; this server serves '{served_id}'"),
                "invalid_request_error",
                "model_not_found",
            ),
        )
        .await;
    }
}

#[tokio::test]
async fn embedding_route_validates_input_before_unloaded_model() {
    assert_response(
        request_json(None, json!({"input": ""})).await,
        StatusCode::BAD_REQUEST,
        error_body(
            "input must not be empty",
            "invalid_request_error",
            "invalid_input",
        ),
    )
    .await;

    assert_response(
        request_json(None, json!({"input": "hello", "model": "another-model"})).await,
        StatusCode::SERVICE_UNAVAILABLE,
        error_body(
            "no embedding model is loaded; start lattice_serve with --embedding-model <dir>",
            "server_error",
            "embedding_model_not_loaded",
        ),
    )
    .await;
}

#[tokio::test]
async fn embedding_route_fixture_usage_matches_tokenizer_and_batch_vectors() {
    let temp = tempfile::tempdir().expect("temporary directory must be created");
    let path = fixture(temp.path(), "all-minilm-route-fixture");
    let embedding = load_embedding_state(&path, None).expect("fixture checkpoint must load");
    let model_id = embedding.model_id.to_string();

    let single_refs = [INPUTS[0]];
    let single_tokens: usize = embedding
        .model
        .tokenizer()
        .tokenize_batch(&single_refs)
        .iter()
        .map(|tokens| tokens.real_length)
        .sum();
    let batch_refs: Vec<&str> = INPUTS.to_vec();
    let batch_tokens: usize = embedding
        .model
        .tokenizer()
        .tokenize_batch(&batch_refs)
        .iter()
        .map(|tokens| tokens.real_length)
        .sum();

    let single = success_body(
        request(embedding.clone(), json!(INPUTS[0])).await,
        &model_id,
        1,
    )
    .await;
    let batch = success_body(
        request(embedding, json!(INPUTS)).await,
        &model_id,
        INPUTS.len(),
    )
    .await;

    assert_eq!(single["usage"]["prompt_tokens"], single_tokens as u64);
    assert_eq!(single["usage"]["total_tokens"], single_tokens as u64);
    assert_eq!(batch["usage"]["prompt_tokens"], batch_tokens as u64);
    assert_eq!(batch["usage"]["total_tokens"], batch_tokens as u64);

    let single_vector: Vec<f32> = serde_json::from_value(single["data"][0]["embedding"].clone())
        .expect("single response embedding must be a vector");
    let batch_first: Vec<f32> = serde_json::from_value(batch["data"][0]["embedding"].clone())
        .expect("first batch response embedding must be a vector");
    assert_vector(&single_vector, &batch_first);
}

#[tokio::test]
async fn embedding_route_fixture_budget_boundary_reaches_admission() {
    let temp = tempfile::tempdir().expect("temporary directory must be created");
    let path = fixture(temp.path(), "all-minilm-budget-fixture");
    let embedding = load_embedding_state(&path, None).expect("fixture checkpoint must load");
    let text = "hello ".repeat(MAX_POSITIONS - 2);
    let inputs = vec![text; MAX_EMBEDDINGS_BATCH_SIZE];
    let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
    let tokens = embedding.model.tokenizer().tokenize_batch(&refs);
    assert_eq!(tokens.len(), MAX_EMBEDDINGS_BATCH_SIZE);
    assert!(tokens.iter().all(|item| {
        item.real_length == MAX_POSITIONS && item.pre_truncation_len == MAX_POSITIONS
    }));
    let total_tokens: usize = tokens.iter().map(|item| item.real_length).sum();
    assert_eq!(
        MAX_EMBEDDINGS_BATCH_SIZE * MAX_POSITIONS,
        MAX_EMBEDDINGS_TOTAL_TOKENS
    );
    assert_eq!(total_tokens, MAX_EMBEDDINGS_TOTAL_TOKENS);

    let _permits = hold_all_embedding_permits(&embedding);
    assert_response(
        request_json(Some(embedding), json!({"input": inputs})).await,
        StatusCode::SERVICE_UNAVAILABLE,
        error_body(
            "too many outstanding /v1/embeddings requests; retry shortly",
            "server_error",
            "server_busy",
        ),
    )
    .await;
}

#[tokio::test]
async fn embedding_route_admission_follows_request_validation() {
    let temp = tempfile::tempdir().expect("temporary directory must be created");
    let path = fixture(temp.path(), "all-minilm-admission-fixture");
    let embedding = load_embedding_state(&path, None).expect("fixture checkpoint must load");
    let _permits = hold_all_embedding_permits(&embedding);

    assert_response(
        request_json(Some(embedding.clone()), json!({"input": "hello"})).await,
        StatusCode::SERVICE_UNAVAILABLE,
        error_body(
            "too many outstanding /v1/embeddings requests; retry shortly",
            "server_error",
            "server_busy",
        ),
    )
    .await;

    assert_response(
        request_json(Some(embedding.clone()), json!({"input": ""})).await,
        StatusCode::BAD_REQUEST,
        error_body(
            "input must not be empty",
            "invalid_request_error",
            "invalid_input",
        ),
    )
    .await;

    let normal = "hello ".repeat(MAX_POSITIONS - 2);
    let over_context = "hello ".repeat(MAX_POSITIONS - 1);
    let mut inputs = vec![normal; MAX_EMBEDDINGS_BATCH_SIZE];
    inputs[0] = over_context;
    let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
    let tokens = embedding.model.tokenizer().tokenize_batch(&refs);
    assert_eq!(tokens[0].pre_truncation_len, MAX_POSITIONS + 1);
    assert_eq!(tokens[0].real_length, MAX_POSITIONS);
    assert!(tokens.iter().skip(1).all(|item| {
        item.real_length == MAX_POSITIONS && item.pre_truncation_len == MAX_POSITIONS
    }));
    assert_eq!(
        tokens.iter().map(|item| item.real_length).sum::<usize>(),
        MAX_EMBEDDINGS_TOTAL_TOKENS
    );
    assert_response(
        request_json(Some(embedding), json!({"input": inputs})).await,
        StatusCode::BAD_REQUEST,
        error_body(
            format!(
                "input item 0 has {} tokens, exceeding the loaded model's context window of {} tokens",
                tokens[0].pre_truncation_len,
                MAX_POSITIONS
            ),
            "invalid_request_error",
            "context_length_exceeded",
        ),
    )
    .await;
}

#[tokio::test]
async fn embedding_route_rejects_unsupported_and_unknown_fields_first() {
    assert_response(
        request_json(None, json!({"dimensions": 3, "encoding_format": "base64"})).await,
        StatusCode::BAD_REQUEST,
        error_body(
            "dimensions is not supported by this server; the loaded model's full output width is always returned",
            "invalid_request_error",
            "unsupported_parameter",
        ),
    )
    .await;

    assert_response(
        request_json(None, json!({"input": "hello", "dimensions": null})).await,
        StatusCode::SERVICE_UNAVAILABLE,
        error_body(
            "no embedding model is loaded; start lattice_serve with --embedding-model <dir>",
            "server_error",
            "embedding_model_not_loaded",
        ),
    )
    .await;

    assert_response(
        request_json(None, json!({"input": "hello", "unknown": true})).await,
        StatusCode::BAD_REQUEST,
        error_body(
            "invalid request body",
            "invalid_request_error",
            "invalid_request_body",
        ),
    )
    .await;
}
