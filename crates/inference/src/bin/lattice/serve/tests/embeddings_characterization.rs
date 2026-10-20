use super::embeddings_route::{json_body, post_embeddings, state_with_embedder};
use super::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use lattice_inference::serve::embeddings::test_support::{tiny_embedding_model, tiny_png_data_uri};
use serde_json::{Value, json};
use tower::ServiceExt as _;

fn raw_embeddings_request(body: Body) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(body)
        .expect("request fixture must build")
}

async fn assert_error_envelope(
    response: axum::response::Response,
    status: StatusCode,
    message: &str,
    error_type: &str,
    code: &str,
) {
    assert_eq!(response.status(), status);
    assert_eq!(
        json_body(response).await,
        json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
                "param": null
            }
        })
    );
}

#[tokio::test]
async fn pooled_route_body_failures_keep_their_full_error_envelopes() {
    let request = raw_embeddings_request(Body::from(vec![
        b'x';
        lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES
            + 1
    ]));
    let response = router(state_with_embedder())
        .oneshot(request)
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::PAYLOAD_TOO_LARGE,
        "request body exceeds 1 MiB limit",
        "invalid_request_error",
        "request_body_too_large",
    )
    .await;

    let unreadable = Body::from_stream(futures::stream::iter([Err::<Vec<u8>, _>(
        std::io::Error::other("fixture body stream failed"),
    )]));
    let response = router(state_with_embedder())
        .oneshot(raw_embeddings_request(unreadable))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "invalid JSON request body",
        "invalid_request_error",
        "invalid_request_body",
    )
    .await;

    let response = router(state_with_embedder())
        .oneshot(raw_embeddings_request(Body::from("{")))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "invalid JSON request body",
        "invalid_request_error",
        "invalid_request_body",
    )
    .await;
}

#[tokio::test]
async fn pooled_route_input_refusals_keep_their_full_error_envelopes() {
    let cases: [(Value, &str, &str); 4] = [
        (
            json!({"input": "a", "pooling": "max"}),
            "pooling must be 'mean_visual' or 'last_token', got 'max'",
            "invalid_pooling",
        ),
        (
            json!({"input": []}),
            "input must not be empty",
            "invalid_input",
        ),
        (
            json!({
                "input": {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"}
                }
            }),
            "image_url.url must be an inline data URI; remote URLs are not accepted",
            "unsupported_image_url_scheme",
        ),
        (
            json!({
                "input": {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png,not-base64-marked"}
                }
            }),
            "image data URI must use the ';base64' encoding marker",
            "invalid_image",
        ),
    ];

    for (body, message, code) in cases {
        let response = router(state_with_embedder())
            .oneshot(post_embeddings(body))
            .await
            .expect("router must return a response");
        assert_error_envelope(
            response,
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            code,
        )
        .await;
    }
}

#[tokio::test]
async fn pooled_route_busy_refusal_keeps_its_full_error_envelope() {
    let state = state_with_embedder();
    let held: Vec<_> = (0..EMBEDDING_MAX_CONCURRENT_JOBS)
        .map(|_| {
            state
                .embedding_admission
                .clone()
                .try_acquire_owned()
                .expect("fixture must hold each admission slot")
        })
        .collect();
    let response = router(state)
        .oneshot(post_embeddings(json!({"input": "a"})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "too many outstanding /v1/embeddings requests; retry shortly",
        "server_error",
        "server_busy",
    )
    .await;
    drop(held);
}

#[tokio::test]
async fn pooled_route_aggregate_budget_refusal_keeps_its_full_error_envelope() {
    let item = "a".repeat(480);
    let items: Vec<_> = std::iter::repeat_n(Value::String(item), 80).collect();
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(json!({"input": items})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "input has 38400 total scaffold tokens across all items; maximum is 32768",
        "invalid_request_error",
        "batch_token_budget_exceeded",
    )
    .await;
}

#[tokio::test]
async fn pooled_route_precedence_keeps_the_current_refusal_order() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(json!({"input": [], "pooling": "max"})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "pooling must be 'mean_visual' or 'last_token', got 'max'",
        "invalid_request_error",
        "invalid_pooling",
    )
    .await;

    let response = router(tiny_state(64))
        .oneshot(post_embeddings(json!({"input": []})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "input must not be empty",
        "invalid_request_error",
        "invalid_input",
    )
    .await;

    let no_model_state = tiny_state(64);
    let held_no_model: Vec<_> = (0..EMBEDDING_MAX_CONCURRENT_JOBS)
        .map(|_| {
            no_model_state
                .embedding_admission
                .clone()
                .try_acquire_owned()
                .expect("fixture must hold each admission slot")
        })
        .collect();
    let response = router(no_model_state)
        .oneshot(post_embeddings(json!({"input": "a"})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "embeddings require a loaded vision-language checkpoint; restart this server with \u{60}--model\u{60} pointed at a vision-language checkpoint directory to enable this route",
        "invalid_request_error",
        "vision_unsupported",
    )
    .await;
    drop(held_no_model);

    let busy_state = state_with_embedder();
    let held_busy: Vec<_> = (0..EMBEDDING_MAX_CONCURRENT_JOBS)
        .map(|_| {
            busy_state
                .embedding_admission
                .clone()
                .try_acquire_owned()
                .expect("fixture must hold each admission slot")
        })
        .collect();
    let response = router(busy_state)
        .oneshot(post_embeddings(json!({"input": []})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "input must not be empty",
        "invalid_request_error",
        "invalid_input",
    )
    .await;
    drop(held_busy);

    let items: Vec<_> = std::iter::repeat_n(Value::String("a".repeat(513)), 65).collect();
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(json!({"input": items})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::BAD_REQUEST,
        "input item 0 has 513 scaffold tokens, exceeding the model's context window of 512 tokens",
        "invalid_request_error",
        "context_length_exceeded",
    )
    .await;
}

#[tokio::test]
async fn pooled_route_uses_the_served_model_and_preserves_mixed_batch_output() {
    let image_url = tiny_png_data_uri(1);
    let image_bytes = lattice_inference::serve::contract::decode_inline_image(&image_url)
        .expect("fixture image URI must decode");
    let fixture = tiny_embedding_model();
    let expected_items = lattice_inference::serve::embeddings::normalize_embedding_items(vec![
        lattice_inference::serve::embeddings::EmbeddingInputItem::Text("a".to_string()),
        lattice_inference::serve::embeddings::EmbeddingInputItem::Image {
            url: image_url.clone(),
        },
        lattice_inference::serve::embeddings::EmbeddingInputItem::Text("b".to_string()),
    ])
    .expect("fixture input must normalize");
    let (expected_data, expected_usage) = lattice_inference::serve::embeddings::embed_items(
        &fixture,
        expected_items,
        lattice_inference::serve::embeddings::parse_pooling(None)
            .expect("default pooling must parse"),
    )
    .expect("fixture inputs must embed");
    assert_eq!(
        expected_usage.prompt_tokens,
        fixture.tokenize_len("a")
            + fixture
                .image_scaffold_token_count(&image_bytes, "")
                .expect("fixture image scaffold must count")
            + fixture.tokenize_len("b")
    );

    let response = router(state_with_embedder())
        .oneshot(post_embeddings(json!({
            "model": "request-model",
            "input": [
                "a",
                {"type": "image_url", "image_url": {"url": image_url}},
                "b"
            ],
            "unrecognized_option": true
        })))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::OK);
    let value = json_body(response).await;
    assert_eq!(value["object"], "list");
    assert_eq!(value["model"], "test-model");
    assert_eq!(
        value["usage"]["prompt_tokens"],
        expected_usage.prompt_tokens
    );
    assert_eq!(value["usage"]["total_tokens"], expected_usage.total_tokens);
    let data = value["data"].as_array().expect("data must be an array");
    assert_eq!(data.len(), expected_data.len());
    for (index, item) in data.iter().enumerate() {
        assert_eq!(item["object"], "embedding");
        assert_eq!(item["index"], index);
        let vector = item["embedding"]
            .as_array()
            .expect("embedding must be an array");
        assert_eq!(vector.len(), fixture.dimensions());
        assert!(
            vector
                .iter()
                .all(|value| value.as_f64().is_some_and(f64::is_finite)),
            "embedding {index} must contain only finite values"
        );
        assert_eq!(
            item["embedding"],
            serde_json::to_value(&expected_data[index].embedding)
                .expect("fixture embedding must serialize")
        );
    }
}

#[tokio::test]
async fn pooled_route_returns_distinct_vectors_for_both_pooling_modes() {
    let mut vectors = Vec::new();
    for pooling in ["mean_visual", "last_token"] {
        let response = router(state_with_embedder())
            .oneshot(post_embeddings(json!({"input": "aa", "pooling": pooling})))
            .await
            .expect("router must return a response");
        assert_eq!(response.status(), StatusCode::OK);
        let value = json_body(response).await;
        let vector = value["data"][0]["embedding"]
            .as_array()
            .expect("pooling must return an embedding vector");
        assert_eq!(vector.len(), tiny_embedding_model().dimensions());
        assert!(
            vector
                .iter()
                .all(|value| value.as_f64().is_some_and(f64::is_finite))
        );
        vectors.push(value["data"][0]["embedding"].clone());
    }
    assert_ne!(vectors[0], vectors[1]);
}

#[tokio::test]
async fn pooled_route_compute_failure_discards_earlier_batch_vectors() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(json!({"input": ["a", ""]})))
        .await
        .expect("router must return a response");
    assert_error_envelope(
        response,
        StatusCode::INTERNAL_SERVER_ERROR,
        "inference failed",
        "server_error",
        "internal_error",
    )
    .await;
}
