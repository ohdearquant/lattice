use super::*;
use axum::body::Body;
use base64::Engine as _;
use lattice_inference::serve::metal_worker::{ContextWindowPolicy, spawn_fake_with_vision};
use std::sync::atomic::{AtomicBool, Ordering};
use tower::ServiceExt as _;

fn inline_png_data_uri() -> String {
    let image = image::RgbImage::new(32, 32);
    let mut bytes = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("PNG fixture must encode");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn request(data_uri: &str) -> axum::http::Request<Body> {
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": "after"}
            ]
        }]
    });
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request fixture must build")
}

#[test]
fn worker_bad_request_remains_a_client_error() {
    let error = map_metal_generation_error(ApiError::BadRequest {
        message: "image geometry is unsupported".to_string(),
        code: "invalid_image",
    });
    assert!(matches!(
        error,
        ApiError::BadRequest {
            code: "invalid_image",
            ..
        }
    ));
}

#[tokio::test]
async fn vision_model_accepts_image_and_enqueues_it_on_the_shared_worker() {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let image_seen = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&image_seen);
    let client = spawn_fake_with_vision(
        ContextWindowPolicy::PromptAndMaxTokens,
        4096,
        tokenizer.clone(),
        move |messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
            let image = messages[0]
                .image
                .as_ref()
                .expect("image must reach the common worker job");
            assert!(image.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
            assert_eq!(image.text_offset, "before".len());
            assert_eq!(messages[0].content, "beforeafter");
            seen.store(true, Ordering::SeqCst);
            Ok(GenerateOutput {
                text: "ok".to_string(),
                token_ids: vec![0],
                prompt_tokens,
                generated_tokens: 1,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            })
        },
    );
    let state = AppState {
        model: ModelBackend::Metal {
            handle: MetalHandle { client },
            tokenizer: Arc::new(tokenizer),
            max_context: 4096,
        },
        default_max_tokens: 16,
        max_tokens_cap: 64,
        model_id: "test-model".to_string(),
        request_counter: Arc::new(AtomicU64::new(0)),
        embedding_model: None,
        embedding_admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        cpu_generation_admission: Arc::new(Semaphore::new(CPU_GENERATION_MAX_CONCURRENT_JOBS)),
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        router_state: None,
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        episodes: episode_store(),
    };

    let response = router(state)
        .oneshot(request(&inline_png_data_uri()))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(image_seen.load(Ordering::SeqCst));
}

#[tokio::test]
async fn text_only_model_rejects_the_same_image_with_capability_code() {
    let response = router(tiny_state(64))
        .oneshot(request(&inline_png_data_uri()))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("error body must be readable");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("error body must be JSON");
    assert_eq!(value["error"]["code"], "vision_unsupported");
    assert_eq!(
        value["error"]["message"],
        "image input requires a vision-capable model"
    );
}
