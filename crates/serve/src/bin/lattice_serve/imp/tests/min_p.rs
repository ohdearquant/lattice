use super::*;
use tower::ServiceExt as _;

fn request_body(streaming: bool, value: Option<serde_json::Value>) -> serde_json::Value {
    let mut body =
        json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":streaming});
    if let Some(value) = value {
        body["min_p"] = value;
    }
    body
}

fn request(body: serde_json::Value) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn standalone_min_p_reaches_worker_in_both_modes() {
    for streaming in [false, true] {
        for (value, expected) in [
            (None, 0.0),
            (Some(serde_json::Value::Null), 0.0),
            (Some(json!(0.0)), 0.0),
            (Some(json!(0.5)), 0.5),
            (Some(json!(1.0)), 1.0),
        ] {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
            let captured = Arc::clone(&observed);
            let jobs = spawn_fake(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                tokenizer,
                move |_messages, cfg, prompt_tokens, on_token, _cancel| {
                    captured.lock().unwrap().push(cfg.min_p);
                    let _ = on_token("ok", 1);
                    Ok(GenerateOutput {
                        text: "ok".into(),
                        token_ids: vec![1],
                        prompt_tokens,
                        generated_tokens: 1,
                        stopped: true,
                        stop_reason: Some(lattice_inference::StopReason::Eos),
                        token_logprobs: vec![],
                    })
                },
            );
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults: GenerationDefaults::standard(100),
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: Arc::new(vec![]),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                embedding: None,
            };
            let response = router(state)
                .oneshot(request(request_body(streaming, value)))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(*observed.lock().unwrap(), vec![expected]);
        }
    }
}

#[tokio::test]
async fn standalone_min_p_rejects_invalid_before_worker_and_preserves_precedence() {
    for streaming in [false, true] {
        for value in [
            json!(-0.1),
            json!(1.1),
            json!("bad"),
            json!(true),
            json!([]),
            json!({}),
        ] {
            let (state, mut jobs) = test_app_state_with_jobs();
            let response = router(state)
                .oneshot(request(request_body(streaming, Some(value))))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["code"], "invalid_min_p");
            assert!(jobs.try_recv().is_err());
        }
        let (state, mut jobs) = test_app_state_with_jobs();
        let mut body = request_body(streaming, Some(json!(2)));
        body["top_p"] = json!(0);
        let response = router(state).oneshot(request(body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["code"], "invalid_top_p");
        assert!(jobs.try_recv().is_err());
    }
}
