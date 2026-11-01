use super::*;
use tower::ServiceExt as _;

#[tokio::test]
async fn tool_history_is_rejected_before_submission_and_empty_is_accepted() {
    for streaming in [false, true] {
        for field in [None, Some(json!(null)), Some(json!([]))] {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            let jobs = spawn_fake(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                tokenizer,
                move |_messages, _cfg, prompt_tokens, on_token, _cancel| {
                    observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
            let mut message = json!({"role":"assistant","content":"hello"});
            if let Some(field) = field {
                message["tool_calls"] = field;
            }
            let body = json!({"model":"test-model","messages":[message,{"role":"user","content":"hi"}],"max_tokens":8,"stream":streaming});
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        }
        for (message, field) in [
            (
                json!({"role":"assistant","content":"hi","tool_calls":[{"id":"call"}]}),
                "tool_calls",
            ),
            (
                json!({"role":"assistant","content":"hi","tool_calls":{}}),
                "tool_calls",
            ),
            (
                json!({"role":"assistant","content":"hi","tool_calls":"bad"}),
                "tool_calls",
            ),
            (
                json!({"role":"assistant","content":null,"tool_calls":[{}]}),
                "tool_calls",
            ),
            (json!({"role":"tool","content":"hi"}), "role"),
            (json!({"role":"tool","content":null}), "role"),
        ] {
            let (state, mut jobs) = test_app_state_with_jobs();
            let body = json!({"model":"test-model","messages":[message,{"role":"user","content":"hi"}],"max_tokens":8,"stream":streaming});
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                router(state).oneshot(request),
            )
            .await
            .expect("unsupported tool-history request reached the unserved worker")
            .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["error"]["code"], "unsupported_feature");
            let expected = if field == "role" {
                "role 'tool' is not supported by this server"
            } else {
                "messages[].tool_calls is not supported by this server"
            };
            assert_eq!(error["error"]["message"], expected);
            assert!(jobs.try_recv().is_err());
        }
    }
}
