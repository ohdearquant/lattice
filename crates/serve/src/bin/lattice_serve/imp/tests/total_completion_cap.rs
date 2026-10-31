use super::*;
use tower::ServiceExt as _;

#[tokio::test]
async fn total_completion_cap_reaches_worker_and_controls_response_in_both_modes() {
    for streaming in [false, true] {
        for (fields, returned, expected_cap, expected_error) in [
            (
                json!({"max_completion_tokens":5,"reasoning_budget":3}),
                5,
                Some(5),
                false,
            ),
            (
                json!({"max_completion_tokens":5,"max_tokens":5,"reasoning_budget":3}),
                5,
                Some(5),
                false,
            ),
            (
                json!({"max_completion_tokens":5,"reasoning_budget":3}),
                6,
                Some(5),
                true,
            ),
            (json!({"max_tokens":5,"reasoning_budget":3}), 6, None, false),
            (
                json!({"max_completion_tokens":5,"reasoning_budget":3,"chat_template_kwargs":{"enable_thinking":false}}),
                5,
                Some(5),
                false,
            ),
            (
                json!({"max_completion_tokens":5,"chat_template_kwargs":{"enable_thinking":true}}),
                5,
                Some(5),
                false,
            ),
        ] {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let observed = Arc::clone(&seen);
            let jobs = spawn_fake(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                tokenizer,
                move |_, cfg, prompt_tokens, on_token, _| {
                    observed.lock().unwrap().push((
                        cfg.total_token_cap,
                        cfg.reasoning_budget,
                        cfg.enable_thinking,
                    ));
                    let _ = on_token("x", 1);
                    Ok(GenerateOutput {
                        text: "x".into(),
                        token_ids: vec![1; returned],
                        prompt_tokens,
                        generated_tokens: returned,
                        stopped: false,
                        stop_reason: Some(lattice_inference::StopReason::Length),
                        token_logprobs: vec![],
                    })
                },
            );
            let defaults = GenerationDefaults {
                reasoning_budget: Some(3),
                ..GenerationDefaults::standard(100)
            };
            let state = AppState {
                jobs,
                model_id: Arc::from("test-model"),
                defaults,
                model_max_context: 4096,
                max_pending: 1_000_000,
                metrics: Arc::new(ServeMetrics::default()),
                vocab_bytes: Arc::new(vec![]),
                grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
                embedding: None,
            };
            let mut body = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":streaming});
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            if streaming {
                body["stream_options"] = json!({"include_usage":true});
            }
            let req = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(req).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let payload = String::from_utf8(bytes.to_vec()).unwrap();
            let observed = seen.lock().unwrap();
            assert_eq!(observed.len(), 1);
            assert_eq!(observed[0].0, expected_cap);
            if fields["chat_template_kwargs"]["enable_thinking"] == json!(false) {
                assert_eq!((observed[0].1, observed[0].2), (None, false));
            }
            if fields["chat_template_kwargs"]["enable_thinking"] == json!(true) {
                assert_eq!((observed[0].1, observed[0].2), (Some(3), true));
            }
            if expected_error {
                if streaming {
                    assert!(payload.contains("internal_error"), "{payload}");
                } else {
                    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
                }
            } else {
                assert_eq!(status, StatusCode::OK, "{payload}");
                if streaming {
                    assert!(
                        payload.contains(&format!("\"completion_tokens\":{returned}")),
                        "{payload}"
                    );
                } else {
                    let json: serde_json::Value = serde_json::from_str(&payload).unwrap();
                    assert_eq!(json["usage"]["completion_tokens"], returned);
                }
            }
        }
    }
}
