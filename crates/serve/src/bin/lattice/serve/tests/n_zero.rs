use super::*;
use tower::ServiceExt as _;

#[tokio::test]
async fn n_zero_rejected_before_worker_in_both_modes() {
    for streaming in [false, true] {
        for (n, expected_error) in [
            (None, None),
            (Some(serde_json::json!(null)), None),
            (Some(serde_json::json!(1)), None),
            (Some(serde_json::json!(0)), Some("n must be at least 1")),
            (Some(serde_json::json!(2)), Some("n > 1 is not supported")),
        ] {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            let state = tiny_state_with_test_generator(64, move |_, _, on_token, _| {
                observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = on_token("ok");
                Ok(GenerateOutput {
                    text: "ok".into(),
                    token_ids: vec![1],
                    prompt_tokens: 1,
                    generated_tokens: 1,
                    stopped: true,
                    stop_reason: Some(lattice_inference::StopReason::Eos),
                    token_logprobs: vec![],
                })
            });
            let mut body = serde_json::json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":streaming});
            if let Some(n) = n {
                body["n"] = n;
            }
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            if let Some(message) = expected_error {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(error["error"]["code"], "unsupported_feature");
                assert_eq!(error["error"]["message"], message);
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
            } else {
                assert_eq!(status, StatusCode::OK);
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            }
        }
    }
}
