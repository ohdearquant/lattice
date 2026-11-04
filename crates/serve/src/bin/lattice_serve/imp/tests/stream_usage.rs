use super::*;
use lattice_serve::serve::stream_usage::test_support::{self, COMPLETIONS, Completion, OPTIONS};
use std::time::Duration;
use tower::ServiceExt as _;

async fn request(
    body: serde_json::Value,
    fail: bool,
    completion: Completion,
) -> (StatusCode, String) {
    let (state, mut jobs) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs.recv().await {
            let _ = job.reply(WorkerEvent::Delta("é".into()));
            if fail {
                let _ = job.reply(WorkerEvent::Failed("fixture failure".into()));
            } else {
                let _ = job.reply(WorkerEvent::Complete(test_support::output(completion)));
            }
        }
    });
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
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn body() -> serde_json::Value {
    json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true})
}

#[tokio::test]
async fn dropping_stream_body_closes_worker_receiver_for_every_usage_option() {
    for option in OPTIONS {
        let (state, mut jobs) = test_app_state_with_jobs();
        let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel::<()>();
        let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(async move {
            if let Some(WorkerMessage::Generate(job)) = jobs.recv().await {
                let sent = job.reply(WorkerEvent::Delta("é".into()));
                let signaled = matches!(
                    tokio::time::timeout(Duration::from_secs(10), dropped_rx).await,
                    Ok(Ok(()))
                );
                let receiver_closed =
                    sent && signaled && !job.reply(WorkerEvent::Delta("late".into()));
                let _ = observed_tx.send(receiver_closed);
            }
        });
        let req = test_support::stream_request(body(), option);
        let response = router(state).oneshot(req).await.unwrap();
        test_support::assert_body_drop(response, option).await;
        dropped_tx.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(10), observed_rx)
                .await
                .unwrap()
                .unwrap()
        );
        worker.await.unwrap();
    }
}

#[tokio::test]
async fn stream_usage_frames_use_authoritative_counts_and_preserve_default() {
    for option in OPTIONS {
        for completion in COMPLETIONS {
            let mut input = body();
            if let Some(enabled) = option {
                input["stream_options"] = json!({"include_usage": enabled});
            }
            let (status, text) = request(input, false, completion).await;
            assert_eq!(status, StatusCode::OK);
            test_support::assert_frames(&text, option, completion);
        }
    }
}

#[tokio::test]
async fn failed_or_interrupted_stream_does_not_claim_final_usage() {
    for (fail, completion) in [(true, Completion::Stop), (false, Completion::Interrupt)] {
        let mut input = body();
        input["stream_options"] = json!({"include_usage":true});
        let (status, text) = request(input, fail, completion).await;
        assert_eq!(status, StatusCode::OK);
        let frames = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(frames.last(), Some(&"[DONE]"));
        let chunks = frames[..frames.len() - 1]
            .iter()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(
            chunks
                .iter()
                .all(|chunk| !chunk["choices"].as_array().is_some_and(Vec::is_empty))
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.get("usage").is_none_or(|usage| usage.is_null()))
        );
        if fail {
            assert!(chunks.iter().any(|chunk| chunk.get("error").is_some()));
        }
    }
}

#[tokio::test]
async fn stream_options_reject_nonstream_and_ignore_unknown_fields() {
    let mut input = body();
    input["stream"] = json!(false);
    input["stream_options"] = json!({"include_usage":true});
    let (status, text) = request(input, false, Completion::Stop).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&text).unwrap()["error"]["code"],
        "unsupported_feature"
    );
    let mut input = body();
    input["stream_options"] =
        json!({"include_usage":true,"include_obfuscation":false,"future_option":"ignored"});
    let (status, text) = request(input, false, Completion::Stop).await;
    assert_eq!(status, StatusCode::OK);
    let frames = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 5, "{text}");
    assert_eq!(frames.last(), Some(&"[DONE]"));
    let usage: serde_json::Value = serde_json::from_str(frames[3]).unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(
        usage["usage"],
        json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5})
    );
}
