//! The `reasoning_content` field of a chat completion: on the message and on stream deltas when
//! the reply has reasoning, in generated order before the content, and absent from both when it
//! has none.

use super::*;
use lattice_serve::serve::stream_usage::test_support::{self, Completion};
use serde_json::{Value, json};
use std::sync::Mutex;
use std::time::Duration;
use tower::ServiceExt as _;

/// A CPU host whose replies pass each of `reasoning` to the reasoning sink, then `é` to the
/// content sink.
fn reasoning_state(reasoning: &'static [&'static str]) -> AppState {
    let model = Arc::new(lattice_inference::model::qwen35::test_support::tiny_zero_model());
    tiny_state_with_cpu(
        64,
        lattice_serve::serving_cpu_host::SharedCpuHandle::with_reasoning_generator_for_test(
            model,
            move |_, _, on_content, on_reasoning, _| {
                for piece in reasoning {
                    on_reasoning(piece);
                }
                on_content("é");
                Ok((test_support::output(Completion::Stop), reasoning.concat()))
            },
        ),
    )
}

/// A CPU host whose runtime has no reasoning field and keeps the default reasoning method.
fn plain_state() -> AppState {
    tiny_state_with_test_generator(64, |_, _, on_token, _| {
        on_token("é");
        Ok(test_support::output(Completion::Stop))
    })
}

fn request(stream: bool) -> axum::http::Request<axum::body::Body> {
    let body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "stream": stream,
    });
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap()
}

async fn body(state: AppState, stream: bool) -> String {
    let response = router(state).oneshot(request(stream)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn message(state: AppState) -> Value {
    let text = body(state, false).await;
    let response: Value = serde_json::from_str(&text).unwrap();
    response["choices"][0]["message"].clone()
}

/// The `delta` of every chunk of a streamed reply, after checking that it ends in `[DONE]`.
async fn deltas(state: AppState) -> (Vec<Value>, String) {
    let text = body(state, true).await;
    let frames: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(frames.last(), Some(&"[DONE]"), "{text}");
    let deltas = frames[..frames.len() - 1]
        .iter()
        .map(|frame| serde_json::from_str::<Value>(frame).unwrap()["choices"][0]["delta"].clone())
        .collect();
    (deltas, text)
}

#[tokio::test]
async fn a_completion_carries_reasoning_content_only_when_the_reply_has_reasoning() {
    assert_eq!(
        message(reasoning_state(&["th", "ink"])).await,
        json!({"role": "assistant", "content": "é", "reasoning_content": "think"})
    );
    assert_eq!(
        message(reasoning_state(&[])).await,
        json!({"role": "assistant", "content": "é"})
    );
}

#[tokio::test]
async fn a_stream_sends_the_reasoning_between_the_role_chunk_and_the_content() {
    let (with_reasoning, _) = deltas(reasoning_state(&["th", "ink"])).await;
    assert_eq!(
        with_reasoning,
        vec![
            json!({"role": "assistant"}),
            json!({"reasoning_content": "th"}),
            json!({"reasoning_content": "ink"}),
            json!({"content": "é"}),
            json!({}),
        ]
    );
    let (without, text) = deltas(reasoning_state(&[])).await;
    assert_eq!(
        without,
        vec![
            json!({"role": "assistant"}),
            json!({"content": "é"}),
            json!({}),
        ]
    );
    assert!(!text.contains("reasoning"), "{text}");
}

/// A runtime with no reasoning field answers exactly as it did before the field existed, on the
/// completion call and on the streaming call.
#[tokio::test]
async fn a_runtime_without_a_reasoning_field_answers_without_the_key() {
    let text = body(plain_state(), false).await;
    assert!(!text.contains("reasoning"), "{text}");
    let response: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        response["choices"][0]["message"],
        json!({"role": "assistant", "content": "é"})
    );
    let (deltas, text) = deltas(plain_state()).await;
    assert!(!text.contains("reasoning"), "{text}");
    assert_eq!(
        deltas,
        vec![
            json!({"role": "assistant"}),
            json!({"content": "é"}),
            json!({}),
        ]
    );
}

/// A client that drops the stream after the first reasoning piece: the next reasoning piece is
/// refused and the cancel flag is set, so a runtime stops before any content.
#[tokio::test]
async fn a_client_that_disconnects_during_reasoning_refuses_the_next_piece_and_cancels() {
    let (sent_tx, sent_rx) = tokio::sync::oneshot::channel();
    let sent_tx = Mutex::new(Some(sent_tx));
    let (dropped_tx, dropped_rx) = std::sync::mpsc::channel::<()>();
    let dropped_rx = Mutex::new(dropped_rx);
    let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
    let observed_tx = Mutex::new(Some(observed_tx));
    let model = Arc::new(lattice_inference::model::qwen35::test_support::tiny_zero_model());
    let state = tiny_state_with_cpu(
        64,
        lattice_serve::serving_cpu_host::SharedCpuHandle::with_reasoning_generator_for_test(
            model,
            move |_, _, _on_content, on_reasoning, should_cancel| {
                let first = on_reasoning("th");
                if let Some(tx) = sent_tx.lock().unwrap().take() {
                    let _ = tx.send(());
                }
                let dropped = dropped_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .is_ok();
                let next = on_reasoning("ink");
                if let Some(tx) = observed_tx.lock().unwrap().take() {
                    let _ = tx.send((first, dropped, next, should_cancel()));
                }
                Ok((
                    test_support::output(Completion::Interrupt),
                    "think".to_owned(),
                ))
            },
        ),
    );
    let response = router(state).oneshot(request(true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    tokio::time::timeout(Duration::from_secs(10), sent_rx)
        .await
        .unwrap()
        .unwrap();
    drop(response);
    dropped_tx.send(()).unwrap();
    let observed = tokio::time::timeout(Duration::from_secs(10), observed_rx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(observed, (true, true, false, true));
}
