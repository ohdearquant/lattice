use super::*;
use crate::serving_runtime_contract::WorkerFailure;
use crate::serving_worker::{
    ScoreOutcome, WorkerClient, WorkerMessage, control_unsupported, run_worker_loop_with_lora,
};
use axum::body::Body;
use axum::http::Request;
use lattice_inference::generation::GenerateOutput;
use lattice_tokenizer::TokenizedInput;
use tokio::sync::{Semaphore, mpsc};
use tower::ServiceExt as _;

/// A byte-level tokenizer, so the route runs without a checkpoint.
struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedInput {
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();
        let length = ids.len();
        TokenizedInput {
            input_ids: ids,
            attention_mask: vec![1; length],
            token_type_ids: vec![0; length],
            real_length: length,
            pre_truncation_len: length,
        }
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }

    fn vocab_size(&self) -> usize {
        4096
    }

    fn max_seq_len(&self) -> usize {
        usize::MAX
    }
}

fn decider() -> Arc<DeciderServing> {
    Arc::new(DeciderServing::from_parts(
        Arc::new(ByteTokenizer),
        DecisionLabelTable {
            labels: (0..255).map(|index| format!("L{index}")).collect(),
            ids: (0..255).map(|index| 1000 + index as u32).collect(),
            open_ids: vec![10, 40],
        },
        DeciderConfig {
            version: "0.8b-v1".to_string(),
            temperature: 1.0,
            max_state_tokens: 32768,
            max_options: 255,
            isolated_levels: true,
        },
    ))
}

/// A served route whose worker answers every row with `logits(row)`.
fn served(
    max_pending: usize,
    score: impl FnMut(&[ScoreRow]) -> ScoreOutcome + Send + 'static,
) -> (Router, std::thread::JoinHandle<()>) {
    let (tx, rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let worker = std::thread::spawn(move || {
        run_worker_loop_with_lora(
            rx,
            |_, _, _, _, _, _, _| -> Result<GenerateOutput, WorkerFailure> {
                panic!("no generation is submitted in these tests")
            },
            control_unsupported,
            score,
        );
    });
    let client = WorkerClient::unattached_for_test(tx, Arc::new(Semaphore::new(max_pending)));
    (
        router(ServingHandle::affine_standalone(client), decider()),
        worker,
    )
}

async fn post_body(app: Router, body: &str) -> (StatusCode, serde_json::Value) {
    let response = app
        .oneshot(
            Request::post("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(body.to_owned()))
                .expect("request"),
        )
        .await
        .expect("served");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, serde_json::from_slice(&bytes).expect("json body"))
}

const ONE_CHOICE: &str = r#"{"state": "The sky is clear.", "questions": {"colour": {"type": "choice", "instructions": "Pick the colour.", "criteria": ["red", "green", "blue"]}}}"#;

#[tokio::test]
async fn a_request_is_scored_by_the_worker_and_answered() {
    let (app, _worker) = served(4, |rows| {
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].labels, vec![1000, 1001, 1002]);
        Ok(vec![vec![0.0, 3.0, 1.0]])
    });
    let (status, body) = post_body(app, ONE_CHOICE).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "decider-0.8b-v1");
    assert_eq!(body["answers"]["colour"]["choice"], "green", "{body}");
    assert!(body["usage"]["input_tokens"].as_u64().expect("count") > 0);
    assert_eq!(body["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn a_malformed_request_is_refused_with_422_and_a_detail() {
    let (app, _worker) = served(4, |_| panic!("a refused request is never scored"));
    let (status, body) = post_body(app, r#"{"state": "x"}"#).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"].is_string(), "{body}");
}

#[tokio::test]
async fn a_scorer_failure_is_a_server_error() {
    let (app, _worker) = served(4, |_| Err("gpu lost".to_string()));
    let (status, body) = post_body(app, ONE_CHOICE).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
}

#[tokio::test]
async fn a_short_worker_reply_is_a_server_error() {
    let (app, _worker) = served(4, |_| Ok(Vec::new()));
    let (status, body) = post_body(app, ONE_CHOICE).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
}

#[test]
fn a_directory_without_a_decider_config_serves_no_route() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(
        DeciderServing::load(dir.path())
            .expect("not an error")
            .is_none()
    );
}

#[test]
fn a_decider_config_without_a_tokenizer_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("decider_config.json"),
        r#"{"temperature": 1.03, "temperature_schema_first": 1.03, "neutralize_none": false, "version": "0.8b-v1", "base": "b", "max_options": 255, "max_state_tokens": 32768, "schema_first": false, "schema_first_trained": true, "isolated_levels": true, "recipe": "r", "release_date": "2026-09-19"}"#,
    )
    .expect("write config");
    assert!(DeciderServing::load(dir.path()).is_err());
}
