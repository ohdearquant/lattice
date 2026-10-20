#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod bert_pooling_tests;

fn input_message(role: String, content: MessageContent) -> InMsg {
    let mut message: InMsg = serde_json::from_str(r#"{"role":"","content":""}"#).unwrap();
    message.role = role;
    message.content = content;
    message
}

fn input_image_url(url: String, detail: Option<String>) -> ImageUrl {
    let mut image: ImageUrl = serde_json::from_str(r#"{"url":""}"#).unwrap();
    image.url = url;
    image.detail = detail;
    image
}

use super::*;
use lattice_inference::model::qwen35_config::Qwen35Config;
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
use lattice_inference::serve::ApiError;
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
use lattice_inference::serve::metal_worker::{
    WorkerMessage, spawn_fake, spawn_fake_with_vision, test_client_and_jobs,
};
use lattice_inference::serve::prepare::build_cfg;
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
use std::sync::Arc;

/// `Content-Type: application/json` `HeaderMap`, for tests that call
/// `chat_completions` directly as a Rust function (bypassing the
/// router) and need to satisfy its `require_json_content_type`
/// guard to exercise the behavior downstream of it.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn test_json_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    headers
}

fn normalize_for_cfg(
    req: &ChatReq,
    defaults: &GenerationDefaults,
    model_max_context: usize,
) -> ValidatedChatRequest {
    normalize_request(
        req,
        *defaults,
        ServeProfile::lattice_serve("", model_max_context),
    )
    .unwrap()
}

// NOTE (issue #832): the FIFO/cancellation/window-check worker-loop
// test suite that used to live here (`fake_generate`,
// `fake_generate_with_prefill_gap`, `make_job`,
// `queued_job_cancelled_before_dequeue_is_skipped_entirely`,
// `running_job_cancelled_midstream_stops_early_and_worker_survives`,
// `running_job_cancelled_during_prefill_like_phase_never_calls_on_token`,
// `fake_generate_fails_once_then_succeeds`,
// `generation_failure_is_reported_as_ev_failed_not_ev_done`, and the
// three `check_prompt_fits_window_*` tests further below) exercised
// this binary's private `Job`/`Ev`/`run_worker_loop`/
// `check_prompt_fits_window`, all of which moved to the shared
// `lattice_inference::serve::metal_worker` module and are covered by
// that module's own `#[cfg(test)] mod tests` instead (ported
// verbatim, same assertions, same fake generators -- see
// `crates/inference/src/serve/metal_worker.rs`). The HTTP-level
// tests below still exercise THIS binary's real `chat_completions`,
// now routed through `MetalWorkerClient`/`WorkerEvent` via the
// `metal_worker` `test-utils` seam (`test_client_and_jobs`/
// `spawn_fake`) instead of this binary's own private job channel.

/// #832 checklist: "Add a Metal-feature test asserting an explicit
/// common-worker marker, so a silently-reintroduced per-binary
/// fallback cannot pass green." `AppState.jobs`'s field type is
/// private, so this test can only be written from inside `mod imp`
/// (same module as `AppState`, so private-field access is allowed)
/// -- it constructs an `AppState` directly from a
/// `lattice_inference::serve::metal_worker::MetalWorkerClient`,
/// which only type-checks if `AppState.jobs` still holds that exact
/// shared type. A private per-binary fallback job channel (any type
/// other than the shared `MetalWorkerClient`) would fail to compile
/// here, not just fail some behavioral assertion at runtime. See
/// `lattice.rs`'s
/// `metal_handle_is_backed_by_the_shared_metal_worker_client` for
/// the sibling marker on the other binary.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[test]
fn app_state_is_backed_by_the_shared_metal_worker_client() {
    fn build_from_shared_client(jobs: MetalWorkerClient) -> AppState {
        AppState {
            jobs,
            model_id: Arc::from("marker-test-model"),
            defaults: GenerationDefaults::standard(100),
            model_max_context: 4096,
            max_pending: 1_000_000,
            metrics: Arc::new(ServeMetrics::default()),
            vocab_bytes: Arc::new(vec![]),
            grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            embedding: None,
        }
    }
    let (jobs, _rx) = test_client_and_jobs();
    let _state: AppState = build_from_shared_client(jobs);
}

// ── #641 / #649 request parsing and clamp tests ──────────────────

#[test]
fn message_content_plain_string_normalizes() {
    let msg = input_message("user".to_string(), MessageContent::Text("hi".to_string()));
    let chat_message =
        normalize_messages(std::slice::from_ref(&msg)).expect("plain string content must parse");
    assert_eq!(
        chat_message[0].role,
        lattice_inference::serve::contract::NormalizedChatRole::User
    );
    assert_eq!(chat_message[0].content, "hi");
}

#[test]
fn message_content_parts_concatenate_in_order() {
    let msg = input_message(
        "user".to_string(),
        MessageContent::Parts(vec![
            Part::Text {
                text: "a".to_string(),
            },
            Part::Text {
                text: "b".to_string(),
            },
        ]),
    );
    let chat_message = normalize_messages(std::slice::from_ref(&msg)).unwrap();
    assert_eq!(chat_message[0].content, "ab");
}

#[test]
fn message_content_image_url_rejected() {
    let msg = input_message(
        "user".to_string(),
        MessageContent::Parts(vec![Part::ImageUrl {
            image_url: input_image_url("https://example.com/cat.png".to_string(), None),
        }]),
    );
    let err = normalize_messages(std::slice::from_ref(&msg)).unwrap_err();
    assert_eq!(err.message(), IMAGE_REQUIRES_VISION_MESSAGE);
}

#[test]
fn message_content_unknown_part_rejected() {
    // The shared normalizer's wording ("content part type 'file' is
    // not supported; ...") differs from this binary's now-removed
    // local test-only copy ("unsupported content part type 'file';
    // ..."); production always went through the shared normalizer
    // (`normalize_request`), so this pins the wording clients
    // actually receive.
    let msg = input_message(
        "user".to_string(),
        MessageContent::Parts(vec![Part::Unsupported {
            kind: "file".to_string(),
        }]),
    );
    let err = normalize_messages(std::slice::from_ref(&msg)).unwrap_err();
    assert_eq!(
        err.message(),
        "content part type 'file' is not supported; only 'text' and 'image_url' parts are \
                 accepted"
    );
}

#[test]
fn message_role_unknown_rejected() {
    // Not an OpenAI chat role at all -- `invalid_role`, matching the
    // shared contract's production normalization for the same case.
    let err = MessageRole::parse("moderator").unwrap_err();
    assert_eq!(
        err.message(),
        "unsupported role 'moderator'; must be 'system', 'user', or 'assistant'"
    );
    assert_eq!(err.code(), "invalid_role");
}

#[test]
fn message_role_tool_and_developer_rejected_as_unsupported_feature() {
    // A real OpenAI role this server does not implement --
    // `unsupported_feature`, matching `lattice.rs`'s split between
    // "not a role" and "a role we don't support" (ADR-080 C2).
    for role in ["tool", "developer"] {
        let err = MessageRole::parse(role).unwrap_err();
        assert_eq!(err.code(), "unsupported_feature");
    }
}

#[test]
fn parse_chat_req_rejects_too_many_parts_before_typed_parse() {
    let parts: Vec<String> = (0..65)
        .map(|i| format!(r#"{{"type":"text","text":"p{i}"}}"#))
        .collect();
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{}]}}]}}"#,
        parts.join(",")
    );
    let err = parse_chat_req(body.as_bytes()).unwrap_err();
    assert_eq!(
        err.message(),
        "messages[0].content has too many parts; maximum is 64"
    );
}

/// #656 boundary fix: exactly `MAX_CONTENT_PARTS_PER_MESSAGE` (64)
/// parts is the documented maximum and MUST be accepted — only
/// MAX+1 (covered above) is rejected.
#[test]
fn parse_chat_req_accepts_max_parts_boundary() {
    assert_eq!(MAX_CONTENT_PARTS_PER_MESSAGE, 64);
    let parts: Vec<String> = (0..MAX_CONTENT_PARTS_PER_MESSAGE)
        .map(|i| format!(r#"{{"type":"text","text":"p{i}"}}"#))
        .collect();
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{}]}}]}}"#,
        parts.join(",")
    );
    let req = parse_chat_req(body.as_bytes()).expect("exactly 64 parts must be accepted");
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn parse_chat_req_rejects_oversized_text_part_before_typed_parse() {
    let big_text = "x".repeat(MAX_CONTENT_PART_BYTES + 1);
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"{big_text}"}}]}}]}}"#,
    );
    let err = parse_chat_req(body.as_bytes()).unwrap_err();
    assert_eq!(err.message(), "messages[0].content[0] exceeds 65536 bytes");
}

#[test]
fn build_cfg_clamps_to_runtime_context() {
    let defaults = GenerationDefaults {
        reasoning_budget: Some(50),
        ..GenerationDefaults::standard(100)
    };
    let mut req = serde_json::from_str::<ChatReq>("{}").unwrap();
    req.messages = vec![input_message(
        "user".to_string(),
        MessageContent::Text("hi".to_string()),
    )];
    req.reasoning_budget =
        Some(serde_json::value::RawValue::from_string("50".to_string()).unwrap());
    let normalized = normalize_for_cfg(&req, &defaults, 8);
    let cfg = build_cfg(&normalized);
    assert!(cfg.max_new_tokens <= 8);
    let reasoning_budget = cfg.reasoning_budget.unwrap_or(0);
    assert!(reasoning_budget + cfg.max_new_tokens < 8);
}

// NOTE (issue #832): the three `check_prompt_fits_window_*` tests
// that used to live here (`build_cfg`-clamped `GenerateConfig`
// values fed into this binary's private `check_prompt_fits_window`)
// no longer compile: the check moved to
// `lattice_inference::serve::metal_worker`, which does not expose it
// (it is a private implementation detail of the shared worker loop,
// not part of that module's public/test-utils surface). The same
// math is covered by `check_prompt_fits_window`'s own pure unit
// tests, ported verbatim to `metal_worker.rs`'s test suite; the same
// `build_cfg` -> worker -> window-check composition this file's
// deleted tests exercised is covered end to end, through a REAL
// worker and a real request, by `real_router_overflow_parity` and
// `parity_table` below. `build_cfg`'s own clamping arithmetic (the
// half of the invariant that stays in THIS file) keeps its direct
// coverage in `build_cfg_clamps_to_runtime_context` above.

#[test]
fn model_context_uses_config_max_position_embeddings() {
    let mut cfg = Qwen35Config::qwen35_2b();
    cfg.max_position_embeddings = 12345;
    assert_eq!(model_context_from_config(Some(&cfg)), 12345);
}

#[test]
fn model_context_falls_back_to_4096_when_absent() {
    assert_eq!(model_context_from_config(None), FALLBACK_MODEL_MAX_CONTEXT);
}

// ── #939 `--max-pending` daemon-arg boundary tests ────────────────
//
// `parse_max_pending` only rejects a MALFORMED string here (this
// binary's own hand-rolled argv parser, unlike `lattice.rs`'s clap
// `value_parser`); zero and above-`Semaphore::MAX_PERMITS` are valid
// `usize`s that parse fine and are instead caught once, downstream,
// by `MetalWorker::spawn`'s own `StartupError::InvalidMaxPending`
// (covered by that module's own boundary tests).

#[test]
fn resident_limits_defaults_and_overrides() {
    let defaults = parse_residency_limits(&[]).unwrap();
    assert_eq!(
        (defaults.max_adapters, defaults.max_bytes),
        (32, 536_870_912)
    );
    let args = [
        "--max-resident-adapters",
        "2",
        "--max-resident-adapter-bytes",
        "128",
    ]
    .map(String::from);
    let limits = parse_residency_limits(&args).unwrap();
    assert_eq!((limits.max_adapters, limits.max_bytes), (2, 128));
}

fn assert_resident_flag_rejects_invalid(flag: &str) {
    let args = [flag, "8"].map(String::from);
    parse_residency_limits(&args).unwrap();
    for value in ["not-a-number", "0", "-1", "18446744073709551616"] {
        let args = [flag, value].map(String::from);
        let err = parse_residency_limits(&args).unwrap_err();
        assert!(err.contains(flag), "{err}");
    }
    assert!(
        parse_residency_limits(&[flag.to_string()])
            .unwrap_err()
            .contains(flag)
    );
}

#[test]
fn resident_count_malformed_is_rejected() {
    assert_resident_flag_rejects_invalid("--max-resident-adapters");
}

#[test]
fn resident_bytes_malformed_is_rejected() {
    assert_resident_flag_rejects_invalid("--max-resident-adapter-bytes");
}

#[test]
fn max_pending_omitted_defaults_to_32() {
    let args: Vec<String> = vec![];
    assert_eq!(
        parse_max_pending(&args).expect("no --max-pending flag"),
        lattice_inference::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS,
    );
}

#[test]
fn max_pending_valid_override_is_accepted() {
    let args: Vec<String> = vec!["--max-pending".to_string(), "8".to_string()];
    assert_eq!(parse_max_pending(&args).expect("8 is well-formed"), 8);
}

#[test]
fn max_pending_negative_is_rejected_not_silently_defaulted() {
    let args: Vec<String> = vec!["--max-pending".to_string(), "-1".to_string()];
    let err = parse_max_pending(&args)
        .expect_err("-1 must be rejected, not silently replaced by the default");
    assert!(
        err.contains("-1"),
        "error must name the offending value: {err}"
    );
}

#[test]
fn max_pending_malformed_is_rejected_not_silently_defaulted() {
    let args: Vec<String> = vec!["--max-pending".to_string(), "not-a-number".to_string()];
    let err = parse_max_pending(&args)
        .expect_err("a non-numeric value must be rejected, not silently replaced");
    assert!(
        err.contains("not-a-number"),
        "error must name the offending value: {err}"
    );
}

#[test]
fn max_pending_zero_parses_ok_and_is_left_to_metal_worker_spawn_to_reject() {
    // Zero is a syntactically valid `usize` -- `parse_max_pending`
    // itself does not range-check it (that check lives once, in
    // `MetalWorker::spawn`, shared with `lattice.rs`). This test
    // pins that division of responsibility down: it must NOT start
    // rejecting zero itself without `MetalWorker::spawn`'s own
    // boundary tests changing too.
    let args: Vec<String> = vec!["--max-pending".to_string(), "0".to_string()];
    assert_eq!(
        parse_max_pending(&args).expect("0 parses as a valid usize"),
        0
    );
}

// ── HTTP-level 400 tests ──────────────────────────────────────────
//
// All three failure modes below (`#641` unknown role, `#649` image
// part, `#649` oversized part) return from `chat_completions` before
// a job is ever submitted to `s.jobs`, so a `MetalWorkerClient` with
// no running worker behind it (issue #832's `test_client_and_jobs`
// seam, receiver half discarded) is a faithful stand-in: no GPU, no
// model load.

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
/// ADR-095 decision 4: every entry in the shared route list is registered
/// in THIS binary. A route added to the list and forgotten here reds this
/// test.
///
/// The assertion is only that the response is not 404. A 405 or a 4xx from
/// validation both count as registered -- the test asks whether the route
/// exists and nothing else, because anything more would need the state the
/// two binaries do not share. Asserting a specific status would make this
/// a behaviour test that passes or fails for reasons unrelated to
/// registration.
#[tokio::test]
async fn every_shared_lora_route_is_registered_in_this_binary() {
    use lattice_inference::serve::lora::LORA_ROUTES;
    use tower::ServiceExt;

    assert!(
        !LORA_ROUTES.is_empty(),
        "the route list is empty, so this test would pass while checking nothing"
    );

    for (path, methods) in LORA_ROUTES {
        for method in *methods {
            let request = axum::http::Request::builder()
                .method(*method)
                .uri(*path)
                .header("content-type", "application/json")
                .body(axum::body::Body::from("{}"))
                .expect("fixture request must build");
            let response = router(test_app_state())
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_ne!(
                response.status(),
                axum::http::StatusCode::NOT_FOUND,
                "{method} {path} is in LORA_ROUTES but not registered in this binary"
            );
            // A 405 means the path exists under a DIFFERENT method, so the listed
            // method is not registered either; checking 404 alone would pass it.
            assert_ne!(
                response.status(),
                axum::http::StatusCode::METHOD_NOT_ALLOWED,
                "{method} {path} is in LORA_ROUTES but this binary registers {path} under another method"
            );
        }
    }

    // The must-not-match control: the same machinery reports 404 for a
    // path nobody registered. Without it, a router that answered
    // everything -- a catch-all fallback, say -- would pass the loop above
    // while proving nothing about any individual route.
    let request = axum::http::Request::builder()
        .method("GET")
        .uri("/v1/lora/definitely-not-a-route")
        .body(axum::body::Body::empty())
        .expect("control request must build");
    let response = router(test_app_state())
        .oneshot(request)
        .await
        .expect("router must produce a response");
    assert_eq!(
        response.status(),
        axum::http::StatusCode::NOT_FOUND,
        "an unregistered path did not 404, so the loop above cannot distinguish \
                 a registered route from a router that answers everything"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn test_app_state() -> AppState {
    let (jobs, _rx) = test_client_and_jobs();
    AppState {
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context: 4096,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: Arc::new(vec![]),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn test_app_state_with_embedding(embedding: Arc<EmbeddingState>) -> AppState {
    let mut state = test_app_state();
    state.embedding = Some(embedding);
    state
}

/// Loads a real embedding model for the `embeddings_*` route tests
/// below, from `LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR`. Mirrors
/// `crate::model::bert`'s own
/// `test_from_bytes_matches_from_directory` convention (an env var
/// pointing at a real downloaded model directory, `None` early-return
/// when unset) rather than a synthetic in-memory fixture:
/// `BertModel` has no all-zero-weight test constructor the way
/// `Qwen35Model`'s `test-utils` feature exposes one for the chat
/// model, and hand-building a synthetic safetensors byte buffer with
/// every tensor `BertWeights::load` requires (~16 tensors per layer)
/// is exactly the kind of unverifiable-in-this-environment risk this
/// route's tests are already carrying -- better to
/// reuse the repo's own established real-model convention than add a
/// second, novel, equally-unrun fixture-construction path.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn test_embedding_state() -> Option<Arc<EmbeddingState>> {
    let dir = std::env::var("LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR").ok()?;
    let model = BertModel::from_directory(std::path::Path::new(&dir))
        .expect("LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR must be a valid BertModel directory");
    Some(Arc::new(EmbeddingState {
        model,
        model_id: Arc::from("test-embedding-model"),
        admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
    }))
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn inline_png_data_uri() -> String {
    use base64::Engine as _;
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

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
async fn error_message_of(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&body).expect("error response must be valid JSON");
    let message = value["error"]["message"]
        .as_str()
        .expect("error response must carry error.message")
        .to_string();
    (status, message)
}

/// Like `error_message_of`, plus the OpenAI-style `error.code` (used
/// by the structured-output v0 tests to assert the exact machine
/// code -- `unsupported_strict_schema`, `blocked_constraint`,
/// `validation_failed`, `length_limit` -- not just the status).
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
async fn error_code_of(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&body).expect("error response must be valid JSON");
    let code = value["error"]["code"]
        .as_str()
        .expect("error response must carry error.code")
        .to_string();
    (status, code)
}

// ── `/v1/embeddings` tests (issue #584) ──────────────────────────
//
// The four error-arm tests below (missing input, empty input, empty
// string inside an array, batch over the limit) and the
// no-model-loaded test need no `BertModel` at all: they return from
// `embeddings` before `s.embedding` is ever dereferenced for
// anything but its `is_some()`-equivalent check, so `test_app_state()`
// (`embedding: None`) is a faithful stand-in. The model-mismatch and
// success-path tests genuinely need a loaded model and are
// `#[ignore]`d behind `LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR` (see
// `test_embedding_state`'s doc comment).

// ── `/v1/lora/*` tests (runtime adapter load, Ocean 2026-09-16) ──
//
// Every arm below returns before the worker is ever reached, except
// the unload arm, which reaches a `test_app_state()` client with no
// worker thread behind it and so exercises the closed-receiver path.
// The success path needs a real Metal engine and a real adapter and
// is covered by the engine tests, not here.

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_load_wrong_content_type_415() {
    let body = Body::from(r#"{"path":"/tmp/a.safetensors"}"#.to_string());
    let response = lora_load(State(test_app_state()), HeaderMap::new(), body).await;
    let (status, _code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod lora_orchestration;

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod adapter_runtime_admission_order;

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_load_missing_path_400() {
    let response = lora_load(
        State(test_app_state()),
        test_json_headers(),
        Body::from(r#"{}"#.to_string()),
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_load_unknown_field_rejected_not_dropped() {
    let response = lora_load(
        State(test_app_state()),
        test_json_headers(),
        Body::from(r#"{"path":"/tmp/a.safetensors","scale":2.0}"#.to_string()),
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_load_empty_path_400() {
    let response = lora_load(
        State(test_app_state()),
        test_json_headers(),
        Body::from(r#"{"path":"   "}"#.to_string()),
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request");
}

/// A path that does not exist must be refused as such, before the file is
/// opened and before the worker is touched: an adapter that was never read
/// and an adapter that failed to parse are different answers.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_load_nonexistent_file_400() {
    let missing = std::env::temp_dir().join("lattice-no-such-adapter-1584.safetensors");
    let _ = std::fs::remove_file(&missing);
    let body = serde_json::json!({"path": missing.to_string_lossy()}).to_string();
    let response = lora_load(
        State(test_app_state()),
        test_json_headers(),
        Body::from(body),
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "lora_adapter_not_found");
}

/// With no worker thread the reply channel closes, and a closed channel means
/// the command was never applied. Reporting that as success would be the
/// worst available answer, so it is a 503 naming the worker.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_unload_without_a_worker_is_503() {
    let response = lora_unload(
        State(test_app_state()),
        test_json_headers(),
        Body::from(r#"{"id":0}"#),
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(code, "server_busy");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_missing_input_400() {
    let body = Body::from(r#"{}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_empty_string_input_400() {
    let body = Body::from(r#"{"input":""}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_input");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_empty_array_input_400() {
    let body = Body::from(r#"{"input":[]}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_input");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_empty_string_inside_array_400() {
    let body = Body::from(r#"{"input":["a","","c"]}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_input");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_batch_over_limit_400() {
    use lattice_inference::serve::embeddings::MAX_EMBEDDINGS_BATCH_SIZE;
    let items: Vec<String> = (0..MAX_EMBEDDINGS_BATCH_SIZE + 1)
        .map(|i| format!("\"{i}\""))
        .collect();
    let body = Body::from(format!(r#"{{"input":[{}]}}"#, items.join(",")));
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "batch_size_exceeds_limit");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_no_model_loaded_503() {
    let body = Body::from(r#"{"input":"hello"}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    // Documented contract (docs/capability-matrix.md): a request
    // reaching this route with no `--embedding-model` loaded gets
    // `embedding_model_not_loaded`, distinct from the Metal
    // worker's admission-cap `server_busy` (issue #932) -- both are
    // 503s, but only one names a route the operator forgot to
    // enable.
    assert_eq!(code, "embedding_model_not_loaded");
}

// ── `POST /v1/embeddings` unsupported-field rejection (contract
// fix: an OpenAI field this route cannot honor must be rejected,
// not silently ignored) -- like the tests above, these return
// before `s.embedding` is dereferenced for anything but its
// `is_some()`-equivalent check, so `test_app_state()` (no loaded
// model) is a faithful stand-in.

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_dimensions_field_400() {
    let body = Body::from(r#"{"input":"hello","dimensions":3}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "unsupported_parameter");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_base64_encoding_format_400() {
    let body = Body::from(r#"{"input":"hello","encoding_format":"base64"}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "unsupported_parameter");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_unknown_field_rejected_not_dropped() {
    // `pooling` is not a declared field of `TextEmbeddingsRequest`;
    // `deny_unknown_fields` must fail the parse rather than
    // silently ignore it (the pre-fix behavior).
    let body = Body::from(r#"{"input":"hello","pooling":"cls"}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request_body");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_accepted_noop_fields_pass_validation() {
    // `user` (caller metadata) and `encoding_format: "float"` (this
    // route's only output shape) are genuine no-ops: a request
    // carrying them must clear field validation and reach the next
    // stage (here, the no-model-loaded 503) rather than being
    // rejected as `unsupported_parameter`.
    let body =
        Body::from(r#"{"input":"hello","user":"trace-123","encoding_format":"float"}"#.to_string());
    let response = embeddings(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(code, "embedding_model_not_loaded");
}

/// Unit-level coverage for the embedding-concurrency admission cap
/// ([`EmbeddingState::admission`] / [`try_acquire_embedding_slot`]):
/// a real end-to-end version of this test (two concurrent
/// `/v1/embeddings` requests, the first held open until the second
/// is rejected) would need the same "block a real worker thread
/// until signaled" rig `single_slot_blocking_state`/
/// `chat_completions_returns_503_json_envelope_when_admission_cap_reached`
/// use for the Metal chat path -- disproportionate here since the
/// admission check itself is a plain `Semaphore::try_acquire_owned`
/// call with no embedding-specific logic between "cap reached" and
/// "rejected", so exercising the semaphore directly covers the same
/// behavior the handler relies on.
#[test]
fn try_acquire_embedding_slot_rejects_once_cap_reached() {
    let admission = Arc::new(Semaphore::new(1));
    let permit1 = try_acquire_embedding_slot(&admission).expect("first slot must be admitted");
    let err = try_acquire_embedding_slot(&admission)
        .expect_err("second slot must be rejected: cap=1, 1 outstanding");
    match err {
        lattice_inference::serve::ApiError::ServiceUnavailable { .. } => {}
        other => panic!("expected ServiceUnavailable, got {other:?}"),
    }
    drop(permit1);
    let _permit2 = try_acquire_embedding_slot(&admission)
        .expect("slot must be admitted again once the first is released");
}

/// Regression test for the admission-control hole where a client
/// disconnecting mid-request released the admission permit
/// immediately (the request future dropping at its `.await`) while
/// the already-started `spawn_blocking` `encode_batch` job kept
/// running to completion -- repeated disconnects could accumulate
/// unbounded concurrent jobs past [`EMBEDDING_MAX_CONCURRENT_JOBS`].
/// Exercises [`run_embedding_encode`] directly (the same helper
/// [`embeddings`] calls) rather than the full HTTP handler:
/// `BertModel` has no synthetic/mock constructor (see
/// `test_embedding_state`'s doc comment above), so there is no way
/// to give the real handler an artificially slow `encode_batch`
/// without a downloaded model directory. `run_embedding_encode` is
/// generic over the blocking closure, so a deterministic
/// channel-gated closure exercises the exact permit-lifetime
/// mechanics the fix changed.
///
/// `tokio::spawn` + `JoinHandle::abort()` simulates axum dropping a
/// disconnected client's handler future mid-`.await`: at the point
/// of `abort()` the task is suspended awaiting the inner
/// `spawn_blocking` `JoinHandle` (not actively running), so tokio
/// drops the task's future there -- the same event a dropped
/// request future produces.
#[tokio::test]
async fn embedding_permit_survives_dropped_request_future() {
    let admission = Arc::new(Semaphore::new(1));
    let permit = admission
        .clone()
        .try_acquire_owned()
        .expect("single slot must be admitted");

    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let mut started_tx = Some(started_tx);

    let task = tokio::spawn(run_embedding_encode(permit, move || {
        started_tx
            .take()
            .expect("closure runs exactly once")
            .send(())
            .expect("test task must still be waiting on started_rx");
        // Deterministic handshake, not a sleep: blocks until the
        // test explicitly lets the "encode_batch" job finish.
        release_rx
            .recv()
            .expect("release_tx must not be dropped before sending");
        Ok(Vec::new())
    }));

    started_rx
        .await
        .expect("job must signal that it started running");

    // Simulate the client disconnecting: cancel the outstanding
    // request future while the blocking job is still in flight.
    task.abort();
    let join_result = task.await;
    assert!(
        join_result.is_err() && join_result.unwrap_err().is_cancelled(),
        "aborted task must resolve as cancelled"
    );

    assert_eq!(
        admission.available_permits(),
        0,
        "admission slot must stay held while the encode_batch job is still running, \
                 not release just because the request future was dropped"
    );

    // Let the blocking job finish and confirm the slot frees up.
    release_tx
        .send(())
        .expect("blocking closure must still be waiting on release_rx");
    for _ in 0..200 {
        if admission.available_permits() == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        admission.available_permits(),
        1,
        "admission slot must free once the encode_batch job actually completes"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
#[ignore = "requires LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR"]
async fn embeddings_total_token_budget_exceeded_400() {
    use lattice_inference::serve::embeddings::MAX_EMBEDDINGS_TOTAL_TOKENS;
    let Some(embedding) = test_embedding_state() else {
        return;
    };
    // Repeated single-character words tokenize to roughly one real
    // token each for a typical BPE vocabulary; comfortably above
    // the budget regardless of the loaded checkpoint's exact
    // tokenization, without needing to know its vocabulary.
    let over_budget_text = "a ".repeat(MAX_EMBEDDINGS_TOTAL_TOKENS + 1000);
    let body = Body::from(serde_json::json!({ "input": over_budget_text }).to_string());
    let response = embeddings(
        State(test_app_state_with_embedding(embedding)),
        test_json_headers(),
        body,
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "batch_token_budget_exceeded");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
#[ignore = "requires LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR"]
async fn embeddings_wrong_model_name_400() {
    let Some(embedding) = test_embedding_state() else {
        return;
    };
    let body = Body::from(r#"{"input":"hello","model":"not-the-loaded-model"}"#.to_string());
    let response = embeddings(
        State(test_app_state_with_embedding(embedding)),
        test_json_headers(),
        body,
    )
    .await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "model_not_found");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn embeddings_item_over_context_window_400() {
    bert_pooling_tests::check_context_window_route().await;
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
#[ignore = "requires LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR"]
async fn embeddings_single_string_200_response_shape() {
    let Some(embedding) = test_embedding_state() else {
        return;
    };
    let dimensions = embedding.model.dimensions();
    let body = Body::from(r#"{"input":"hello world"}"#.to_string());
    let response = embeddings(
        State(test_app_state_with_embedding(embedding)),
        test_json_headers(),
        body,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&body).expect("response must be valid JSON");
    assert_eq!(value["object"], "list");
    assert_eq!(value["model"], "test-embedding-model");
    let data = value["data"].as_array().expect("data must be an array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(
        data[0]["embedding"]
            .as_array()
            .expect("embedding must be an array")
            .len(),
        dimensions
    );
    let prompt_tokens = value["usage"]["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens must be an integer");
    assert!(prompt_tokens > 0);
    assert_eq!(value["usage"]["total_tokens"], prompt_tokens);
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
#[ignore = "requires LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR"]
async fn embeddings_array_input_preserves_index_order() {
    let Some(embedding) = test_embedding_state() else {
        return;
    };
    let body = Body::from(r#"{"input":["alpha","beta","gamma"]}"#.to_string());
    let response = embeddings(
        State(test_app_state_with_embedding(embedding)),
        test_json_headers(),
        body,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&body).expect("response must be valid JSON");
    let data = value["data"].as_array().expect("data must be an array");
    assert_eq!(data.len(), 3);
    for (expected_index, item) in data.iter().enumerate() {
        assert_eq!(item["index"], expected_index);
    }
    let first = data[0]["embedding"]
        .as_array()
        .expect("embedding must be an array");
    let second = data[1]["embedding"]
        .as_array()
        .expect("embedding must be an array");
    assert_ne!(
        first, second,
        "distinct input texts must not produce identical embeddings"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_unknown_role_400() {
    // A role string that is not an OpenAI chat role at all (as
    // opposed to "developer"/"tool", which ARE real OpenAI roles
    // this server just doesn't implement -- see
    // `chat_completions_tool_and_developer_role_400_unsupported_feature`
    // below for that split, ADR-080 C2).
    let body = Body::from(r#"{"messages":[{"role":"moderator","content":"hi"}]}"#.to_string());
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        message,
        "unsupported role 'moderator'; must be 'system', 'user', or 'assistant'"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_tool_and_developer_role_400_unsupported_feature() {
    for role in ["tool", "developer"] {
        let body = Body::from(format!(
            r#"{{"messages":[{{"role":"{role}","content":"hi"}}]}}"#
        ));
        let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body must be readable");
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response body must be valid JSON");
        assert_eq!(
            v["error"]["code"], "unsupported_feature",
            "role '{role}' must be reported as unsupported_feature, not invalid_role"
        );
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_image_url_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}
                ]}]}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("error response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("error response must be JSON");
    assert_eq!(value["error"]["code"], "vision_unsupported");
    assert_eq!(
        value["error"]["message"],
        serde_json::Value::String(IMAGE_REQUIRES_VISION_MESSAGE.to_string())
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_vision_model_enqueues_image_on_shared_worker() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let image_seen = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&image_seen);
    let jobs = spawn_fake_with_vision(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        tokenizer,
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
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(16),
        model_max_context: 4096,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: Arc::new(vec![]),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(
        serde_json::json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": inline_png_data_uri()}},
                    {"type": "text", "text": "after"}
                ]
            }]
        })
        .to_string(),
    );

    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(image_seen.load(Ordering::SeqCst));
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_oversized_part_400() {
    let big_text = "x".repeat(MAX_CONTENT_PART_BYTES + 1);
    let body = Body::from(format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"{big_text}"}}]}}]}}"#,
    ));
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(message, "messages[0].content[0] exceeds 65536 bytes");
}

// ── #656: known-but-unsupported OpenAI field rejection ────────────
//
// Same fail-closed contract as the three tests above: these fields
// are parsed (not silently dropped by serde's unknown-field
// default), so a client asking for tool calls / JSON mode / N
// completions / stop sequences gets an explicit 400 naming the
// unsupported field instead of a plain-text completion that quietly
// ignored the request's actual contract.

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_tools_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],
                    "tools":[{"type":"function","function":{"name":"f"}}]}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        message,
        "tools and tool_choice are not supported by this server"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_tool_choice_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#.to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        message,
        "tools and tool_choice are not supported by this server"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_json_response_format_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_object"}}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        message,
        "response_format.type 'json_object' is not supported; use 'text'"
    );
}

// ── structured-output v0 (design note) ─────────────────────────────

/// A synthetic vocabulary sized for `V0_ROUTE_SCHEMA` specifically:
/// literal multi-byte tokens for exactly the bytes a closed
/// `{"ok": <boolean>}` value needs. Decoupled from whatever
/// tokenizer a given test's fake worker uses for prompt-side
/// tokenization -- `AppState.vocab_bytes` and the worker's
/// `BpeTokenizer` are two independent concerns. Deliberately NOT a
/// full byte-range vocabulary: `GrammarEngine::new`'s reachable-
/// state BFS fans out per distinguishable token at every state, and
/// an open `type: "string"` property (arbitrary content) combined
/// with a wide single-byte vocabulary blows `MAX_GRAMMAR_STATES`
/// even for a schema this small -- a real tokenizer's vocabulary is
/// large but mostly irrelevant to any one schema's few reachable
/// states, which a literal-token synthetic vocab does not model.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn route_test_vocab() -> Arc<Vec<Vec<u8>>> {
    Arc::new(
        ["{", "}", "\"", ":", "ok", "true", "false", ","]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect(),
    )
}

const V0_OBJECT_SCHEMA: &str = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;
/// Compiles within `route_test_vocab`'s small literal-token budget
/// (see that function's doc comment): used only by tests that
/// actually run admission through `GrammarEngine::new`, as opposed
/// to `V0_OBJECT_SCHEMA`, which is used by tests that stop at pure
/// schema-shape validation (`admit_v0_schema`) or reject before
/// compilation is ever attempted.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
const V0_ROUTE_SCHEMA: &str = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn structured_body(schema: &str, strict: Option<&str>, stream: Option<bool>) -> String {
    let strict_field = match strict {
        Some(raw) => format!(r#","strict":{raw}"#),
        None => String::new(),
    };
    let stream_field = match stream {
        Some(v) => format!(r#","stream":{v}"#),
        None => String::new(),
    };
    format!(
        r#"{{"messages":[{{"role":"user","content":"hi"}}]{stream_field},
                    "response_format":{{"type":"json_schema","json_schema":{{
                        "name":"result"{strict_field},"schema":{schema}}}}}}}"#
    )
}

/// Regression: a duplicate root `"type"`
/// member inside the submitted schema must be rejected with 400
/// before any worker job is submitted. `test_app_state()`'s worker
/// receiver is dropped immediately, so a request that reached
/// admission/job-submission would simply have its `send` silently
/// fail rather than this test observing a clean 400 -- reaching the
/// 400 assertion is itself proof no job was submitted.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_rejects_duplicate_root_type_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_schema","json_schema":{
                        "name":"result","strict":true,
                        "schema":{"type":"boolean","type":"string"}}}}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request_body");
}

/// Regression: a duplicate nested
/// `"properties"` member inside an object schema must be rejected
/// with 400 before any worker job is submitted.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_rejects_duplicate_nested_properties_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_schema","json_schema":{
                        "name":"result","strict":true,
                        "schema":{"type":"object",
                            "properties":{"a":{"type":"boolean"}},
                            "properties":{"a":{"type":"string"}},
                            "required":["a"],"additionalProperties":false}}}}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request_body");
}

/// Regression: a duplicate key inside a
/// nested `properties.<name>` schema object must be rejected with
/// 400 before any worker job is submitted.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_rejects_duplicate_key_inside_property_schema_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],
                    "response_format":{"type":"json_schema","json_schema":{
                        "name":"result","strict":true,
                        "schema":{"type":"object",
                            "properties":{"a":{"type":"boolean","type":"string"}},
                            "required":["a"],"additionalProperties":false}}}}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_request_body");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_missing_strict_400() {
    let body = Body::from(structured_body(V0_OBJECT_SCHEMA, None, None));
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(message.contains("strict must be true"), "{message}");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_strict_false_400() {
    let body = Body::from(structured_body(V0_OBJECT_SCHEMA, Some("false"), None));
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(message.contains("strict must be true"), "{message}");
}

/// Spec §"Accepted request profile" item 1: `stream: true` combined
/// with a structured request is rejected before a worker job is
/// submitted (`test_app_state()`'s worker has no running loop
/// behind it at all -- a job send here would simply vanish silently
/// rather than this test observing a 400, so reaching the 400
/// assertion below is itself proof admission ran first).
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_streaming_400() {
    let body = Body::from(structured_body(V0_OBJECT_SCHEMA, Some("true"), Some(true)));
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(message.contains("non-streaming"), "{message}");
}

/// Rejection coverage per unsupported keyword (design note
/// §"Correctness tests", item 1), driven through the real HTTP
/// route (not just the pure `admit_v0_schema` unit tests below) so
/// the `unsupported_strict_schema` code and pre-worker-submission
/// contract are both proven at the boundary the design note names.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_pattern_keyword_400() {
    let schema = r#"{"type":"object","properties":{"name":{"type":"string","pattern":"^a"}},"required":["name"],"additionalProperties":false}"#;
    let body = Body::from(structured_body(schema, Some("true"), None));
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "unsupported_strict_schema");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_enum_keyword_400() {
    let schema = r#"{"type":"string","enum":["a","b"]}"#;
    let body = Body::from(structured_body(schema, Some("true"), None));
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "unsupported_strict_schema");
}

/// Route test (design note §"Serve and backend integration", item
/// 1): an admitted strict request's worker job carries
/// `grammar.is_some()`, `enable_thinking == false`,
/// and the successful response carries `constraint_applied` --
/// asserted from a real, compiled `GrammarEngine`, not a stub, so a
/// route that merely returns schema-shaped JSON without actually
/// wiring the grammar would fail this test's `grammar.is_some()`
/// assertion even before the marker check.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_applies_grammar_and_marker() {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let jobs = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        tokenizer,
        move |_messages, cfg, _prompt_tokens, on_token, _should_cancel| {
            assert!(
                cfg.grammar.is_some(),
                "structured request must attach a compiled grammar to GenerateConfig"
            );
            assert!(
                !cfg.enable_thinking,
                "strict structured requests must force enable_thinking off (Q5)"
            );
            // The route handler builds response `content` from
            // streamed `Delta` events, not `GenerateOutput::text`
            // directly -- mirror that by pushing the delta through
            // `on_token` before returning, exactly like a real
            // generate closure would.
            on_token(r#"{"ok":true}"#, 0);
            Ok(GenerateOutput {
                text: r#"{"ok":true}"#.to_string(),
                token_ids: vec![0],
                prompt_tokens: 1,
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
        vocab_bytes: route_test_vocab(),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response must be valid JSON");
    assert_eq!(
        value["constraint_applied"],
        serde_json::Value::Bool(true),
        "a route returning valid JSON without the constraint_applied marker must fail: {value}"
    );
    assert_eq!(value["choices"][0]["message"]["content"], r#"{"ok":true}"#);
}

/// A `WorkerEvent::ConstraintBlocked` event must surface
/// as HTTP 500 with the `blocked_constraint` machine code for a
/// structured request, never a 200 with partial/absent JSON. Replies
/// via the raw `WorkerJob::reply`
/// seam (not `spawn_fake`'s `Result<_, String>` closure, which can
/// only ever produce a generic `WorkerEvent::Failed`) so this test
/// exercises the real production classification -- a distinct
/// `WorkerEvent` variant, not a message-text sniff.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_blocked_constraint_500() {
    let (jobs, mut jobs_rx) = test_client_and_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::ConstraintBlocked(
                "grammar constraint blocked every token; \
                         no legal continuation exists in the current grammar state"
                    .to_string(),
            ));
        }
    });
    let state = AppState {
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context: 4096,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: route_test_vocab(),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
    let response = chat_completions(State(state), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(code, "blocked_constraint");
}

/// A `WorkerEvent::Failed` (the
/// generic variant) whose message text happens to contain the
/// engine's grammar-exhausted-mask wording must NOT be classified as
/// `blocked_constraint` -- only the distinct `ConstraintBlocked`
/// variant may. Mutation-sensitive companion to the test above: if
/// the classification ever regresses back to sniffing `message`,
/// this test starts failing (this exact message string used to
/// produce `blocked_constraint` under the old `.contains(..)` check;
/// it must now produce `internal_error`).
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_failed_with_blocked_wording_stays_internal_error() {
    let (jobs, mut jobs_rx) = test_client_and_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Failed(
                "grammar constraint blocked every token; \
                         no legal continuation exists in the current grammar state"
                    .to_string(),
            ));
        }
    });
    let state = AppState {
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context: 4096,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: route_test_vocab(),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
    let response = chat_completions(State(state), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(code, "internal_error");
}

/// `stopped == false` (length/KV-window exhaustion)
/// on a structured request must surface as HTTP 500
/// `length_limit`, never HTTP 200 with truncated JSON.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_length_limit_500() {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let jobs = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        tokenizer,
        move |_messages, _cfg, _prompt_tokens, _on_token, _should_cancel| {
            Ok(GenerateOutput {
                text: r#"{"ok":tr"#.to_string(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Length),
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
        vocab_bytes: route_test_vocab(),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
    let response = chat_completions(State(state), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(code, "length_limit");
}

/// Independent-validation failure path (design note's
/// "independent validation" requirement): the engine claims a
/// clean stop, but the produced text does not conform to the
/// admitted schema -- must still be a 500 `validation_failed`, not
/// a 200 with nonconforming JSON.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_structured_validation_failed_500() {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let jobs = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        tokenizer,
        move |_messages, _cfg, _prompt_tokens, on_token, _should_cancel| {
            on_token(r#"{"unexpected":1}"#, 0);
            Ok(GenerateOutput {
                // Conforms to neither `additionalProperties:false`
                // nor the required `ok` boolean field.
                text: r#"{"unexpected":1}"#.to_string(),
                token_ids: vec![0],
                prompt_tokens: 1,
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
        vocab_bytes: route_test_vocab(),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None));
    let response = chat_completions(State(state), test_json_headers(), body).await;
    let (status, code) = error_code_of(response).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(code, "validation_failed");
}

// ── GrammarCache (stage 2: LRU + single-flight) ─────────────────────

const SCHEMA_CACHE_TEST_BOOL: &str = r#"{"type":"boolean"}"#;
const SCHEMA_CACHE_TEST_NULL: &str = r#"{"type":"null"}"#;
const SCHEMA_CACHE_TEST_EMPTY_OBJECT: &str =
    r#"{"type":"object","properties":{},"required":[],"additionalProperties":false}"#;

/// A vocabulary sized only for the tiny literal schemas above --
/// same rationale as `route_test_vocab`: a wide vocabulary combined
/// with these schemas' few reachable states is irrelevant work, not
/// a more faithful test.
fn cache_test_vocab() -> Arc<Vec<Vec<u8>>> {
    Arc::new(
        ["true", "false", "null", "{", "}"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect(),
    )
}

fn compile_schema_for_cache_test(schema_json: &str) -> Result<GrammarEngine, String> {
    let schema: Value = serde_json::from_str(schema_json).expect("test schema must parse");
    let spec = GrammarSpec::JsonSchema(schema);
    GrammarEngine::new(&spec, (*cache_test_vocab()).clone())
        .map_err(|e| format!("test schema failed to compile: {e}"))
}

#[test]
fn grammar_cache_hit_returns_same_engine_without_recompiling() {
    let cache = GrammarCache::new(GRAMMAR_CACHE_CAPACITY);
    let vocab = cache_test_vocab();
    let key = grammar_cache_key(
        &vocab,
        &serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap(),
    );
    let first = cache
        .get_or_compile(key.clone(), || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_BOOL)
        })
        .expect("first compile must succeed");
    let second = cache
        .get_or_compile(key, || {
            panic!("cache hit must not invoke the compile closure again")
        })
        .expect("cache hit must succeed");
    assert!(
        Arc::ptr_eq(&first, &second),
        "cache hit must return the exact same Arc"
    );
    assert_eq!(cache.compile_count(), 1);
}

#[test]
fn grammar_cache_evicts_least_recently_used_at_capacity() {
    let cache = GrammarCache::new(2);
    let vocab = cache_test_vocab();
    let key_a = grammar_cache_key(
        &vocab,
        &serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap(),
    );
    let key_b = grammar_cache_key(
        &vocab,
        &serde_json::from_str(SCHEMA_CACHE_TEST_NULL).unwrap(),
    );
    let key_c = grammar_cache_key(
        &vocab,
        &serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap(),
    );

    cache
        .get_or_compile(key_a.clone(), || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_BOOL)
        })
        .unwrap();
    cache
        .get_or_compile(key_b.clone(), || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_NULL)
        })
        .unwrap();
    cache
        .get_or_compile(key_c, || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
        })
        .unwrap();
    assert_eq!(
        cache.compile_count(),
        3,
        "three distinct schemas must each compile exactly once"
    );

    // `a` was the least-recently-used entry when `c` was inserted
    // into a capacity-2 cache, so `b` must still be cached.
    cache
        .get_or_compile(key_b, || {
            panic!("b must still be cached: a, not b, was the LRU entry evicted by c")
        })
        .unwrap();
    assert_eq!(cache.compile_count(), 3);

    // `a` was evicted, so requesting it again must recompile.
    cache
        .get_or_compile(key_a, || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_BOOL)
        })
        .unwrap();
    assert_eq!(
        cache.compile_count(),
        4,
        "evicted key must recompile on next request"
    );
}

#[test]
fn grammar_cache_single_flight_collapses_concurrent_identical_requests() {
    let cache = Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY));
    let vocab = cache_test_vocab();
    let key = grammar_cache_key(
        &vocab,
        &serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap(),
    );
    const N: usize = 8;
    let barrier = Arc::new(std::sync::Barrier::new(N));
    let handles: Vec<_> = (0..N)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let key = key.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                cache
                    .get_or_compile(key, || {
                        std::thread::sleep(std::time::Duration::from_millis(30));
                        compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
                    })
                    .expect("compile must succeed")
            })
        })
        .collect();
    let engines: Vec<Arc<GrammarEngine>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(
        cache.compile_count(),
        1,
        "{N} concurrent identical requests must collapse into exactly 1 compile"
    );
    for engine in &engines[1..] {
        assert!(
            Arc::ptr_eq(&engines[0], engine),
            "every waiter must observe the same compiled engine"
        );
    }
}

/// Exercises the `catch_unwind` single-flight
/// recovery path (`get_or_compile`'s `Action::Compile` arm) against
/// an actual owner-closure panic. `compile` is an
/// injectable `FnOnce` argument to `get_or_compile` already -- no
/// extra test-only seam is needed, since single-flight semantics
/// mean only the one thread that wins `Action::Compile` ever
/// invokes its closure, so every concurrent caller can safely be
/// handed the same panicking closure. Every caller's result is
/// collected over a channel with a bounded `recv_timeout` rather
/// than a plain `.join()`, so a regression that reintroduces a
/// missed `notify_all` fails this test deterministically instead of
/// hanging the suite.
#[test]
fn grammar_cache_panic_in_compile_notifies_waiters_with_internal_error() {
    let cache = Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY));
    let vocab = cache_test_vocab();
    let key = grammar_cache_key(
        &vocab,
        &serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap(),
    );
    const N: usize = 8;
    let barrier = Arc::new(std::sync::Barrier::new(N));
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the expected injected panic
    let (tx, rx) = std::sync::mpsc::channel();
    for _ in 0..N {
        let cache = Arc::clone(&cache);
        let key = key.clone();
        let barrier = Arc::clone(&barrier);
        let tx = tx.clone();
        std::thread::spawn(move || {
            barrier.wait();
            let result = cache.get_or_compile(key, || {
                std::thread::sleep(std::time::Duration::from_millis(30));
                panic!("injected compile panic for single-flight regression test");
            });
            let _ = tx.send(result);
        });
    }
    drop(tx);
    for _ in 0..N {
        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(Err(CacheError::Internal(_))) => {}
            Ok(Err(other)) => panic!("expected CacheError::Internal, got Err({other:?})"),
            Ok(Ok(_)) => panic!("expected CacheError::Internal, got Ok(engine)"),
            Err(_) => {
                panic!("caller did not return within timeout -- single-flight panic recovery hung")
            }
        }
    }
    std::panic::set_hook(prev_hook);
    assert_eq!(
        cache.compile_count(),
        1,
        "panic-injected compile must run exactly once across all waiters"
    );

    // Poison-recovery: retrying the same key with a working compile
    // must succeed, proving the slot was removed (not left poisoned
    // or permanently occupied) after the panic.
    cache
        .get_or_compile(key, || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
        })
        .expect("retry after a panicked compile must succeed");
    assert_eq!(
        cache.compile_count(),
        2,
        "retry after panic recovery must actually recompile, not replay a poisoned entry"
    );
}

#[test]
fn grammar_cache_key_is_invariant_to_object_property_order() {
    let v1: Value = serde_json::from_str(
                r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a","b"],"additionalProperties":false}"#,
            )
            .unwrap();
    let v2: Value = serde_json::from_str(
                r#"{"additionalProperties":false,"required":["a","b"],"properties":{"b":{"type":"null"},"a":{"type":"boolean"}},"type":"object"}"#,
            )
            .unwrap();
    assert_eq!(
        canonical_schema_key(&v1),
        canonical_schema_key(&v2),
        "reordering object keys (including nested 'properties') must not change the cache key"
    );

    let different: Value = serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap();
    assert_ne!(
        canonical_schema_key(&v1),
        canonical_schema_key(&different),
        "structurally different schemas must have different cache keys"
    );
}

#[test]
fn grammar_cache_hits_across_reordered_but_equivalent_top_level_keys() {
    let cache = GrammarCache::new(GRAMMAR_CACHE_CAPACITY);
    let vocab = cache_test_vocab();
    const EMPTY_OBJ_REORDERED: &str =
        r#"{"additionalProperties":false,"required":[],"properties":{},"type":"object"}"#;
    let v1: Value = serde_json::from_str(SCHEMA_CACHE_TEST_EMPTY_OBJECT).unwrap();
    let v2: Value = serde_json::from_str(EMPTY_OBJ_REORDERED).unwrap();
    assert_eq!(canonical_schema_key(&v1), canonical_schema_key(&v2));

    let first = cache
        .get_or_compile(grammar_cache_key(&vocab, &v1), || {
            compile_schema_for_cache_test(SCHEMA_CACHE_TEST_EMPTY_OBJECT)
        })
        .expect("v1 must compile");
    let second = cache
        .get_or_compile(grammar_cache_key(&vocab, &v2), || {
            panic!("v2's canonical key must hit the entry compiled for v1")
        })
        .expect("v2 must hit cache");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(cache.compile_count(), 1);
}

/// #1557: two different vocabularies compiled against the *same*
/// schema must never share a cache entry. Before the fix, the cache
/// was keyed on `canonical_schema_key(schema)` alone, so `vocab_b`'s
/// lookup would hit the entry `vocab_a` had already compiled and
/// silently hand back a `GrammarEngine` built for `vocab_a`'s
/// (different-sized) vocabulary.
#[test]
fn grammar_cache_does_not_share_entries_across_different_vocabularies() {
    let vocab_a: Arc<Vec<Vec<u8>>> = Arc::new(
        ["true", "false", "null", "{", "}", "0", "1", "2", "3"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect(),
    );
    let vocab_b = cache_test_vocab();
    assert_ne!(
        vocab_a.len(),
        vocab_b.len(),
        "test fixture needs two distinguishably-sized vocabularies"
    );

    let schema: Value = serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap();
    let cache = GrammarCache::new(GRAMMAR_CACHE_CAPACITY);

    let engine_a = cache
        .get_or_compile(grammar_cache_key(&vocab_a, &schema), || {
            let spec = GrammarSpec::JsonSchema(schema.clone());
            GrammarEngine::new(&spec, (*vocab_a).clone())
                .map_err(|e| format!("vocab_a compile failed: {e}"))
        })
        .expect("vocab_a must compile");
    let engine_b = cache
        .get_or_compile(grammar_cache_key(&vocab_b, &schema), || {
            let spec = GrammarSpec::JsonSchema(schema.clone());
            GrammarEngine::new(&spec, (*vocab_b).clone())
                .map_err(|e| format!("vocab_b compile failed: {e}"))
        })
        .expect("vocab_b must compile");

    assert_eq!(
        cache.compile_count(),
        2,
        "two different vocabularies against the same schema must each compile their \
                 own entry, not share one"
    );
    assert!(
        !Arc::ptr_eq(&engine_a, &engine_b),
        "vocab_a's and vocab_b's lookups must not resolve to the same cached engine"
    );

    // Directly exercise the returned engine rather than trusting the
    // pointer check alone: `mask_logits` rejects a `logits` buffer
    // shorter than the engine's own `vocab_size`
    // (`validate_logits_len`), so handing it a buffer sized to
    // `vocab_b.len()` (5) only succeeds if `engine_b` really was
    // compiled against `vocab_b`. Had the cache wrongly served
    // `vocab_a`'s entry (`vocab_size` 9), this same call would
    // reject the length-5 buffer.
    let mut state = engine_b.initial_state();
    let mut logits = vec![0.0f32; vocab_b.len()];
    engine_b
        .mask_logits(&mut state, &mut logits)
        .expect("engine_b must accept a logits buffer sized to vocab_b, not vocab_a");
}

/// The cache key must pin the exact `vocab_bytes` allocation it
/// names, not merely record its address. Written against
/// `grammar_cache_key`/`get_or_compile` as opaque behaviour (never
/// naming `GrammarCacheKey`'s fields or type) so an address-only
/// `String` key would still compile here and fail by assertion.
///
/// A `Weak` observes the allocation from outside without itself
/// keeping it alive. If the cache's key holds only an address, the
/// allocation dies the moment every other handle to it (here, the
/// test's own `vocab_a`) is dropped, and a later, unrelated
/// `vocab_bytes` allocation can legitimately land at that freed
/// address -- reintroducing #1557's original collision through the
/// address alone, exactly because the two are then *not*
/// simultaneously alive.
#[test]
fn grammar_cache_key_keeps_its_vocabulary_allocation_alive() {
    let vocab_a: Arc<Vec<Vec<u8>>> = Arc::new(
        ["true", "false", "null", "{", "}", "0", "1", "2", "3"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect(),
    );
    let weak = Arc::downgrade(&vocab_a);
    let schema: Value = serde_json::from_str(SCHEMA_CACHE_TEST_BOOL).unwrap();
    let cache = GrammarCache::new(GRAMMAR_CACHE_CAPACITY);

    cache
        .get_or_compile(grammar_cache_key(&vocab_a, &schema), || {
            let spec = GrammarSpec::JsonSchema(schema.clone());
            GrammarEngine::new(&spec, (*vocab_a).clone())
                .map_err(|e| format!("vocab_a compile failed: {e}"))
        })
        .expect("vocab_a must compile");

    // Drop the caller's only other handle. The cache's own key is
    // now the sole thing that can still be keeping this allocation
    // alive.
    drop(vocab_a);

    assert!(
        weak.upgrade().is_some(),
        "the grammar cache's key must keep its vocabulary allocation alive for as long \
                 as the entry sits in the cache -- otherwise the allocator can hand the freed \
                 address to an unrelated later vocabulary, and an address-only key would then \
                 alias the two"
    );
}

// ── canonical_schema_key: property-based differential tests (#1557) ──
//
// The hand-written tests above pin specific, hand-picked schemas.
// This module generates schema pairs across the shapes JSON admits
// and pins two properties none of the hand-written tests state:
// that the key never collides two schemas that are not
// `serde_json::Value`-equal (injectivity), and that the key
// survives a full round trip back through `serde_json::from_str`
// (which is what actually exercises the two
// `serde_json::to_string(..).unwrap_or_default()` calls on the
// string and object-key escaping paths through a real parser,
// rather than trusting the printed text by inspection).
mod canonical_schema_key_properties;

mod v0_schema;

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod characterization;

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_n_greater_than_one_400() {
    let body = Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#.to_string());
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(message, "n > 1 is not supported");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_logprobs_400() {
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#.to_string());
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        message,
        "logprobs/top_logprobs are not supported by this server"
    );
}

// #831: `lattice_serve` now accepts and applies `stop`, aligned with
// `lattice serve` (was previously rejected outright with "stop is
// not supported by this server"). This drives the real worker
// thread (`spawn_fake`, issue #832's cross-binary test seam) end to
// end through `router()`, so the generate closure captures the
// `stop_strings` this binary's real `build_cfg` actually produced --
// not a value reconstructed independently in the test. Replaces the
// old `chat_completions_stop_400` (`test_app_state()`'s dropped-
// receiver job queue can't reach this far, since the request no
// longer short-circuits with a 400 before submission).
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_stop_is_accepted_and_reaches_generate_config() {
    use tower::ServiceExt as _;
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let observed_stop: Arc<std::sync::Mutex<Option<Vec<String>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let observed_for_closure = Arc::clone(&observed_stop);
    let jobs = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        tokenizer,
        move |_messages, cfg, prompt_tokens, _on_token, _should_cancel| {
            *observed_for_closure
                .lock()
                .expect("observation mutex poisoned") = Some(cfg.stop_strings.clone());
            Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens,
                generated_tokens: 0,
                stopped: true,
                stop_reason: None,
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
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stop":"\n"}"#.to_string());
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(body)
        .expect("fixture request must build");
    let response = router(state)
        .oneshot(request)
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        observed_stop
            .lock()
            .expect("observation mutex poisoned")
            .clone(),
        Some(vec!["\n".to_string()]),
        "stop must reach the real GenerateConfig the worker's generate closure receives"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_conflicting_max_tokens_400() {
    let body = Body::from(
        r#"{"messages":[{"role":"user","content":"hi"}],
                    "max_tokens":10,"max_completion_tokens":20}"#
            .to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        message,
        "max_tokens (10) and max_completion_tokens (20) differ; supply only one"
    );
}

// ── ADR-080 C2 (#782): max_tokens=0 rejection + finish_reason round-trip ──

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_max_tokens_zero_400() {
    // #745: previously clamped straight through into a zero-budget
    // completion instead of being rejected.
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#.to_string());
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    let (status, message) = error_message_of(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(message, "max_tokens must be at least 1");
}

/// Builds an `AppState` wired to a fresh raw job channel plus the
/// receiving half (issue #832's `test_client_and_jobs` seam), so
/// tests can stand in for the worker: reply with whatever
/// `WorkerEvent` sequence the test wants (via `WorkerJob::reply`,
/// also `test-utils`-gated) without a real GPU/model.
///
/// The channel carries `WorkerMessage`, not `WorkerJob`: the worker's
/// single receiver also carries adapter control commands. A caller
/// matching only `WorkerMessage::Generate` replies to nothing if a
/// control command arrives instead, which fails the test loudly rather
/// than passing over a message it ignored.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_unknown_id_is_400_before_streaming() {
    for stream in [false, true] {
        let body = Body::from(serde_json::json!({"messages":[{"role":"user","content":"hi"}],"lora":[{"id":99,"scale":1.0}],"stream":stream}).to_string());
        let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
        let (status, message) = error_message_of(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(message.contains("99"));
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_duplicate_id_is_400_before_admission() {
    use lattice_inference::serve::lora::AdapterMetadata;
    use lattice_inference::serve::metal_worker::test_client_and_jobs_with_adapters;
    for stream in [false, true] {
        let (client, mut jobs) = test_client_and_jobs_with_adapters(vec![AdapterMetadata {
            id: 7,
            name: "test".into(),
            path: "test.safetensors".into(),
            rank: 1,
            layers: 1,
        }]);
        let mut state = test_app_state();
        state.jobs = client;
        let body = Body::from(
            serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "lora": [{"id": 7, "scale": 1.0}, {"id": 7, "scale": -0.25}],
                "stream": stream,
            })
            .to_string(),
        );
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            chat_completions(State(state.clone()), test_json_headers(), body),
        )
        .await
        .expect("duplicate selection must return before a worker reply");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["code"], "lora_duplicate_adapter_id");
        assert_eq!(error["error"]["message"], "duplicate LoRA adapter id 7");
        assert!(matches!(
            jobs.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_list_reads_confirmed_index() {
    let response = lora_list(State(test_app_state())).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        value,
        serde_json::json!({"adapters":[],"applied":[],"router":{"enabled":false}})
    );
}

/// The worker's runtime names the mode of each request it routes, so
/// the handler must tell the worker whether the response is streamed.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn streaming_flag_reaches_worker() {
    for stream in [false, true] {
        let (state, mut jobs_rx) = test_app_state_with_jobs();
        let worker = tokio::spawn(async move {
            let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await else {
                panic!("missing generation")
            };
            let streamed = job.is_streamed();
            job.reply(WorkerEvent::Complete(GenerateOutput {
                text: "ok".into(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            }));
            streamed
        });
        let body = Body::from(
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"stream":stream})
                .to_string(),
        );
        let response = chat_completions(State(state), test_json_headers(), body).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(worker.await.unwrap(), stream);
    }
}

#[test]
fn gemma_startup_refuses_options_it_cannot_honor_by_name() {
    let directory = tempfile::tempdir().expect("create Gemma checkpoint directory");
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut weights = (header.len() as u64).to_le_bytes().to_vec();
    weights.extend_from_slice(header);
    weights.extend_from_slice(&[0u8; 4]);
    std::fs::write(directory.path().join("model.safetensors"), weights)
        .expect("write minimal safetensors");
    std::fs::write(
        directory.path().join("config.json"),
        br#"{"model_type":"gemma4"}"#,
    )
    .expect("write Gemma config");
    let selected = serving_provider::select(
        serving_provider::inspect_checkpoint(directory.path()),
        ServingEntry::Standalone,
    )
    .expect("Gemma safetensors selects its standalone provider");
    let presence = |flag: Option<&str>| StandaloneOptionPresence {
        preload_vision: flag == Some("--preload-vision"),
        tokenizer_dir: flag == Some("--tokenizer-dir"),
        resident_count: flag == Some("--max-resident-adapters"),
        resident_bytes: flag == Some("--max-resident-adapter-bytes"),
        positive_reasoning_budget: flag == Some("--reasoning-budget"),
    };
    assert_eq!(
        selected.validate_standalone_options(ServingEntry::Standalone, &presence(None)),
        Ok(())
    );
    for flag in [
        "--preload-vision",
        "--tokenizer-dir",
        "--max-resident-adapters",
        "--max-resident-adapter-bytes",
        "--reasoning-budget",
    ] {
        let error = selected
            .validate_standalone_options(ServingEntry::Standalone, &presence(Some(flag)))
            .expect_err("an option with no Gemma meaning ends startup");
        assert_eq!(
            error,
            format!("unsupported_feature: {flag} is not supported for Gemma 4 checkpoints")
        );
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_selection_reaches_worker() {
    use lattice_inference::serve::lora::{AdapterMetadata, LoraSelection};
    use lattice_inference::serve::metal_worker::test_client_and_jobs_with_adapters;
    for (stream, reject) in [(false, false), (true, false), (false, true), (true, true)] {
        let (client, mut jobs) = test_client_and_jobs_with_adapters(vec![AdapterMetadata {
            id: 7,
            name: "test".into(),
            path: "test.safetensors".into(),
            rank: 1,
            layers: 1,
        }]);
        let mut state = test_app_state();
        state.jobs = client;
        let worker = tokio::spawn(async move {
            let Some(WorkerMessage::Generate(job)) = jobs.recv().await else {
                panic!("missing generation")
            };
            assert_eq!(
                job.lora_selection(),
                &[serde_json::from_str::<LoraSelection>(r#"{"id":7,"scale":0.25}"#).unwrap()]
            );
            if reject {
                job.reply(WorkerEvent::Rejected(
                    lattice_inference::serve::lora::unknown_adapter(7),
                ));
                return;
            }
            job.reply(WorkerEvent::Complete(GenerateOutput {
                text: "ok".into(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            }));
        });
        let body = Body::from(serde_json::json!({"messages":[{"role":"user","content":"hi"}],"lora":[{"id":7,"scale":0.25}],"stream":stream}).to_string());
        let response = chat_completions(State(state), test_json_headers(), body).await;
        if reject {
            let (status, message) = error_message_of(response).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(message.contains("7"));
            worker.await.unwrap();
            continue;
        }
        assert_eq!(response.status(), StatusCode::OK);
        let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        worker.await.unwrap();
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
fn test_app_state_with_jobs() -> (AppState, mpsc::UnboundedReceiver<WorkerMessage>) {
    let (jobs, jobs_rx) = test_client_and_jobs();
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
    (state, jobs_rx)
}

/// #746 (ADR-080 C2): the non-streaming JSON response's
/// `finish_reason` must reflect the engine's actual `stopped` flag,
/// not a hardcoded `"stop"`. A fake worker task stands in for the
/// GPU, replying with a crafted `WorkerEvent::Complete` (`stopped`)
/// directly.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
async fn non_streaming_finish_reason_for(stopped: bool) -> String {
    let (state, mut jobs_rx) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("hi".to_string()));
            let _ = job.reply(WorkerEvent::Complete(GenerateOutput {
                text: "hi".to_string(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped,
                stop_reason: None,
                token_logprobs: vec![],
            }));
        }
    });
    let body = Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string());
    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response body must be valid JSON");
    v["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason must be a string")
        .to_string()
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_non_streaming_finish_reason_stop_when_engine_stopped() {
    assert_eq!(non_streaming_finish_reason_for(true).await, "stop");
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_non_streaming_finish_reason_length_when_not_stopped() {
    // Previously this hardcoded "stop" unconditionally, so a
    // length-capped or cancelled completion was misreported as an
    // explicit stop condition (#746).
    assert_eq!(non_streaming_finish_reason_for(false).await, "length");
}

/// #583: `GET /metrics` must reflect REAL traffic recorded through
/// `emit_serve_event`, not a static template -- mutation check: if
/// `emit_serve_event` stops forwarding into `ServeMetrics` (or
/// `metrics_handler` stops rendering it), every assertion below
/// regresses to "0"/absent while the route itself would still
/// return 200, so this would NOT be caught by a route-existence
/// check alone.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn metrics_endpoint_reports_requests_tokens_and_errors() {
    let (state, mut jobs_rx) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Complete(GenerateOutput {
                text: "hi".to_string(),
                token_ids: vec![0],
                prompt_tokens: 3,
                generated_tokens: 2,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            }));
        }
    });
    let ok_body = Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string());
    let ok_response = chat_completions(State(state.clone()), test_json_headers(), ok_body).await;
    assert_eq!(ok_response.status(), StatusCode::OK);

    let bad_body = Body::from(r#"{"messages":[]}"#.to_string());
    let bad_response = chat_completions(State(state.clone()), test_json_headers(), bad_body).await;
    assert_eq!(bad_response.status(), StatusCode::BAD_REQUEST);

    let _ = health(State(state.clone())).await;

    let metrics_response = metrics_handler(State(state.clone())).await.into_response();
    let bytes = axum::body::to_bytes(metrics_response.into_body(), usize::MAX)
        .await
        .expect("metrics body must be readable");
    let body = String::from_utf8(bytes.to_vec()).expect("metrics body must be UTF-8");

    assert!(
                body.contains(
                    "lattice_http_requests_total{method=\"POST\",route=\"/v1/chat/completions\",status=\"200\",model=\"test-model\"} 1"
                ),
                "missing successful chat_completions counter:\n{body}"
            );
    assert!(
                body.contains(
                    "lattice_http_requests_total{method=\"GET\",route=\"/health\",status=\"200\",model=\"test-model\"} 1"
                ),
                "missing health counter:\n{body}"
            );
    assert!(
        body.contains("lattice_prompt_tokens_total{model=\"test-model\"} 3"),
        "missing prompt token count:\n{body}"
    );
    assert!(
        body.contains("lattice_completion_tokens_total{model=\"test-model\"} 2"),
        "missing completion token count:\n{body}"
    );
    assert!(
        body.contains("lattice_errors_total{code=\"invalid_messages\",model=\"test-model\"} 1"),
        "missing invalid_messages error count:\n{body}"
    );
}

/// Same round-trip as above, but through the SSE streaming path
/// (`Phase::Body`'s `Ev::Done` arm) instead of the non-streaming
/// JSON body.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
async fn streaming_finish_reason_for(stopped: bool) -> String {
    let (state, mut jobs_rx) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("hi".to_string()));
            let _ = job.reply(WorkerEvent::Complete(GenerateOutput {
                text: "hi".to_string(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped,
                stop_reason: None,
                token_logprobs: vec![],
            }));
        }
    });
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string());
    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("SSE response body must be readable");
    String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8")
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_streaming_finish_reason_stop_when_engine_stopped() {
    let text = streaming_finish_reason_for(true).await;
    assert!(
        text.contains("\"finish_reason\":\"stop\""),
        "SSE body must carry finish_reason: stop; got: {text}"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_streaming_finish_reason_length_when_not_stopped() {
    let text = streaming_finish_reason_for(false).await;
    assert!(
        text.contains("\"finish_reason\":\"length\""),
        "SSE body must carry finish_reason: length, not a hardcoded \"stop\" \
                 (#746); got: {text}"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_streaming_failure_emits_error_event() {
    let (state, mut jobs_rx) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("partial".to_string()));
            let _ = job.reply(WorkerEvent::Failed("blocked by grammar".to_string()));
        }
    });
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string());
    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("SSE response body must be readable");
    let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
    assert!(
        text.contains("\"content\":\"partial\""),
        "partial output must precede the error event; got: {text}"
    );
    let error_payload = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .find(|payload| payload.contains("\"error\""))
        .expect("a failed generation must emit an SSE error payload");
    let error: serde_json::Value =
        serde_json::from_str(error_payload).expect("SSE error payload must be valid JSON");
    assert_eq!(error["error"]["type"], "server_error");
    assert_eq!(error["error"]["code"], "internal_error");
    assert!(
        !text.contains("\"finish_reason\":\"stop\""),
        "generation failure must not masquerade as a clean stop; got: {text}"
    );
}

/// A mid-stream `WorkerEvent::Failed` (or
/// `Rejected`/worker-gone) emits a client-facing SSE error event
/// (asserted above) but, before this fix, the terminal `Phase::End`
/// still recorded the request in `/metrics` as a plain 200 with no
/// error code -- `lattice_errors_total` never incremented for a
/// failed stream. Mutation-sensitive: reverting the `Phase::Done`/
/// `Phase::End` error-outcome threading (while keeping this test)
/// makes this fail, because the recorded counter reverts to
/// `status="200"` and `lattice_errors_total{code="internal_error"}`
/// stays absent.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_streaming_failure_records_failed_metric() {
    let (state, mut jobs_rx) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("partial".to_string()));
            let _ = job.reply(WorkerEvent::Failed("blocked by grammar".to_string()));
        }
    });
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string());
    let response = chat_completions(State(state.clone()), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);
    // Drain the full SSE body so the stream runs to `Phase::End`,
    // which is where the metric gets recorded.
    let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("SSE response body must be readable");

    let metrics_response = metrics_handler(State(state.clone())).await.into_response();
    let bytes = axum::body::to_bytes(metrics_response.into_body(), usize::MAX)
        .await
        .expect("metrics body must be readable");
    let text = String::from_utf8(bytes.to_vec()).expect("metrics body must be UTF-8");

    assert!(
                !text.contains(
                    "lattice_http_requests_total{method=\"POST\",route=\"/v1/chat/completions\",status=\"200\""
                ),
                "a failed stream must NOT be recorded as a 200; got:\n{text}"
            );
    assert!(
                text.contains(
                    "lattice_http_requests_total{method=\"POST\",route=\"/v1/chat/completions\",status=\"500\",model=\"test-model\"} 1"
                ),
                "failed stream must be recorded with the same 500 status the non-streaming \
                 `Failed` arm records; got:\n{text}"
            );
    assert!(
        text.contains("lattice_errors_total{code=\"internal_error\",model=\"test-model\"} 1"),
        "lattice_errors_total must increment for a failed stream; got:\n{text}"
    );
}

/// ADR-080 C2: a `stream:
/// true` request whose prompt overflows the model's context window
/// must return HTTP 400 `context_length_exceeded` BEFORE any SSE
/// stream is committed -- not silently commit to a 200 SSE response
/// that only discovers the overflow later via `WorkerEvent::Rejected`
/// mid-stream and terminates with `finish_reason: "length"` (the
/// exact drift this test catches). This fakes the worker side of
/// the `WorkerEvent::Rejected` contract (production code sends it
/// via `metal_worker::check_prompt_fits_window`'s typed `ApiError`,
/// issue #832) so the composition under test is purely
/// `chat_completions`'s streaming branch: does it await the
/// worker's first event and inspect it for `WorkerEvent::Rejected`
/// BEFORE calling `Sse::new(..).into_response()`, or does it commit
/// unconditionally?
///
/// ADR-080 C2: this test
/// pins ONLY that response-mapping contract with a request that
/// would NOT genuinely overflow in production (`max_tokens`
/// defaults to 100 against a 4096-token `model_max_context`) and a
/// fake task that manufactures `WorkerEvent::Rejected` directly --
/// it does NOT exercise the real worker's `check_prompt_fits_window`
/// call, so it is not, by itself, same-input real-router parity with
/// `lattice.rs`'s equivalent test (an earlier version of this
/// comment incorrectly claimed "identical request body"; it is not
/// -- `real_router_overflow_parity` below is the test that actually
/// is). Kept because it is still the cheapest, fastest pin of the
/// response-mapping contract in isolation.
///
/// Mutation-sensitive: reverting the pre-`Sse::new()` `first_ev`
/// preflight (going back to building `stream::unfold` and
/// returning `Sse::new(stream).into_response()` unconditionally,
/// discovering `WorkerEvent::Rejected` only inside `Phase::Body`)
/// makes this fail -- the response status would be 200 with an SSE body
/// carrying a `finish_reason: "length"` terminal chunk instead of a
/// 400 error envelope, so this test fails if the preflight
/// regresses.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn chat_completions_streaming_context_overflow_returns_400_before_committing_sse() {
    let (state, mut jobs_rx) = test_app_state_with_jobs();
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Rejected(ApiError::BadRequest {
                message: "prompt has 4090 tokens, leaving 6 of the 4096-token \
                                  context window for generation, but this request needs \
                                  100 generated tokens plus 1 (total 4191); reduce \
                                  max_tokens/reasoning_budget or shorten the prompt"
                    .to_string(),
                code: "context_length_exceeded",
            }));
        }
    });
    let body =
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#.to_string());
    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "an Ev::Rejected worker reply for a stream:true request must surface \
                 as a pre-commit HTTP 400, not a committed 200 SSE stream"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("error response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("error response must be JSON");
    assert_eq!(value["error"]["code"], "context_length_exceeded");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("context window"),
        "error message must carry the worker's actual overflow explanation, \
                 got: {value}"
    );
}

/// ADR-080 C2: the
/// same-input real-router parity the test above does NOT provide.
/// Drives `lattice_inference::serve::OVERFLOW_PARITY_REQUEST_BODY`
/// -- the SAME fixture `lattice.rs`'s `streaming_context_overflow`
/// module drives through its own real `Router` -- through THIS
/// binary's real `router()`, backed by a REAL worker thread running
/// the actual `run_worker_loop` + `check_prompt_fits_window`
/// production code (via `metal_worker::spawn_fake`, issue #832's
/// cross-binary test seam; a real `BpeTokenizer` from
/// `test_support::tiny_zero_model`, no Metal engine involved since
/// the request is expected to be rejected before any generation
/// call). `AppState.model_max_context` is set to
/// `OVERFLOW_PARITY_CONTEXT_WINDOW` (1024) -- the same effective
/// limit `lattice.rs`'s tiny test model's fixed context window uses
/// -- so both sides genuinely share the same input AND the same
/// effective context limit, not just the same JSON body.
///
/// Mutation-sensitive: removing `check_prompt_fits_window`'s
/// overflow check (i.e. having it unconditionally return `Ok(())`)
/// makes this fail -- the real worker would accept the request,
/// `chat_completions` would commit a 200 SSE response, and this
/// test's `StatusCode::BAD_REQUEST` assertion would fail. This is
/// the exact production worker-side check that removing
/// `MetalWorker::spawn`'s reliance on it would leave undetected --
/// `MetalWorker::spawn`'s real Metal closure and this test's
/// `spawn_fake` seam both route through the SAME `run_worker_loop`
/// wrapper (`metal_worker::run_worker_loop`, private to that
/// module) that calls `check_prompt_fits_window`, not two
/// independent copies of the check, so a mutation to the shared
/// function is observed here too.
///
/// Gated behind `test-utils` (see
/// `lattice_inference::model::qwen35::test_support`) for the same
/// reason as `lattice.rs`'s equivalent test modules: this needs a
/// real (tiny) `BpeTokenizer`, which is only constructible outside
/// this crate's own `#[cfg(test)]` build via that feature.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod real_router_overflow_parity {
    use super::*;
    use lattice_inference::serve::{OVERFLOW_PARITY_CONTEXT_WINDOW, OVERFLOW_PARITY_REQUEST_BODY};
    use tower::ServiceExt as _;

    /// A real worker thread running the actual `run_worker_loop` +
    /// `check_prompt_fits_window` production code (via
    /// `metal_worker::spawn_fake`, issue #832's cross-binary test
    /// seam), with a real (tiny) tokenizer and NO Metal engine --
    /// the generate closure is never reached by an overflowing
    /// request, so it only needs a trivial success stand-in for a
    /// non-overflowing one.
    fn real_worker_state(model_max_context: usize) -> AppState {
        let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
            .tokenizer()
            .clone();
        let jobs = spawn_fake(
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            model_max_context,
            tokenizer,
            |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
                Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens,
                    generated_tokens: 0,
                    stopped: true,
                    stop_reason: None,
                    token_logprobs: vec![],
                })
            },
        );
        AppState {
            jobs,
            model_id: Arc::from("test-model"),
            defaults: GenerationDefaults::standard(100),
            model_max_context,
            max_pending: 1_000_000,
            metrics: Arc::new(ServeMetrics::default()),
            vocab_bytes: Arc::new(vec![]),
            grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
            embedding: None,
        }
    }

    #[tokio::test]
    async fn chat_completions_streaming_context_overflow_matches_lattice_real_router() {
        let body = Body::from(OVERFLOW_PARITY_REQUEST_BODY.to_string());
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(body)
            .expect("fixture request must build");
        let response = router(real_worker_state(OVERFLOW_PARITY_CONTEXT_WINDOW))
            .oneshot(request)
            .await
            .expect("router must produce a response, not a transport error");
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "the shared overflow-parity request, driven through the real \
                     worker's production enforce_prompt_window check, must be \
                     rejected with HTTP 400 before any SSE stream is committed -- \
                     matching lattice.rs's real-router result for the identical \
                     request body and effective context limit"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error response body must be readable");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("error response must be JSON");
        assert_eq!(value["error"]["code"], "context_length_exceeded");
    }
}

/// HTTP-level admission/backpressure test (issue #932): a real
/// worker thread running the actual production `MetalWorkerClient`
/// (via `metal_worker::spawn_fake_with_cap`, the same cross-binary
/// test seam `real_router_overflow_parity` above uses, with an
/// explicit small cap instead of the effectively-unbounded
/// default), driven through this binary's real `router()` end to
/// end. Proves the 503 JSON envelope shape a real OpenAI-compatible
/// client would see, and that it is returned fail-fast (before the
/// worker thread's `generate` closure -- which would tokenize the
/// prompt -- is ever reached for the rejected request).
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod real_router_admission_cap;

#[test]
fn build_cfg_aliases_max_completion_tokens_when_max_tokens_absent() {
    let defaults = GenerationDefaults::standard(100);
    let mut req = serde_json::from_str::<ChatReq>("{}").unwrap();
    req.messages = vec![input_message(
        "user".to_string(),
        MessageContent::Text("hi".to_string()),
    )];
    req.max_completion_tokens = Some(42);
    let normalized = normalize_for_cfg(&req, &defaults, 4096);
    let cfg = build_cfg(&normalized);
    assert_eq!(cfg.max_new_tokens, 42);
}

// -----------------------------------------------------------------------
// Capability-matrix fixtures (#654). The `chat_completions_*_400` tests
// above this block are ALSO capability-matrix fixtures in their own
// right — their names are the fixture IDs `docs/capability-matrix.md`'s
// Fixture manifest section cites for the tools/tool_choice/n/
// response_format/logprobs/stop/role/content-part rows on this surface.
// `scripts/check-capability-matrix.sh` greps this file for
// `fn <fixture_id>` and fails the build if a matrix row cites an ID
// that no longer exists here.
// -----------------------------------------------------------------------

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn cm_lattice_serve_model_mismatch_rejected() {
    let body = Body::from(
        r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}]}"#.to_string(),
    );
    let response = chat_completions(State(test_app_state()), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("error response must be valid JSON");
    assert_eq!(value["error"]["code"], "model_not_found");
    assert_eq!(
        value["error"]["message"],
        "model 'wrong-model' is not loaded; this server serves 'test-model'"
    );
}

// ── cross-binary parity table (ADR-080 C2) ─────────────────────────
//
// Drives every fixture body in
// `lattice_inference::serve::CHAT_COMPLETIONS_PARITY_CASES` through
// THIS binary's real `Router` (via `router()`, extracted from
// `run()` specifically so it's testable in isolation from process
// startup) and compares the resulting status + error code against
// the case's `lattice_serve`-side expectation. `lattice.rs`'s own
// test module runs the SAME table against its own router, asserting
// the `lattice`-side expectation -- together the two prove
// same-input parity (or a documented, intentional divergence) at
// the real HTTP layer.
//
// Issue #828's `Json`/`Sse` rows need a REAL tiny tokenizer (via
// `test_support::tiny_zero_model`, gated behind the `test-utils`
// feature -- see that module's own doc comment) to drive
// `check_prompt_fits_window` for real, so this whole module now
// requires `test-utils`, matching `lattice.rs`'s own
// `parity_table` module gating. CI's "serve-surface capability-
// matrix fixtures" step adds `test-utils` alongside `f16,metal-gpu`
// for this binary to keep running it (see `.github/workflows/ci.yml`).
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod parity_table;

// -----------------------------------------------------------------------
// Production-adapter observation (issue #828): proves the shared
// `ProductionAdapterObservation`/`GenerateConfigSnapshot` types
// capture what THIS binary's real `chat_completions` -> `build_cfg`
// construction produces, not a value the test independently
// reconstructs. The injected worker-loop `generate` closure below
// runs strictly BELOW that real path -- it records the actual
// `&[ChatMessage]`/`&GenerateConfig` it was called with, then
// returns a canned result; it never recomputes `build_cfg` itself.
// -----------------------------------------------------------------------
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod production_adapter_observation;
