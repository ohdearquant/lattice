//! Shared HTTP serving contract for the `lattice` unified server
//! (`crates/inference/src/bin/lattice.rs`) and the `lattice_serve` daemon
//! (`crates/inference/src/bin/lattice_serve.rs`) -- ADR-080 cluster C2 (#782).
//!
//! Both binaries speak a subset of the OpenAI chat-completions wire format,
//! and both previously carried independent copies of: the error envelope
//! shape, the `finish_reason` mapping from the engine's `stopped` flag, the
//! `max_tokens == 0` rejection, and the `/v1/models` response body -- with
//! real drift between the copies (#744, #745, #746: `lattice_serve.rs`
//! discarded the engine's stop cause and hardcoded `finish_reason: "stop"`,
//! silently accepted `max_tokens: 0`, and never installed `/v1/models`'s
//! sibling route on the other binary). This module is the single source of
//! truth for those contracts; each binary still owns its own router wiring
//! and backend-specific generation dispatch (CPU/Metal dispatch for
//! `lattice.rs`, the daemon job queue for `lattice_serve.rs`) -- per the
//! ADR, only the request/response CONTRACT is shared, not the kernels or
//! scheduling behind it.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::watch;

use crate::model::qwen35_config::GenerateConfig;

/// Request body size cap shared by both HTTP servers: 1 MiB. Both binaries
/// already enforced this exact limit independently (`lattice.rs` via
/// `DefaultBodyLimit::max`, `lattice_serve.rs` via `to_bytes(body, LIMIT)`);
/// centralizing the constant removes one silent-drift vector even though the
/// two binaries still wire it into axum differently.
pub const REQUEST_BODY_LIMIT_BYTES: usize = 1_048_576;

/// Shared streaming context-overflow parity fixture (ADR-080 C2): both binaries' real-router
/// context-overflow tests build their request from these SAME constants and
/// configure their real (tiny, test-only CPU) model's effective context
/// window to this SAME value, so "same input, same effective limit" is
/// enforced by shared constants rather than by two independently-typed
/// literals that could silently drift apart. `lattice.rs`'s tiny test model
/// (`lattice_inference::model::qwen35::test_support::tiny_zero_model`) has a
/// fixed 1024-token context window; `lattice_serve.rs`'s real-worker test
/// configures its `AppState.model_max_context` to the same figure.
pub const OVERFLOW_PARITY_CONTEXT_WINDOW: usize = 1024;
/// `max_tokens` for the shared overflow-parity request: equal to the whole
/// context window, so any non-empty prompt pushes `prompt_len + max_tokens`
/// past it once the worker's full-window check (not just `build_cfg`'s
/// in-isolation clamp) runs.
pub const OVERFLOW_PARITY_MAX_TOKENS: usize = OVERFLOW_PARITY_CONTEXT_WINDOW;
/// Request-level `max_tokens` cap, kept well above
/// [`OVERFLOW_PARITY_MAX_TOKENS`] so a cap-rejection (`max_tokens_exceeds_limit`
/// / equivalent) never fires first and masks the context-window check this
/// fixture exists to isolate.
pub const OVERFLOW_PARITY_MAX_TOKENS_CAP: usize = 4096;
/// The exact request body both binaries' overflow-parity tests send.
pub const OVERFLOW_PARITY_REQUEST_BODY: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1024,"stream":true}"#;

/// Structured HTTP error shared by both binaries, serializing to the OpenAI
/// error envelope: `{"error": {"message", "type", "code", "param"}}`.
#[derive(Debug)]
pub enum ApiError {
    /// Caller mistake — HTTP 400.
    BadRequest { message: String, code: &'static str },
    /// Request body exceeds size limit — HTTP 413.
    PayloadTooLarge { message: String },
    /// Server-side failure — HTTP 500.
    Internal { message: String },
}

impl ApiError {
    /// The human-readable message, regardless of variant. Used by tests and
    /// by callers that need the text without matching on the variant.
    pub fn message(&self) -> &str {
        match self {
            ApiError::BadRequest { message, .. } => message,
            ApiError::PayloadTooLarge { message } => message,
            ApiError::Internal { message } => message,
        }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: String,
    param: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::BadRequest { message, code } => {
                let body = Json(ErrorBody {
                    error: ErrorDetail {
                        message,
                        r#type: "invalid_request_error",
                        code: code.to_string(),
                        param: None,
                    },
                });
                (StatusCode::BAD_REQUEST, body).into_response()
            }
            ApiError::PayloadTooLarge { message } => {
                let body = Json(ErrorBody {
                    error: ErrorDetail {
                        message,
                        r#type: "invalid_request_error",
                        code: "request_body_too_large".to_string(),
                        param: None,
                    },
                });
                (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
            }
            ApiError::Internal { message } => {
                let body = Json(ErrorBody {
                    error: ErrorDetail {
                        message,
                        r#type: "server_error",
                        code: "internal_error".to_string(),
                        param: None,
                    },
                });
                (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
            }
        }
    }
}

/// Maps a generation's `stopped` flag to the OpenAI `finish_reason` string:
/// `"stop"` when the engine explicitly ended generation via a stop condition
/// (EOS, stop-token-id, or stop-string match); `"length"` when the token
/// budget was exhausted without one, or generation was interrupted by the
/// caller (a disconnect is not an OpenAI "stop condition" either). The
/// ENGINE-reported `stopped` flag is the single source of truth --
/// `lattice.rs`'s `finish_reason_for` already carried it through correctly;
/// `lattice_serve.rs`'s worker previously discarded it entirely and
/// hardcoded `"stop"` unconditionally in both SSE and JSON responses (#746).
pub fn finish_reason(stopped: bool) -> &'static str {
    if stopped { "stop" } else { "length" }
}

/// Rejects a resolved `effective` `max_tokens` value of zero (#745).
/// `lattice.rs`'s `validate_max_tokens` already enforced this;
/// `lattice_serve.rs`'s `build_cfg` silently let a client-supplied
/// `max_tokens: 0` (or `max_completion_tokens: 0`) clamp through unchanged,
/// producing a zero-budget completion instead of a clear rejection.
///
/// Scoped narrowly to the zero case only, matching #745's triaged scope:
/// the two binaries' cap/alias-conflict policies differ intentionally
/// (`lattice.rs` rejects a request whose resolved `max_tokens` exceeds its
/// server cap; `lattice_serve.rs` clamps it to the model's context window
/// instead) and are deliberately NOT unified by this helper.
pub fn reject_zero_max_tokens(effective: usize) -> Result<(), ApiError> {
    if effective == 0 {
        return Err(ApiError::BadRequest {
            message: "max_tokens must be at least 1".to_string(),
            code: "invalid_max_tokens",
        });
    }
    Ok(())
}

/// `GET /` response body: a minimal engine-identity/endpoint-discovery
/// document. Shared so both binaries expose a byte-identical root route
/// (ADR-080 C2): `lattice_serve.rs` already
/// served this; `lattice.rs` had no `GET /` route at all, an undocumented
/// divergence a route-set audit caught. Both binaries expose the
/// same three routes, so the endpoint list is a fixed constant here rather
/// than a per-binary parameter.
pub fn root_body() -> Value {
    serde_json::json!({
        "name": "lattice",
        "object": "engine",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
    })
}

/// `GET /v1/models` response body: advertises the single loaded model.
/// Shared so both binaries expose byte-identical shapes for the same model
/// id and `created` timestamp -- previously only `lattice_serve.rs`
/// installed this route at all; `lattice.rs` had no equivalent endpoint.
pub fn models_list_body(model_id: &str, created: u64) -> Value {
    serde_json::json!({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "lattice",
        }],
    })
}

/// Disconnect-cancellation contract shared by both HTTP servers (#744):
/// flips the paired [`watch::Receiver<bool>`] to `true` the moment this guard
/// is dropped. Held inside the per-request SSE stream state (streaming) or
/// the handler's local scope (non-streaming) so it drops exactly when axum
/// stops caring about the response -- on client disconnect, or harmlessly
/// after the request already finished normally. `lattice_serve.rs` already
/// had this exact type as a private struct; `lattice.rs`'s CPU streaming
/// path had no equivalent at all and kept generating to the token cap after
/// a client left (its own comment documented this as "a future
/// refinement") -- this hoists the ONE contract both binaries now share.
pub struct CancelOnDrop(pub watch::Sender<bool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(true);
    }
}

/// Fresh cancel-on-drop guard/receiver pair for one request. The receiver is
/// threaded into the engine's `should_cancel` predicate (checked before
/// prefill, immediately after prefill, and at the top of every decode
/// iteration); the guard is held for the lifetime of the response so it
/// fires the moment axum drops it.
pub fn cancel_pair() -> (CancelOnDrop, watch::Receiver<bool>) {
    let (tx, rx) = watch::channel(false);
    (CancelOnDrop(tx), rx)
}

/// Which binary a [`ParityCase`] expectation applies to. Both binaries build
/// their own `Router` and drive it independently (bins can't cross-import
/// each other's `chat_completions`/router as a normal dependency), so a case
/// carries per-binary expected outcomes rather than one shared HTTP call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    Lattice,
    LatticeServe,
}

/// A parity case's request body. Most cases pin a small literal fixture;
/// [`CaseBody::Oversized`] generates a body larger than
/// [`REQUEST_BODY_LIMIT_BYTES`] at test time instead of embedding a >1MiB
/// string literal in source (ADR-080 C2: neither binary's parity table
/// exercised the oversized-body case at all, so restoring the daemon's old
/// 400/`invalid_request` mapping left its parity test green).
pub enum CaseBody {
    Fixed(&'static str),
    /// A `messages` array whose `content` field alone exceeds
    /// `REQUEST_BODY_LIMIT_BYTES`, forcing both binaries' body-limit
    /// enforcement (`DefaultBodyLimit` on `lattice.rs`, manual
    /// `to_bytes(.., LIMIT)` on `lattice_serve.rs`) to trip.
    Oversized,
}

impl CaseBody {
    pub fn build(&self) -> Vec<u8> {
        match self {
            CaseBody::Fixed(s) => s.as_bytes().to_vec(),
            CaseBody::Oversized => {
                let filler = "x".repeat(REQUEST_BODY_LIMIT_BYTES + 1);
                format!(
                    r#"{{"model":"test-model","messages":[{{"role":"user","content":"{filler}"}}]}}"#
                )
                .into_bytes()
            }
        }
    }
}

/// A scalar JSON value an [`FieldExpectation::Eq`] compares against. Only
/// the three primitive shapes the shared response contracts actually emit
/// (OpenAI-style string enums/ids, integer counts, booleans) -- not a full
/// `serde_json::Value` -- so a fixture row can be built entirely from
/// `'static` literals in the shared const table below.
#[derive(Debug, Clone, Copy)]
pub enum Scalar {
    Str(&'static str),
    U64(u64),
    Bool(bool),
}

impl Scalar {
    fn matches(&self, value: &Value) -> bool {
        match self {
            Scalar::Str(s) => value.as_str() == Some(*s),
            Scalar::U64(n) => value.as_u64() == Some(*n),
            Scalar::Bool(b) => value.as_bool() == Some(*b),
        }
    }
}

/// One field-level assertion against a successful JSON response body
/// (issue #828): a richer replacement for "just check status/error_code"
/// that can pin the actual shape of a 2xx response -- the gap #828's `Why`
/// section names (`CHAT_COMPLETIONS_PARITY_CASES` previously asserted
/// nothing at all about a successful response's fields). `json_pointer`
/// uses [`Value::pointer`]'s RFC 6901 syntax (e.g. `"/choices/0/finish_reason"`).
#[derive(Debug, Clone, Copy)]
pub enum FieldExpectation {
    /// The pointed-to field must exist and equal `scalar`.
    Eq {
        json_pointer: &'static str,
        scalar: Scalar,
    },
    /// The pointed-to field must not exist (`Value::pointer` returns `None`).
    Absent { json_pointer: &'static str },
    /// The pointed-to field must be a JSON array of exactly `len` elements.
    ArrayLen {
        json_pointer: &'static str,
        len: usize,
    },
    /// The pointed-to field must be a JSON string starting with `prefix`
    /// (issue #828: dynamic response IDs are checked
    /// by type/prefix, not exact value -- a response ID's suffix varies by
    /// request timestamp/sequence).
    StringPrefix {
        json_pointer: &'static str,
        prefix: &'static str,
    },
    /// The pointed-to field must be a JSON number representable as `u64`
    /// (issue #828: timestamps are checked by type,
    /// not exact value).
    UnsignedInt { json_pointer: &'static str },
}

impl FieldExpectation {
    /// Checks this expectation against a decoded response body, returning a
    /// human-readable failure description (never panics itself -- callers
    /// decide how to surface it, e.g. via `assert!`/`panic!` with the
    /// owning case's name for context).
    pub fn check(&self, body: &Value) -> Result<(), String> {
        match self {
            FieldExpectation::Eq {
                json_pointer,
                scalar,
            } => match body.pointer(json_pointer) {
                Some(value) if scalar.matches(value) => Ok(()),
                Some(value) => Err(format!(
                    "field '{json_pointer}': expected {scalar:?}, got {value} (body: {body})"
                )),
                None => Err(format!(
                    "field '{json_pointer}': expected {scalar:?}, field is absent (body: {body})"
                )),
            },
            FieldExpectation::Absent { json_pointer } => match body.pointer(json_pointer) {
                None => Ok(()),
                Some(value) => Err(format!(
                    "field '{json_pointer}': expected absent, got {value} (body: {body})"
                )),
            },
            FieldExpectation::ArrayLen { json_pointer, len } => match body.pointer(json_pointer) {
                Some(Value::Array(arr)) if arr.len() == *len => Ok(()),
                Some(Value::Array(arr)) => Err(format!(
                    "field '{json_pointer}': expected array of length {len}, got length {} \
                     (body: {body})",
                    arr.len()
                )),
                Some(other) => Err(format!(
                    "field '{json_pointer}': expected an array of length {len}, got {other} \
                     (body: {body})"
                )),
                None => Err(format!(
                    "field '{json_pointer}': expected an array of length {len}, field is \
                     absent (body: {body})"
                )),
            },
            FieldExpectation::StringPrefix {
                json_pointer,
                prefix,
            } => match body.pointer(json_pointer).and_then(Value::as_str) {
                Some(s) if s.starts_with(prefix) => Ok(()),
                Some(s) => Err(format!(
                    "field '{json_pointer}': expected a string starting with '{prefix}', got \
                     '{s}' (body: {body})"
                )),
                None => Err(format!(
                    "field '{json_pointer}': expected a string starting with '{prefix}', field \
                     is absent or not a string (body: {body})"
                )),
            },
            FieldExpectation::UnsignedInt { json_pointer } => {
                match body.pointer(json_pointer).and_then(Value::as_u64) {
                    Some(_) => Ok(()),
                    None => Err(format!(
                        "field '{json_pointer}': expected an unsigned integer, field is \
                         absent or not representable as u64 (body: {body})"
                    )),
                }
            }
        }
    }
}

/// One expected SSE chunk phase, in the order OpenAI's `chat.completion.chunk`
/// stream actually emits them (issue #828: "SSE expectations are ordered").
/// `ContentDelta` matches one-or-more consecutive content-delta chunks --
/// [`check_sse_events`] greedily consumes every consecutive chunk that
/// classifies as a content delta before moving to the next expected phase --
/// so a fixture only ever needs a single `ContentDelta` entry regardless of
/// how many tokens the generation seam actually emitted.
#[derive(Debug, Clone, Copy)]
pub enum EventExpectation {
    /// `delta: {"role":"assistant"}`, `finish_reason: null`, no `content`.
    RoleOpener,
    /// `delta: {"content": "..."}`, `finish_reason: null`, no `role`.
    ContentDelta,
    /// `delta: {}` (both `role`/`content` absent), `finish_reason` set to
    /// this exact string.
    Finish { finish_reason: &'static str },
    /// The literal `data: [DONE]` sentinel event.
    Done,
}

/// One decoded SSE `data:` payload: either a `chat.completion.chunk` JSON
/// object, or the literal `[DONE]` sentinel.
enum SseFrame {
    Chunk(Value),
    Done,
}

/// Splits a raw SSE response body into its `data:` payloads. Both binaries'
/// SSE bodies are `axum::response::sse::Event::default().data(..)` events,
/// which serialize as one `data: <payload>` line per event (a bare newline
/// is never embedded in either binary's payloads: `serde_json::to_string`
/// output for `lattice.rs`, `json!(..).to_string()` for `lattice_serve.rs`),
/// so splitting on lines starting with `data: ` is sufficient -- no SSE
/// multi-line/`id:`/`event:` framing to reassemble.
fn parse_sse_frames(body: &str) -> Vec<SseFrame> {
    body.lines()
        .filter_map(|line| {
            line.strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
        })
        .map(|payload| {
            let payload = payload.trim();
            if payload == "[DONE]" {
                SseFrame::Done
            } else {
                SseFrame::Chunk(
                    serde_json::from_str(payload).unwrap_or_else(|e| {
                        panic!("SSE data payload must be JSON: {e} ({payload})")
                    }),
                )
            }
        })
        .collect()
}

/// Classification of one decoded `chat.completion.chunk` object, used
/// internally by [`check_sse_events`]. Distinct from the public
/// [`EventExpectation`] (whose `Finish` carries a `&'static str`) because a
/// chunk's actual `finish_reason` is only known at parse time.
enum ChunkKind {
    RoleOpener,
    ContentDelta,
    Finish {
        finish_reason: String,
    },
    /// Matches none of the three shapes above (malformed/unexpected chunk).
    Other,
}

fn classify_chunk(chunk: &Value) -> ChunkKind {
    let Some(choice) = chunk.pointer("/choices/0") else {
        return ChunkKind::Other;
    };
    let finish_reason = choice.pointer("/finish_reason");
    let role = choice.pointer("/delta/role");
    let content = choice.pointer("/delta/content");
    if finish_reason.is_none_or(Value::is_null) {
        if role.and_then(Value::as_str) == Some("assistant") && content.is_none() {
            return ChunkKind::RoleOpener;
        }
        if content.and_then(Value::as_str).is_some() && role.is_none() {
            return ChunkKind::ContentDelta;
        }
        ChunkKind::Other
    } else {
        match finish_reason.and_then(Value::as_str) {
            Some(reason) if role.is_none() && content.is_none() => ChunkKind::Finish {
                finish_reason: reason.to_string(),
            },
            _ => ChunkKind::Other,
        }
    }
}

/// Asserts an SSE response body matches `expected`, in order. `ContentDelta`
/// greedily consumes every consecutive actual chunk that classifies as a
/// content delta (issue #828: "at least one content-delta chunk" -- the
/// fixture only lists one `ContentDelta` phase regardless of how many
/// tokens were actually streamed). Returns a human-readable failure
/// description on the first mismatch.
pub fn check_sse_events(body: &str, expected: &[EventExpectation]) -> Result<(), String> {
    let frames = parse_sse_frames(body);
    let mut idx = 0usize;
    for (phase_idx, exp) in expected.iter().enumerate() {
        match exp {
            EventExpectation::ContentDelta => {
                let start = idx;
                while idx < frames.len()
                    && matches!(
                        &frames[idx],
                        SseFrame::Chunk(c) if matches!(classify_chunk(c), ChunkKind::ContentDelta)
                    )
                {
                    idx += 1;
                }
                if idx == start {
                    return Err(format!(
                        "expected phase {phase_idx} (ContentDelta) to match at least one \
                         content-delta chunk at frame index {start}, but none matched"
                    ));
                }
            }
            EventExpectation::Done => match frames.get(idx) {
                Some(SseFrame::Done) => idx += 1,
                Some(SseFrame::Chunk(c)) => {
                    return Err(format!(
                        "expected phase {phase_idx} (Done) at frame index {idx}, got a \
                         chunk instead: {c}"
                    ));
                }
                None => {
                    return Err(format!(
                        "expected phase {phase_idx} (Done) at frame index {idx}, but the \
                         stream ended"
                    ));
                }
            },
            EventExpectation::RoleOpener => {
                let frame = frames.get(idx).ok_or_else(|| {
                    format!(
                        "expected phase {phase_idx} (RoleOpener) at frame index {idx}, but \
                         the stream ended"
                    )
                })?;
                match frame {
                    SseFrame::Chunk(c) if matches!(classify_chunk(c), ChunkKind::RoleOpener) => {
                        // Issue #828: dynamic /id and
                        // /created are type/prefix-checked on the opener
                        // chunk too, not only the non-streaming JSON
                        // baseline -- every `chat.completion.chunk` this
                        // binary emits carries both fields.
                        FieldExpectation::StringPrefix {
                            json_pointer: "/id",
                            prefix: "chatcmpl-",
                        }
                        .check(c)
                        .map_err(|e| {
                            format!("phase {phase_idx} (RoleOpener) chunk field check failed: {e}")
                        })?;
                        FieldExpectation::UnsignedInt {
                            json_pointer: "/created",
                        }
                        .check(c)
                        .map_err(|e| {
                            format!("phase {phase_idx} (RoleOpener) chunk field check failed: {e}")
                        })?;
                        idx += 1;
                    }
                    other => {
                        return Err(sse_phase_mismatch(phase_idx, "RoleOpener", idx, other));
                    }
                }
            }
            EventExpectation::Finish { finish_reason } => {
                let frame = frames.get(idx).ok_or_else(|| {
                    format!(
                        "expected phase {phase_idx} (Finish) at frame index {idx}, but the \
                         stream ended"
                    )
                })?;
                match frame {
                    SseFrame::Chunk(c) => match classify_chunk(c) {
                        ChunkKind::Finish {
                            finish_reason: actual,
                        } if actual == *finish_reason => {
                            idx += 1;
                        }
                        _ => {
                            return Err(format!(
                                "expected phase {phase_idx} (Finish {{ finish_reason: \
                                 \"{finish_reason}\" }}) at frame index {idx}, got: {c}"
                            ));
                        }
                    },
                    SseFrame::Done => {
                        return Err(sse_phase_mismatch(phase_idx, "Finish", idx, frame));
                    }
                }
            }
        }
    }
    if idx != frames.len() {
        return Err(format!(
            "expected exactly {idx} SSE frames but the stream carried {} \
             (trailing frames beyond every listed phase)",
            frames.len()
        ));
    }
    Ok(())
}

fn sse_phase_mismatch(phase_idx: usize, phase: &str, idx: usize, frame: &SseFrame) -> String {
    match frame {
        SseFrame::Chunk(c) => {
            format!("expected phase {phase_idx} ({phase}) at frame index {idx}, got chunk: {c}")
        }
        SseFrame::Done => format!(
            "expected phase {phase_idx} ({phase}) at frame index {idx}, got the [DONE] sentinel"
        ),
    }
}

/// A row's expected outcome for one binary (issue #828): the original
/// coarse `(status, error_code)` pair -- now [`ExpectedResponse::Error`] --
/// plus two richer variants for a successful response's actual JSON/SSE
/// shape. Every pre-#828 case keeps using `Error`; the shared table's
/// baseline/boundary rows below use `Json`/`Sse`.
#[derive(Debug, Clone, Copy)]
pub enum ExpectedResponse {
    /// A non-2xx error envelope: `{"error": {"code", ...}}`.
    Error { status: u16, code: &'static str },
    /// A 2xx JSON body, field-checked via `fields` (empty = status-only,
    /// e.g. `GET /`'s `root_body()` shape, which this table does not pin
    /// field-by-field).
    Json {
        status: u16,
        fields: &'static [FieldExpectation],
    },
    /// A 2xx SSE body, phase-checked via `events` (see [`check_sse_events`]).
    Sse {
        status: u16,
        events: &'static [EventExpectation],
    },
}

impl ExpectedResponse {
    pub fn status(&self) -> u16 {
        match self {
            ExpectedResponse::Error { status, .. } => *status,
            ExpectedResponse::Json { status, .. } => *status,
            ExpectedResponse::Sse { status, .. } => *status,
        }
    }
}

/// One row of the cross-binary HTTP parity table (ADR-080 C2): a request
/// `method`/`path`/`body`, and the expected outcome for each binary. A case
/// whose `divergence_reason` is `None` means both binaries must produce an
/// identical outcome for this request (the common case, post-alignment);
/// `Some` documents an intentional, reviewed per-binary difference -- an
/// undocumented divergence is exactly the drift this table exists to catch.
/// `method`/`path` were added after an unguarded `GET /` route removal on
/// `lattice.rs` left the table green: every case before that was implicitly
/// `POST /v1/chat/completions`, so route exposure itself was never actually
/// checked.
pub struct ParityCase {
    pub name: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub body: CaseBody,
    lattice: ExpectedResponse,
    lattice_serve: ExpectedResponse,
    /// `Some` only for a documented intentional divergence; explains WHY the
    /// two expected outcomes differ (reviewed alongside the table, not left
    /// to be inferred from the two variants).
    pub divergence_reason: Option<&'static str>,
}

impl ParityCase {
    pub fn expected(&self, binary: Binary) -> ExpectedResponse {
        match binary {
            Binary::Lattice => self.lattice,
            Binary::LatticeServe => self.lattice_serve,
        }
    }
}

/// Plain-data mirror of every [`GenerateConfig`] field (issue #828), so a
/// test can assert against an observed config without threading a whole
/// `GenerateConfig` -- whose `grammar: Option<Arc<GrammarEngine>>` field has
/// no `PartialEq`/`Eq` -- through assertion machinery. `has_grammar`
/// records only whether a grammar engine was attached, not its identity.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateConfigSnapshot {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,
    pub enable_thinking: bool,
    pub enable_mtp: Option<bool>,
    pub has_grammar: bool,
    pub stop_strings: Vec<String>,
    pub reasoning_budget: Option<usize>,
    pub logprobs: Option<usize>,
}

impl From<&GenerateConfig> for GenerateConfigSnapshot {
    fn from(cfg: &GenerateConfig) -> Self {
        GenerateConfigSnapshot {
            max_new_tokens: cfg.max_new_tokens,
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            repetition_penalty: cfg.repetition_penalty,
            seed: cfg.seed,
            stop_token_ids: cfg.stop_token_ids.clone(),
            enable_thinking: cfg.enable_thinking,
            enable_mtp: cfg.enable_mtp,
            has_grammar: cfg.grammar.is_some(),
            stop_strings: cfg.stop_strings.clone(),
            reasoning_budget: cfg.reasoning_budget,
            logprobs: cfg.logprobs,
        }
    }
}

/// A snapshot of exactly what one request handed a binary's production
/// generation adapter, captured from inside each binary's own deterministic
/// test-only generation seam (issue #828) -- strictly BELOW the real
/// request-parse/normalize/`build_cfg`-or-equivalent/handler/serialization
/// path, which still runs unmodified for every field this struct reports.
///
/// The two binaries' adapters receive genuinely different shapes at that
/// seam (`lattice.rs`'s CPU `generate`/`generate_streaming_with_cancel`
/// entry points take an already-rendered ChatML string; `lattice_serve.rs`'s
/// worker `generate` takes structured per-message data and renders ChatML
/// itself further downstream) -- exactly one of `rendered_prompt`/`messages`
/// is `Some` per capture, reflecting which shape that binary's real adapter
/// actually receives, not a missing capture.
#[derive(Debug, Clone)]
pub struct ProductionAdapterObservation {
    pub rendered_prompt: Option<String>,
    pub messages: Option<Vec<(String, String)>>,
    pub gen_cfg: GenerateConfigSnapshot,
    /// The rendered prompt's tokenized length, measured by the real
    /// tokenizer against the real rendered prompt (not a canned figure).
    pub prompt_tokens: usize,
    /// Whether the (canned) terminal outcome this capture's caller chose to
    /// report was an explicit stop condition (`true`) vs. exhausting the
    /// token budget (`false`) -- mirrors [`GenerateOutput::stopped`] /
    /// `Ev::Done`'s `stopped` field.
    pub stopped: bool,
}

/// The exact ChatML rendering both binaries' production code produces for a
/// single `{role: "user", content: "hi there"}` message -- `lattice.rs`'s
/// `render_prompt` and `lattice_serve.rs`'s (shared-engine)
/// `format_chat_template` use the identical
/// `"<|im_start|>{role}\n{content}<|im_end|>\n"` + trailing
/// `"<|im_start|>assistant\n"` template. A ground-truth literal, not a call
/// into either binary's render function, so a template regression in either
/// binary is visible against this fixture instead of round-tripping through
/// the same (possibly mutated) function that produced it (issue #828).
pub const OBSERVATION_GOLDEN_USER_HI_THERE_CHATML: &str =
    "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n";

/// Full expected value for a [`ProductionAdapterObservation`] (issue #828):
/// every `GenerateConfigSnapshot` field, the exact
/// rendered prompt or normalized message list, the exact prompt-token count,
/// and the terminal outcome -- one shared comparison both binaries' tests
/// call, instead of each asserting a different hand-picked subset of fields.
pub struct ExpectedObservation<'a> {
    pub gen_cfg: GenerateConfigSnapshot,
    pub rendered_prompt: Option<&'a str>,
    pub messages: Option<&'a [(&'a str, &'a str)]>,
    pub prompt_tokens: usize,
    pub stopped: bool,
}

/// Asserts every field of `obs` against `expected`, panicking with a
/// specific field name on the first mismatch (issue #828). Used by both
/// `lattice.rs`'s and `lattice_serve.rs`'s
/// `production_adapter_observation` test modules so neither binary can drift
/// back to asserting only a hand-picked subset of `GenerateConfigSnapshot`'s
/// thirteen fields.
pub fn assert_observation_matches(
    obs: &ProductionAdapterObservation,
    expected: &ExpectedObservation<'_>,
) {
    assert_eq!(
        obs.gen_cfg, expected.gen_cfg,
        "GenerateConfigSnapshot mismatch: observed {:?}, expected {:?}",
        obs.gen_cfg, expected.gen_cfg
    );
    assert_eq!(
        obs.rendered_prompt.as_deref(),
        expected.rendered_prompt,
        "rendered_prompt mismatch: observed {:?}, expected {:?}",
        obs.rendered_prompt,
        expected.rendered_prompt
    );
    let expected_messages: Option<Vec<(String, String)>> = expected.messages.map(|m| {
        m.iter()
            .map(|(r, c)| (r.to_string(), c.to_string()))
            .collect()
    });
    assert_eq!(
        obs.messages, expected_messages,
        "messages mismatch: observed {:?}, expected {:?}",
        obs.messages, expected_messages
    );
    assert_eq!(
        obs.prompt_tokens, expected.prompt_tokens,
        "prompt_tokens mismatch: observed {}, expected {}",
        obs.prompt_tokens, expected.prompt_tokens
    );
    assert_eq!(
        obs.stopped, expected.stopped,
        "stopped (terminal outcome) mismatch: observed {}, expected {}",
        obs.stopped, expected.stopped
    );
}

/// Shared fixture table for both binaries' `/v1/chat/completions` HTTP
/// contract, driven through each binary's real `Router` via
/// `tower::ServiceExt::oneshot` in `lattice.rs`'s and `lattice_serve.rs`'s
/// own test modules. Every case that ISN'T a documented divergence must
/// resolve to the SAME `(status, code)` on both binaries -- this table closes
/// concrete drift found across the two binaries: oversized body (413/`request_body_too_large`
/// vs 400/`invalid_request`), zero `max_tokens` (`invalid_max_tokens` vs
/// erased to `invalid_request`), and unknown role (generic message/no code
/// on one side).
pub const CHAT_COMPLETIONS_PARITY_CASES: &[ParityCase] = &[
    ParityCase {
        name: "unknown_role_not_openai",
        method: "POST",
        path: "/v1/chat/completions",
        // A trailing valid `user` turn keeps this isolated to the role
        // check: `lattice.rs` separately requires the conversation's LAST
        // message to have role `user` (a Qwen ChatML constraint, unrelated
        // to and checked before role-validity), so a single-message
        // `moderator` body would fail on THAT check first with
        // `invalid_messages` instead of exercising role validation at all.
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"moderator","content":"hi"},{"role":"user","content":"hi"}]}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_role",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "invalid_role",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "developer_role_unsupported_feature",
        method: "POST",
        path: "/v1/chat/completions",
        // See `unknown_role_not_openai`'s comment on the trailing `user` turn.
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"developer","content":"hi"},{"role":"user","content":"hi"}]}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "unsupported_feature",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "unsupported_feature",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "empty_messages",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(r#"{"model":"test-model","messages":[]}"#),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_messages",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "invalid_messages",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "max_tokens_zero",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_max_tokens",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "invalid_max_tokens",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "max_tokens_and_max_completion_tokens_conflict",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_request",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "invalid_request",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "tools_unsupported",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "unsupported_feature",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "unsupported_feature",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "malformed_json_body",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(r#"{"model":"test-model","messages":"#),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_request_body",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 400,
            code: "invalid_request_body",
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "max_tokens_over_cap_reject_vs_clamp",
        method: "POST",
        path: "/v1/chat/completions",
        // Both servers are configured (in each binary's own oneshot test
        // harness) with a small cap/context window; this body's max_tokens
        // exceeds it. `lattice.rs` rejects before ever touching the
        // model/worker; `lattice_serve.rs` clamps to the model's context
        // window in `build_cfg` and proceeds past validation entirely
        // (#745's triaged scope, kept deliberately unnified by
        // `reject_zero_max_tokens`'s doc comment). `lattice_serve`'s
        // expected (500, "internal_error") here is a harness artifact, not
        // real-server behavior: the router-level test fixture has no live
        // worker behind its job queue (matching this test module's existing
        // `test_app_state()` convention -- HTTP-level 400 tests only, no
        // GPU/model load), so a request that clears validation and reaches
        // `jobs.send(..)` fails there instead. The signal this case actually
        // proves is "not a 400 at the validation cascade" -- clamp-not-reject
        // -- which the diverging (500 vs 400) outcome demonstrates without
        // needing a real model.
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":999999}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "max_tokens_exceeds_limit",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 500,
            code: "internal_error",
        },
        divergence_reason: Some(
            "lattice.rs rejects max_tokens above its server cap at validation time; \
             lattice_serve.rs clamps the resolved value to the model's context \
             window and proceeds past validation instead of rejecting -- an \
             intentional per-binary policy difference, not drift (see \
             reject_zero_max_tokens's doc comment). The lattice_serve 500 here is \
             this router-level fixture's no-live-worker harness artifact once past \
             validation, not the divergence itself.",
        ),
    },
    ParityCase {
        name: "get_root_route_exposed",
        // ADR-080 C2, mutation-proven: every case above targets only `POST
        // /v1/chat/completions`, so removing `lattice.rs`'s `.route("/",
        // get(root))` entirely left the parity test green -- route exposure
        // itself was never actually checked. Both binaries must expose
        // `GET /` and return the shared `root_body()` shape (200; no error
        // envelope to check, so no error `code` is meaningful here).
        method: "GET",
        path: "/",
        body: CaseBody::Fixed(""),
        lattice: ExpectedResponse::Json {
            status: 200,
            fields: &[],
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: &[],
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "oversized_body_over_limit",
        // ADR-080 C2, mutation-proven: no case above sent a body over
        // `REQUEST_BODY_LIMIT_BYTES`, so restoring `lattice_serve.rs`'s old
        // 400/`invalid_request` oversized-body mapping (instead of the
        // current 413/`request_body_too_large`) also left the parity test
        // green. Both binaries enforce the same 1 MiB cap today (`lattice.rs`
        // via `DefaultBodyLimit`, `lattice_serve.rs` via a manual
        // `to_bytes(.., LIMIT)` check) and must report it identically.
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Oversized,
        lattice: ExpectedResponse::Error {
            status: 413,
            code: "request_body_too_large",
        },
        lattice_serve: ExpectedResponse::Error {
            status: 413,
            code: "request_body_too_large",
        },
        divergence_reason: None,
    },
    // -------------------------------------------------------------------
    // Field-level rows (issue #828): every case above only ever asserted
    // `(status, error_code)`, so a successful response's actual JSON/SSE
    // shape -- `object`, `model`, assistant role/content, `finish_reason`,
    // usage counts, SSE chunk ordering -- was never checked at all. Every
    // binary that runs this table drives these through a deterministic
    // test-only generation seam (canned content/token counts), never the
    // real request-parse/normalize/`build_cfg`-equivalent/handler path,
    // which stays exactly as exercised by every case above.
    // -------------------------------------------------------------------
    ParityCase {
        name: "baseline_non_streaming_200",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        lattice: ExpectedResponse::Json {
            status: 200,
            fields: &[
                FieldExpectation::StringPrefix {
                    json_pointer: "/id",
                    prefix: "chatcmpl-",
                },
                FieldExpectation::UnsignedInt {
                    json_pointer: "/created",
                },
                FieldExpectation::Eq {
                    json_pointer: "/object",
                    scalar: Scalar::Str("chat.completion"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/model",
                    scalar: Scalar::Str("test-model"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/choices/0/message/role",
                    scalar: Scalar::Str("assistant"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/choices/0/message/content",
                    scalar: Scalar::Str(BASELINE_CANNED_TEXT),
                },
                FieldExpectation::Eq {
                    json_pointer: "/choices/0/finish_reason",
                    scalar: Scalar::Str("stop"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/usage/prompt_tokens",
                    scalar: Scalar::U64(BASELINE_CANNED_PROMPT_TOKENS),
                },
                FieldExpectation::Eq {
                    json_pointer: "/usage/completion_tokens",
                    scalar: Scalar::U64(BASELINE_CANNED_COMPLETION_TOKENS),
                },
                FieldExpectation::Eq {
                    json_pointer: "/usage/total_tokens",
                    scalar: Scalar::U64(
                        BASELINE_CANNED_PROMPT_TOKENS + BASELINE_CANNED_COMPLETION_TOKENS,
                    ),
                },
            ],
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: &[
                FieldExpectation::StringPrefix {
                    json_pointer: "/id",
                    prefix: "chatcmpl-",
                },
                FieldExpectation::UnsignedInt {
                    json_pointer: "/created",
                },
                FieldExpectation::Eq {
                    json_pointer: "/object",
                    scalar: Scalar::Str("chat.completion"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/model",
                    scalar: Scalar::Str("test-model"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/choices/0/message/role",
                    scalar: Scalar::Str("assistant"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/choices/0/message/content",
                    scalar: Scalar::Str(BASELINE_CANNED_TEXT),
                },
                FieldExpectation::Eq {
                    json_pointer: "/choices/0/finish_reason",
                    scalar: Scalar::Str("stop"),
                },
                FieldExpectation::Eq {
                    json_pointer: "/usage/prompt_tokens",
                    scalar: Scalar::U64(BASELINE_CANNED_PROMPT_TOKENS),
                },
                FieldExpectation::Eq {
                    json_pointer: "/usage/completion_tokens",
                    scalar: Scalar::U64(BASELINE_CANNED_COMPLETION_TOKENS),
                },
                FieldExpectation::Eq {
                    json_pointer: "/usage/total_tokens",
                    scalar: Scalar::U64(
                        BASELINE_CANNED_PROMPT_TOKENS + BASELINE_CANNED_COMPLETION_TOKENS,
                    ),
                },
            ],
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "baseline_streaming_200",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        ),
        lattice: ExpectedResponse::Sse {
            status: 200,
            events: BASELINE_SSE_EVENTS,
        },
        lattice_serve: ExpectedResponse::Sse {
            status: 200,
            events: BASELINE_SSE_EVENTS,
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "temperature_boundary_zero_accepted",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"temperature":0.0}"#,
        ),
        lattice: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "temperature_boundary_two_accepted",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"temperature":2.0}"#,
        ),
        lattice: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "temperature_out_of_range_rejected",
        // `lattice.rs`'s `validate_temperature` enforces `[0.0, 2.0]`
        // (`crates/inference/src/bin/lattice.rs`); `lattice_serve.rs`'s
        // `build_cfg` has NO temperature range check at all -- a
        // caller-supplied value flows straight into `GenerateConfig`
        // unclamped. This is a genuine, pre-existing divergence (not
        // introduced by #828), pinned here the same way
        // `max_tokens_over_cap_reject_vs_clamp` pins its own divergence.
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"temperature":2.5}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_temperature",
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        divergence_reason: Some(
            "lattice.rs's validate_temperature rejects outside [0.0, 2.0]; \
             lattice_serve.rs's build_cfg has no temperature range check at \
             all and passes any client-supplied value straight into \
             GenerateConfig -- a pre-existing gap, not introduced by #828.",
        ),
    },
    ParityCase {
        name: "top_p_boundary_one_accepted",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"top_p":1.0}"#,
        ),
        lattice: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        divergence_reason: None,
    },
    ParityCase {
        name: "top_p_zero_rejected",
        // Same divergence shape as `temperature_out_of_range_rejected`:
        // `lattice.rs`'s `validate_top_p` enforces `(0.0, 1.0]`;
        // `lattice_serve.rs` has no top_p range check at all.
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"top_p":0.0}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_top_p",
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        divergence_reason: Some(
            "lattice.rs's validate_top_p rejects a top_p of 0.0 ((0.0, 1.0] \
             is half-open at zero); lattice_serve.rs's build_cfg has no \
             top_p range check at all -- a pre-existing gap, not introduced \
             by #828.",
        ),
    },
    ParityCase {
        name: "top_p_above_one_rejected",
        method: "POST",
        path: "/v1/chat/completions",
        body: CaseBody::Fixed(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"top_p":1.5}"#,
        ),
        lattice: ExpectedResponse::Error {
            status: 400,
            code: "invalid_top_p",
        },
        lattice_serve: ExpectedResponse::Json {
            status: 200,
            fields: ACCEPTED_MINIMAL_FIELDS,
        },
        divergence_reason: Some(
            "lattice.rs's validate_top_p rejects anything above 1.0; \
             lattice_serve.rs's build_cfg has no top_p range check at all \
             -- a pre-existing gap, not introduced by #828.",
        ),
    },
];

/// Canned non-streaming completion text/token counts every binary's
/// deterministic test-only generation seam returns for
/// `baseline_non_streaming_200` (issue #828). Arbitrary but fixed, so the
/// row's `Eq` field checks are exact-match, not shape-only.
pub const BASELINE_CANNED_TEXT: &str = "hello world";
pub const BASELINE_CANNED_PROMPT_TOKENS: u64 = 7;
pub const BASELINE_CANNED_COMPLETION_TOKENS: u64 = 2;

/// Ordered SSE phases every binary's deterministic streaming seam must
/// produce for `baseline_streaming_200`: role opener, one-or-more content
/// deltas, the finish chunk (canned `stopped: true` -> `"stop"`), then
/// `[DONE]`.
pub const BASELINE_SSE_EVENTS: &[EventExpectation] = &[
    EventExpectation::RoleOpener,
    EventExpectation::ContentDelta,
    EventExpectation::Finish {
        finish_reason: "stop",
    },
    EventExpectation::Done,
];

/// Minimal field list for a boundary row that only needs to prove
/// "request was accepted and a real chat-completion object came back", not
/// pin every field the way `baseline_non_streaming_200` does.
const ACCEPTED_MINIMAL_FIELDS: &[FieldExpectation] = &[FieldExpectation::Eq {
    json_pointer: "/object",
    scalar: Scalar::Str("chat.completion"),
}];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_stopped_true_is_stop() {
        assert_eq!(finish_reason(true), "stop");
    }

    #[test]
    fn finish_reason_stopped_false_is_length() {
        assert_eq!(finish_reason(false), "length");
    }

    #[test]
    fn reject_zero_max_tokens_rejects_zero() {
        let err = reject_zero_max_tokens(0).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_max_tokens",
                ..
            }
        ));
    }

    #[test]
    fn reject_zero_max_tokens_accepts_positive() {
        assert!(reject_zero_max_tokens(1).is_ok());
        assert!(reject_zero_max_tokens(4096).is_ok());
    }

    #[test]
    fn root_body_shape() {
        let body = root_body();
        assert_eq!(body["name"], "lattice");
        assert_eq!(body["object"], "engine");
        assert_eq!(
            body["endpoints"],
            serde_json::json!(["/v1/chat/completions", "/v1/models", "/health"])
        );
    }

    #[test]
    fn models_list_body_shape() {
        let body = models_list_body("my-model", 1_700_000_000);
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "my-model");
        assert_eq!(body["data"][0]["object"], "model");
        assert_eq!(body["data"][0]["created"], 1_700_000_000);
        assert_eq!(body["data"][0]["owned_by"], "lattice");
    }

    #[test]
    fn api_error_bad_request_envelope_shape() {
        let err = ApiError::BadRequest {
            message: "bad".to_string(),
            code: "some_code",
        };
        assert_eq!(err.message(), "bad");
        // IntoResponse is exercised at the HTTP layer in each binary's own
        // tests (axum::response::Response has no public body-introspection
        // API worth duplicating here); this pins the pure data this module
        // owns instead.
    }

    #[test]
    fn api_error_internal_message() {
        let err = ApiError::Internal {
            message: "oops".to_string(),
        };
        assert_eq!(err.message(), "oops");
    }

    #[test]
    fn api_error_payload_too_large_message() {
        let err = ApiError::PayloadTooLarge {
            message: "too big".to_string(),
        };
        assert_eq!(err.message(), "too big");
    }

    #[test]
    fn cancel_pair_receiver_starts_false() {
        let (_guard, rx) = cancel_pair();
        assert!(!*rx.borrow());
    }

    #[test]
    fn cancel_on_drop_flips_receiver_true_on_drop() {
        let (guard, rx) = cancel_pair();
        assert!(!*rx.borrow());
        drop(guard);
        assert!(*rx.borrow());
    }

    #[test]
    fn cancel_on_drop_leaves_receiver_false_while_alive() {
        let (guard, rx) = cancel_pair();
        assert!(!*rx.borrow());
        // Guard still in scope -- receiver must not have flipped yet.
        assert!(!*rx.borrow());
        drop(guard);
    }

    // -------------------------------------------------------------------
    // FieldExpectation / ExpectedResponse / check_sse_events (issue #828)
    // -------------------------------------------------------------------

    #[test]
    fn field_expectation_eq_matches_and_reports_mismatch() {
        let body = serde_json::json!({"object": "chat.completion", "usage": {"total_tokens": 9}});
        assert!(
            FieldExpectation::Eq {
                json_pointer: "/object",
                scalar: Scalar::Str("chat.completion"),
            }
            .check(&body)
            .is_ok()
        );
        assert!(
            FieldExpectation::Eq {
                json_pointer: "/usage/total_tokens",
                scalar: Scalar::U64(9),
            }
            .check(&body)
            .is_ok()
        );
        let err = FieldExpectation::Eq {
            json_pointer: "/object",
            scalar: Scalar::Str("chat.completion.chunk"),
        }
        .check(&body)
        .unwrap_err();
        assert!(err.contains("/object"), "error must name the field: {err}");
    }

    #[test]
    fn field_expectation_eq_missing_field_is_an_error() {
        let body = serde_json::json!({});
        let err = FieldExpectation::Eq {
            json_pointer: "/model",
            scalar: Scalar::Str("x"),
        }
        .check(&body)
        .unwrap_err();
        assert!(err.contains("absent"));
    }

    #[test]
    fn field_expectation_absent_passes_when_missing_fails_when_present() {
        let body = serde_json::json!({"logprobs": null});
        assert!(
            FieldExpectation::Absent {
                json_pointer: "/choices"
            }
            .check(&body)
            .is_ok()
        );
        // `Value::pointer` finds `null` -- present-but-null is still
        // "present" for this check (matches `#[serde(skip_serializing_if =
        // "Option::is_none")]`'s OMITTED contract, not a `null` literal).
        let err = FieldExpectation::Absent {
            json_pointer: "/logprobs",
        }
        .check(&body)
        .unwrap_err();
        assert!(err.contains("expected absent"));
    }

    #[test]
    fn field_expectation_array_len_checks_exact_length() {
        let body = serde_json::json!({"choices": [{"index": 0}]});
        assert!(
            FieldExpectation::ArrayLen {
                json_pointer: "/choices",
                len: 1,
            }
            .check(&body)
            .is_ok()
        );
        assert!(
            FieldExpectation::ArrayLen {
                json_pointer: "/choices",
                len: 2,
            }
            .check(&body)
            .is_err()
        );
    }

    fn sse_body(lines: &[&str]) -> String {
        lines.iter().map(|l| format!("data: {l}\n\n")).collect()
    }

    #[test]
    fn check_sse_events_accepts_well_formed_baseline_stream() {
        let body = sse_body(&[
            r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"hel"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);
        check_sse_events(&body, BASELINE_SSE_EVENTS).expect("well-formed stream must pass");
    }

    #[test]
    fn check_sse_events_requires_at_least_one_content_delta() {
        let body = sse_body(&[
            r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);
        let err = check_sse_events(&body, BASELINE_SSE_EVENTS).unwrap_err();
        assert!(err.contains("ContentDelta"), "error: {err}");
    }

    #[test]
    fn check_sse_events_rejects_wrong_finish_reason() {
        let body = sse_body(&[
            r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
            "[DONE]",
        ]);
        assert!(check_sse_events(&body, BASELINE_SSE_EVENTS).is_err());
    }

    #[test]
    fn check_sse_events_rejects_missing_done_sentinel() {
        let body = sse_body(&[
            r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let err = check_sse_events(&body, BASELINE_SSE_EVENTS).unwrap_err();
        assert!(
            err.contains("Done") || err.contains("stream ended"),
            "error: {err}"
        );
    }

    #[test]
    fn check_sse_events_rejects_role_opener_missing_id_or_created() {
        let missing_id = sse_body(&[
            r#"{"created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        ]);
        let err = check_sse_events(&missing_id, &[EventExpectation::RoleOpener]).unwrap_err();
        assert!(err.contains("/id"), "error: {err}");

        let missing_created = sse_body(&[
            r#"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        ]);
        let err = check_sse_events(&missing_created, &[EventExpectation::RoleOpener]).unwrap_err();
        assert!(err.contains("/created"), "error: {err}");
    }

    #[test]
    fn generate_config_snapshot_captures_every_field() {
        let cfg = GenerateConfig {
            max_new_tokens: 42,
            temperature: 1.3,
            top_k: 7,
            top_p: 0.55,
            repetition_penalty: 1.05,
            seed: Some(9),
            stop_token_ids: vec![100],
            enable_thinking: false,
            enable_mtp: Some(true),
            grammar: None,
            stop_strings: vec!["STOP".to_string()],
            reasoning_budget: Some(3),
            logprobs: Some(2),
        };
        let snapshot = GenerateConfigSnapshot::from(&cfg);
        assert_eq!(snapshot.max_new_tokens, 42);
        assert_eq!(snapshot.temperature, 1.3);
        assert_eq!(snapshot.top_k, 7);
        assert_eq!(snapshot.top_p, 0.55);
        assert_eq!(snapshot.repetition_penalty, 1.05);
        assert_eq!(snapshot.seed, Some(9));
        assert_eq!(snapshot.stop_token_ids, vec![100]);
        assert!(!snapshot.enable_thinking);
        assert_eq!(snapshot.enable_mtp, Some(true));
        assert!(!snapshot.has_grammar);
        assert_eq!(snapshot.stop_strings, vec!["STOP".to_string()]);
        assert_eq!(snapshot.reasoning_budget, Some(3));
        assert_eq!(snapshot.logprobs, Some(2));
    }

    #[test]
    fn chat_completions_parity_cases_expected_status_matches_variant() {
        // Every case's declared status must agree with its own variant --
        // a cheap sanity check that catches a copy-paste status/variant
        // mismatch in the const table itself, independent of any HTTP call.
        for case in CHAT_COMPLETIONS_PARITY_CASES {
            for binary in [Binary::Lattice, Binary::LatticeServe] {
                let expected = case.expected(binary);
                match expected {
                    ExpectedResponse::Error { status, .. } => assert!(
                        !(200..300).contains(&status),
                        "case '{}': Error variant must not carry a 2xx status",
                        case.name
                    ),
                    ExpectedResponse::Json { status, .. }
                    | ExpectedResponse::Sse { status, .. } => {
                        assert_eq!(
                            expected.status(),
                            status,
                            "case '{}': status() must match the variant's own status",
                            case.name
                        );
                    }
                }
            }
        }
    }
}
